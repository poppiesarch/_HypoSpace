import sys
import json
import argparse
import os
import re
import yaml
import random
import numpy as np
import ast
import sympy as sp
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any
from datetime import datetime
from textwrap import dedent
from collections import Counter, defaultdict
from scipy import stats
import traceback

# Add parent directory to path if needed
sys.path.append(str(Path(__file__).parent))

from modules.llm_interface import LLMInterface, OpenRouterLLM, OpenAILLM, AnthropicLLM
from boolean_dataset import BooleanExpression, BooleanObservation

###########################################################################
# ACE Playbook: Cross-basic/extended/full semantic notes + dynamic hints  #
###########################################################################
class ACEPlaybook:
    """
    A lightweight ACE-style prompt augmenter that:
    - Provides cross-dataset semantic notes (hard constraints reminder)
    - Generates dynamic suggestions based on last round errors and coverage state
    It does not change metrics definitions and keeps output format untouched.
    """
    def __init__(self, variables: List[str], operators: Set[str], max_depth: int):
        self.variables = variables
        self.operators = operators
        self.max_depth = max_depth
        # Keep only the last few dynamic suggestions to avoid prompt bloat
        self.dynamic_suggestions: List[str] = []
        self.max_suggestions = 3

    def _semantic_notes(self) -> List[str]:
        # Cross-dataset, operator-agnostic hard constraints reminders
        notes = [
            f"Use ONLY variables: {', '.join(self.variables)}",
            f"Use ONLY allowed operators: {', '.join(sorted(self.operators))}",
            f"Expression must match ALL given observations",
            f"Expression depth ≤ {self.max_depth} (count by nesting of operations/parentheses)",
            "Do NOT use boolean constants True/False or numeric literals",
            "Return exactly one line starting with 'Expression: ' followed by the expression",
            "Avoid LaTeX/markdown/symbolic glyphs; use uppercase operators, lowercase variables"
        ]
        return notes

    def _truncate_push(self, text: str):
        self.dynamic_suggestions.append(text)
        if len(self.dynamic_suggestions) > self.max_suggestions:
            self.dynamic_suggestions = self.dynamic_suggestions[-self.max_suggestions:]

    def reflect_and_update(
        self,
        last_prompt: str,
        last_response: Optional[str],
        parsed_expr: Optional[str],
        in_space: bool,
        valid_against_obs: bool,
        error_type: Optional[str],
        seen_mechanistic_keys: Set[Tuple]
    ):
        """
        Update dynamic suggestions based on last round signals.
        error_type: classified error name or None
        seen_mechanistic_keys: mechanistic keys already explored in-space (for novelty guidance)
        """
        # Parse/format errors
        if error_type:
            et = error_type.lower()
            if "json" in et or "parse" in et:
                self._truncate_push("Strictly follow the output format: a single line starting with 'Expression: ' and nothing else.")
            elif "variable" in et:
                self._truncate_push("Use ONLY the allowed variables; remove any unknown symbols or constants.")
            elif "operator" in et or "forbidden" in et:
                self._truncate_push("Use ONLY the allowed operators; replace any unsupported operator with an allowed one.")
            elif "constant" in et:
                self._truncate_push("Do NOT use True/False constants or numeric literals in the expression.")
            elif "depth" in et:
                self._truncate_push(f"Reduce nesting to keep depth ≤ {self.max_depth}; simplify parentheses and avoid redundant NOT chains.")
            elif "timeout" in et or "rate_limit" in et:
                # Non-behavioral; avoid spamming the prompt
                pass

        # Space violation without a specific error type
        if parsed_expr and not in_space:
            self._truncate_push("Ensure the expression stays within the space: allowed variables/operators only, no constants, and depth within limit.")

        # Mismatch with observations
        if parsed_expr and in_space and not valid_against_obs:
            # Cross-dataset discriminative guidance
            self._truncate_push(
                "Use a discriminative plan: ensure consistency with observed points; if multiple candidates remain, mentally check unobserved inputs in a fixed order (e.g., lexicographic) and prefer expressions that differentiate those points the most."
            )

        # Novelty encouragement against mechanistic duplicates
        if seen_mechanistic_keys:
            self._truncate_push("Avoid mechanisms equivalent to previously attempted ones; do not only re-parenthesize or reorder the same operator. Change the operator mix or decision structure meaningfully.")

    def get_prompt_blocks(self) -> Tuple[str, str]:
        semantic_notes = self._semantic_notes()
        sem_block = "- " + "\n- ".join(semantic_notes)
        sug_block = "None" if not self.dynamic_suggestions else "- " + "\n- ".join(self.dynamic_suggestions)
        return sem_block, sug_block


#############################################################
# Self-consistency majority vote within a single LLM query  #
#############################################################
class SelfConsistencyModule:
    """
    For one logical query, sample multiple completions from the LLM and select
    a single final expression by majority vote over mechanistic keys.
    - Keeps per-query output cardinality = 1 (compatible with existing metrics)
    """
    def __init__(self, num_samples: int = 5, temperature: float = 0.9):
        self.num_samples = max(1, int(num_samples))
        self.temperature = temperature

    def query_with_vote(
        self,
        llm: LLMInterface,
        prompt: str,
        parse_fn,
        in_space_fn,
        mech_key_fn,
        validate_fn
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Returns:
          - chosen expression string or None
          - usage dict: prompt_tokens, completion_tokens, total_tokens, cost
        """
        candidates: List[str] = []
        mech_keys: List[Optional[Tuple]] = []
        valid_flags: List[bool] = []
        usage_aggr = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost": 0.0
        }

        # Temporarily override llm temperature if supported
        set_temp = getattr(llm, "set_temperature", None)
        orig_temp = getattr(llm, "temperature", None)
        if callable(set_temp) and self.temperature is not None:
            try:
                set_temp(self.temperature)
            except:
                pass

        try:
            for _ in range(self.num_samples):
                if hasattr(llm, 'query_with_usage'):
                    result = llm.query_with_usage(prompt)
                    response = result['response']
                    usage_aggr["prompt_tokens"] += result['usage']['prompt_tokens']
                    usage_aggr["completion_tokens"] += result['usage']['completion_tokens']
                    usage_aggr["total_tokens"] += result['usage']['total_tokens']
                    usage_aggr["cost"] += result.get('cost', 0.0)
                else:
                    response = llm.query(prompt)

                if isinstance(response, str) and response.startswith("Error querying"):
                    # do not early stop; continue sampling to allow others to succeed
                    continue

                expr = parse_fn(response)
                if not expr:
                    candidates.append(None)  # keep position for tally
                    mech_keys.append(None)
                    valid_flags.append(False)
                    continue

                # in-space filter
                in_space = in_space_fn(expr)
                if not in_space:
                    candidates.append(None)
                    mech_keys.append(None)
                    valid_flags.append(False)
                    continue

                # validate against observations inside validate_fn
                is_valid, _ = validate_fn(expr)
                candidates.append(expr)
                mk = mech_key_fn(expr)
                mech_keys.append(mk)
                valid_flags.append(bool(is_valid))

            # Majority voting over mechanistic keys among valid ones; fallback to any in-space
            counter = Counter()
            for mk, ok in zip(mech_keys, valid_flags):
                if mk is not None and ok:
                    counter[mk] += 1

            chosen_expr = None
            if counter:
                # choose mk with highest count; pick first candidate with that mk and valid
                best_mk, _ = counter.most_common(1)[0]
                for expr, mk, ok in zip(candidates, mech_keys, valid_flags):
                    if ok and mk == best_mk and expr is not None:
                        chosen_expr = expr
                        break
            else:
                # fallback: choose any in-space candidate (even if invalid) with most common mk
                counter_any = Counter([mk for mk in mech_keys if mk is not None])
                if counter_any:
                    best_mk, _ = counter_any.most_common(1)[0]
                    for expr, mk in zip(candidates, mech_keys):
                        if expr is not None and mk == best_mk:
                            chosen_expr = expr
                            break
                else:
                    # last resort: first non-empty expr
                    for expr in candidates:
                        if expr:
                            chosen_expr = expr
                            break

            return chosen_expr, usage_aggr
        finally:
            if callable(set_temp) and orig_temp is not None:
                try:
                    set_temp(orig_temp)
                except:
                    pass


#########################################
# Refined Boolean Benchmark (merged)    #
#########################################
class BooleanBenchmarkRefined:
    def __init__(self, complete_dataset_path: str, enable_ace: bool = True, enable_self_consistency: bool = True, sc_num_samples: int = 5, sc_temperature: float = 0.9):
        """
        Initialize benchmark with a complete dataset.
        
        Args:
            complete_dataset_path: Path to the complete Boolean dataset JSON file
            enable_ace: whether to enable ACE prompt augmentation
            enable_self_consistency: whether to enable self-consistency module
            sc_num_samples: samples per query when self-consistency enabled
            sc_temperature: temperature used for self-consistency sampling
        """
        with open(complete_dataset_path, 'r') as f:
            self.complete_dataset = json.load(f)
        
        # Extract metadata
        self.mechanistic_opts = self.complete_dataset['metadata']['mechanistic_opts']
        self.metadata = self.complete_dataset['metadata']
        self.variables = self.metadata['variables']
        self.operators = set(self.metadata['operators'])
        self.max_depth = self.metadata['max_depth']
        
        # Flatten all datasets into a single list for sampling
        self.all_observation_sets = []
        for n_obs, datasets in self.complete_dataset['datasets_by_n_observations'].items():
            self.all_observation_sets.extend(datasets)
        
        # ACE and SC modules
        self.enable_ace = enable_ace
        self.ace = ACEPlaybook(self.variables, self.operators, self.max_depth) if enable_ace else None
        self.enable_self_consistency = enable_self_consistency
        self.sc = SelfConsistencyModule(num_samples=sc_num_samples, temperature=sc_temperature) if enable_self_consistency else None

        # Basic/extended/full profile (for logs only)
        self.profile = self._infer_profile(self.operators)

        print(f"Loaded complete dataset with {len(self.all_observation_sets)} observation sets")
        print(f"Variables: {', '.join(self.variables)}")
        print(f"Operators: {', '.join(self.operators)}")
        print(f"Max depth: {self.max_depth}")
        print(f"Profile: {self.profile}")
        print(f"ACE: {'on' if self.enable_ace else 'off'}, Self-consistency: {'on' if self.enable_self_consistency else 'off'}")
    
    def _infer_profile(self, ops: Set[str]) -> str:
        # Heuristic tag
        base = {"AND", "OR"}
        if ops == base:
            return "basic"
        if "NOT" in ops or "XOR" in ops or "NOR" in ops:
            return "extended/full"
        return "custom"
    
    def sample_observation_sets(self, n_samples: int, seed: Optional[int] = None) -> List[Dict]:
        """
        Sample n observation sets from the complete dataset.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Sample without replacement
        n_available = len(self.all_observation_sets)
        n_to_sample = min(n_samples, n_available)
        
        if n_to_sample < n_samples:
            print(f"Warning: Requested {n_samples} samples but only {n_available} available")
        
        sampled = random.sample(self.all_observation_sets, n_to_sample)
        return sampled

    def create_prompt(self, observations: List[BooleanObservation], 
                     prior_hypotheses: List[str]) -> str:
        # Build obs_block, prior_block, operators_str ...
        # Format observations
        obs_lines = []
        for obs in observations:
            obs_lines.append(obs.to_string())
        obs_block = "\n".join(obs_lines)
        
        # Format prior hypotheses
        if prior_hypotheses:
            prior_block = "\n".join([f"Expression: {h}" for h in prior_hypotheses])
        else:
            prior_block = "None"
        
        # Build operator description
        op_descriptions = []
        if 'AND' in self.operators: op_descriptions.append("AND (conjunction)")
        if 'OR'  in self.operators: op_descriptions.append("OR (disjunction)")
        if 'NOT' in self.operators: op_descriptions.append("NOT (negation)")
        if 'XOR' in self.operators: op_descriptions.append("XOR (exclusive or)")
        if 'NOR' in self.operators: op_descriptions.append("NOR (NOT (x OR y))")
        
        operators_str = ", ".join(op_descriptions)

        # Pull mechanistic opts from the loaded dataset
        mech = self.complete_dataset.get('metadata', {}).get('mechanistic_opts', {})
        comm = mech.get('apply_commutativity', True)
        idem = mech.get('apply_idempotence_and_or', True)
        flat = mech.get('flatten_associativity', True)

        eq_rules = []
        if comm:
            eq_rules.append(
                "- Commutativity: reorderings are the SAME (e.g., x AND y = y AND x)."
            )
        if idem:
            eq_rules.append(
                "- Idempotence: duplicates of the same input under AND/OR collapse (e.g., x AND x = x; x OR x = x)."
            )
        if flat:
            eq_rules.append(
                "- Associativity flattening: cascades of the SAME operator are the SAME regardless of parentheses "
                "(e.g., (x AND y) AND x = x AND (y AND x) = x AND x AND y)."
            )
        else:
            eq_rules.append(
                "- No associativity flattening: different parenthesizations of the SAME operator are DIFFERENT "
                "(e.g., (x AND y) AND x ≠ x AND (y AND x))."
            )

        eq_rules.append(
            "- No distributivity, absorption, or De Morgan normalization: mixing operators keeps expressions DIFFERENT "
            "(e.g., (x AND y) OR z ≠ x AND (y OR z))."
        )

        rules_text = "\n        ".join(eq_rules)

        # ACE blocks
        ace_sem_block = ""
        ace_sug_block = ""
        if self.enable_ace and self.ace:
            sem_block, sug_block = self.ace.get_prompt_blocks()
            ace_sem_block = f"""
            ACE Semantic Notes:
            {sem_block}
            """.rstrip()
            ace_sug_block = f"""
            ACE Suggestions (dynamic, optional to follow for better coverage/novelty):
            {sug_block}
            """.rstrip()

        prompt = f"""You are given partial observations of a Boolean function with variables: {', '.join(self.variables)}

            Allowed operators: {operators_str}

            Observations (input -> output):
            {obs_block}

            Prior expressions generated (avoid repeating any expression 
            that is equivalent under the rules below):
            {prior_block}

            {ace_sem_block}

            {ace_sug_block}

            Task: Generate a single Boolean expression that is consistent with ALL observations.

            Requirements:
            1. Use ONLY the variables: {', '.join(self.variables)}
            2. Use ONLY these operators: {', '.join(self.operators)}
            3. The expression must match all given observations
            4. Structural uniqueness is judged by these rules:
            {rules_text}
            5. Expression depth should be at most {self.max_depth} levels of nesting
            6. Do not use boolean constants True or False anywhere in the expression.

            Output format: 
            - Return ONLY the Boolean expression on a single line
            - Use plain text format (no LaTeX, no markdown, no special formatting)
            - Use uppercase for operators. 
            - Use lowercase for variables: {', '.join(self.variables)}
            - Use parentheses for grouping when needed
            - Start your response with "Expression: " followed by the expression
            
            Examples of correct format:
            Expression: x AND y
            Expression: (x OR y) AND NOT x
            Expression: NOT (x AND y)
            Expression: NOR(x, y)
            
            DO NOT use formats like:
            - \\(x \\land y\\)  (LaTeX)
            - `x AND y`  (markdown)
            - x ∧ y  (mathematical symbols)
            """
        return prompt
    
    def parse_llm_response(self, response: str) -> Optional[str]:
        """Parse LLM response to extract Boolean expression."""
        if not isinstance(response, str):
            return None
        
        response = response.strip()
        
        # Clean common formatting issues
        def clean_expression(expr: str) -> str:
            # Remove LaTeX delimiters
            expr = re.sub(r'\\[()\[\]]', '', expr)
            expr = re.sub(r'\$+', '', expr)
            
            # Remove markdown backticks
            expr = expr.strip('`')
            
            # Convert LaTeX/math symbols to text
            expr = expr.replace('\\land', 'AND')
            expr = expr.replace('\\lor', 'OR')
            expr = expr.replace('\\lnot', 'NOT')
            expr = expr.replace('\\neg', 'NOT')
            expr = expr.replace('\\wedge', 'AND')
            expr = expr.replace('\\vee', 'OR')
            expr = expr.replace('∧', 'AND')
            expr = expr.replace('∨', 'OR')
            expr = expr.replace('¬', 'NOT')
            expr = expr.replace('⊕', 'XOR')
            
            # Remove \text{} wrappers
            expr = re.sub(r'\\text\{([^}]+)\}', r'\1', expr)
            
            # Clean up extra spaces and symbols
            expr = re.sub(r'\*+$', '', expr)  # Remove trailing asterisks
            expr = expr.strip('"\'').rstrip('.,;')
            
            return expr.strip()
        
        # Look for "Expression:" line
        lines = response.split('\n')
        for line in lines:
            if 'Expression:' in line or 'expression:' in line:
                parts = re.split(r'[Ee]xpression:', line)
                if len(parts) >= 2:
                    expr = clean_expression(parts[1])
                    if expr:
                        return expr
        
        # Fallback: try first non-empty line with operators
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                cleaned = clean_expression(line)
                if any(op in cleaned.upper() for op in ['AND','OR','NOT','XOR','NOR']):
                    return cleaned
                if all(c.isalnum() or c in '() ' for c in cleaned):
                    return cleaned
        
        return None

    def get_expression_mechanistic_key(self, expr_str: str) -> Optional[Tuple]:
        try:
            expr = BooleanExpression(expr_str, self.variables, self.operators)
            return expr.mechanistic_key(**self.mechanistic_opts)
        except:
            return None
    
    def _in_space(self, expr_str: str) -> bool:
        """Check if expression meets all space constraints (variables, operators, depth, no constants)."""
        try:
            expr = BooleanExpression(expr_str, self.variables, self.operators)
            
            if expr.sympy_expr is None:
                return False
            
            # Check variables
            allowed_symbols = {sp.symbols(v) for v in self.variables}
            if not expr.sympy_expr.free_symbols.issubset(allowed_symbols):
                return False
            
            # Check operators
            if not self._check_operators_in_ast(expr.sympy_expr, self.operators):
                return False
            
            # Check no constants
            for node in sp.preorder_traversal(expr.sympy_expr):
                if isinstance(node, (sp.logic.boolalg.BooleanTrue, sp.logic.boolalg.BooleanFalse)):
                    return False
            
            # Check depth
            if self._get_ast_depth(expr.sympy_expr) > self.max_depth:
                return False
            
            return True
        except:
            return False
    
    def _get_ast_depth(self, sympy_expr) -> int:
        """Calculate the true AST depth of an expression."""
        if sympy_expr is None:
            return 0
        
        # Leaf nodes (variables, constants) have depth 0
        if isinstance(sympy_expr, sp.Symbol):
            return 0
        if isinstance(sympy_expr, (sp.logic.boolalg.BooleanTrue, sp.logic.boolalg.BooleanFalse)):
            return 0
        
        # For operators, depth is 1 + max depth of children
        if hasattr(sympy_expr, 'args') and len(sympy_expr.args) > 0:
            child_depths = [self._get_ast_depth(arg) for arg in sympy_expr.args]
            return 1 + max(child_depths)
        
        return 0
    
    def _check_operators_in_ast(self, sympy_expr, allowed_ops: Set[str]) -> bool:
        if sympy_expr is None:
            return False
        stack = [sympy_expr]
        while stack:
            node = stack.pop()

            # NOR pattern: Not(Or(...))
            if isinstance(node, sp.Not) and isinstance(node.args[0], sp.Or):
                if 'NOR' not in allowed_ops:
                    return False
                # treat as primitive; do NOT traverse inside
                continue

            # primitive checks
            if isinstance(node, sp.Not) and 'NOT' not in allowed_ops:
                return False
            elif isinstance(node, sp.And) and 'AND' not in allowed_ops:
                return False
            elif isinstance(node, sp.Or) and 'OR' not in allowed_ops:
                return False
            elif isinstance(node, sp.Xor) and 'XOR' not in allowed_ops:
                return False
            elif isinstance(node, sp.Implies) and 'IMPLIES' not in allowed_ops:
                return False
            elif isinstance(node, sp.Equivalent) and 'EQUIV' not in allowed_ops:
                return False

            if hasattr(node, 'args'):
                stack.extend(node.args)
        return True
    
    def validate_expression(self, expr_str: str, observations: List[BooleanObservation]) -> Tuple[bool, Optional[Dict]]:
        """Validate if expression is consistent with observations and meets constraints."""
        try:
            # Now create expression and validate
            expr = BooleanExpression(expr_str, self.variables, self.operators)
            
            # Check if expression uses only allowed variables
            if expr.sympy_expr is not None:
                allowed_symbols = {sp.symbols(v) for v in self.variables}
                free_symbols = expr.sympy_expr.free_symbols
                if not free_symbols.issubset(allowed_symbols):
                    return False, None
                
                # Check operators using AST traversal
                if not self._check_operators_in_ast(expr.sympy_expr, self.operators):
                    return False, None
                
                # Disallow constants (True/False) to maintain consistent space
                for node in sp.preorder_traversal(expr.sympy_expr):
                    if isinstance(node, (sp.logic.boolalg.BooleanTrue, sp.logic.boolalg.BooleanFalse)):
                        return False, None
            
            # Check expression depth using AST
            if expr.sympy_expr is not None:
                ast_depth = self._get_ast_depth(expr.sympy_expr)
                if ast_depth > self.max_depth:
                    return False, None
            
            # Check consistency with observations
            for obs in observations:
                if expr.evaluate(obs.inputs) != obs.output:
                    return False, None
            
            return True, expr.truth_table
        except:
            return False, None
    
    def _classify_error(self, error_message: str) -> str:
        """Classify the type of error from the error message with more detail."""
        if "Expecting value" in error_message:
            # Extract line and column info if available
            match = re.search(r'line (\d+) column (\d+)', error_message)
            if match:
                return f"json_parse_error (line {match.group(1)}, col {match.group(2)})"
            return "json_parse_error"
        elif "Rate limit" in error_message.lower():
            return "rate_limit"
        elif "timeout" in error_message.lower():
            return "timeout"
        elif "401" in error_message or "unauthorized" in error_message.lower():
            return "auth_error"
        elif "403" in error_message or "forbidden" in error_message.lower():
            return "forbidden_error"
        elif "404" in error_message:
            return "not_found_error"
        elif "429" in error_message:
            return "rate_limit_429"
        elif "500" in error_message or "internal server error" in error_message.lower():
            return "server_error_500"
        elif "502" in error_message or "bad gateway" in error_message.lower():
            return "bad_gateway_502"
        elif "503" in error_message or "service unavailable" in error_message.lower():
            return "service_unavailable_503"
        elif "connection" in error_message.lower():
            return "connection_error"
        elif "JSONDecodeError" in error_message:
            return "json_decode_error"
        else:
            # Try to extract HTTP status code
            match = re.search(r'\b(\d{3})\b', error_message)
            if match:
                return f"http_error_{match.group(1)}"
            return "unknown_error"
    
    def get_expression_depth(self, expr_str: str) -> int:
        """Calculate the depth of a Boolean expression."""
        # Simple heuristic: count maximum nesting level of parentheses
        max_depth = 0
        current_depth = 0
        
        for char in expr_str:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth -= 1
        
        # Also check for operators without parentheses
        if max_depth == 0:
            # Count operators to estimate depth
            expr_upper = expr_str.upper()
            if any(op in expr_upper for op in ['AND','OR','NOT','XOR','NOR']):
                max_depth = 1
        
        return max_depth
    
    def evaluate_single_observation_set(
        self,
        llm: LLMInterface,
        observation_set: Dict,
        n_queries: int = 10,
        verbose: bool = True,
        max_retries: int = 5
    ) -> Dict:
        """
        Evaluate LLM on a single observation set.
        
        Returns:
            Dictionary with evaluation results for this observation set
        """
        # Extract observations and ground truths
        observations = [
            BooleanObservation(obs['inputs'], obs['output']) 
            for obs in observation_set['observations']
        ]
        
        ground_truth_expressions = observation_set['ground_truth_expressions']
        gt_truth_tables = []
        gt_mechanistic_keys = []
        for gt in ground_truth_expressions:
            truth_table = {}
            for key_str, value in gt['truth_table'].items():
                key_tuple = ast.literal_eval(key_str)
                truth_table[key_tuple] = value
            gt_truth_tables.append(truth_table)
            # Extract mechanistic key for structural comparison
            gt_mech_key = gt.get('mechanistic_key', None)
            if gt_mech_key:
                # Convert string representation back to tuple if needed (safe parsing)
                gt_mechanistic_keys.append(ast.literal_eval(gt_mech_key) if isinstance(gt_mech_key, str) else gt_mech_key)
        
        # Track results
        all_hypotheses = []
        valid_hypotheses = []
        unique_mechanistic_keys = set()
        unique_valid_expressions = []
        all_unique_mechanistic_keys = set()
        unique_all_expressions = []
        parse_success_count = 0
        in_space_count = 0
        
        # Track token usage and costs
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_cost = 0.0
        
        # Track errors
        errors = []  # List of error details
        error_counts = {}  # Count of each error type

        # Keep set of seen mechanistic keys for ACE novelty guidance
        seen_mech_keys_in_space: Set[Tuple] = set()

        for i in range(n_queries):
            prompt = self.create_prompt(observations, all_hypotheses)
            
            # Try to get a valid response
            hypothesis_str = None
            query_error = None

            # Self-consistency path
            if self.enable_self_consistency and self.sc is not None:
                def parse_fn(resp): 
                    return self.parse_llm_response(resp)
                def in_space_fn(expr):
                    return self._in_space(expr)
                def mech_key_fn(expr):
                    return self.get_expression_mechanistic_key(expr)
                def validate_fn(expr):
                    return self.validate_expression(expr, observations)

                chosen_expr, usage = self.sc.query_with_vote(
                    llm, prompt, parse_fn, in_space_fn, mech_key_fn, validate_fn
                )

                # Aggregate usage
                total_prompt_tokens += usage.get("prompt_tokens", 0)
                total_completion_tokens += usage.get("completion_tokens", 0)
                total_tokens += usage.get("total_tokens", 0)
                total_cost += usage.get("cost", 0.0)

                hypothesis_str = chosen_expr

                # If we failed to get any expression, mark an error for visibility
                if hypothesis_str is None:
                    query_error = {
                        'query_index': i,
                        'attempt': 1,
                        'error_message': "self_consistency_failed_to_select",
                        'error_type': "selection_failure"
                    }
                    error_type = query_error['error_type']
                    error_counts[error_type] = error_counts.get(error_type, 0) + 1
                    errors.append(query_error)

            else:
                # Original retry loop path
                for attempt in range(max_retries):
                    # Use query_with_usage if available
                    if hasattr(llm, 'query_with_usage'):
                        result = llm.query_with_usage(prompt)
                        response = result['response']
                        # Track usage
                        total_prompt_tokens += result['usage']['prompt_tokens']
                        total_completion_tokens += result['usage']['completion_tokens']
                        total_tokens += result['usage']['total_tokens']
                        total_cost += result.get('cost', 0.0)
                    else:
                        response = llm.query(prompt)
                    
                    # Check if response is an error
                    if isinstance(response, str) and response.startswith("Error querying"):
                        query_error = {
                            'query_index': i,
                            'attempt': attempt + 1,
                            'error_message': response,
                            'error_type': self._classify_error(response)
                        }
                        # Track error type count
                        error_type = query_error['error_type']
                        error_counts[error_type] = error_counts.get(error_type, 0) + 1
                        continue  # Try again
                    
                    hypothesis_str = self.parse_llm_response(response)
                    if hypothesis_str:
                        break

                # If all attempts failed, record the error
                if not hypothesis_str and query_error:
                    errors.append(query_error)
                    # Store detailed error info in all_hypotheses
                    error_detail = f"[ERROR after {max_retries} attempts] Type: {query_error['error_type']} | {query_error['error_message']}"
                    all_hypotheses.append(error_detail)
                    # Reflect to ACE
                    if self.enable_ace and self.ace:
                        self.ace.reflect_and_update(
                            last_prompt=prompt,
                            last_response=None,
                            parsed_expr=None,
                            in_space=False,
                            valid_against_obs=False,
                            error_type=query_error['error_type'],
                            seen_mechanistic_keys=seen_mech_keys_in_space
                        )
                    continue  # next query

            if hypothesis_str:
                parse_success_count += 1
                all_hypotheses.append(hypothesis_str)
                
                # Only count novelty for in-space expressions
                in_space_ok = self._in_space(hypothesis_str)
                if in_space_ok:
                    in_space_count += 1
                    
                    # Check uniqueness among in-space hypotheses (for novelty calculation)
                    all_mech_key = self.get_expression_mechanistic_key(hypothesis_str)
                    if all_mech_key and all_mech_key not in all_unique_mechanistic_keys:
                        all_unique_mechanistic_keys.add(all_mech_key)
                        unique_all_expressions.append(hypothesis_str)
                        # record for ACE novelty guidance
                        seen_mech_keys_in_space.add(all_mech_key)
                
                # Validate expression against observations
                is_valid, truth_table = self.validate_expression(hypothesis_str, observations)
                
                if is_valid:
                    valid_hypotheses.append(hypothesis_str)
                    
                    # Check uniqueness among valid hypotheses
                    mech_key = self.get_expression_mechanistic_key(hypothesis_str)
                    if mech_key and mech_key not in unique_mechanistic_keys:
                        unique_mechanistic_keys.add(mech_key)
                        unique_valid_expressions.append(hypothesis_str)

                # Reflect to ACE with signals
                if self.enable_ace and self.ace:
                    self.ace.reflect_and_update(
                        last_prompt=prompt,
                        last_response=None,
                        parsed_expr=hypothesis_str,
                        in_space=in_space_ok,
                        valid_against_obs=bool(is_valid),
                        error_type=None,
                        seen_mechanistic_keys=seen_mech_keys_in_space
                    )

        # Calculate metrics
        n_queries_eff = n_queries if n_queries > 0 else 1
        parse_success_rate = parse_success_count / n_queries_eff
        in_space_rate = in_space_count / n_queries_eff
        valid_rate = len(valid_hypotheses) / n_queries_eff
        novelty_rate = len(unique_all_expressions) / n_queries_eff
        recovery_rate = 0
        
        # Check recovery against ground truths using mechanistic keys (structural matching)
        recovered_gts = set()
        recovered_gt_keys = set()
        
        # Collect mechanistic keys of generated expressions
        generated_mech_keys = set()
        for expr in unique_valid_expressions:
            mech_key = self.get_expression_mechanistic_key(expr)
            if mech_key:
                generated_mech_keys.add(mech_key)
        
        # Check which ground truths were recovered (structurally)
        for j, gt_key in enumerate(gt_mechanistic_keys):
            if gt_key in generated_mech_keys:
                recovered_gts.add(j)
                recovered_gt_keys.add(gt_key)
        
        recovery_rate = len(recovered_gts) / len(gt_mechanistic_keys) if gt_mechanistic_keys else 0
        
        return {
            'observation_set_id': observation_set['observation_set_id'],
            'n_observations': observation_set['n_observations'],
            'n_ground_truths': len(ground_truth_expressions),
            'n_queries': n_queries,
            'n_valid': len(valid_hypotheses),
            'n_unique_valid': len(unique_valid_expressions),  # Unique among valid hypotheses
            'n_unique_all': len(unique_all_expressions),  # Unique among in-space hypotheses
            'n_recovered_gts': len(recovered_gts),
            'parse_success_rate': parse_success_rate,  # Diagnostic: how many parsed
            'in_space_rate': in_space_rate,  # Diagnostic: how many met space constraints
            'valid_rate': valid_rate,
            'novelty_rate': novelty_rate,  # Now using unique in-space / n_queries
            'recovery_rate': recovery_rate,
            'all_hypotheses': all_hypotheses,
            'valid_hypotheses': valid_hypotheses,
            'unique_valid_expressions': unique_valid_expressions,
            'unique_all_expressions': unique_all_expressions,
            # Token usage and cost tracking
            'token_usage': {
                'prompt_tokens': total_prompt_tokens,
                'completion_tokens': total_completion_tokens,
                'total_tokens': total_tokens
            },
            'cost': total_cost,
            # Error tracking
            'errors': errors,
            'error_summary': {
                'total_errors': len(errors),
                'error_types': error_counts
            }
        }
    
    def run_benchmark(
        self,
        llm: LLMInterface,
        n_samples: int = 10,
        n_queries_per_sample: Optional[int] = None,
        query_multiplier: float = 2.0,
        seed: Optional[int] = None,
        verbose: bool = True,
        checkpoint_dir: str = "checkpoints",
        run_id: Optional[str] = None
    ) -> Dict:
        """
        Run the benchmark with sampling.
        
        Args:
            llm: LLM interface to use
            n_samples: Number of observation sets to sample
            n_queries_per_sample: Fixed number of queries per observation set (if None, uses query_multiplier)
            query_multiplier: Multiplier for n_gt to determine queries (default 2.0 means 2x ground truths)
            seed: Random seed
            verbose: Print progress
            checkpoint_dir: Directory to save checkpoints
        
        Returns:
            Dictionary with complete benchmark results
        """
        # Create checkpoint directory
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Generate run ID and safe filename
        if not run_id:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize LLM name for filename
        safe_llm_name = llm.get_name().replace('/', '_').replace('(', '_').replace(')', '_').replace(' ', '_')
        checkpoint_file = checkpoint_path / f"checkpoint_{safe_llm_name}_{run_id}.json"
        
        print(f"\nRunning refined Boolean benchmark")
        print(f"LLM: {llm.get_name()}")
        print(f"Sampling {n_samples} observation sets")
        if n_queries_per_sample is not None:
            print(f"Queries per sample: {n_queries_per_sample} (fixed)")
        else:
            print(f"Queries per sample: {query_multiplier}x number of ground truths (adaptive)")
        print(f"Checkpoint file: {checkpoint_file}")
        print("-" * 50)
        
        # Sample observation sets
        sampled_sets = self.sample_observation_sets(n_samples, seed)
        
        # Initialize results tracking
        all_results = []
        valid_rates = []
        novelty_rates = []
        recovery_rates = []
        
        # Initialize token and cost tracking
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_cost = 0.0
        
        # Initialize error tracking
        all_errors = []
        total_error_counts = {}
        
        # Load existing checkpoint if it exists
        start_idx = 0
        if checkpoint_file.exists():
            print(f"Using Recovered checkpoint file: {checkpoint_file}")
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                all_results = checkpoint_data['results']
                start_idx = len(all_results)
                print(f"Resuming from checkpoint: {start_idx}/{n_samples} completed")

                # Recalculate rates from checkpoint
                for result in all_results:
                    valid_rates.append(result['valid_rate'])
                    novelty_rates.append(result['novelty_rate'])
                    recovery_rates.append(result['recovery_rate'])

                # RESTORE TOKEN AND COST TOTALS
                if 'total_token_usage' in checkpoint_data:
                    total_prompt_tokens = checkpoint_data['total_token_usage'].get('prompt_tokens', 0)
                    total_completion_tokens = checkpoint_data['total_token_usage'].get('completion_tokens', 0)
                    total_tokens = checkpoint_data['total_token_usage'].get('total_tokens', 0)
                if 'total_cost' in checkpoint_data:
                    total_cost = checkpoint_data['total_cost']
        
        # Process each sampled observation set
        for idx in range(start_idx, len(sampled_sets)):
            obs_set = sampled_sets[idx]
            
            if verbose:
                print(f"\nSample {idx + 1}/{n_samples}")
                print(f"  Observation set ID: {obs_set['observation_set_id']}")
                print(f"  Number of observations: {obs_set['n_observations']}")
                print(f"  Number of ground truths: {obs_set['n_compatible_expressions']}")
            
            try:
                # Determine number of queries for this observation set
                if n_queries_per_sample is not None:
                    # Use fixed number
                    n_queries = n_queries_per_sample
                else:
                    # Adaptive: use multiplier of ground truths
                    n_gt = obs_set['n_compatible_expressions']
                    n_queries = max(1, int(n_gt * query_multiplier))
                    if verbose:
                        print(f"  Using {n_queries} queries ({query_multiplier}x {n_gt} ground truths)")
                
                # Evaluate on this observation set
                result = self.evaluate_single_observation_set(
                    llm, obs_set, n_queries, verbose=False
                )
                
                all_results.append(result)
                valid_rates.append(result['valid_rate'])
                novelty_rates.append(result['novelty_rate'])
                recovery_rates.append(result['recovery_rate'])
                
                # Aggregate token usage and costs
                if 'token_usage' in result:
                    total_prompt_tokens += result['token_usage']['prompt_tokens']
                    total_completion_tokens += result['token_usage']['completion_tokens']
                    total_tokens += result['token_usage']['total_tokens']
                if 'cost' in result:
                    total_cost += result['cost']
                
                # Aggregate errors
                if 'errors' in result and result['errors']:
                    all_errors.extend(result['errors'])
                    # Update error type counts
                    if 'error_summary' in result:
                        for error_type, count in result['error_summary']['error_types'].items():
                            total_error_counts[error_type] = total_error_counts.get(error_type, 0) + count
                
                if verbose:
                    print(f"  Valid rate: {result['valid_rate']:.2%}")
                    print(f"  Novelty rate: {result['novelty_rate']:.2%}")
                    print(f"  Recovery rate: {result['recovery_rate']:.2%}")
                
                # Save checkpoint after each sample
                checkpoint_data = {
                    'run_id': run_id,
                    'llm_name': llm.get_name(),
                    'n_samples': n_samples,
                    'n_queries_per_sample': n_queries_per_sample,
                    'query_multiplier': query_multiplier if n_queries_per_sample is None else None,
                    'seed': seed,
                    'timestamp': datetime.now().isoformat(),
                    'results': all_results,
                    'total_token_usage': {
                        'prompt_tokens': total_prompt_tokens,
                        'completion_tokens': total_completion_tokens,
                        'total_tokens': total_tokens
                    },
                    'total_cost': total_cost,
                    'total_errors': len(all_errors),
                    'error_types': total_error_counts
                }
                
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)

            except Exception as e:
                print(f"  Error processing sample {idx + 1}: {str(e)}")
                traceback.print_exc()
                continue
        
        # Calculate statistics
        def calculate_stats(rates):
            if not rates:
                return {'mean': 0, 'std': 0, 'var': 0, 'min': 0, 'max': 0}
            return {
                'mean': np.mean(rates),
                'std': np.std(rates),
                'var': np.var(rates),
                'min': np.min(rates),
                'max': np.max(rates)
            }
        
        # Calculate p-values (one-sample t-test against null hypothesis of 0)
        def calculate_p_value(rates):
            if not rates or len(rates) < 2:
                return None
            t_stat, p_val = stats.ttest_1samp(rates, 0)
            return p_val
        
        # Compile final results
        final_results = {
            'run_id': run_id,
            'llm_name': llm.get_name(),
            'n_samples': len(all_results),
            'n_queries_per_sample': n_queries_per_sample,
            'query_multiplier': query_multiplier if n_queries_per_sample is None else None,
            'query_mode': 'fixed' if n_queries_per_sample is not None else f'adaptive_{query_multiplier}x',
            'seed': seed,
            'timestamp': datetime.now().isoformat(),
            'statistics': {
                'valid_rate': {
                    **calculate_stats(valid_rates),
                    'p_value': calculate_p_value(valid_rates)
                },
                'novelty_rate': {
                    **calculate_stats(novelty_rates),
                    'p_value': calculate_p_value(novelty_rates)
                },
                'recovery_rate': {
                    **calculate_stats(recovery_rates),
                    'p_value': calculate_p_value(recovery_rates)
                }
            },
            'token_usage': {
                'prompt_tokens': total_prompt_tokens,
                'completion_tokens': total_completion_tokens,
                'total_tokens': total_tokens,
                'avg_tokens_per_sample': total_tokens / len(all_results) if all_results else 0,
                'avg_tokens_per_query': total_tokens / (len(all_results) * (n_queries_per_sample or 1)) if all_results else 0
            },
            'cost': {
                'total_cost': total_cost,
                'avg_cost_per_sample': total_cost / len(all_results) if all_results else 0,
                'avg_cost_per_query': total_cost / (len(all_results) * (n_queries_per_sample or 1)) if all_results else 0
            },
            'error_summary': {
                'total_errors': len(all_errors),
                'error_types': total_error_counts,
                'error_rate': len(all_errors) / (len(all_results) * (n_queries_per_sample or 1)) if all_results else 0
            },
            'per_sample_results': all_results
        }
        
        # Print summary
        print("\n" + "=" * 50)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 50)
        print(f"Samples evaluated: {len(all_results)}")
        
        for metric_name, metric_key in [('Valid Rate', 'valid_rate'), 
                                        ('Novelty Rate', 'novelty_rate'), 
                                        ('Recovery Rate', 'recovery_rate')]:
            stats_dict = final_results['statistics'][metric_key]
            print(f"\n{metric_name}:")
            print(f"  Mean ± Std: {stats_dict['mean']:.3f} ± {stats_dict['std']:.3f}")
            print(f"  Variance: {stats_dict['var']:.3f}")
            print(f"  Range: [{stats_dict['min']:.3f}, {stats_dict['max']:.3f}]")
            if stats_dict['p_value'] is not None:
                print(f"  p-value: {stats_dict['p_value']:.4f}")
        
        # Print token usage and cost summary
        print(f"\nToken Usage:")
        print(f"  Total tokens: {final_results['token_usage']['total_tokens']:,}")
        print(f"  Prompt tokens: {final_results['token_usage']['prompt_tokens']:,}")
        print(f"  Completion tokens: {final_results['token_usage']['completion_tokens']:,}")
        print(f"  Avg tokens/sample: {final_results['token_usage']['avg_tokens_per_sample']:.1f}")
        print(f"  Avg tokens/query: {final_results['token_usage']['avg_tokens_per_query']:.1f}")
        
        print(f"\nCost:")
        print(f"  Total cost: ${final_results['cost']['total_cost']:.4f}")
        print(f"  Avg cost/sample: ${final_results['cost']['avg_cost_per_sample']:.4f}")
        print(f"  Avg cost/query: ${final_results['cost']['avg_cost_per_query']:.6f}")
        
        # Print error summary if there were errors
        if final_results['error_summary']['total_errors'] > 0:
            print(f"\nErrors:")
            print(f"  Total errors: {final_results['error_summary']['total_errors']}")
            print(f"  Error rate: {final_results['error_summary']['error_rate']:.2%}")
            if final_results['error_summary']['error_types']:
                print(f"  Error types:")
                for error_type, count in final_results['error_summary']['error_types'].items():
                    print(f"    - {error_type}: {count}")
        
        print("=" * 50)
        
        return final_results


def setup_llm(llm_type: str, **kwargs) -> LLMInterface:
    """Set up the LLM interface based on type."""
    
    if llm_type == "openai":
        api_key = kwargs.get('api_key') or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key required")
        
        return OpenAILLM(
            model=kwargs.get('model', 'gpt-4'),
            api_key=api_key,
            temperature=kwargs.get('temperature', 0.7)
        )
    
    elif llm_type == "anthropic":
        api_key = kwargs.get('api_key') or os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("Anthropic API key required")
        
        return AnthropicLLM(
            model=kwargs.get('model', 'claude-3-opus-20240229'),
            api_key=api_key,
            temperature=kwargs.get('temperature', 0.7)
        )
    
    elif llm_type == "openrouter":
        api_key = kwargs.get('api_key') or os.environ.get('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OpenRouter API key required")
        
        return OpenRouterLLM(
            model=kwargs.get('model', 'anthropic/claude-3.5-sonnet'),
            api_key=api_key,
            temperature=kwargs.get('temperature', 0.7)
        )
    
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Run refined Boolean discovery benchmark with sampling")
    parser.add_argument("--dataset", required=True, help="Path to complete Boolean dataset JSON file")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--n-samples", type=int, default=10, help="Number of observation sets to sample")
    parser.add_argument("--n-queries", type=int, default=None, help="Fixed number of queries per observation set (if not set, uses adaptive)")
    parser.add_argument("--query-multiplier", type=float, default=1.0, help="Multiplier for adaptive queries (n_queries = n_gt * multiplier)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    # New flags for self-consistency
    parser.add_argument("--no-self-consistency", action="store_true", help="Disable self-consistency sampling")
    parser.add_argument("--sc-num-samples", type=int, default=5, help="Number of samples per query for self-consistency")
    parser.add_argument("--sc-temperature", type=float, default=0.9, help="Temperature for self-consistency sampling")
    # Optionally allow disabling ACE (mostly for ablations)
    parser.add_argument("--no-ace", action="store_true", help="Disable ACE prompt augmentation")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    llm_type = config.get('llm', {}).get('type', 'openrouter')
    
    model = config.get('llm', {}).get('models', {}).get(llm_type)
    if not model:
        default_models = {
            'openrouter': 'openai/gpt-3.5-turbo',
            'openai': 'gpt-4',
            'anthropic': 'claude-3-opus-20240229'
        }
        model = default_models.get(llm_type)
    
    api_key = config.get('llm', {}).get('api_keys', {}).get(llm_type)
    if not api_key:
        env_vars = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'openrouter': 'OPENROUTER_API_KEY'
        }
        if llm_type in env_vars:
            api_key = os.environ.get(env_vars[llm_type])
    
    temperature = config.get('llm', {}).get('temperature', 0.7)
    checkpoint_dir = config.get('benchmark', {}).get('checkpoint', 'checkpoints')
    run_id = config.get('benchmark', {}).get('run_id', None)
    verbose = config.get('benchmark', {}).get('verbose', True)

    if not Path(args.dataset).exists():
        print(f"Error: Dataset file not found: {args.dataset}")
        sys.exit(1)
    dataset_name = Path(args.dataset).stem
    model_name = Path(model).stem if model else llm_type
    output_pattern = config.get('benchmark', {}).get('output_pattern', 'results_{dataset_name}_{model}.json')
    output = output_pattern.format(dataset_name=dataset_name, model=model_name, llm_type=llm_type)

    # Generate output filename if not specified
    if output is None:
        dataset_name = Path(args.dataset).stem
        model_name = Path(model).stem if model else llm_type
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = f"results/{dataset_name}_{model_name}_{timestamp}.json"

    # Initialize benchmark
    benchmark = BooleanBenchmarkRefined(
        args.dataset,
        enable_ace=(not args.no_ace),
        enable_self_consistency=(not args.no_self_consistency),
        sc_num_samples=args.sc_num_samples,
        sc_temperature=args.sc_temperature
    )
     
    # Print configuration
    print("\n" + "=" * 60)
    print("REFINED BOOLEAN BENCHMARK CONFIGURATION")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"LLM Type: {llm_type}")
    print(f"Model: {model}")
    print(f"Temperature: {temperature}")
    print(f"Samples: {args.n_samples}")
    print(f"Queries per sample: {args.n_queries}")
    print(f"Seed: {args.seed}")
    print(f"Output: {output}")
    print(f"ACE: {'on' if not args.no_ace else 'off'}")
    print(f"Self-consistency: {'on' if not args.no_self_consistency else 'off'} (num_samples={args.sc_num_samples}, temp={args.sc_temperature})")
    print("=" * 60)
    
    # Set up LLM
    llm = setup_llm(
        llm_type,
        model=model,
        api_key=api_key,
        temperature=temperature
    )
    
    # Run benchmark
    results = benchmark.run_benchmark(
        llm=llm,
        n_samples=args.n_samples,
        n_queries_per_sample=args.n_queries,
        query_multiplier=args.query_multiplier,
        seed=args.seed,
        verbose=verbose,
        checkpoint_dir=checkpoint_dir,
        run_id=run_id
    )
    
    # Save final results
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nFinal results saved to: {output}")


if __name__ == "__main__":
    main()