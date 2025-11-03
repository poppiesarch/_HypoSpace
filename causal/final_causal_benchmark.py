import sys
import json
import argparse
import os
import re
import yaml
import random
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

import networkx as nx
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any
from datetime import datetime
from textwrap import dedent
from collections import Counter
from scipy import stats
import traceback
from dataclasses import dataclass
from modules.models import CausalGraph
from modules.llm_interface import LLMInterface, OpenRouterLLM, OpenAILLM, AnthropicLLM
from generate_causal_dataset import PerturbationObservation, CausalDatasetGenerator


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ACEBullet:
    """
    Structured context item in ACE playbook.
    
    Attributes:
        bullet_id: Unique identifier (e.g., "sem-00001", "rul-00001")
        section: Module name (e.g., "strategies_and_hard_rules")
        content: Actual content (strategy, code snippet, troubleshooting tip)
        type: "semantic" (fixed) or "rule" (dynamic)
        helpful_count: Number of times marked as helpful
        harmful_count: Number of times marked as harmful
        created_at: Creation timestamp
        last_updated: Last update timestamp
    """
    bullet_id: str
    section: str
    content: str
    type: str
    helpful_count: int = 0
    harmful_count: int = 0
    created_at: str = datetime.now().isoformat()
    last_updated: str = datetime.now().isoformat()


# ============================================================================
# Main Benchmark Class
# ============================================================================

class CausalBenchmarkEnhanced:
    """
    Enhanced benchmark for evaluating LLM creativity in causal graph discovery.
    
    Features:
    - ACE (Autonomous Contextualization and Execution) framework
    - Self-consistency sampling for diversity
    - Response aggregation for robustness
    - Dynamic playbook management
    - Comprehensive tracking (tokens, costs, errors)
    """
    
    def __init__(self, complete_dataset_path: Optional[str] = None, 
                 n_observations_filter: Optional[List[int]] = None,
                 gt_filter: Optional[Tuple] = None):
        """
        Initialize benchmark with optional dataset and filters.
        
        Args:
            complete_dataset_path: Path to complete causal dataset JSON
            n_observations_filter: List of n_observations values to include
            gt_filter: (min, max) for GT range or (list, None) for specific values
        """
        self.n_observations_filter = n_observations_filter
        self.gt_filter = gt_filter
        self.filtered_observation_sets = []
        self.excluded_observation_sets = []
        
        # *** ACE Playbook Initialization ***
        self.ace_playbook: Dict[str, ACEBullet] = {}
        self.bullet_counter = 0
        self._init_semantic_rules()
        self._init_empty_rules()
        
        # *** Embedding Model for Semantic Deduplication ***
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.bullet_embeddings: Dict[str, np.ndarray] = {}
        for bullet_id, bullet in self.ace_playbook.items():
            self.bullet_embeddings[bullet_id] = self.embedding_model.encode(
                bullet.content, convert_to_tensor=False
            )
        
        # *** Dataset Loading and Filtering ***
        if complete_dataset_path:
            with open(complete_dataset_path, 'r') as f:
                self.complete_dataset = json.load(f)
            
            self.metadata = self.complete_dataset.get('metadata', {})
            self.nodes = self.metadata.get('nodes', [])
            self.max_edges = self.metadata.get('max_edges', None)
            
            # Flatten all datasets
            self.all_observation_sets = []
            if 'datasets_by_n_observations' in self.complete_dataset:
                for n_obs, datasets in self.complete_dataset['datasets_by_n_observations'].items():
                    self.all_observation_sets.extend(datasets)
            elif 'datasets' in self.complete_dataset:
                self.all_observation_sets = self.complete_dataset['datasets']
            elif 'sampled_datasets' in self.complete_dataset:
                self.all_observation_sets = self.complete_dataset['sampled_datasets']
            
            # Apply two-stage filtering
            stage1_filtered = self.all_observation_sets
            
            if n_observations_filter:
                stage1_filtered = [
                    obs_set for obs_set in self.all_observation_sets
                    if obs_set.get('n_observations') in n_observations_filter
                ]
                print(f"Stage 1: Filtered to {len(stage1_filtered)} observation sets "
                      f"with n_observations in {n_observations_filter}")
            
            if gt_filter:
                if gt_filter[1] is not None:
                    min_gt, max_gt = gt_filter
                    self.filtered_observation_sets = [
                        obs_set for obs_set in stage1_filtered
                        if min_gt <= obs_set.get('n_compatible_graphs', 0) <= max_gt
                    ]
                    print(f"Stage 2: Filtered to {len(self.filtered_observation_sets)} "
                          f"observation sets with n_compatible_graphs in [{min_gt}, {max_gt}]")
                else:
                    allowed_values = gt_filter[0] if isinstance(gt_filter[0], list) else []
                    self.filtered_observation_sets = [
                        obs_set for obs_set in stage1_filtered
                        if obs_set.get('n_compatible_graphs', 0) in allowed_values
                    ]
                    print(f"Stage 2: Filtered to {len(self.filtered_observation_sets)} "
                          f"observation sets with n_compatible_graphs in {allowed_values}")
                
                self.excluded_observation_sets = [
                    obs_set for obs_set in self.all_observation_sets
                    if obs_set not in self.filtered_observation_sets
                ]
            else:
                self.filtered_observation_sets = stage1_filtered
                self.excluded_observation_sets = []
            
            # Infer max_edges if not specified
            if self.max_edges is None and self.all_observation_sets:
                max_edges_in_gts = 0
                for obs_set in self.all_observation_sets:
                    for gt in obs_set.get('ground_truth_graphs', []):
                        num_edges = len(gt.get('edges', []))
                        max_edges_in_gts = max(max_edges_in_gts, num_edges)
                self.max_edges = max_edges_in_gts
                print(f"Inferred max_edges={self.max_edges} from ground truth graphs")
            
            print(f"Loaded complete dataset with {len(self.all_observation_sets)} observation sets")
            if n_observations_filter or gt_filter:
                print(f"After filtering: {len(self.filtered_observation_sets)} observation sets meet criteria")
                if self.excluded_observation_sets:
                    print(f"  ({len(self.excluded_observation_sets)} observation sets available for backfill)")
            print(f"Nodes: {', '.join(self.nodes)}")
            print(f"Max edges in hypothesis space: {self.max_edges if self.max_edges is not None else 'unlimited'}")
        else:
            self.complete_dataset = None
            self.all_observation_sets = []
            self.filtered_observation_sets = []
            print("Initialized empty benchmark - will generate datasets on demand")

    # ========================================================================
    # ACE Playbook Management
    # ========================================================================
    
    def _init_empty_rules(self):
        """Initialize empty dynamic rules (will be populated during execution)."""
        print("Initialized empty dynamic rules (will be updated dynamically)")

    def _init_semantic_rules(self):
        """Initialize 3 fixed semantic rules (never change)."""
        semantic_timestamp = datetime(2000, 1, 1).isoformat()
        semantic_rules = [
            (
                "strategies_and_hard_rules",
                "When a node is perturbed, the perturbed node is 0.",
                "semantic"
            ),
            (
                "strategies_and_hard_rules",
                "A node is 1 if it is a downstream descendant of the perturbed node in the causal graph.",
                "semantic"
            ),
            (
                "strategies_and_hard_rules",
                "All other nodes are 0.",
                "semantic"
            )
        ]
        for section, content, type_ in semantic_rules:
            self.bullet_counter += 1
            bullet_id = f"sem-{str(self.bullet_counter).zfill(5)}"
            self.ace_playbook[bullet_id] = ACEBullet(
                bullet_id=bullet_id,
                section=section,
                content=content,
                type=type_,
                created_at=semantic_timestamp,
                last_updated=semantic_timestamp
            )
        print(f"Initialized 3 fixed semantic rules")

    def _calculate_content_similarity(self, content: str) -> float:
        """
        Calculate max cosine similarity between new content and existing bullets.
        
        Args:
            content: New content to compare
            
        Returns:
            Maximum similarity score (0.0 to 1.0)
        """
        if not self.bullet_embeddings:
            return 0.0
        
        new_embedding = self.embedding_model.encode(content, convert_to_tensor=False).reshape(1, -1)
        max_sim = 0.0
        for emb in self.bullet_embeddings.values():
            sim = cosine_similarity(new_embedding, emb.reshape(1, -1))[0][0]
            if sim > max_sim:
                max_sim = sim
        return max_sim
    
    def update_bullet_feedback(self, bullet_id: str, is_helpful: bool):
        """Update helpful/harmful count for a bullet."""
        if bullet_id not in self.ace_playbook:
            print(f"Warning: Bullet {bullet_id} not found in playbook")
            return
        
        bullet = self.ace_playbook[bullet_id]
        if is_helpful:
            bullet.helpful_count += 1
        else:
            bullet.harmful_count += 1
        bullet.last_updated = datetime.now().isoformat()
        self.ace_playbook[bullet_id] = bullet

    # ========================================================================
    # ACE Curator: Hypothesis and Reflection Management
    # ========================================================================
    
    def curate_hypotheses(self, new_hypothesis: CausalGraph, 
                         existing_hypotheses: List[CausalGraph]) -> Tuple[List[CausalGraph], bool]:
        """
        ACE Curator: Incrementally update and deduplicate hypothesis list.
        
        Args:
            new_hypothesis: Newly generated hypothesis
            existing_hypotheses: Current hypothesis list
            
        Returns:
            (curated_list, is_new): Updated list and whether hypothesis was added
        """
        # Semantic deduplication: same edge set = duplicate
        new_edges = frozenset(new_hypothesis.edges)
        for h in existing_hypotheses:
            if frozenset(h.edges) == new_edges:
                return existing_hypotheses, False
        
        # Incremental addition (delta update)
        curated = existing_hypotheses + [new_hypothesis]

        # Redundancy pruning: keep at most 30 representative hypotheses
        if len(curated) > 30:
            edge_count_map = defaultdict(list)
            for h in curated:
                edge_count = len(h.edges)
                edge_count_map[edge_count].append(h)
            curated = [h_list[0] for h_list in edge_count_map.values()][:10]
        
        return curated, True
    
    def curate_reflection(self, reflection: Dict) -> bool:
        """
        ACE Curator: Integrate reflection insights into playbook.
        
        Args:
            reflection: Structured reflection output from reflector
            
        Returns:
            success: Whether new insight was added
        """
        # 1. Update bullet feedback (only for dynamic rules)
        for feedback in reflection["bullet_feedback"]:
            bullet_id = feedback["bullet_id"]
            if bullet_id in self.ace_playbook and self.ace_playbook[bullet_id].type == "rule":
                self.update_bullet_feedback(bullet_id, feedback["is_helpful"])
            else:
                print(f"Skipping feedback for non-rule bullet: {bullet_id}")

        # 2. Process new insight candidate
        new_content = reflection.get("new_insight_candidate", "").strip()
        if not new_content:
            print("No new insight candidate in reflection—skipping curation")
            return False

        # 3. Semantic deduplication (only compare with existing rules)
        rule_embeddings = {
            bid: emb for bid, emb in self.bullet_embeddings.items()
            if self.ace_playbook[bid].type == "rule"
        }
        max_sim = 0.0
        if rule_embeddings:
            new_embedding = self.embedding_model.encode(new_content, convert_to_tensor=False).reshape(1, -1)
            for emb in rule_embeddings.values():
                sim = cosine_similarity(new_embedding, emb.reshape(1, -1))[0][0]
                max_sim = max(max_sim, sim)
        if max_sim > 0.85:
            print(f"New rule is redundant (similarity={max_sim:.2f})—skipping")
            return False

        # 4. Determine section and add as dynamic rule
        if "format" in new_content.lower() or "output" in new_content.lower():
            section = "troubleshooting"
        elif "cycle" in new_content.lower() or "dag" in new_content.lower():
            section = "strategies_and_hard_rules"
        elif "def " in new_content.lower() or "code" in new_content.lower():
            section = "code_snippets"
        else:
            section = "strategies_and_hard_rules"

        self.bullet_counter += 1
        new_bullet_id = f"rul-{str(self.bullet_counter).zfill(5)}"
        new_timestamp = datetime.now().isoformat()
        new_bullet = ACEBullet(
            bullet_id=new_bullet_id,
            section=section,
            content=new_content,
            type="rule",
            created_at=new_timestamp,
            last_updated=new_timestamp
        )
        self.ace_playbook[new_bullet_id] = new_bullet
        self.bullet_embeddings[new_bullet_id] = self.embedding_model.encode(new_content, convert_to_tensor=False)
        print(f"Added new dynamic rule: [{new_bullet_id}] (section: {section})")

        # 5. Keep only latest 3 rules (semantic rules excluded)
        rule_bullets = [
            (bullet.created_at, bullet_id, bullet)
            for bullet_id, bullet in self.ace_playbook.items()
            if bullet.type == "rule"
        ]
        rule_bullets.sort(reverse=True, key=lambda x: x[0])

        if len(rule_bullets) > 3:
            to_delete = rule_bullets[3:]
            for _, bid, _ in to_delete:
                del self.ace_playbook[bid]
                del self.bullet_embeddings[bid]
            print(f"Kept latest 3 rules, deleted {len(to_delete)} old rules")
        else:
            print(f"Current rules count: {len(rule_bullets)} (≤3, no deletion)")

        return True
    
    def prune_playbook(self, max_bullets_per_section: int = 3, min_helpful_ratio: float = 0.3):
        """
        Prune playbook: only prune dynamic rules, preserve semantic rules.
        
        Args:
            max_bullets_per_section: Max bullets to keep per section
            min_helpful_ratio: Minimum helpful/(helpful+harmful) ratio
            
        Returns:
            Number of pruned bullets
        """
        section_bullets = defaultdict(list)
        for bullet in self.ace_playbook.values():
            if bullet.type == "rule":
                section_bullets[bullet.section].append(bullet)
        
        pruned_bullet_ids = []
        for section, bullets in section_bullets.items():
            # Filter by helpful ratio
            filtered = []
            for bullet in bullets:
                total_feedback = bullet.helpful_count + bullet.harmful_count
                if total_feedback == 0:
                    filtered.append(bullet)
                else:
                    helpful_ratio = bullet.helpful_count / total_feedback
                    if helpful_ratio >= min_helpful_ratio:
                        filtered.append(bullet)
            
            # Sort by helpful ratio and keep top N
            sorted_bullets = sorted(
                filtered, 
                key=lambda x: (x.helpful_count / (x.helpful_count + x.harmful_count + 1e-6)), 
                reverse=True
            )
            kept_bullets = sorted_bullets[:max_bullets_per_section]
            
            # Mark bullets for deletion
            kept_ids = {b.bullet_id for b in kept_bullets}
            for bullet in bullets:
                if bullet.bullet_id not in kept_ids:
                    pruned_bullet_ids.append(bullet.bullet_id)
        
        # Execute deletion
        for bullet_id in pruned_bullet_ids:
            del self.ace_playbook[bullet_id]
            if bullet_id in self.bullet_embeddings:
                del self.bullet_embeddings[bullet_id]
        
        print(f"Pruned {len(pruned_bullet_ids)} rule bullets from playbook—"
              f"remaining: {len(self.ace_playbook)} (semantic rules preserved)")
        return len(pruned_bullet_ids)

    # ========================================================================
    # Dataset Sampling
    # ========================================================================
    
    def sample_observation_sets(self, n_samples: int, seed: Optional[int] = None) -> List[Dict]:
        """
        Sample n observation sets with smart backfill.
        
        Args:
            n_samples: Number of observation sets to sample
            seed: Random seed for reproducibility
        
        Returns:
            List of sampled observation sets
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        if self.n_observations_filter or self.gt_filter:
            primary_pool = self.filtered_observation_sets
            backup_pool = self.excluded_observation_sets
        else:
            primary_pool = self.all_observation_sets
            backup_pool = []
        
        sampled = []
        
        # Sample from primary pool
        n_primary = len(primary_pool)
        if n_primary > 0:
            n_from_primary = min(n_samples, n_primary)
            sampled_primary = random.sample(primary_pool, n_from_primary)
            sampled.extend(sampled_primary)
            
            for obs_set in sampled_primary:
                obs_set['meets_filter_criteria'] = True
        
        # Backfill from backup pool if needed
        n_still_needed = n_samples - len(sampled)
        if n_still_needed > 0 and backup_pool:
            n_backup = len(backup_pool)
            n_from_backup = min(n_still_needed, n_backup)
            
            if n_from_backup > 0:
                print(f"\nBackfilling: Only {n_primary} datasets met filter criteria.")
                print(f"  Adding {n_from_backup} randomly selected datasets from outside the filter range.")
                
                sampled_backup = random.sample(backup_pool, n_from_backup)
                
                for obs_set in sampled_backup:
                    obs_set['meets_filter_criteria'] = False
                    obs_set['backfilled'] = True
                
                sampled.extend(sampled_backup)
        
        if len(sampled) < n_samples:
            total_available = len(primary_pool) + len(backup_pool)
            print(f"\nWarning: Requested {n_samples} samples but only {total_available} total datasets available.")
            print(f"  Returning {len(sampled)} datasets.")
        
        return sampled
    
    # ========================================================================
    # Prompt Generation
    # ========================================================================
    
    def create_prompt(self, observations: List[Dict], 
                     prior_hypotheses_with_validity: List[Tuple]) -> Tuple[str, List[str]]:
        """
        Generate ACE-style prompt (split semantics and dynamic rules).
        
        Args:
            observations: Perturbation observation data
            prior_hypotheses_with_validity: Historical hypotheses with validity flags
            
        Returns:
            (prompt, used_bullet_ids): Prompt text and list of used bullet IDs
        """
        nodes_str = ", ".join(self.nodes)
        obs_block = "\n".join(obs["string"] for obs in observations)
        used_bullet_ids = []

        # Fixed semantic rules (always displayed)
        semantic_bullets = [
            bullet for bullet in self.ace_playbook.values()
            if bullet.type == "semantic"
        ]
        semantic_block = "Semantics (must follow):\n"
        for bullet in semantic_bullets:
            semantic_block += f"- {bullet.content.strip()}\n"
            used_bullet_ids.append(bullet.bullet_id)

        # Dynamic rules (latest 3)
        rule_bullets = [
            bullet for bullet in self.ace_playbook.values()
            if bullet.type == "rule"
        ]
        rule_bullets.sort(key=lambda x: x.created_at, reverse=True)
        current_rules = rule_bullets[:3]

        rule_block = "\nSuggestions (to avoid invalid predictions):\n"
        if current_rules:
            for bullet in current_rules:
                rule_block += f"- {bullet.content.strip()}\n"
                used_bullet_ids.append(bullet.bullet_id)
        else:
            rule_block += "- No dynamic rules yet (will be updated as you generate insights)\n"

        # Historical hypotheses (mark valid/invalid)
        invalid_block = ""
        valid_block = ""
        for idx, (hypo, is_valid, reason) in enumerate(prior_hypotheses_with_validity, 1):
            edges_str = ", ".join([f"{s}->{d}" for s, d in hypo.edges]) or "No edges"
            if is_valid:
                valid_block += f"Graph: {edges_str}\n"
            else:
                invalid_block += f"Graph: {edges_str}\n"

        constraint_info = f"\nConstraint: The graph should have at most {self.max_edges} edges."
        
        diversity_guidance = """
        Diversity enhancement rules:
        - Generate hypotheses with DIFFERENT CAUSAL MECHANISMS: e.g., if prior graphs have "A->C", try "A->B->C" or "B->A->C" (different paths).
        - Avoid only changing edge order (e.g., "A->B, B->C" is same mechanism as "B->C, A->B" — do not repeat).
        - Prioritize adding/removing edges that change the causal chain (e.g., add "C->A" if no prior graph has this edge).
        """

        prompt = dedent(f"""
        You are given observations from perturbation experiments on a causal system. Your task is to generate a valid causal graph that explains all observations.

        {semantic_block}
        {rule_block}

        Nodes: {nodes_str}{constraint_info}

        Observation:
        {obs_block}

        Previous valid predictions (do NOT repeat these):
        {valid_block if valid_block else "None"}

        Previous invalid predictions (avoid these mistakes):
        {invalid_block if invalid_block else "None"}
        
        Task:
        Output a single directed acyclic graph (DAG) over the nodes above that explains all observations.
        (important!) Generate a DIFFERENT valid structure from the prior attempts shown above

        Diversity requirement:
        - A "diverse" graph must have a unique edge set not present in previous valid or invalid predictions.
        - Prioritize graphs with different causal mechanisms (e.g., new paths or edge directions).

        {diversity_guidance}

        Formatting rules (must follow strictly):
        1) Use only the listed nodes. No self-loops. No cycles.
        2) Respond with exactly one line:
        - If there are edges: Graph: A->B, B->C
        - If there are no edges: Graph: No edges
        """).strip()

        return prompt, used_bullet_ids

    # ========================================================================
    # Self-Consistency Sampling and Response Aggregation
    # ========================================================================
    
    def generate_with_self_consistency(self, llm: LLMInterface, prompt: str, 
                                      num_samples: int = 3, 
                                      temp_override: float = 0.8) -> List[Tuple[str, str]]:
        """
        *** Self-Consistency Sampling ***
        Generate multiple responses with higher temperature for diversity.
        
        Args:
            llm: LLM interface
            prompt: Input prompt
            num_samples: Number of samples to generate
            temp_override: Temperature for sampling (higher = more diverse)
            
        Returns:
            List of (trajectory, hypothesis_str) tuples
        """
        responses = []
        for i in range(num_samples):
            try:
                # Query with temperature override
                if hasattr(llm, 'query_with_params'):
                    result = llm.query_with_params(
                        prompt, 
                        temperature=temp_override,
                        top_p=0.95,  # Nucleus sampling
                        repetition_penalty=1.1  # Reduce repetition
                    )
                    full_response = result['response']
                else:
                    # Fallback to standard query
                    full_response = llm.query(prompt)
                
                # Split trajectory and hypothesis
                if "Graph:" in full_response:
                    trajectory_part, hypothesis_part = full_response.split("Graph:", 1)
                    trajectory = trajectory_part.strip()
                    hypothesis_str = "Graph:" + hypothesis_part.strip()
                else:
                    trajectory = full_response.strip()
                    hypothesis_str = ""
                
                responses.append((trajectory, hypothesis_str))
                
            except Exception as e:
                print(f"  Warning: Self-consistency sample {i+1} failed: {str(e)[:100]}")
                continue
        
        return responses
    
    def aggregate_responses(self, responses: List[Tuple[str, str]], 
                           observations: List[Dict]) -> Optional[CausalGraph]:
        """
        *** Response Aggregation ***
        Extract all unique valid hypotheses from multiple samples.
        
        Args:
            responses: List of (trajectory, hypothesis_str) tuples
            observations: Observation data for validation
            
        Returns:
            Best hypothesis (most common valid one) or None
        """
        valid_hypotheses = []
        
        for trajectory, hypothesis_str in responses:
            hypothesis, cycle_exist = self.parse_llm_response(hypothesis_str)
            if hypothesis and not cycle_exist:
                is_valid, _ = self.validate_hypothesis_with_reason(hypothesis, observations)
                if is_valid:
                    valid_hypotheses.append(hypothesis)
        
        if not valid_hypotheses:
            return None
        
        # Return most common valid hypothesis (majority voting)
        hypothesis_counts = Counter([h.get_hash() for h in valid_hypotheses])
        most_common_hash = hypothesis_counts.most_common(1)[0][0]
        
        for h in valid_hypotheses:
            if h.get_hash() == most_common_hash:
                return h
        
        return None

    # ========================================================================
    # ACE Generator and Reflector
    # ========================================================================
    
    def generate_reasoning_trajectory(self, llm: LLMInterface, prompt: str) -> Tuple[str, str]:
        """
        Generate reasoning trajectory (LLM's thought process).
        
        Returns:
            (trajectory, final_hypothesis_str): Reasoning process and final hypothesis
        """
        response = llm.query(prompt)
        
        if "Graph:" in response:
            trajectory_part, hypothesis_part = response.split("Graph:", 1)
            trajectory = trajectory_part.strip()
            final_hypothesis_str = "Graph:" + hypothesis_part.strip()
        else:
            trajectory = response.strip()
            final_hypothesis_str = ""
        
        return trajectory, final_hypothesis_str
    
    def reflect_on_trajectory(self, trajectory: str, hypothesis: Optional[CausalGraph], 
                             observations: List[Dict], used_bullet_ids: List[str]) -> Dict:
        """
        ACE Reflector: Analyze trajectory and extract insights.
        
        Args:
            trajectory: Generator's reasoning process
            hypothesis: Generated hypothesis (may be None if parsing failed)
            observations: Perturbation observation data
            used_bullet_ids: Bullet IDs used in prompt
            
        Returns:
            Structured reflection result (error analysis, improvement suggestions, bullet feedback)
        """
        reflection = {
            "reasoning": "",
            "error_identification": "",
            "root_cause": "",
            "improvement_suggestion": "",
            "bullet_feedback": [],
            "new_insight_candidate": ""
        }

        # Validate hypothesis
        is_valid, error_reason = self.validate_hypothesis_with_reason(hypothesis, observations)
        if is_valid:
            reflection["reasoning"] = f"The generated hypothesis (edges: {hypothesis.edges}) is valid."
            reflection["error_identification"] = "No errors: The hypothesis matches all perturbation observations and is a valid DAG."
            reflection["root_cause"] = "Successful application of Playbook strategies."
            reflection["improvement_suggestion"] = "Continue using the current Playbook strategies."
            
            for bullet_id in used_bullet_ids:
                reflection["bullet_feedback"].append({"bullet_id": bullet_id, "is_helpful": True})
            return reflection

        # Handle invalid hypothesis: cycle error
        if "cycle" in error_reason.lower() or not nx.is_directed_acyclic_graph(nx.DiGraph(hypothesis.edges)):
            reflection["error_identification"] = error_reason
            
            cycle_reflection_templates = [
                {
                    "root_cause": "Possible oversight in Playbook application: The 'strategies_and_hard_rules' require explicit cycle checks, but the reasoning trajectory did not mention verifying node paths.",
                    "improvement_suggestion": "After drafting edges, list all node paths (e.g., A→B→C) and check if any path starts and ends at the same node—this catches hidden cycles.",
                    "new_insight_candidate": "Cycle detection tip: For small graphs, draw edges on paper and trace from each node; if you return to the start, it's a cycle (e.g., B→C→B)."
                },
                {
                    "root_cause": "Lack of step-by-step validation: The Playbook's 'is_dag()' logic was not applied, leading to missed cycles in the final hypothesis.",
                    "improvement_suggestion": "Add a mandatory step: After generating edges, run through the 'is_dag()' code mentally—check if removing any edge breaks a potential cycle.",
                    "new_insight_candidate": "Common cycle patterns to watch for: Mutually connected nodes (A→B and B→A) or triangular loops (A→B, B→C, C→A)."
                },
                {
                    "root_cause": "Misinterpretation of 'acyclic' constraint: The trajectory assumed 'no direct self-loops' (e.g., A→A) is sufficient, but indirect cycles were ignored.",
                    "improvement_suggestion": "Explicitly verify indirect paths: For each node X, check if there's a path from X back to X through other nodes (e.g., X→Y→X).",
                    "new_insight_candidate": "Cycle prevention: Build the graph incrementally—add one edge at a time and check for cycles after each addition to catch issues early."
                }
            ]
            
            selected_template = random.choice(cycle_reflection_templates)
            reflection["root_cause"] = selected_template["root_cause"]
            reflection["improvement_suggestion"] = selected_template["improvement_suggestion"]
            reflection["new_insight_candidate"] = selected_template["new_insight_candidate"]
            
            cycle_bullet_id = next((b.bullet_id for b in self.ace_playbook.values() if "cycles" in b.content), None)
            for bullet_id in used_bullet_ids:
                reflection["bullet_feedback"].append({
                    "bullet_id": bullet_id, 
                    "is_helpful": False if bullet_id == cycle_bullet_id else True
                })

        # Handle invalid hypothesis: perturbation effect mismatch
        elif "mismatch" in error_reason.lower() or "perturb" in error_reason.lower():
            reflection["error_identification"] = error_reason
            
            mismatch_reflection_templates = [
                {
                    "root_cause": "Incomplete verification of descendants: The Playbook requires checking all downstream nodes of the perturbed node, but some indirect descendants were missed.",
                    "improvement_suggestion": "For each perturbed node S: List direct children (S→A) and indirect children (A→B) separately, then confirm both are marked as 1 in observations.",
                    "new_insight_candidate": "Effect validation trick: Draw the graph, highlight the perturbed node, and color all reachable nodes—only these should have effect 1; others must be 0."
                },
                {
                    "root_cause": "Confusion between 'direct' and 'indirect' effects: The trajectory only checked direct children of the perturbed node, ignoring nodes affected through intermediaries.",
                    "improvement_suggestion": "Use a 'chain check': For perturbed node S, follow each edge step-by-step (S→A→B→C) and mark all nodes in the chain as needing effect 1.",
                    "new_insight_candidate": "Common mistake: If perturbing S causes B to be 1, there must be a path S→...→B (even if indirect); missing this path causes effect mismatch."
                },
                {
                    "root_cause": "Neglect of 'non-descendant' nodes: The Playbook requires non-descendants of the perturbed node to have effect 0, but some were incorrectly marked as 1.",
                    "improvement_suggestion": "After listing descendants of S, cross-verify all other nodes (non-descendants) to ensure their effect is 0 in the hypothesis.",
                    "new_insight_candidate": "Two-step validation: 1) All descendants of S must be 1; 2) All non-descendants must be 0. Both steps are required to avoid mismatch."
                },
                {
                    "root_cause": "Mismatch between graph structure and effects: The trajectory assumed a node is a descendant but didn't confirm the path in the graph (e.g., claiming S affects B but no S→...→B edge).",
                    "improvement_suggestion": "For each node marked as 1 in observations, explicitly check if there's a path from the perturbed node to it in the hypothesis graph.",
                    "new_insight_candidate": "Effect → graph consistency: If observation says 'perturb S makes B=1', the graph must contain a path S→...→B (add missing edges if this is violated)."
                }
            ]
            
            selected_template = random.choice(mismatch_reflection_templates)
            reflection["root_cause"] = selected_template["root_cause"]
            reflection["improvement_suggestion"] = selected_template["improvement_suggestion"]
            reflection["new_insight_candidate"] = selected_template["new_insight_candidate"]
            
            effect_bullet_id = next((b.bullet_id for b in self.ace_playbook.values() if "perturbation effects" in b.content), None)
            for bullet_id in used_bullet_ids:
                reflection["bullet_feedback"].append({
                    "bullet_id": bullet_id, 
                    "is_helpful": False if bullet_id == effect_bullet_id else True
                })

        return reflection
    
    # ========================================================================
    # Response Parsing and Validation
    # ========================================================================
    
    def parse_llm_response(self, response: str) -> Tuple[Optional[CausalGraph], bool]:
        """
        Parse LLM response to extract causal graph.
        
        Returns:
            (hypothesis, cycle_exist): Parsed graph and whether it contains cycles
        """
        if not isinstance(response, str):
            return None, False
        
        s = response.replace("```", "").strip()
        
        m = re.search(r'(?i)\bgraph\s*:\s*(.+)$', s, flags=re.MULTILINE)
        line = m.group(1).strip() if m else (s.splitlines()[0].strip() if s.splitlines() else "")
        if not line:
            return None, False
        
        if (line.startswith('"') and line.endswith('"')) or (line.startswith("'") and line.endswith("'")):
            line = line[1:-1].strip()
        line = (line
                .replace("→", "->")
                .replace("-->", "->")
                .replace("=>", "->")
                .rstrip(" .;"))
        
        if re.search(r'\b(no\s+edges?|empty|none|null)\b', line, re.I):
            return CausalGraph(nodes=self.nodes, edges=[]), False
        
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if not parts:
            return None, False
        
        edges = []
        for part in parts:
            m = re.fullmatch(r'([A-Za-z0-9_]+)\s*->\s*([A-Za-z0-9_]+)', part)
            if not m:
                return None, False
            u, v = m.group(1), m.group(2)
            if u not in self.nodes or v not in self.nodes or u == v:
                return None, False
            edges.append((u, v))
        
        edges = list(dict.fromkeys(edges))
        
        if self.max_edges is not None and len(edges) > self.max_edges:
            return None, False
        
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from(edges)
        if not nx.is_directed_acyclic_graph(G):
            return CausalGraph(nodes=self.nodes, edges=edges), True

        return CausalGraph(nodes=self.nodes, edges=edges), False
    
    def validate_hypothesis(self, hypothesis: CausalGraph, observations: List[Dict]) -> bool:
        """Check if hypothesis is consistent with observations."""
        for obs_dict in observations:
            perturbed_node = obs_dict['perturbed_node']
            expected_effects = obs_dict['effects']
            
            hypothesis_obs = CausalDatasetGenerator.get_perturbation_effects(hypothesis, perturbed_node)
            
            if hypothesis_obs.effects != expected_effects:
                return False
        
        return True
    
    def validate_hypothesis_with_reason(self, hypothesis: CausalGraph, 
                                       observations: List[Dict]) -> Tuple[bool, str]:
        """
        Validate hypothesis and return error reason.
        
        Returns:
            (is_valid, error_reason): Validation result and error description
        """
        G = nx.DiGraph(hypothesis.edges)
        if not nx.is_directed_acyclic_graph(G):
            cycles = list(nx.simple_cycles(G))
            cycle_str = " -> ".join(cycles[0] + [cycles[0][0]])
            return False, f"Cycle detected in hypothesis: {cycle_str} (violates DAG constraint)"
        
        for obs_dict in observations:
            perturbed_node = obs_dict['perturbed_node']
            expected_effects = obs_dict['effects']
            predicted_obs = CausalDatasetGenerator.get_perturbation_effects(hypothesis, perturbed_node)
            predicted_effects = predicted_obs.effects
            
            if predicted_effects != expected_effects:
                error_nodes = [node for node in expected_effects 
                            if predicted_effects.get(node) != expected_effects[node]]
                return False, f"When perturbing {perturbed_node}, causal effects of {error_nodes} not match"
        
        return True, "Consistent with all observation data"
    
    def _classify_error(self, error_message: str) -> str:
        """Classify error type from error message."""
        if "Expecting value" in error_message:
            match = re.search(r'line (\d+) column (\d+)', error_message)
            if match:
                return f"json_parse_error (line {match.group(1)}, col {match.group(2)})"
            return "json_parse_error"
        elif "Rate limit" in error_message.lower() or "rate_limit" in error_message.lower():
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
            match = re.search(r'\b(\d{3})\b', error_message)
            if match:
                return f"http_error_{match.group(1)}"
            return "unknown_error"
    
    # ========================================================================
    # Single Observation Set Evaluation
    # ========================================================================
    
    def evaluate_single_observation_set(
        self,
        llm: LLMInterface,
        observation_set: Dict,
        n_queries: int = 10,
        verbose: bool = True,
        max_retries: int = 5,
        use_self_consistency: bool = True,
        num_consistency_samples: int = 3
    ) -> Dict:
        """
        *** Main Evaluation Loop with Self-Consistency ***
        Evaluate LLM on a single observation set.
        
        Args:
            llm: LLM interface
            observation_set: Observation data
            n_queries: Number of queries
            verbose: Print progress
            max_retries: Max retries per query
            use_self_consistency: Whether to use self-consistency sampling
            num_consistency_samples: Number of samples for self-consistency
            
        Returns:
            Evaluation results dictionary
        """
        observations = observation_set['observations']
        ground_truth_graphs = [
            CausalGraph.from_dict(g) for g in observation_set['ground_truth_graphs']
        ]
        
        gt_hashes = {g.get_hash() for g in ground_truth_graphs}
        
        hypothesis_with_validity = []
        curated_hypotheses = []
        
        all_hypotheses = []
        valid_hypotheses = []
        unique_hashes = set()
        unique_valid_graphs = []
        all_unique_hashes = set()
        unique_all_graphs = []
        parse_success_count = 0
        
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_cost = 0.0
        
        errors = []
        error_counts = {}
        
        for i in range(n_queries):
            prompt, used_bullet_ids = self.create_prompt(observations, hypothesis_with_validity)

            # Save prompt
            prompt_dir = Path("prompt_logs")
            prompt_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            prompt_file = prompt_dir / f"llm_prompt_{timestamp}.txt"
            with open(prompt_file, "w", encoding="utf-8") as f:
                print(f"query_times={i + 1}", file=f)
                print("\n", file=f)
                print(prompt, file=f)
            
            hypothesis = None
            query_error = None
            trajectory = ""
            
            for attempt in range(max_retries):
                try:
                    # *** Self-Consistency Sampling ***
                    if use_self_consistency and attempt == 0:
                        responses = self.generate_with_self_consistency(
                            llm, prompt, 
                            num_samples=num_consistency_samples,
                            temp_override=0.8
                        )
                        
                        # *** Response Aggregation ***
                        hypothesis = self.aggregate_responses(responses, observations)
                        
                        if hypothesis:
                            trajectory = f"[Self-consistency aggregated from {len(responses)} samples]"
                            response = f"Graph: {', '.join([f'{s}->{d}' for s, d in hypothesis.edges]) or 'No edges'}"
                            cycle_exist = False
                        else:
                            # Fallback to standard query if aggregation fails
                            if hasattr(llm, 'query_with_usage'):
                                result = llm.query_with_usage(prompt)
                                full_response = result['response']
                                if "Graph:" in full_response:
                                    trajectory_part, hypothesis_part = full_response.split("Graph:", 1)
                                    trajectory = trajectory_part.strip()
                                    response = "Graph:" + hypothesis_part.strip()
                                
                                usage = result.get('usage', {})
                                total_prompt_tokens += usage.get('prompt_tokens', 0)
                                total_completion_tokens += usage.get('completion_tokens', 0)
                                total_tokens += usage.get('total_tokens', 0)
                                total_cost += result.get('cost', 0.0)
                            else:
                                trajectory, response = self.generate_reasoning_trajectory(llm, prompt)
                            
                            hypothesis, cycle_exist = self.parse_llm_response(response)
                    else:
                        # Standard query
                        if hasattr(llm, 'query_with_usage'):
                            result = llm.query_with_usage(prompt)
                            full_response = result['response']
                            if "Graph:" in full_response:
                                trajectory_part, hypothesis_part = full_response.split("Graph:", 1)
                                trajectory = trajectory_part.strip()
                                response = "Graph:" + hypothesis_part.strip()
                            
                            usage = result.get('usage', {})
                            total_prompt_tokens += usage.get('prompt_tokens', 0)
                            total_completion_tokens += usage.get('completion_tokens', 0)
                            total_tokens += usage.get('total_tokens', 0)
                            total_cost += result.get('cost', 0.0)
                        else:
                            trajectory, response = self.generate_reasoning_trajectory(llm, prompt)
                        
                        hypothesis, cycle_exist = self.parse_llm_response(response)
                    
                    # Save response
                    response_file = prompt_dir / f"llm_response_{timestamp}.txt"
                    with open(response_file, "w", encoding="utf-8") as f:
                        f.write(trajectory)
                    
                    if response and response.startswith("Error querying"):
                        print(f"  ⚠ LLM returned an error on query {i + 1}, attempt {attempt + 1}: {response}")
                        query_error = {
                            'query_index': i,
                            'attempt': attempt + 1,
                            'error_message': response,
                            'error_type': self._classify_error(response)
                        }
                        error_type = query_error['error_type']
                        error_counts[error_type] = error_counts.get(error_type, 0) + 1
                        continue
                    
                    # Reflect and curate
                    reflection = self.reflect_on_trajectory(
                        trajectory=trajectory,
                        hypothesis=hypothesis,
                        observations=observations,
                        used_bullet_ids=used_bullet_ids
                    )
                    
                    self.curate_reflection(reflection)
                    curated_hypotheses, is_new = self.curate_hypotheses(hypothesis, curated_hypotheses)

                    if not cycle_exist:
                        print("Parse successful: Valid hypothesis")
                        parse_success_count += 1
                        break
                    else:
                        print(f"Parse failed: Invalid hypothesis, response={response}")
                        
                except Exception as e:
                    query_error = {
                        'query_index': i,
                        'attempt': attempt + 1,
                        'error_message': str(e),
                        'error_type': self._classify_error(str(e))
                    }
                    if verbose:
                        print(f"  ⚠ Exception on query {i + 1}: {str(e)[:100]}")

            if not hypothesis and query_error:
                errors.append(query_error)
                continue
            
            if hypothesis:
                if is_new:
                    all_hypotheses.append(hypothesis)
                
                all_h_hash = hypothesis.get_hash()
                if all_h_hash not in all_unique_hashes:
                    all_unique_hashes.add(all_h_hash)
                    unique_all_graphs.append(hypothesis)
                
                is_valid, error_reason = self.validate_hypothesis_with_reason(hypothesis, observations)
                hypothesis_with_validity.append((hypothesis, is_valid, reflection["error_identification"]))
                
                if is_valid:
                    valid_hypotheses.append(hypothesis)
                    
                    h_hash = hypothesis.get_hash()
                    if h_hash not in unique_hashes:
                        unique_hashes.add(h_hash)
                        unique_valid_graphs.append(hypothesis)
        
        # Calculate metrics
        valid_rate = len(valid_hypotheses) / n_queries if n_queries > 0 else 0
        novelty_rate = len(unique_all_graphs) / n_queries if n_queries > 0 else 0
        parse_success_rate = parse_success_count / n_queries if n_queries > 0 else 0
        
        recovered_gts = set()
        for graph in unique_valid_graphs:
            if graph.get_hash() in gt_hashes:
                recovered_gts.add(graph.get_hash())
        
        recovery_rate = len(recovered_gts) / len(gt_hashes) if gt_hashes else 0

        pruned_count = self.prune_playbook()
        if verbose:
            print(f"Pruned {pruned_count} low-value bullets, remaining: {len(self.ace_playbook)}")

        result_dict = {
            'observation_set_id': observation_set.get('observation_set_id', 'unknown'),
            'n_observations': len(observations),
            'n_ground_truths': len(ground_truth_graphs),
            'n_queries': n_queries,
            'n_valid': len(valid_hypotheses),
            'n_unique_valid': len(unique_valid_graphs),
            'n_unique_all': len(unique_all_graphs),
            'n_recovered_gts': len(recovered_gts),
            'parse_success_count': parse_success_count,
            'parse_success_rate': parse_success_rate,
            'valid_rate': valid_rate,
            'novelty_rate': novelty_rate,
            'recovery_rate': recovery_rate,
            'token_usage': {
                'prompt_tokens': total_prompt_tokens,
                'completion_tokens': total_completion_tokens,
                'total_tokens': total_tokens
            },
            'cost': total_cost,
            'errors': errors,
            'error_summary': {
                'total_errors': len(errors),
                'error_types': error_counts
            },
            'all_hypotheses': [h.to_dict() for h in all_hypotheses],
            'valid_hypotheses': [h.to_dict() for h in valid_hypotheses],
            'unique_graphs': [g.to_dict() for g in unique_valid_graphs]
        }

        # Save results
        save_dir = Path("sample_results")
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"evaluation_result_{timestamp}.json"
        save_path = save_dir / file_name
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        return result_dict
    
    # ========================================================================
    # Full Benchmark Execution
    # ========================================================================
    
    def run_benchmark(
        self,
        llm: LLMInterface,
        n_samples: int = 10,
        n_queries_per_sample: Optional[int] = None,
        query_multiplier: float = 2.0,
        seed: Optional[int] = None,
        verbose: bool = True,
        checkpoint_dir: str = "checkpoints",
        max_retries: int = 3,
        use_self_consistency: bool = True,
        num_consistency_samples: int = 3
    ) -> Dict:
        """
        *** Run Full Benchmark with Self-Consistency ***
        
        Args:
            llm: LLM interface
            n_samples: Number of observation sets to sample
            n_queries_per_sample: Fixed number of queries per set
            query_multiplier: Multiplier for adaptive queries
            seed: Random seed
            verbose: Print progress
            checkpoint_dir: Checkpoint directory
            max_retries: Max retries per query
            use_self_consistency: Enable self-consistency sampling
            num_consistency_samples: Number of samples for self-consistency
            
        Returns:
            Complete benchmark results
        """
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_llm_name = llm.get_name().replace('/', '_').replace('(', '_').replace(')', '_').replace(' ', '_')
        checkpoint_file = checkpoint_path / f"checkpoint_causal_enhanced_{safe_llm_name}_{run_id}.json"
        
        print(f"\nRunning Enhanced Causal Benchmark")
        print(f"LLM: {llm.get_name()}")
        print(f"Sampling {n_samples} observation sets")
        if n_queries_per_sample is not None:
            print(f"Queries per sample: {n_queries_per_sample} (fixed)")
        else:
            print(f"Queries per sample: {query_multiplier}x number of ground truths (adaptive)")
        print(f"Max retries: {max_retries}")
        print(f"Self-consistency: {'Enabled' if use_self_consistency else 'Disabled'}")
        if use_self_consistency:
            print(f"  Consistency samples: {num_consistency_samples}")
        print(f"Checkpoint file: {checkpoint_file}")
        print("-" * 50)
        
        sampled_sets = self.sample_observation_sets(n_samples, seed)
        
        all_results = []
        valid_rates = []
        novelty_rates = []
        recovery_rates = []
        parse_success_rates = []
        
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_cost = 0.0
        
        all_errors = []
        total_error_counts = {}
        
        start_idx = 0
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                    all_results = checkpoint_data.get('results', [])
                    start_idx = len(all_results)
                    
                    total_prompt_tokens = checkpoint_data.get('total_prompt_tokens', 0)
                    total_completion_tokens = checkpoint_data.get('total_completion_tokens', 0)
                    total_tokens = checkpoint_data.get('total_tokens', 0)
                    total_cost = checkpoint_data.get('total_cost', 0.0)
                    
                    all_errors = checkpoint_data.get('all_errors', [])
                    total_error_counts = checkpoint_data.get('total_error_counts', {})
                    
                    print(f"Resuming from checkpoint: {start_idx}/{n_samples} completed")
                    
                    for result in all_results:
                        valid_rates.append(result['valid_rate'])
                        novelty_rates.append(result['novelty_rate'])
                        recovery_rates.append(result['recovery_rate'])
                        parse_success_rates.append(result.get('parse_success_rate', 1.0))
            except Exception as e:
                print(f"Warning: Failed to load checkpoint: {e}")
                print("Starting from beginning...")
        
        for idx in range(start_idx, len(sampled_sets)):
            obs_set = sampled_sets[idx]
            
            if verbose:
                print(f"\nSample {idx + 1}/{n_samples}")
                print(f"  Observation set ID: {obs_set.get('observation_set_id', 'unknown')}")
                print(f"  Number of observations: {len(obs_set['observations'])}")
                print(f"  Number of ground truths: {obs_set['n_compatible_graphs']}")
            
            try:
                if n_queries_per_sample is not None:
                    n_queries = n_queries_per_sample
                else:
                    n_gt = obs_set['n_compatible_graphs']
                    n_queries = max(1, int(n_gt * query_multiplier))
                    if verbose:
                        print(f"  Using {n_queries} queries ({query_multiplier}x {n_gt} ground truths)")
                
                result = self.evaluate_single_observation_set(
                    llm, obs_set, n_queries, verbose=False, max_retries=max_retries,
                    use_self_consistency=use_self_consistency,
                    num_consistency_samples=num_consistency_samples
                )
                
                all_results.append(result)
                valid_rates.append(result['valid_rate'])
                novelty_rates.append(result['novelty_rate'])
                recovery_rates.append(result['recovery_rate'])
                parse_success_rates.append(result['parse_success_rate'])
                
                if 'token_usage' in result:
                    total_prompt_tokens += result['token_usage']['prompt_tokens']
                    total_completion_tokens += result['token_usage']['completion_tokens']
                    total_tokens += result['token_usage']['total_tokens']
                if 'cost' in result:
                    total_cost += result['cost']
                
                if 'errors' in result and result['errors']:
                    all_errors.extend(result['errors'])
                    if 'error_summary' in result:
                        for error_type, count in result['error_summary']['error_types'].items():
                            total_error_counts[error_type] = total_error_counts.get(error_type, 0) + count
                
                if verbose:
                    print(f"  Parse success rate: {result['parse_success_rate']:.2%}")
                    print(f"  Valid rate: {result['valid_rate']:.2%}")
                    print(f"  Novelty rate: {result['novelty_rate']:.2%}")
                    print(f"  Recovery rate: {result['recovery_rate']:.2%}")
                    if result['cost'] > 0:
                        print(f"  Cost: ${result['cost']:.6f}")
                
                checkpoint_data = {
                    'run_id': run_id,
                    'llm_name': llm.get_name(),
                    'n_samples': n_samples,
                    'n_queries_per_sample': n_queries_per_sample,
                    'query_multiplier': query_multiplier if n_queries_per_sample is None else None,
                    'seed': seed,
                    'timestamp': datetime.now().isoformat(),
                    'results': all_results,
                    'total_prompt_tokens': total_prompt_tokens,
                    'total_completion_tokens': total_completion_tokens,
                    'total_tokens': total_tokens,
                    'total_cost': total_cost,
                    'all_errors': all_errors,
                    'total_error_counts': total_error_counts
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
        
        def calculate_p_value(rates):
            if not rates or len(rates) < 2:
                return None
            t_stat, p_val = stats.ttest_1samp(rates, 0)
            return p_val
        
        final_results = {
            'run_id': run_id,
            'llm_name': llm.get_name(),
            'n_samples': len(all_results),
            'n_queries_per_sample': n_queries_per_sample,
            'query_multiplier': query_multiplier if n_queries_per_sample is None else None,
            'query_mode': 'fixed' if n_queries_per_sample is not None else f'adaptive_{query_multiplier}x',
            'seed': seed,
            'timestamp': datetime.now().isoformat(),
            'max_edges_constraint': self.max_edges,
            'self_consistency_enabled': use_self_consistency,
            'num_consistency_samples': num_consistency_samples if use_self_consistency else None,
            'statistics': {
                'parse_success_rate': {
                    **calculate_stats(parse_success_rates),
                    'p_value': calculate_p_value(parse_success_rates)
                },
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
        
        # Print comprehensive summary
        print("\n" + "=" * 60)
        print("ENHANCED BENCHMARK RESULTS SUMMARY")
        print("=" * 60)
        print(f"Samples evaluated: {len(all_results)}/{n_samples}")
        print(f"Max edges constraint: {self.max_edges if self.max_edges is not None else 'unlimited'}")
        print(f"Self-consistency: {'Enabled' if use_self_consistency else 'Disabled'}")
        if use_self_consistency:
            print(f"  Consistency samples: {num_consistency_samples}")
        
        for metric_name, metric_key in [('Parse Success Rate', 'parse_success_rate'),
                                        ('Valid Rate', 'valid_rate'), 
                                        ('Novelty Rate', 'novelty_rate'), 
                                        ('Recovery Rate', 'recovery_rate')]:
            stats_dict = final_results['statistics'][metric_key]
            print(f"\n{metric_name}:")
            print(f"  Mean ± Std: {stats_dict['mean']:.3f} ± {stats_dict['std']:.3f}")
            print(f"  Variance: {stats_dict['var']:.3f}")
            print(f"  Range: [{stats_dict['min']:.3f}, {stats_dict['max']:.3f}]")
            if stats_dict['p_value'] is not None:
                print(f"  p-value: {stats_dict['p_value']:.4f}")
        
        print(f"\nToken Usage:")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Prompt tokens: {total_prompt_tokens:,}")
        print(f"  Completion tokens: {total_completion_tokens:,}")
        print(f"  Avg tokens/sample: {final_results['token_usage']['avg_tokens_per_sample']:.1f}")
        print(f"  Avg tokens/query: {final_results['token_usage']['avg_tokens_per_query']:.1f}")
        
        print(f"\nCost:")
        print(f"  Total cost: ${total_cost:.4f}")
        print(f"  Avg cost/sample: ${final_results['cost']['avg_cost_per_sample']:.4f}")
        print(f"  Avg cost/query: ${final_results['cost']['avg_cost_per_query']:.6f}")
        
        if all_errors:
            print(f"\nErrors:")
            print(f"  Total errors: {len(all_errors)}")
            print(f"  Error rate: {final_results['error_summary']['error_rate']:.2%}")
            if total_error_counts:
                print(f"  Error types:")
                for error_type, count in sorted(total_error_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"    - {error_type}: {count}")
        
        print("=" * 60)
        
        # Clean up checkpoint file after successful completion
        if checkpoint_file.exists():
            try:
                checkpoint_file.unlink()
                print(f"\nCleaned up checkpoint: {checkpoint_file}")
            except Exception:
                pass
        
        return final_results


# ============================================================================
# Utility Functions
# ============================================================================

def setup_llm(llm_type: str, **kwargs) -> LLMInterface:
    """
    Set up LLM interface based on type.
    
    Args:
        llm_type: Type of LLM ('openai', 'anthropic', 'openrouter')
        **kwargs: Additional parameters (api_key, model, temperature, etc.)
        
    Returns:
        Configured LLM interface
    """
    
    if llm_type == "openai":
        api_key = kwargs.get('api_key') or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key required")
        
        return OpenAILLM(
            model=kwargs.get('model', 'gpt-4o'),
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


def parse_n_observations_filter(filter_str: str) -> List[int]:
    """
    Parse n_observations filter string.
    
    Args:
        filter_str: String like "2,3,5" or "2-5" or "2,4-6,8"
        
    Returns:
        List of n_observations values to include
    """
    if not filter_str:
        return []
    
    result = []
    parts = filter_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part and not part.startswith('-'):
            start, end = part.split('-')
            start, end = int(start.strip()), int(end.strip())
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    
    return sorted(list(set(result)))


def parse_gt_filter(filter_str: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse ground truth filter string.
    
    Args:
        filter_str: String like "10-16" for range or "1,2,4" for specific values
        
    Returns:
        Tuple of (min_gt, max_gt) for range, or (values_list, None) for specific values
    """
    if not filter_str:
        return None, None
    
    if '-' in filter_str and not filter_str.startswith('-'):
        parts = filter_str.split('-')
        if len(parts) == 2:
            try:
                min_gt = int(parts[0].strip())
                max_gt = int(parts[1].strip())
                return min_gt, max_gt
            except ValueError:
                pass
    
    try:
        values = []
        for part in filter_str.split(','):
            values.append(int(part.strip()))
        return sorted(values), None
    except ValueError:
        print(f"Warning: Invalid GT filter format: {filter_str}")
        return None, None


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run enhanced causal discovery benchmark with self-consistency sampling\n\n"
                    "Features:\n"
                    "- ACE (Autonomous Contextualization and Execution) framework\n"
                    "- Self-consistency sampling for diversity\n"
                    "- Response aggregation for robustness\n"
                    "- Token usage and cost tracking\n"
                    "- Checkpoint mechanism for resuming\n"
                    "- Enhanced error handling\n"
                    "- Statistical analysis of results",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument("--dataset", required=True, help="Path to complete causal dataset JSON file")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--n-samples", type=int, default=30, help="Number of observation sets to sample")
    parser.add_argument("--n-observations-filter", type=str, default=None, 
                       help="Filter datasets by n_observations (e.g., '2,3,5' or '2-5')")
    parser.add_argument("--gt-filter", type=str, default=None, 
                       help="Filter datasets by number of ground truth graphs (e.g., '10-16' or '1,2,4')")
    parser.add_argument("--n-queries", type=int, default=None, 
                       help="Fixed number of queries per observation set")
    parser.add_argument("--query-multiplier", type=float, default=2.0, 
                       help="Multiplier for adaptive queries")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retries per query")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--output", default=None, help="Output file path")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose output")
    parser.add_argument("--use-self-consistency", action="store_true", default=True,
                       help="Enable self-consistency sampling")
    parser.add_argument("--no-self-consistency", action="store_true",
                       help="Disable self-consistency sampling")
    parser.add_argument("--num-consistency-samples", type=int, default=3,
                       help="Number of samples for self-consistency (default: 3)")
    
    args = parser.parse_args()
    
    # Handle verbose/quiet flags
    if args.quiet:
        args.verbose = False
    
    # Handle self-consistency flag
    if args.no_self_consistency:
        args.use_self_consistency = False
    
    # Load configuration
    config = load_config(args.config)
    llm_type = config.get('llm', {}).get('type', 'openrouter')
    
    model = config.get('llm', {}).get('models', {}).get(llm_type)
    if not model:
        default_models = {
            'openrouter': 'openai/gpt-3.5-turbo',
            'openai': 'gpt-4o',
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
    checkpoint_dir = args.checkpoint_dir or config.get('benchmark', {}).get('checkpoint_dir', 'checkpoints')
    verbose = args.verbose and config.get('benchmark', {}).get('verbose', True)
    
    if not Path(args.dataset).exists():
        print(f"Error: Dataset file not found: {args.dataset}")
        sys.exit(1)
    
    # Generate output filename if not specified
    if args.output is None:
        dataset_name = Path(args.dataset).stem
        model_name = Path(model).stem if model else llm_type
        output_pattern = config.get('benchmark', {}).get("output_pattern", "results/{dataset_name}_{model}.json")
        output = output_pattern.format(dataset_name=dataset_name, model=model_name)
    else:
        output = args.output
    
    # Parse filters if provided
    n_observations_filter = None
    if args.n_observations_filter:
        n_observations_filter = parse_n_observations_filter(args.n_observations_filter)
        print(f"Filtering for n_observations: {n_observations_filter}")
    
    gt_filter = None
    if args.gt_filter:
        gt_filter = parse_gt_filter(args.gt_filter)
        if gt_filter[0] is not None:
            if gt_filter[1] is not None:
                print(f"Filtering for n_compatible_graphs: [{gt_filter[0]}, {gt_filter[1]}]")
            else:
                print(f"Filtering for n_compatible_graphs: {gt_filter[0]}")
    
    # Initialize benchmark with filters
    benchmark = CausalBenchmarkEnhanced(args.dataset, 
                                       n_observations_filter=n_observations_filter,
                                       gt_filter=gt_filter)
    
    # Print configuration
    print("\n" + "=" * 60)
    print("ENHANCED CAUSAL BENCHMARK CONFIGURATION")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"LLM Type: {llm_type}")
    print(f"Model: {model}")
    print(f"Temperature: {temperature}")
    print(f"Samples: {args.n_samples}")
    
    if args.n_queries is not None:
        print(f"Queries per sample: {args.n_queries} (fixed)")
    else:
        print(f"Queries per sample: {args.query_multiplier}x ground truths (adaptive)")
    
    print(f"Max retries: {args.max_retries}")
    print(f"Self-consistency: {'Enabled' if args.use_self_consistency else 'Disabled'}")
    if args.use_self_consistency:
        print(f"  Consistency samples: {args.num_consistency_samples}")
    print(f"Seed: {args.seed}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Output: {output}")
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
        max_retries=args.max_retries,
        use_self_consistency=args.use_self_consistency,
        num_consistency_samples=args.num_consistency_samples
    )
    
    # Save final results
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nFinal results saved to: {output}")


if __name__ == "__main__":
    main()
