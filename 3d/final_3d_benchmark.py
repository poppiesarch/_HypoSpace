#!/usr/bin/env python3
"""
3D Structure Discovery Benchmark - HSP-Guided Sampling Strategy
HSP provides systematic sampling strategies, LLM generates structures following guidance
"""

import sys
import json
import argparse
import os
import re
import yaml
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any
from datetime import datetime
from scipy import stats
from itertools import product
import traceback
import hashlib

sys.path.append(str(Path(__file__).parent))
from modules.llm_interface import LLMInterface, OpenRouterLLM, OpenAILLM, AnthropicLLM


# ******************************************************************************
# * CORE DATA STRUCTURES
# ******************************************************************************

class Structure3D:
    """Represents a 3D structure"""
    
    def __init__(self, layers: List[List[List[int]]]):
        if not layers:
            self.layers = []
            self.height = 0
            self.shape = (0, 0)
            return
            
        # Find maximum dimensions
        max_rows = 0
        max_cols = 0
        for layer in layers:
            if layer:
                max_rows = max(max_rows, len(layer))
                for row in layer:
                    if row:
                        max_cols = max(max_cols, len(row))
        
        # Pad all layers to consistent shape
        self.layers = []
        for layer in layers:
            padded_layer = np.zeros((max_rows, max_cols), dtype=int)
            for i, row in enumerate(layer[:max_rows]):
                for j, val in enumerate(row[:max_cols]):
                    padded_layer[i, j] = val
            self.layers.append(padded_layer)
        
        self.height = len(self.layers)
        self.shape = (max_rows, max_cols) if self.layers else (0, 0)
    
    def to_string(self) -> str:
        """Convert to string representation"""
        result = []
        for i, layer in enumerate(self.layers):
            result.append(f"Layer {i+1}:")
            for row in layer:
                result.append(" ".join(str(cell) for cell in row))
        return "\n".join(result)
    
    def get_top_view(self) -> np.ndarray:
        """Get top view (OR projection)"""
        if not self.layers:
            return np.zeros((0, 0), dtype=int)
        
        top = np.zeros_like(self.layers[0], dtype=int)
        for L in self.layers:
            if L.shape == top.shape:
                top |= (L.astype(bool)).astype(int)
            else:
                min_rows = min(L.shape[0], top.shape[0])
                min_cols = min(L.shape[1], top.shape[1])
                top[:min_rows, :min_cols] |= (L[:min_rows, :min_cols].astype(bool)).astype(int)
        return top
    
    def normalize(self) -> 'Structure3D':
        """Remove trailing zero layers"""
        if not self.layers:
            return Structure3D([])
        
        last_non_zero = -1
        for i in range(len(self.layers) - 1, -1, -1):
            if np.any(self.layers[i] != 0):
                last_non_zero = i
                break
        
        if last_non_zero == -1:
            return Structure3D([self.layers[0] * 0])
        
        return Structure3D([layer.tolist() for layer in self.layers[:last_non_zero + 1]])
    
    def get_hash(self) -> str:
        """Get normalized hash"""
        normalized = self.normalize()
        
        if not normalized.layers:
            return "empty000"
        
        arr = np.stack(normalized.layers, axis=0).astype(np.uint8)
        h = hashlib.md5()
        h.update(arr.tobytes())
        h.update(np.array(arr.shape, dtype=np.int64).tobytes())
        return h.hexdigest()[:8]


# ******************************************************************************
# * HSP SAMPLING STRATEGY GENERATOR
# ******************************************************************************

class HSPSamplingStrategy:
    """
    Generates systematic sampling strategies based on HSP principles
    Provides concrete height assignments for LLM to follow
    """
    
    def __init__(self, default_grid_size: int = 3, max_height: int = 3):
        self.default_grid_size = default_grid_size  # Only used as fallback
        self.max_height = max_height
    
    def generate_sampling_plan(
        self,
        observation: np.ndarray,
        n_samples: int,
        seed: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate a systematic sampling plan for LLM to follow
        
        Args:
            observation: 2D numpy array of the top view
            n_samples: number of samples to generate
            seed: random seed for reproducibility
        
        Returns:
            List of sampling instructions, each specifying:
            - block_positions: which positions have blocks
            - height_assignment: specific heights for each block
            - strategy_type: which sampling strategy this follows
            - instruction: natural language instruction for LLM
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # ✅ FIX: Infer grid_size from observation shape
        if observation.ndim == 2:
            grid_size = observation.shape[0]
        elif observation.ndim == 1:
            total_cells = len(observation)
            grid_size = int(np.sqrt(total_cells))
            if grid_size * grid_size != total_cells:
                raise ValueError(f"Cannot infer grid_size from observation length {total_cells}")
            observation = observation.reshape(grid_size, grid_size)
        else:
            raise ValueError(f"Unexpected observation shape: {observation.shape}")
        
        # ✅ FIX: Extract block positions using the inferred grid_size
        block_positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                if observation[i, j] == 1:
                    block_positions.append((i, j))
        
        n_blocks = len(block_positions)
        
        if n_blocks == 0:
            return []
        
        # Calculate total hypothesis space
        total_space = self.max_height ** n_blocks
        
        # Generate sampling strategies
        sampling_plan = []
        
        # Strategy 1: Systematic height enumeration (for small spaces)
        if total_space <= 50:  # Can enumerate most/all
            all_combinations = list(product(range(1, self.max_height + 1), repeat=n_blocks))
            
            # Sample systematically
            if n_samples >= len(all_combinations):
                # Take all combinations
                selected = all_combinations
            else:
                # Stratified sampling: ensure coverage of different patterns
                selected = self._stratified_sample(all_combinations, n_samples)
            
            for heights in selected:
                sampling_plan.append({
                    'block_positions': block_positions,
                    'height_assignment': dict(zip(block_positions, heights)),
                    'strategy_type': 'systematic_enumeration',
                    'instruction': self._create_height_instruction(block_positions, heights)
                })
        
        else:  # Large space: use diverse sampling strategies
            strategies = self._generate_diverse_strategies(block_positions, n_samples)
            sampling_plan.extend(strategies)
        
        # Shuffle to avoid order bias
        random.shuffle(sampling_plan)
        
        return sampling_plan[:n_samples]
    
    def _stratified_sample(
        self,
        combinations: List[Tuple[int, ...]],
        n_samples: int
    ) -> List[Tuple[int, ...]]:
        """
        Stratified sampling to ensure diverse coverage
        Groups by patterns: uniform, increasing, decreasing, mixed
        """
        # Categorize combinations
        uniform = []
        increasing = []
        decreasing = []
        mixed = []
        
        for combo in combinations:
            if len(set(combo)) == 1:
                uniform.append(combo)
            elif all(combo[i] <= combo[i+1] for i in range(len(combo)-1)):
                increasing.append(combo)
            elif all(combo[i] >= combo[i+1] for i in range(len(combo)-1)):
                decreasing.append(combo)
            else:
                mixed.append(combo)
        
        # Allocate samples proportionally
        total = len(combinations)
        strata = [uniform, increasing, decreasing, mixed]
        strata_sizes = [len(s) for s in strata]
        
        selected = []
        remaining = n_samples
        
        for i, stratum in enumerate(strata):
            if not stratum:
                continue
            
            # Proportional allocation
            if total > 0:
                n_from_stratum = max(1, int(n_samples * strata_sizes[i] / total))
                n_from_stratum = min(n_from_stratum, len(stratum), remaining)
            else:
                n_from_stratum = 0
            
            selected.extend(random.sample(stratum, n_from_stratum))
            remaining -= n_from_stratum
            
            if remaining <= 0:
                break
        
        # Fill remaining with random samples
        if remaining > 0:
            available = [c for c in combinations if c not in selected]
            if available:
                selected.extend(random.sample(available, min(remaining, len(available))))
        
        return selected
    
    def _generate_diverse_strategies(
        self,
        block_positions: List[Tuple[int, int]],
        n_samples: int
    ) -> List[Dict[str, Any]]:
        """
        Generate diverse sampling strategies for large hypothesis spaces
        """
        strategies = []
        n_blocks = len(block_positions)
        
        # Calculate how many samples per strategy type
        strategy_types = [
            'uniform_heights',      # All blocks same height
            'increasing_pattern',   # Heights increase
            'decreasing_pattern',   # Heights decrease
            'alternating_pattern',  # Alternate high/low
            'random_diverse',       # Random but diverse
            'extreme_values',       # Focus on min/max heights
            'clustered_heights'     # Groups of similar heights
        ]
        
        samples_per_strategy = max(1, n_samples // len(strategy_types))
        
        # 1. Uniform heights
        for h in range(1, self.max_height + 1):
            if len(strategies) >= samples_per_strategy:
                break
            heights = tuple([h] * n_blocks)
            strategies.append({
                'block_positions': block_positions,
                'height_assignment': dict(zip(block_positions, heights)),
                'strategy_type': 'uniform_heights',
                'instruction': self._create_height_instruction(block_positions, heights)
            })
        
        # 2. Increasing patterns
        for _ in range(samples_per_strategy):
            heights = sorted([random.randint(1, self.max_height) for _ in range(n_blocks)])
            strategies.append({
                'block_positions': block_positions,
                'height_assignment': dict(zip(block_positions, heights)),
                'strategy_type': 'increasing_pattern',
                'instruction': self._create_height_instruction(block_positions, heights)
            })
        
        # 3. Decreasing patterns
        for _ in range(samples_per_strategy):
            heights = sorted([random.randint(1, self.max_height) for _ in range(n_blocks)], reverse=True)
            strategies.append({
                'block_positions': block_positions,
                'height_assignment': dict(zip(block_positions, heights)),
                'strategy_type': 'decreasing_pattern',
                'instruction': self._create_height_instruction(block_positions, heights)
            })
        
        # 4. Alternating patterns
        for _ in range(samples_per_strategy):
            heights = []
            for i in range(n_blocks):
                if i % 2 == 0:
                    heights.append(random.randint(1, self.max_height // 2 + 1))
                else:
                    heights.append(random.randint(self.max_height // 2, self.max_height))
            strategies.append({
                'block_positions': block_positions,
                'height_assignment': dict(zip(block_positions, heights)),
                'strategy_type': 'alternating_pattern',
                'instruction': self._create_height_instruction(block_positions, heights)
            })
        
        # 5. Random diverse (ensure good coverage)
        for _ in range(samples_per_strategy):
            heights = tuple([random.randint(1, self.max_height) for _ in range(n_blocks)])
            strategies.append({
                'block_positions': block_positions,
                'height_assignment': dict(zip(block_positions, heights)),
                'strategy_type': 'random_diverse',
                'instruction': self._create_height_instruction(block_positions, heights)
            })
        
        # 6. Extreme values (all 1s, all max, mixed extremes)
        extreme_patterns = [
            [1] * n_blocks,
            [self.max_height] * n_blocks,
            [1 if i % 2 == 0 else self.max_height for i in range(n_blocks)],
            [self.max_height if i % 2 == 0 else 1 for i in range(n_blocks)]
        ]
        for pattern in extreme_patterns[:samples_per_strategy]:
            strategies.append({
                'block_positions': block_positions,
                'height_assignment': dict(zip(block_positions, pattern)),
                'strategy_type': 'extreme_values',
                'instruction': self._create_height_instruction(block_positions, pattern)
            })
        
        # 7. Clustered heights (groups of similar heights)
        for _ in range(samples_per_strategy):
            # Divide blocks into groups
            n_groups = random.randint(1, min(3, n_blocks))
            group_heights = [random.randint(1, self.max_height) for _ in range(n_groups)]
            heights = []
            for i in range(n_blocks):
                group_idx = i % n_groups
                heights.append(group_heights[group_idx])
            strategies.append({
                'block_positions': block_positions,
                'height_assignment': dict(zip(block_positions, heights)),
                'strategy_type': 'clustered_heights',
                'instruction': self._create_height_instruction(block_positions, heights)
            })
        
        # Ensure we have exactly n_samples
        random.shuffle(strategies)
        return strategies[:n_samples]
    
    def _create_height_instruction(
        self,
        block_positions: List[Tuple[int, int]],
        heights: Tuple[int, ...]
    ) -> str:
        """
        Create natural language instruction for specific height assignment
        """
        instruction = "Generate a structure with the following EXACT height configuration:\n"
        
        for (row, col), height in zip(block_positions, heights):
            instruction += f"  - Position ({row},{col}): height = {height} (blocks from layer 1 to layer {height})\n"
        
        instruction += "\nIMPORTANT: Follow this height assignment EXACTLY. "
        instruction += "Each position must have blocks stacked from layer 1 up to the specified height."
        
        return instruction


# ******************************************************************************
# * MAIN BENCHMARK CLASS
# ******************************************************************************

class Benchmark3D:
    """3D structure discovery benchmark with HSP-guided sampling"""
    
    def __init__(self, dataset_path: str):
        with open(dataset_path, 'r') as f:
            self.complete_dataset = json.load(f)
        
        self.metadata = self.complete_dataset.get('metadata', {})
        self.all_observation_sets = self.complete_dataset.get('observation_sets', [])
        
        print(f"Loaded dataset with {len(self.all_observation_sets)} observation sets")
        
        # Initialize HSP sampling strategy generator
        grid_size = self.metadata.get('grid_size', 3)
        max_height = self.metadata.get('max_height', 3)
        self.hsp_sampler = HSPSamplingStrategy(default_grid_size=grid_size, max_height=max_height)
    
    def sample_observation_sets(
        self,
        n_samples: int,
        observation_type: str = "top",
        seed: Optional[int] = None
    ) -> List[Dict]:
        """Sample observation sets from dataset"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        filtered_sets = self.all_observation_sets
        n_available = len(filtered_sets)
        n_to_sample = min(n_samples, n_available)
        
        if n_to_sample < n_samples:
            print(f"Warning: Requested {n_samples} samples but only {n_available} available")
        
        if n_available == 0:
            print(f"Error: No observation sets found")
            return []
        
        print(f"Sampling {n_to_sample} observation sets")
        return random.sample(filtered_sets, n_to_sample)
    
    def _parse_observation(self, obs_data):
        """Parse observation data to numpy array"""
        if isinstance(obs_data, str):
            length = len(obs_data)
            grid_size = int(length ** 0.5)
            observation = []
            for i in range(grid_size):
                row = [int(obs_data[i * grid_size + j]) for j in range(grid_size)]
                observation.append(row)
            return np.array(observation)
        elif isinstance(obs_data, list):
            return np.array(obs_data)
        else:
            return np.array(obs_data.get('observation', []))
    
    def create_prompt_with_sampling_strategy(
        self,
        observations,
        sampling_instruction: Dict[str, Any],
        prior_structures: Optional[List[Structure3D]] = None
    ) -> str:
        """
        Create prompt with specific HSP sampling strategy
        """
        # Determine grid size
        if isinstance(observations, str):
            grid_size = int(len(observations) ** 0.5)
        elif isinstance(observations, list) and observations:
            obs_data = self._parse_observation(observations[0])
            grid_size = obs_data.shape[0] if hasattr(obs_data, 'shape') else 3
        else:
            obs_data = self._parse_observation(observations)
            grid_size = obs_data.shape[0] if hasattr(obs_data, 'shape') else 3
        
        max_height = self.metadata.get('max_height', 3)
        
        prompt = f"""You are given observations of a 3D structure made of unit blocks on a {grid_size}x{grid_size} grid.
    Each observation shows a top view (1 if ANY layer has a block at that position).
    Maximum height: {max_height} layers.

    Observations (Top View):
    """
        
        # Add observation data
        if isinstance(observations, str):
            observation = self._parse_observation(observations)
            for row in observation:
                prompt += " ".join(str(cell) for cell in row) + "\n"
        elif isinstance(observations, list):
            for obs in observations:
                observation = np.array(obs['observation']) if isinstance(obs, dict) else self._parse_observation(obs)
                for row in observation:
                    prompt += " ".join(str(cell) for cell in row) + "\n"
        else:
            observation = self._parse_observation(observations)
            for row in observation:
                prompt += " ".join(str(cell) for cell in row) + "\n"
        
        # ⭐⭐⭐ 添加清晰的坐标系说明 ⭐⭐⭐
        prompt += f"\n{'='*60}\n"
        prompt += "COORDINATE SYSTEM (IMPORTANT):\n"
        prompt += f"{'='*60}\n"
        prompt += f"Position (row, col) uses 0-based indexing:\n"
        prompt += f"  - (0,0) = top-left corner\n"
        prompt += f"  - (0,{grid_size-1}) = top-right corner\n"
        prompt += f"  - ({grid_size-1},0) = bottom-left corner\n"
        prompt += f"  - ({grid_size-1},{grid_size-1}) = bottom-right corner\n\n"
        
        # 添加观察字符串到坐标的映射示例
        if isinstance(observations, str):
            prompt += f"The observation string maps to the grid as follows:\n"
            prompt += f"String: \"{observations}\"\n"
            prompt += f"Grid positions:\n"
            for i in range(grid_size):
                row_positions = []
                for j in range(grid_size):
                    idx = i * grid_size + j
                    row_positions.append(f"({i},{j})")
                prompt += "  " + " ".join(row_positions) + "\n"
            prompt += "\n"
        
        # Add HSP sampling strategy instruction
        prompt += f"{'='*60}\n"
        prompt += "SAMPLING STRATEGY (Follow this EXACTLY):\n"
        prompt += f"{'='*60}\n"
        prompt += f"Strategy Type: {sampling_instruction['strategy_type']}\n\n"
        prompt += sampling_instruction['instruction']
        prompt += f"\n{'='*60}\n"
        
        # Add prior structures if any
        if prior_structures and len(prior_structures) > 0:
            prompt += "\n\nPreviously generated structures (for reference, generate different if possible):\n"
            for idx, struct in enumerate(prior_structures[-3:], 1):  # Show last 3
                prompt += f"\nStructure {idx}:\n"
                prompt += struct.to_string() + "\n"
        
        # Task description
        prompt += f"""

    Task: Generate the 3D structure following the EXACT height assignment specified above.

    Structure specifications:
    - Grid size: {grid_size}x{grid_size}
    - Maximum height: {max_height} layers
    - Layers stacked from bottom (Layer 1) to top (Layer N)
    - Each layer: {grid_size}x{grid_size} grid with 0 (empty) or 1 (block)

    Critical constraints:
    1. Layer 1 (bottom) must contain blocks at the specified positions
    2. Blocks at height h require support at height h-1 (same position)
    3. Follow the EXACT height assignment given in the sampling strategy
    4. Top view must match the observation
    5. Do not add unnecessary empty layers

    Output format (MANDATORY):
    Structure:
    Layer 1:
    [row 1: space-separated 0s and 1s]
    [row 2: space-separated 0s and 1s]
    ...
    Layer 2:
    [row 1: space-separated 0s and 1s]
    ...

    Example for position (0,0) with height 2, position (2,2) with height 1:
    Structure:
    Layer 1:
    1 0 0
    0 0 0
    0 0 1
    Layer 2:
    1 0 0
    0 0 0
    0 0 0

    CRITICAL: 
    - Position (row, col) uses 0-based indexing
    - Follow the height assignment from the sampling strategy EXACTLY
    - Each layer must be a {grid_size}x{grid_size} grid

    Start your response with "Structure:" and provide the layer-by-layer specification.
    """
        
        return prompt

    
    def validate_structure_matches_observations(
        self,
        structure: Structure3D,
        observations
    ) -> bool:
        """Validate structure against observations and physical constraints"""
        # Check top view match
        if isinstance(observations, str):
            observed = self._parse_observation(observations)
            generated = structure.get_top_view()
            if generated.shape != observed.shape:
                return False
            if not np.array_equal(generated, observed):
                return False
        elif isinstance(observations, list):
            for obs in observations:
                observed = np.array(obs['observation']) if isinstance(obs, dict) else self._parse_observation(obs)
                generated = structure.get_top_view()
                if generated.shape != observed.shape:
                    return False
                if not np.array_equal(generated, observed):
                    return False
        else:
            observed = self._parse_observation(observations)
            generated = structure.get_top_view()
            if generated.shape != observed.shape:
                return False
            if not np.array_equal(generated, observed):
                return False
        
        # Validate physical support
        for z in range(1, len(structure.layers)):
            current = structure.layers[z]
            below = structure.layers[z-1]
            for r in range(current.shape[0]):
                for c in range(current.shape[1]):
                    if current[r, c] == 1 and below[r, c] != 1:
                        return False
        
        return True
    
    def _classify_error(self, error_message: str) -> str:
        """Classify error type"""
        if "Expecting value" in error_message:
            return "json_parse_error"
        elif "Rate limit" in error_message.lower():
            return "rate_limit"
        elif "timeout" in error_message.lower():
            return "timeout"
        elif "401" in error_message:
            return "auth_error"
        elif "429" in error_message:
            return "rate_limit_429"
        elif "500" in error_message:
            return "server_error_500"
        else:
            return "unknown_error"
    
    def parse_llm_response(self, response: str) -> Optional[Structure3D]:
        """Parse LLM response to extract structure"""
        if not isinstance(response, str):
            return None
        
        try:
            lines = response.strip().split('\n')
            
            # Find structure start
            structure_start = -1
            for i, line in enumerate(lines):
                if 'Structure:' in line:
                    structure_start = i
                    break
            
            if structure_start == -1:
                for i, line in enumerate(lines):
                    if line.strip().startswith('Layer 1:'):
                        structure_start = i
                        break
            
            if structure_start == -1:
                return None
            
            # Parse layers
            layers = []
            current_layer = []
            in_layer = False
            layer_num = 0
            
            for line in lines[structure_start:]:
                line = line.strip()
                
                if not line or line.startswith(('Note:', 'Reasoning:', 'Explanation:')):
                    continue
                
                if line.startswith('Layer '):
                    match = re.match(r'Layer\s+(\d+)', line)
                    if match:
                        new_layer_num = int(match.group(1))
                        if new_layer_num == layer_num + 1 or (new_layer_num == 1 and layer_num == 0):
                            if current_layer:
                                layers.append(current_layer)
                                current_layer = []
                            in_layer = True
                            layer_num = new_layer_num
                        else:
                            break
                elif in_layer and line:
                    reasoning_indicators = ['therefore', 'because', 'reasoning:', 'explanation:']
                    if any(indicator in line.lower() for indicator in reasoning_indicators):
                        break
                    
                    try:
                        if len(line) > 100:
                            continue
                        
                        if ',' in line:
                            parts = [x.strip() for x in line.split(',')]
                        elif '|' in line:
                            parts = [x.strip() for x in line.split('|')]
                        else:
                            parts = line.split()
                        
                        row = [int(x) for x in parts if x.strip() and int(x) in [0, 1]]
                        if row:
                            current_layer.append(row)
                    except:
                        pass
            
            if current_layer:
                layers.append(current_layer)
            
            if layers:
                return Structure3D(layers)
            
        except Exception as e:
            print(f"Error parsing response: {e}")
        
        return None
    
    def evaluate_single_observation_set(
        self,
        llm: LLMInterface,
        observation_set: Dict,
        n_queries: int = 10,
        verbose: bool = True,
        max_retries: int = 3,
        sampling_seed: Optional[int] = None
    ) -> Dict:
        """
        Evaluate single observation set with HSP-guided sampling
        """
        
        observations = observation_set.get('observation', observation_set.get('observations', []))
        ground_truth_structures = observation_set.get('ground_truth_structures', [])
        
        # Parse observation
        obs_array = self._parse_observation(observations)
        
        # Generate HSP sampling plan
        sampling_plan = self.hsp_sampler.generate_sampling_plan(
            obs_array,
            n_samples=n_queries,
            seed=sampling_seed
        )
        
        if verbose and sampling_plan:
            strategy_counts = {}
            for plan in sampling_plan:
                st = plan['strategy_type']
                strategy_counts[st] = strategy_counts.get(st, 0) + 1
            print(f"  [HSP] Sampling strategies: {strategy_counts}")
        
        # Get ground truth hashes
        gt_hashes = set()
        for gt in ground_truth_structures:
            if 'layers' in gt and isinstance(gt['layers'][0], str):
                layers = []
                grid_size = int(len(gt['layers'][0]) ** 0.5)
                for layer_str in gt['layers']:
                    layer = []
                    for i in range(grid_size):
                        row = [int(layer_str[i * grid_size + j]) for j in range(grid_size)]
                        layer.append(row)
                    layers.append(layer)
                struct = Structure3D(layers)
                gt_hashes.add(struct.get_hash())
            else:
                struct = Structure3D(gt['layers'])
                gt_hashes.add(struct.get_hash())
        
        # Track results
        all_hypotheses = []
        valid_hypotheses = []
        unique_hashes = set()
        unique_structures = []
        parse_success_count = 0
        
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_cost = 0.0
        
        errors = []
        error_counts = {}
        
        # Query LLM with each sampling strategy
        for i, sampling_instruction in enumerate(sampling_plan):
            prompt = self.create_prompt_with_sampling_strategy(
                observations,
                sampling_instruction,
                prior_structures=unique_structures
            )
            
            structure = None
            query_error = None
            
            for attempt in range(max_retries):
                try:
                    if hasattr(llm, 'query_with_usage'):
                        result = llm.query_with_usage(prompt)
                        response = result['response']
                        
                        usage = result.get('usage', {})
                        total_prompt_tokens += usage.get('prompt_tokens', 0)
                        total_completion_tokens += usage.get('completion_tokens', 0)
                        total_tokens += usage.get('total_tokens', 0)
                        total_cost += result.get('cost', 0.0)
                    else:
                        response = llm.query(prompt)
                    
                    if response and response.startswith("Error querying"):
                        query_error = {
                            'query_index': i,
                            'attempt': attempt + 1,
                            'error_message': response,
                            'error_type': self._classify_error(response)
                        }
                        error_type = query_error['error_type']
                        error_counts[error_type] = error_counts.get(error_type, 0) + 1
                        continue
                    
                    structure = self.parse_llm_response(response)
                    if structure:
                        structure = structure.normalize()
                        parse_success_count += 1
                        break
                
                except Exception as e:
                    query_error = {
                        'query_index': i,
                        'attempt': attempt + 1,
                        'error_message': str(e),
                        'error_type': self._classify_error(str(e))
                    }
                    if verbose:
                        print(f"  ⚠ Exception: {str(e)[:100]}")
            
            if not structure and query_error:
                errors.append(query_error)
            
            if structure:
                all_hypotheses.append(structure)
                
                # Check uniqueness
                s_hash = structure.get_hash()
                if s_hash not in unique_hashes:
                    unique_hashes.add(s_hash)
                    unique_structures.append(structure)
                
                # Validate
                if self.validate_structure_matches_observations(structure, observations):
                    valid_hypotheses.append(structure)
        
        # Calculate metrics (same as original benchmark)
        parse_success_rate = parse_success_count / n_queries if n_queries > 0 else 0
        valid_rate = len(valid_hypotheses) / n_queries if n_queries > 0 else 0
        novelty_rate = len(unique_structures) / n_queries if n_queries > 0 else 0
        
        recovered_gts = set()
        for struct in valid_hypotheses:
            s_hash = struct.get_hash()
            if s_hash in gt_hashes:
                recovered_gts.add(s_hash)
        
        recovery_rate = len(recovered_gts) / len(gt_hashes) if gt_hashes else 0
        
        obs_id = observation_set.get('observation_id', observation_set.get('observation_set_id', 'unknown'))
        n_obs = 1 if isinstance(observations, str) else len(observations)
        
        return {
            'observation_set_id': obs_id,
            'n_observations': n_obs,
            'n_ground_truths': len(ground_truth_structures),
            'n_queries': n_queries,
            'n_valid': len(valid_hypotheses),
            'n_unique': len(unique_structures),
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
            'all_hypotheses': [h.to_string() for h in all_hypotheses],
            'unique_structures': [s.to_string() for s in unique_structures]
        }
    
    def run_benchmark(
        self,
        llm: LLMInterface,
        n_samples: int = 10,
        n_queries_per_sample: Optional[int] = None,
        query_multiplier: float = 2.0,
        observation_type: str = "top",
        seed: Optional[int] = None,
        verbose: bool = True,
        checkpoint_dir: str = "checkpoints",
        run_id: Optional[str] = None,
        max_retries: int = 3
    ) -> Dict:
        """Run benchmark with HSP-guided sampling (output format same as original)"""
        
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        if not run_id:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_llm_name = llm.get_name().replace('/', '_').replace('(', '_').replace(')', '_').replace(' ', '_')
        checkpoint_file = checkpoint_path / f"checkpoint_3d_{safe_llm_name}_{run_id}.json"
        
        print(f"\n{'='*70}")
        print("3D Structure Discovery Benchmark - HSP-Guided Sampling")
        print(f"{'='*70}")
        print(f"LLM: {llm.get_name()}")
        print(f"Samples: {n_samples}")
        if n_queries_per_sample is not None:
            print(f"Queries per sample: {n_queries_per_sample} (fixed)")
        else:
            print(f"Queries per sample: {query_multiplier}x ground truths (adaptive)")
        print(f"Sampling strategy: HSP-guided systematic sampling")
        print(f"Checkpoint: {checkpoint_file}")
        print(f"{'='*70}\n")
        
        sampled_sets = self.sample_observation_sets(n_samples, observation_type, seed)
        
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
        
        # Load checkpoint
        start_idx = 0
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                    all_results = checkpoint_data.get('results', [])
                    start_idx = len(all_results)
                    
                    if 'total_token_usage' in checkpoint_data:
                        total_prompt_tokens = checkpoint_data['total_token_usage'].get('prompt_tokens', 0)
                        total_completion_tokens = checkpoint_data['total_token_usage'].get('completion_tokens', 0)
                        total_tokens = checkpoint_data['total_token_usage'].get('total_tokens', 0)
                    if 'total_cost' in checkpoint_data:
                        total_cost = checkpoint_data['total_cost']
                    
                    all_errors = checkpoint_data.get('all_errors', [])
                    total_error_counts = checkpoint_data.get('total_error_counts', {})
                    
                    print(f"✓ Resuming from checkpoint: {start_idx}/{n_samples} completed\n")
                    
                    for result in all_results:
                        valid_rates.append(result['valid_rate'])
                        novelty_rates.append(result['novelty_rate'])
                        recovery_rates.append(result['recovery_rate'])
                        parse_success_rates.append(result.get('parse_success_rate', 1.0))
            except Exception as e:
                print(f"⚠ Warning: Failed to load checkpoint: {e}")
                print("Starting from beginning...\n")
        
        # Process samples
        for idx in range(start_idx, len(sampled_sets)):
            obs_set = sampled_sets[idx]
            
            if verbose:
                print(f"\n{'─'*70}")
                print(f"Sample {idx + 1}/{n_samples}")
                print(f"{'─'*70}")
                obs_id = obs_set.get('observation_id', obs_set.get('observation_set_id', 'unknown'))
                n_obs = 1 if 'observation' in obs_set and isinstance(obs_set['observation'], str) else obs_set.get('n_observations', 1)
                n_gts = obs_set.get('n_compatible_structures', 0)
                
                print(f"  ID: {obs_id}")
                print(f"  Observations: {n_obs}")
                print(f"  Ground truth structures: {n_gts}")
            
            try:
                if n_queries_per_sample is not None:
                    n_queries = n_queries_per_sample
                else:
                    n_gt = obs_set.get('n_compatible_structures', 1)
                    n_queries = max(1, int(n_gt * query_multiplier))
                    if verbose:
                        print(f"  Queries: {n_queries} ({query_multiplier}x {n_gt} GTs)")
                
                result = self.evaluate_single_observation_set(
                    llm,
                    obs_set,
                    n_queries,
                    verbose=verbose,
                    max_retries=max_retries,
                    sampling_seed=seed
                )
                
                all_results.append(result)
                
                valid_rates.append(result['valid_rate'])
                novelty_rates.append(result['novelty_rate'])
                recovery_rates.append(result['recovery_rate'])
                parse_success_rates.append(result.get('parse_success_rate', 1.0))
                
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
                    print(f"\n  Results:")
                    print(f"    Parse success: {result.get('parse_success_rate', 1.0):.1%}")
                    print(f"    Valid rate:    {result['valid_rate']:.1%}")
                    print(f"    Novelty rate:  {result['novelty_rate']:.1%}")
                    print(f"    Recovery rate: {result['recovery_rate']:.1%}")
                    if result.get('cost', 0) > 0:
                        print(f"    Cost: ${result['cost']:.6f}")
                
                # Save checkpoint
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
                    'all_errors': all_errors,
                    'total_error_counts': total_error_counts
                }
                
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
                    
            except Exception as e:
                print(f"\n  ✗ Error processing sample {idx + 1}: {str(e)}")
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
        
        # Final results (same format as original)
        final_results = {
            'run_id': run_id,
            'llm_name': llm.get_name(),
            'n_samples': len(all_results),
            'n_queries_per_sample': n_queries_per_sample,
            'query_multiplier': query_multiplier if n_queries_per_sample is None else None,
            'query_mode': 'fixed' if n_queries_per_sample is not None else f'adaptive_{query_multiplier}x',
            'seed': seed,
            'timestamp': datetime.now().isoformat(),
            'metadata': self.metadata,
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
        
        # Print summary
        print(f"\n{'='*70}")
        print("BENCHMARK RESULTS SUMMARY")
        print(f"{'='*70}")
        print(f"Samples evaluated: {len(all_results)}/{n_samples}")
        
        for metric_name, metric_key in [
            ('Parse Success Rate', 'parse_success_rate'),
            ('Valid Rate', 'valid_rate'),
            ('Novelty Rate', 'novelty_rate'),
            ('Recovery Rate', 'recovery_rate')
        ]:
            stats_dict = final_results['statistics'][metric_key]
            print(f"\n{metric_name}:")
            print(f"  Mean ± Std: {stats_dict['mean']:.3f} ± {stats_dict['std']:.3f}")
            print(f"  Variance:   {stats_dict['var']:.3f}")
            print(f"  Range:      [{stats_dict['min']:.3f}, {stats_dict['max']:.3f}]")
            if stats_dict['p_value'] is not None:
                print(f"  p-value:    {stats_dict['p_value']:.4f}")
        
        print(f"\nToken Usage:")
        print(f"  Total:            {total_tokens:,}")
        print(f"  Avg per sample:   {final_results['token_usage']['avg_tokens_per_sample']:.1f}")
        print(f"  Avg per query:    {final_results['token_usage']['avg_tokens_per_query']:.1f}")
        
        print(f"\nCost:")
        print(f"  Total:            ${total_cost:.4f}")
        print(f"  Avg per sample:   ${final_results['cost']['avg_cost_per_sample']:.4f}")
        print(f"  Avg per query:    ${final_results['cost']['avg_cost_per_query']:.6f}")
        
        if all_errors:
            print(f"\nErrors:")
            print(f"  Total:            {len(all_errors)}")
            print(f"  Error rate:       {final_results['error_summary']['error_rate']:.2%}")
            if total_error_counts:
                print(f"  Error types:")
                for error_type, count in sorted(total_error_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"    - {error_type}: {count}")
        
        print(f"{'='*70}\n")
        
        return final_results


# ******************************************************************************
# * UTILITY FUNCTIONS
# ******************************************************************************

def setup_llm(llm_type: str, **kwargs) -> LLMInterface:
    """Setup LLM interface"""
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


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ******************************************************************************
# * MAIN ENTRY POINT
# ******************************************************************************

def main():
    parser = argparse.ArgumentParser(
        description="3D Structure Benchmark - HSP-Guided Sampling Strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
HSP Sampling Strategy:
  - For simple problems (1-3 blocks): Systematic enumeration with stratified sampling
  - For complex problems (4+ blocks): Diverse sampling strategies (uniform, increasing, etc.)
  - LLM generates structures following exact height assignments from HSP
  - Maintains original metric formulas for fair comparison
        """
    )
    parser.add_argument("--dataset", required=True, help="Path to 3D dataset JSON")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--n-samples", type=int, default=10, help="Number of samples")
    parser.add_argument("--n-queries", type=int, default=None, help="Fixed queries per sample")
    parser.add_argument("--query-multiplier", type=float, default=1.0, help="Adaptive query multiplier")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per query")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--output", type=str, default=None, help="Output file")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose")
    
    args = parser.parse_args()
    
    if args.quiet:
        args.verbose = False
    
    # Load config
    config = load_config(args.config)
    llm_type = config.get('llm', {}).get('type', 'openrouter')
    model = config.get('llm', {}).get('models', {}).get(llm_type) or {
        'openrouter': 'openai/gpt-3.5-turbo',
        'openai': 'gpt-4',
        'anthropic': 'claude-3-opus-20240229'
    }[llm_type]
    
    env_vars = {
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
        'openrouter': 'OPENROUTER_API_KEY'
    }
    api_key = config.get('llm', {}).get('api_keys', {}).get(llm_type) or os.environ.get(env_vars[llm_type])
    
    temperature = config.get('llm', {}).get('temperature', 0.7)
    checkpoint_dir = args.checkpoint_dir
    run_id = config.get('benchmark', {}).get('run_id')
    
    # Generate output filename
    if args.output is None:
        dataset_name = Path(args.dataset).stem
        model_name = model.replace('/', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/{dataset_name}_{model_name}_hsp_{timestamp}.json"
    
    # Initialize benchmark
    benchmark = Benchmark3D(args.dataset)
    
    print("\n" + "=" * 70)
    print("3D STRUCTURE BENCHMARK - HSP-GUIDED SAMPLING")
    print("=" * 70)
    print(f"Dataset:  {args.dataset}")
    print(f"LLM:      {llm_type} ({model})")
    print(f"Samples:  {args.n_samples}")
    print(f"Queries:  {args.n_queries or f'{args.query_multiplier}x GT'}")
    print(f"Strategy: HSP-guided systematic sampling")
    print(f"Output:   {args.output}")
    print("=" * 70)
    
    # Setup LLM
    llm = setup_llm(llm_type, model=model, api_key=api_key, temperature=temperature)
    
    # Run benchmark
    results = benchmark.run_benchmark(
        llm=llm,
        n_samples=args.n_samples,
        n_queries_per_sample=args.n_queries,
        query_multiplier=args.query_multiplier,
        seed=args.seed,
        verbose=args.verbose,
        checkpoint_dir=checkpoint_dir,
        run_id=run_id,
        max_retries=args.max_retries
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {args.output}")
    print(f"✓ Checkpoint saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()
