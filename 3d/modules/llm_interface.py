import traceback
import random
import requests
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
import re
from .models import CausalGraph


def _extract_text(resp) -> str:
    # 1) Responses API convenience
    t = getattr(resp, "output_text", None)
    if t:
        return t

    # 2) Responses API: walk output -> message -> content
    try:
        pieces = []
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", None) == "message":
                for c in getattr(item, "content", []) or []:
                    ctype = getattr(c, "type", None) or (isinstance(c, dict) and c.get("type"))
                    if ctype in ("output_text", "text"):
                        # pydantic object or dict
                        text = getattr(c, "text", None) if hasattr(c, "text") else c.get("text")
                        if text:
                            pieces.append(text)
        if pieces:
            return "".join(pieces)
    except Exception:
        pass

    # 3) Chat Completions fallback (if you switch endpoints)
    try:
        return resp.choices[0].message.content
    except Exception:
        pass

    return str(resp)


def _extract_usage(resp):
    u = getattr(resp, "usage", None)
    if not u:
        return 0, 0, 0
    # Responses API names
    input_tokens = getattr(u, "input_tokens", getattr(u, "prompt_tokens", 0))
    output_tokens = getattr(u, "output_tokens", getattr(u, "completion_tokens", 0))
    total_tokens = getattr(u, "total_tokens", input_tokens + output_tokens)
    return input_tokens, output_tokens, total_tokens


class LLMInterface(ABC):
    """Abstract interface for LLM interaction."""
    
    @abstractmethod
    def query(self, prompt: str, **kwargs) -> str:
        """
        Query the LLM with a prompt and return response.
        
        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Optional parameters (temperature, top_p, etc.)
        
        Returns:
            The LLM's response as a string
        """
        pass
    
    def query_with_usage(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Query the LLM and return response with usage stats.
        
        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Optional parameters (temperature, top_p, repetition_penalty, etc.)
        
        Returns:
            Dict with 'response', 'usage', and 'cost'
        """
        # Default implementation for backward compatibility
        return {
            'response': self.query(prompt, **kwargs),
            'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
            'cost': 0.0
        }
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name/identifier of the LLM."""
        pass
    
    def get_model_pricing(self) -> Dict[str, float]:
        """Get pricing per 1M tokens for this model."""
        # Default pricing (can be overridden by subclasses)
        return {'input': 0.0, 'output': 0.0}
    
    def reset(self):
        """Reset any internal state (optional)."""
        pass


class OpenRouterLLM(LLMInterface):
    """
    OpenRouter API interface for various LLM models.
    
    OpenRouter provides access to multiple models through a single API.
    """
    
    def __init__(
        self,
        model: str = "anthropic/claude-3.5-sonnet",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 40960,
        base_url: str = "https://openrouter.ai/api/v1"
    ):
        """
        Initialize OpenRouter LLM interface.
        
        Args:
            model: Model identifier (e.g., "anthropic/claude-3.5-sonnet", "openai/gpt-4")
            api_key: OpenRouter API key
            temperature: Sampling temperature (default)
            max_tokens: Maximum tokens in response
            base_url: OpenRouter API base URL
        """
        if not api_key:
            raise ValueError("OpenRouter API key is required")
        
        self.model = model
        self.api_key = api_key
        self.default_temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def query(self, prompt: str, **kwargs) -> str:
        """Query OpenRouter API."""
        result = self.query_with_usage(prompt, **kwargs)
        return result['response']
    
    def query_with_usage(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Query OpenRouter API with usage tracking.
        
        Args:
            prompt: The prompt to send
            temperature: Sampling temperature (overrides default if provided)
            top_p: Nucleus sampling parameter (0.0-1.0)
            repetition_penalty: Penalty for repeating tokens (>1.0 discourages repetition)
            **kwargs: Additional parameters
        """
        try:
            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are an expert in causal inference and graph theory."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature if temperature is not None else self.default_temperature,
                "max_tokens": self.max_tokens
            }
            
            # Add optional parameters if provided
            if top_p is not None:
                payload["top_p"] = top_p
            
            # Note: OpenRouter may not support repetition_penalty directly
            # Use frequency_penalty as approximation (supported by most models)
            if repetition_penalty is not None and repetition_penalty != 1.0:
                # Convert repetition_penalty (typically 1.0-1.5) to frequency_penalty (typically 0.0-2.0)
                payload["frequency_penalty"] = min((repetition_penalty - 1.0) * 2, 2.0)
            
            response = requests.post(url, headers=self.headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            # Extract usage information
            usage = result.get('usage', {})
            usage_data = {
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0)
            }
            
            # Calculate cost based on model pricing
            pricing = self.get_model_pricing()
            cost = (usage_data['prompt_tokens'] * pricing['input'] + 
                   usage_data['completion_tokens'] * pricing['output']) / 1_000_000
            
            return {
                'response': result['choices'][0]['message']['content'],
                'usage': usage_data,
                'cost': cost
            }
            
        except requests.exceptions.RequestException as e:
            return {
                'response': f"Error querying OpenRouter: {str(e)}",
                'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
                'cost': 0.0
            }
        except (KeyError, IndexError) as e:
            return {
                'response': f"Error parsing OpenRouter response: {str(e)}",
                'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
                'cost': 0.0
            }
    
    def get_name(self) -> str:
        """Get the model name."""
        return f"OpenRouter({self.model})"
    
    def get_model_pricing(self) -> Dict[str, float]:
        """Get pricing per 1M tokens for common models."""
        # Pricing in dollars per 1M tokens
        pricing_map = {
            'anthropic/claude-3.5-sonnet': {'input': 3.0, 'output': 15.0},
            'anthropic/claude-3-opus': {'input': 15.0, 'output': 75.0},
            'openai/gpt-4o': {'input': 2.5, 'output': 10.0},
            'meta-llama/llama-3.3-70b-instruct': {'input': 0.038, 'output': 0.12},
            'google/gemini-2.5-pro': {'input': 1.25, 'output': 10.0},
            'deepseek/deepseek-r1': {'input': 0.4, 'output': 2},
        }
        return pricing_map.get(self.model, {'input': 1.0, 'output': 1.0})


class OpenAILLM(LLMInterface):
    """
    OpenAI API interface for GPT models.
    
    Requires openai package and API key.
    """
    
    def __init__(
        self, 
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 40960
    ):
        """
        Initialize OpenAI LLM interface.
        
        Args:
            model: OpenAI model to use
            api_key: OpenAI API key (uses environment variable if not provided)
            temperature: Sampling temperature (default)
            max_tokens: Maximum tokens in response
        """
        try:
            import openai
        except ImportError:
            raise ImportError("Please install openai package: pip install openai")
        
        self.model = model
        self.default_temperature = temperature
        self.max_tokens = max_tokens
        
        if not api_key:
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
        
        self.client = openai.OpenAI(api_key=api_key)
    
    def query(self, prompt: str, **kwargs) -> str:
        """Query OpenAI API."""
        result = self.query_with_usage(prompt, **kwargs)
        return result['response']
    
    def query_with_usage(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Query OpenAI API with usage tracking.
        
        Args:
            prompt: The prompt to send
            temperature: Sampling temperature (overrides default if provided)
            top_p: Nucleus sampling parameter (0.0-1.0)
            repetition_penalty: Penalty for repeating tokens (>1.0 discourages repetition)
            **kwargs: Additional parameters
        """
        try:
            # Build request parameters
            request_params = {
                "model": self.model,
                "input": [
                    {"role": "system", "content": "You are an expert in causal inference and graph theory."},
                    {"role": "user", "content": prompt},
                ],
                "reasoning": {"effort": "medium"},
                "max_output_tokens": self.max_tokens
            }
            
            # Add temperature if provided (otherwise use model default)
            if temperature is not None:
                request_params["temperature"] = temperature
            elif self.default_temperature != 0.7:  # Only set if non-default
                request_params["temperature"] = self.default_temperature
            
            # Note: OpenAI Responses API may not support all sampling parameters
            # top_p and repetition_penalty might not be available
            # Check API documentation for current support
            
            resp = self.client.responses.create(**request_params)

            text = _extract_text(resp)
            in_tok, out_tok, tot_tok = _extract_usage(resp)

            pricing = self.get_model_pricing()
            cost = (in_tok * pricing['input'] + out_tok * pricing['output']) / 1_000_000

            return {
                "response": text,
                "usage": {
                    "prompt_tokens": in_tok,
                    "completion_tokens": out_tok,
                    "total_tokens": tot_tok,
                },
                "cost": cost,
            }
        except Exception as e:
            traceback.print_exc()
            return {
                "response": f"Error querying OpenAI: {str(e)}",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "cost": 0.0,
            }
    
    def get_name(self) -> str:
        """Get the model name."""
        return f"OpenAI({self.model})"
    
    def get_model_pricing(self) -> Dict[str, float]:
        """Get pricing per 1M tokens for OpenAI models."""
        # Pricing in dollars per 1M tokens
        pricing_map = {
            'gpt-4o': {'input': 2.5, 'output': 10.0},
            'gpt-4o-mini': {'input': 0.15, 'output': 0.6},
            'gpt-5': {'input': 1.25, 'output': 10.0}
        }
        return pricing_map.get(self.model, {'input': 10.0, 'output': 30.0})


class AnthropicLLM(LLMInterface):
    """
    Anthropic Claude API interface.
    
    Requires anthropic package and API key.
    """
    
    def __init__(
        self,
        model: str = "claude-3-opus-20240229",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096
    ):
        """
        Initialize Anthropic LLM interface.
        
        Args:
            model: Anthropic model to use
            api_key: Anthropic API key (uses environment variable if not provided)
            temperature: Sampling temperature (default)
            max_tokens: Maximum tokens in response
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError("Please install anthropic package: pip install anthropic")
        
        self.model = model
        self.default_temperature = temperature
        self.max_tokens = max_tokens
        
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def query(self, prompt: str, **kwargs) -> str:
        """Query Anthropic API."""
        result = self.query_with_usage(prompt, **kwargs)
        return result['response']
    
    def query_with_usage(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Query Anthropic API with usage tracking.
        
        Args:
            prompt: The prompt to send
            temperature: Sampling temperature (overrides default if provided)
            top_p: Nucleus sampling parameter (0.0-1.0)
            repetition_penalty: Not supported by Anthropic API (ignored)
            **kwargs: Additional parameters
        """
        try:
            request_params = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": temperature if temperature is not None else self.default_temperature,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            
            # Add top_p if provided
            if top_p is not None:
                request_params["top_p"] = top_p
            
            # Note: Anthropic doesn't support repetition_penalty directly
            # The API uses temperature and top_p for diversity control
            
            response = self.client.messages.create(**request_params)
            
            # Extract usage information
            usage = {
                'prompt_tokens': response.usage.input_tokens if hasattr(response, 'usage') else 0,
                'completion_tokens': response.usage.output_tokens if hasattr(response, 'usage') else 0,
                'total_tokens': (response.usage.input_tokens + response.usage.output_tokens) if hasattr(response, 'usage') else 0
            }
            
            # Calculate cost
            pricing = self.get_model_pricing()
            cost = (usage['prompt_tokens'] * pricing['input'] + 
                   usage['completion_tokens'] * pricing['output']) / 1_000_000
            
            return {
                'response': response.content[0].text,
                'usage': usage,
                'cost': cost
            }
        except Exception as e:
            traceback.print_exc()
            return {
                'response': f"Error querying Anthropic: {str(e)}",
                'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
                'cost': 0.0
            }
    
    def get_name(self) -> str:
        """Get the model name."""
        return f"Anthropic({self.model})"
    
    def get_model_pricing(self) -> Dict[str, float]:
        """Get pricing per 1M tokens for Anthropic models."""
        # Pricing in dollars per 1M tokens
        pricing_map = {
            'claude-3-opus-20240229': {'input': 15.0, 'output': 75.0},
            'claude-3-sonnet-20240229': {'input': 3.0, 'output': 15.0},
            'claude-3-haiku-20240307': {'input': 0.25, 'output': 1.25},
            'claude-3.5-sonnet-20241022': {'input': 3.0, 'output': 15.0},
        }
        return pricing_map.get(self.model, {'input': 3.0, 'output': 15.0})


class ResponseParser:
    """Parser for extracting causal graphs from LLM responses."""
    
    @staticmethod
    def parse_response(response: str) -> Optional[CausalGraph]:
        """
        Parse LLM response to extract causal graph.
        
        Handles various response formats and edge notations.
        
        Args:
            response: The LLM's response text
        
        Returns:
            CausalGraph if successfully parsed, None otherwise
        """
        try:
            # Extract nodes
            nodes = ResponseParser._extract_nodes(response)
            if not nodes:
                return None
            
            # Extract edges
            edges = ResponseParser._extract_edges(response)
            if not edges:
                # Try alternative extraction methods
                edges = ResponseParser._extract_edges_alternative(response)
            
            if nodes and edges:
                # Validate that edge nodes are in the node list
                edge_nodes = set()
                for src, dst in edges:
                    edge_nodes.add(src)
                    edge_nodes.add(dst)
                
                # Add any missing nodes
                for node in edge_nodes:
                    if node not in nodes:
                        nodes.append(node)
                
                return CausalGraph(nodes=sorted(nodes), edges=edges)
            
        except Exception as e:
            print(f"Error parsing response: {e}")
        
        return None
    
    @staticmethod
    def _extract_nodes(response: str) -> Optional[List[str]]:
        """Extract node list from response."""
        # Try different patterns
        patterns = [
            r'nodes?\s*\[([^\]]+)\]',
            r'nodes?\s*:\s*\[([^\]]+)\]',
            r'nodes?\s+(?:are\s+)?(\w+(?:,\s*\w+)*)',
            r'variables?\s*\[([^\]]+)\]',
            r'variables?\s+(?:are\s+)?(\w+(?:,\s*\w+)*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                nodes_str = match.group(1)
                # Clean and split
                nodes = [n.strip().strip("'\"") for n in nodes_str.split(',')]
                return [n for n in nodes if n]  # Filter empty strings
        
        return None
    
    @staticmethod
    def _extract_edges(response: str) -> List[tuple]:
        """Extract edges from response."""
        edges = []
        
        # Edge patterns to look for
        edge_patterns = [
            r'(\w+)\s*->\s*(\w+)',
            r'(\w+)\s*→\s*(\w+)',
            r'(\w+)\s+causes?\s+(\w+)',
            r'(\w+)\s+affects?\s+(\w+)',
            r'(\w+)\s+influences?\s+(\w+)'
        ]
        
        for pattern in edge_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                src, dst = match[0].strip(), match[1].strip()
                if src and dst and src != dst:  # Avoid self-loops
                    edges.append((src, dst))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_edges = []
        for edge in edges:
            if edge not in seen:
                seen.add(edge)
                unique_edges.append(edge)
        
        return unique_edges
    
    @staticmethod
    def _extract_edges_alternative(response: str) -> List[tuple]:
        """Alternative method for extracting edges."""
        edges = []
        
        # Look for edges in format "edges: A->B, C->D, ..."
        edges_match = re.search(r'edges?:?\s*([^.]+)', response, re.IGNORECASE)
        if edges_match:
            edges_str = edges_match.group(1)
            
            # Split by comma and parse each edge
            edge_parts = edges_str.split(',')
            for part in edge_parts:
                # Try to extract edge from each part
                edge_match = re.search(r'(\w+)\s*(?:->|→)\s*(\w+)', part)
                if edge_match:
                    edges.append((edge_match.group(1), edge_match.group(2)))
        
        return edges
