#!/usr/bin/env python3
"""
OpenRouter API Client
=====================

Handles all OpenRouter API interactions with:
- Automatic retries with exponential backoff
- Cost tracking from response headers
- Support for reasoning models (Anthropic/OpenAI style)
- Support for web search plugin
- Proper error handling

Author: Akram Naoufel Tabet
Version: 1.3.0
"""

import os
import re
import time
import random
from typing import Any, Dict, List, Optional

import httpx


class OpenRouterError(Exception):
    """Base exception for OpenRouter errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class OpenRouterRateLimitError(OpenRouterError):
    """Rate limit exceeded."""
    pass


def extract_urls_from_text(text: str) -> List[Dict[str, Any]]:
    """Extract URLs mentioned in text as pseudo-citations."""
    if not text:
        return []
    
    url_pattern = r'https?://[^\s\)\]\>\"\']+[^\s\.\,\)\]\>\"\':]'
    urls = re.findall(url_pattern, text)
    
    # Deduplicate while preserving order
    seen = set()
    unique_urls = []
    for url in urls:
        # Clean trailing punctuation
        url = url.rstrip('.,;:!?')
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    
    return [{"url": url, "title": None, "source": "extracted"} for url in unique_urls]


class OpenRouterClient:
    """
    OpenRouter API client with retry logic and cost tracking.
    
    Usage:
        client = OpenRouterClient()
        response = client.chat(
            model="anthropic/claude-sonnet-4",
            messages=[{"role": "user", "content": "Hello"}],
            plugins=[{"id": "web", "max_results": 5}]  # Optional web search
        )
        print(response["content"])
        print(response["annotations"])  # Web search citations
        print(response["usage"]["cost"])
    """
    
    BASE_URL = "https://openrouter.ai/api/v1"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 30.0,
        debug: bool = False,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")
        
        self.base_url = base_url or self.BASE_URL
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        self.debug = debug
        
        self._last_headers: Dict[str, str] = {}
    
    def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic."""
        
        url = f"{self.base_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/forecasting",
            "X-Title": "Forecasting Bot",
        }
        
        last_error = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.request(
                        method,
                        url,
                        headers=headers,
                        **kwargs
                    )
                    
                    # Store headers for cost extraction
                    self._last_headers = dict(response.headers)
                    
                    # Check for errors
                    if response.status_code == 429:
                        retry_after = float(response.headers.get("retry-after", 5))
                        raise OpenRouterRateLimitError(
                            f"Rate limited. Retry after {retry_after}s",
                            status_code=429
                        )
                    
                    if response.status_code >= 400:
                        error_body = response.json() if response.content else {}
                        error_msg = error_body.get("error", {}).get("message", response.text[:200])
                        raise OpenRouterError(
                            f"API error {response.status_code}: {error_msg}",
                            status_code=response.status_code,
                            response=error_body
                        )
                    
                    return response.json()
                    
            except OpenRouterRateLimitError as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = min(self.retry_max_delay, self.retry_base_delay * (2 ** (attempt - 1)))
                    delay += random.uniform(0, 1)
                    time.sleep(delay)
                    continue
                raise
                
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_error = OpenRouterError(f"Connection error: {e}")
                if attempt < self.max_retries:
                    delay = min(self.retry_max_delay, self.retry_base_delay * (2 ** (attempt - 1)))
                    delay += random.uniform(0, 1)
                    time.sleep(delay)
                    continue
                raise last_error
        
        raise last_error or OpenRouterError("Max retries exceeded")
    
    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        reasoning: Optional[Dict[str, Any]] = None,
        plugins: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send chat completion request.
        
        Args:
            model: Model ID (e.g., "anthropic/claude-sonnet-4")
            messages: List of message dicts with "role" and "content"
            reasoning: Optional reasoning config
            plugins: Optional list of plugins for web search
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        
        Returns:
            Dict with keys: content, reasoning, usage, annotations, message
        """
        
        # Build request body
        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        
        # Add standard params
        for key in ["temperature", "top_p", "max_tokens", "seed", "stop"]:
            if key in kwargs:
                body[key] = kwargs.pop(key)
        
        # Add reasoning config if provided
        if reasoning:
            if "max_tokens" in reasoning:
                body["reasoning"] = {"max_tokens": reasoning["max_tokens"]}
            elif "effort" in reasoning:
                body["reasoning_effort"] = reasoning["effort"]
        
        # Add plugins if provided
        if plugins:
            body["plugins"] = plugins
        
        # Add any remaining kwargs
        body.update(kwargs)
        
        # ============================================================
        # DEBUG: Log request when using plugins
        # ============================================================
        if plugins and self.debug:
            print("\n" + "="*70)
            print("DEBUG: REQUEST TO OPENROUTER")
            print("="*70)
            print(f"Model: {model}")
            print(f"Plugins: {plugins}")
            print(f"Messages count: {len(messages)}")
            if messages:
                last_msg = messages[-1].get("content", "")
                print(f"Last message (first 500 chars):\n{last_msg[:500]}")
            print("="*70 + "\n")
        
        # Make request
        response = self._request("POST", "/chat/completions", json=body)
        
        # ============================================================
        # DEBUG: Log full response when using plugins
        # ============================================================
        if plugins and self.debug:
            print("\n" + "="*70)
            print("DEBUG: RESPONSE FROM OPENROUTER")
            print("="*70)
            
            print(f"Response keys: {list(response.keys())}")
            
            choices = response.get("choices", [])
            print(f"Choices count: {len(choices)}")
            
            if choices:
                choice = choices[0]
                print(f"Choice keys: {list(choice.keys())}")
                
                message = choice.get("message", {})
                print(f"Message keys: {list(message.keys())}")
                
                annotations = message.get("annotations", [])
                print(f"Annotations count: {len(annotations)}")
                
                if annotations:
                    print("\n--- ANNOTATIONS (ALL) ---")
                    for i, ann in enumerate(annotations):
                        if isinstance(ann, dict):
                            url = ann.get("url") or ann.get("url_citation", {}).get("url", "N/A")
                            title = ann.get("title") or ann.get("url_citation", {}).get("title", "N/A")
                            print(f"  [{i+1}] {title[:60]}")
                            print(f"      URL: {url}")
                        else:
                            print(f"  [{i+1}] {ann}")
                    print("--- END ANNOTATIONS ---\n")
                
                content = message.get("content", "")
                print(f"Content length: {len(content)}")
                print(f"Content (first 500 chars):\n{content[:500]}")
            
            print("="*70 + "\n")
        
        # Parse response
        choices = response.get("choices", [])
        if not choices:
            return {
                "content": None,
                "reasoning": None,
                "usage": response.get("usage"),
                "annotations": [],
                "message": {},
            }
        
        message = choices[0].get("message", {})
        content = message.get("content") or ""
        
        # Extract reasoning
        reasoning_content = None
        if "reasoning" in message:
            reasoning_content = message["reasoning"]
        elif "reasoning_content" in message:
            reasoning_content = message["reasoning_content"]
        
        # Extract annotations
        annotations = message.get("annotations", [])
        
        # Fallback: extract URLs from content if no annotations
        if not annotations and content and plugins:
            annotations = extract_urls_from_text(content)
        
        # Build usage dict
        usage = response.get("usage", {})
        cost_header = self._last_headers.get("x-openrouter-cost")
        if cost_header:
            try:
                usage["cost"] = float(cost_header)
            except (ValueError, TypeError):
                pass
        
        if "cost" in response:
            usage["cost"] = response["cost"]
        
        return {
            "content": content,
            "reasoning": reasoning_content,
            "usage": usage,
            "annotations": annotations,
            "message": message,
        }
    
    def get_models(self) -> List[Dict[str, Any]]:
        """Get list of available models."""
        response = self._request("GET", "/models")
        return response.get("data", [])
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get info for a specific model."""
        models = self.get_models()
        for model in models:
            if model.get("id") == model_id:
                return model
        return None