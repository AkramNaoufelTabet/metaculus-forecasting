"""
OpenRouter API Client
=====================

Thread-safe client for OpenRouter chat completions API.
Supports extended thinking / reasoning models.
"""

import os
import requests
from typing import Any, Dict, List, Optional

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TIMEOUT = 180  # 3 minutes for reasoning models


class OpenRouterClient:
    """
    Minimal, thread-safe OpenRouter client.
    
    Each instance maintains its own session, making it safe to create
    one instance per thread.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
        app_name: str = "metaculus-forecasting",
        site_url: str = "https://metaculus.com",
    ):
        """
        Initialize client.
        
        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            timeout: Request timeout in seconds
            app_name: App name for OpenRouter attribution
            site_url: Site URL for OpenRouter attribution
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("Missing OPENROUTER_API_KEY in environment")
        
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": site_url,
            "X-Title": app_name,
        })
    
    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 4096,
        seed: Optional[int] = None,
        reasoning: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Send chat completion request.
        
        Args:
            model: OpenRouter model ID
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0 = deterministic)
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens in response
            seed: Random seed for reproducibility
            reasoning: OpenRouter reasoning config, e.g.:
                {"effort": "high", "exclude": False}
                {"max_tokens": 4000, "exclude": False}
            **kwargs: Additional parameters passed to API
        
        Returns:
            Dict with keys: content, reasoning, usage, raw
        
        Raises:
            RuntimeError: On API error or missing response data
        """
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        
        if seed is not None:
            payload["seed"] = seed
        
        if reasoning is not None:
            payload["reasoning"] = reasoning
        
        payload.update(kwargs)
        
        response = self.session.post(
            OPENROUTER_API_URL,
            json=payload,
            timeout=self.timeout,
        )
        
        # Handle errors
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", response.text)
            except Exception:
                error_msg = response.text[:500]
            raise RuntimeError(f"OpenRouter API error {response.status_code}: {error_msg}")
        
        data = response.json()
        
        # Extract response
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"OpenRouter response missing choices: {data}")
        
        message = choices[0].get("message", {})
        
        return {
            "content": message.get("content"),
            "reasoning": message.get("reasoning"),
            "usage": data.get("usage"),
            "raw": data,
        }
    
    def __repr__(self) -> str:
        return f"OpenRouterClient(timeout={self.timeout})"