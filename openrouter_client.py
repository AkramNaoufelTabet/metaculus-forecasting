#!/usr/bin/env python3
"""
OpenRouter API Client with Native Web Search Support
"""

import os
import requests
from typing import Any, Dict, List, Optional


class OpenRouterClient:
    """OpenRouter API client."""
    
    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set")
        
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "https://github.com/forecasting"),
            "X-Title": os.getenv("OPENROUTER_TITLE", "Forecasting Research"),
        })
    
    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        reasoning: Optional[Dict[str, Any]] = None,
        web_search: bool = False,
        web_search_context_size: str = "high",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make chat completion request.
        
        Args:
            model: Model ID (e.g., "anthropic/claude-opus-4.5")
            messages: Chat messages
            reasoning: Reasoning config for supported models
            web_search: Enable native web search (appends :online to model)
            web_search_context_size: "low", "medium", or "high"
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        
        Returns:
            Dict with content, reasoning, annotations, usage
        """
        # Append :online for web search (simplest method per docs)
        model_to_use = f"{model}:online" if web_search else model
        
        payload: Dict[str, Any] = {
            "model": model_to_use,
            "messages": messages,
            **kwargs
        }
        
        # Add reasoning config if provided
        if reasoning:
            payload["reasoning"] = reasoning
        
        # Add search context size for native search pricing/depth
        if web_search:
            payload["web_search_options"] = {
                "search_context_size": web_search_context_size
            }
        
        # Make request
        response = self.session.post(
            self.BASE_URL,
            json=payload,
            timeout=300,
        )
        
        if not response.ok:
            print(f"[DEBUG] {response.status_code} {model}: {response.text[:500]}")
        response.raise_for_status()
   
        data = response.json()
        
        # Extract response components
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        
        return {
            "content": message.get("content", ""),
            "reasoning": message.get("reasoning"),
            "annotations": message.get("annotations", []),
            "usage": data.get("usage"),
            "model": data.get("model"),
            "id": data.get("id"),
        }