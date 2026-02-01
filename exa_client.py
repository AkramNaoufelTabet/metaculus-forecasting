#!/usr/bin/env python3
"""
Exa Search Client
=================

Direct Exa API client for forecasting research.
Bypasses OpenRouter plugin for full control over queries.

Author: Forecasting Team
Version: 1.0.0
"""

import os
import time
from typing import Any, Dict, List, Optional

import httpx


class ExaError(Exception):
    """Exa API error."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class ExaClient:
    """
    Direct Exa API client.
    
    Usage:
        client = ExaClient()
        results = client.search("Grammy Awards 2026 video game soundtrack")
        for r in results:
            print(r["title"], r["url"])
    """
    
    BASE_URL = "https://api.exa.ai"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 2,
    ):
        self.api_key = api_key or os.environ.get("EXA_API_KEY")
        if not self.api_key:
            raise ValueError("EXA_API_KEY not found in environment")
        
        self.timeout = timeout
        self.max_retries = max_retries
    
    def search(
        self,
        query: str,
        num_results: int = 10,
        search_type: str = "auto",
        use_autoprompt: bool = False,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        include_text: bool = True,
        text_max_chars: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Execute a single search query.
        
        Args:
            query: Search query string
            num_results: Number of results to return (max 100)
            search_type: "auto", "neural", or "keyword"
            use_autoprompt: Let Exa enhance the query (we disable for control)
            start_published_date: Filter by date (ISO format)
            end_published_date: Filter by date (ISO format)
            include_domains: Only search these domains
            exclude_domains: Exclude these domains
            include_text: Include text snippets
            text_max_chars: Max chars per snippet
        
        Returns:
            List of result dicts with keys: title, url, snippet, published_date, score
        """
        payload = {
            "query": query,
            "numResults": min(num_results, 100),
            "type": search_type,
            "useAutoprompt": use_autoprompt,
            "contents": {
                "text": {"maxCharacters": text_max_chars} if include_text else False,
            },
        }
        
        if start_published_date:
            payload["startPublishedDate"] = start_published_date
        if end_published_date:
            payload["endPublishedDate"] = end_published_date
        if include_domains:
            payload["includeDomains"] = include_domains
        if exclude_domains:
            payload["excludeDomains"] = exclude_domains
        
        # Make request with retries
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        f"{self.BASE_URL}/search",
                        headers={
                            "x-api-key": self.api_key,
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
                    
                    if response.status_code == 429:
                        wait = 2 ** attempt
                        time.sleep(wait)
                        continue
                    
                    if response.status_code >= 400:
                        error_msg = response.text[:200]
                        raise ExaError(f"Exa API error {response.status_code}: {error_msg}", response.status_code)
                    
                    data = response.json()
                    results = data.get("results", [])
                    
                    # Normalize result format
                    normalized = []
                    for r in results:
                        normalized.append({
                            "title": r.get("title", ""),
                            "url": r.get("url", ""),
                            "snippet": r.get("text", ""),
                            "published_date": r.get("publishedDate"),
                            "score": r.get("score", 0),
                            "author": r.get("author"),
                        })
                    
                    return normalized
                    
            except httpx.TimeoutException as e:
                last_error = ExaError(f"Timeout: {e}")
                if attempt < self.max_retries:
                    time.sleep(1)
                    continue
            except httpx.ConnectError as e:
                last_error = ExaError(f"Connection error: {e}")
                if attempt < self.max_retries:
                    time.sleep(1)
                    continue
        
        raise last_error or ExaError("Max retries exceeded")
    
    def search_multiple(
        self,
        queries: List[str],
        num_results_per_query: int = 10,
        dedupe: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute multiple queries and merge results.
        
        Args:
            queries: List of search queries
            num_results_per_query: Results per query
            dedupe: Remove duplicate URLs
            **kwargs: Additional args passed to search()
        
        Returns:
            {
                "results": [...],
                "by_query": {"query1": [...], "query2": [...]},
                "total_raw": 50,
                "total_deduped": 35,
                "queries_executed": ["q1", "q2", ...],
                "queries_failed": ["q3"],
            }
        """
        all_results = []
        by_query = {}
        failed_queries = []
        
        for query in queries:
            try:
                results = self.search(query, num_results=num_results_per_query, **kwargs)
                by_query[query] = results
                all_results.extend(results)
            except ExaError as e:
                print(f"  [EXA] Query failed: {query[:50]}... - {e}")
                by_query[query] = []
                failed_queries.append(query)
        
        total_raw = len(all_results)
        
        # Deduplicate by URL
        if dedupe:
            seen_urls = set()
            unique_results = []
            for r in all_results:
                url = r.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_results.append(r)
            all_results = unique_results
        
        return {
            "results": all_results,
            "by_query": by_query,
            "total_raw": total_raw,
            "total_deduped": len(all_results),
            "queries_executed": [q for q in queries if q not in failed_queries],
            "queries_failed": failed_queries,
        }


def test_exa():
    """Quick test of Exa client."""
    client = ExaClient()
    
    print("Testing Exa search...")
    results = client.search(
        "68th Grammy Awards video game soundtrack nominees 2026",
        num_results=5
    )
    
    print(f"\nFound {len(results)} results:")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] {r['title'][:60]}")
        print(f"      {r['url']}")
        print(f"      {r['snippet'][:100]}...")
        print()


if __name__ == "__main__":
    test_exa()