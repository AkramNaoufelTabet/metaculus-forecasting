#!/usr/bin/env python3
"""
Search Pipeline
===============

Orchestrates the full search flow:
1. Generate queries (LLM)
2. Execute searches (Exa)
3. Validate relevance (sanity gate)
4. Retry if needed
5. Format for curator

Author: Forecasting Team
Version: 1.1.0
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from openrouter_client import OpenRouterClient
from exa_client import ExaClient, ExaError
from search_query_generator import generate_search_queries


class SearchPipeline:
    """
    Full search pipeline with query generation, retrieval, and validation.
    
    Usage:
        pipeline = SearchPipeline(openrouter_client, exa_client)
        result = pipeline.search_for_question(question_dict)
        
        if result["success"]:
            formatted_context = result["formatted_context"]
            # Pass to curator
    """
    
    def __init__(
        self,
        openrouter_client: OpenRouterClient,
        exa_client: Optional[ExaClient] = None,
        cache_dir: Optional[Path] = None,
        query_model: str = "anthropic/claude-sonnet-4.5",
        results_per_query: int = 8,
        max_total_results: int = 25,
        relevance_threshold: int = 2,
        enable_retry: bool = True,
        logger: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize search pipeline.
        
        Args:
            openrouter_client: Client for LLM calls
            exa_client: Client for Exa search (created if not provided)
            cache_dir: Directory for caching search results
            query_model: Model ID for query generation
            results_per_query: Number of results per search query
            max_total_results: Maximum total results after deduplication
            relevance_threshold: Minimum must-hit tokens to consider relevant
            enable_retry: Whether to retry on low relevance
            logger: Optional logging callback (receives string messages)
        """
        self.openrouter = openrouter_client
        self.exa = exa_client or ExaClient()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.query_model = query_model
        self.results_per_query = results_per_query
        self.max_total_results = max_total_results
        self.relevance_threshold = relevance_threshold
        self.enable_retry = enable_retry
        self._log = logger  # Optional logger callback
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _log_msg(self, msg: str) -> None:
        """Log a message if logger is configured."""
        if self._log:
            self._log(msg)
    
    def search_for_question(
        self,
        question: Dict[str, Any],
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Full search pipeline for a forecasting question.
        
        Args:
            question: Question dict
            use_cache: Whether to use cached results
        
        Returns:
            {
                "success": True/False,
                "formatted_context": "## SEARCH RESULTS...",
                "results": [...],
                "queries": [...],
                "must_hit_tokens": [...],
                "relevance": {...},
                "retried": False,
                "cost": 0.01,
                "cached": False,
            }
        """
        question_id = question.get("question_id", "unknown")
        
        # Step 1: Generate queries
        query_result = generate_search_queries(
            self.openrouter,
            question,
            model_id=self.query_model,
        )
        
        queries = query_result["queries"]
        must_hit_tokens = query_result["must_hit_tokens"]
        total_cost = query_result.get("cost", 0)
        
        # Check cache
        cache_key = self._cache_key(question_id, queries)
        if use_cache and self.cache_dir:
            cached = self._load_cache(cache_key)
            if cached:
                # Regenerate formatted context from cached results
                cached["formatted_context"] = self._format_for_curator(
                    cached.get("results", []), 
                    cached.get("queries", queries)
                )
                cached["cached"] = True
                return cached
        
        # Step 2: Execute searches
        search_result = self.exa.search_multiple(
            queries,
            num_results_per_query=self.results_per_query,
            dedupe=True,
        )
        
        results = search_result["results"][:self.max_total_results]
        
        # Step 3: Check relevance
        relevance = self._check_relevance(results, must_hit_tokens)
        
        # Step 4: Retry if needed
        retried = False
        if not relevance["relevant"] and self.enable_retry:
            retried = True
            
            # Generate stricter queries
            strict_queries = self._make_stricter_queries(question, must_hit_tokens)
            
            # Re-search
            retry_result = self.exa.search_multiple(
                strict_queries,
                num_results_per_query=self.results_per_query,
                dedupe=True,
            )
            
            # Merge with original results
            all_urls = {r["url"] for r in results}
            for r in retry_result["results"]:
                if r["url"] not in all_urls:
                    results.append(r)
                    all_urls.add(r["url"])
            
            results = results[:self.max_total_results]
            
            # Re-check relevance
            relevance = self._check_relevance(results, must_hit_tokens)
        
        # Step 5: Format for curator
        formatted_context = self._format_for_curator(results, queries)
        
        # Build result
        result = {
            "success": relevance["relevant"] or len(results) > 0,
            "formatted_context": formatted_context,
            "results": results,
            "queries": queries,
            "must_hit_tokens": must_hit_tokens,
            "relevance": relevance,
            "retried": retried,
            "cost": total_cost,
            "cached": False,
        }
        
        # Save to cache
        if self.cache_dir:
            self._save_cache(cache_key, result)
        
        return result
    
    def _check_relevance(
        self,
        results: List[Dict[str, Any]],
        must_hit_tokens: List[str],
    ) -> Dict[str, Any]:
        """Check if results contain must-hit tokens."""
        if not must_hit_tokens:
            return {"relevant": True, "match_count": 0, "matches": [], "missing": []}
        
        # Build text from all results
        all_text = ""
        for r in results:
            all_text += f" {r.get('title', '')} {r.get('snippet', '')} {r.get('url', '')} "
        all_text = all_text.lower()
        
        # Check each token
        matches = []
        missing = []
        for token in must_hit_tokens:
            if token.lower() in all_text:
                matches.append(token)
            else:
                missing.append(token)
        
        return {
            "relevant": len(matches) >= self.relevance_threshold,
            "match_count": len(matches),
            "matches": matches,
            "missing": missing,
            "threshold": self.relevance_threshold,
        }
    
    def _make_stricter_queries(
        self,
        question: Dict[str, Any],
        must_hit_tokens: List[str],
    ) -> List[str]:
        """Generate stricter queries for retry."""
        q_text = question.get("question", "")
        
        # Take first 3 must-hit tokens and combine with question
        key_terms = " ".join(must_hit_tokens[:3])
        
        strict_queries = [
            f"{key_terms} official results announcement",
            f"{key_terms} winner predictions 2025 2026",
        ]
        
        if len(must_hit_tokens) >= 2:
            strict_queries.append(f'"{must_hit_tokens[0]}" {must_hit_tokens[1]}')
        
        # Add domain-specific queries based on question content
        q_lower = q_text.lower()
        if "grammy" in q_lower:
            strict_queries.append(f"{key_terms} site:grammy.com OR site:billboard.com")
        elif "oscar" in q_lower:
            strict_queries.append(f"{key_terms} site:oscars.org OR site:variety.com")
        elif any(term in q_lower for term in ["capex", "earnings", "revenue"]):
            strict_queries.append(f"{key_terms} site:sec.gov OR site:reuters.com")
        
        return strict_queries[:4]
    
    def _format_for_curator(
        self,
        results: List[Dict[str, Any]],
        queries: List[str],
    ) -> str:
        """Format search results for injection into curator prompt."""
        if not results:
            return (
                "## WEB SEARCH RESULTS\n\n"
                "**No results found.** The search did not return relevant information. "
                "Use the background information provided in the question and note high uncertainty.\n\n"
                f"Queries attempted: {queries}\n"
            )
        
        lines = [
            "## WEB SEARCH RESULTS",
            "",
            f"Found {len(results)} sources. Use these to build your evidence packet.",
            "",
        ]
        
        for i, r in enumerate(results, 1):
            title = r.get("title", "Unknown Title")
            url = r.get("url", "")
            snippet = r.get("snippet", "")[:500]
            pub_date = r.get("published_date", "")
            
            lines.append(f"### Source [{i}]")
            lines.append(f"**Title:** {title}")
            if pub_date:
                lines.append(f"**Date:** {pub_date}")
            lines.append(f"**URL:** {url}")
            lines.append(f"**Content:** {snippet}")
            lines.append("")
        
        lines.append("---")
        lines.append("")
        lines.append("Use the sources above to build your evidence packet. Cite sources by number [1], [2], etc.")
        
        return "\n".join(lines)
    
    def _cache_key(self, question_id: str, queries: List[str]) -> str:
        """Generate cache key from question ID and queries."""
        query_hash = hashlib.md5(json.dumps(sorted(queries)).encode()).hexdigest()[:8]
        return f"{question_id}_{query_hash}"
    
    def _load_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load cached search result."""
        if not self.cache_dir:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text())
            except Exception:
                return None
        return None
    
    def _save_cache(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Save search result to cache."""
        if not self.cache_dir:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            # Don't cache formatted_context (regenerate from results)
            cache_data = {k: v for k, v in result.items() if k != "formatted_context"}
            cache_file.write_text(json.dumps(cache_data, indent=2, default=str))
        except Exception:
            pass  # Silently ignore cache save failures


def test_pipeline():
    """Test the full search pipeline."""
    from openrouter_client import OpenRouterClient
    
    client = OpenRouterClient()
    
    # With a simple print logger for testing
    def test_logger(msg: str) -> None:
        print(f"[TEST] {msg}")
    
    pipeline = SearchPipeline(client, logger=test_logger)
    
    question = {
        "question_id": "41331",
        "question": "Which video game soundtrack will win Best Score Soundtrack for Video Games at the 68th Grammy Awards?",
        "options": "Avatar: Frontiers of Pandora, Helldivers 2, Indiana Jones And The Great Circle, Star Wars Outlaws, Sword of the Sea",
        "fine_print": "Resolution based on official Grammy winner announcement",
        "close_date": "2026-02-01",
        "background": "The 68th Grammy Awards will be held on February 1, 2026.",
    }
    
    print("Testing search pipeline...")
    result = pipeline.search_for_question(question, use_cache=False)
    
    print(f"\n=== RESULT ===")
    print(f"Success: {result['success']}")
    print(f"Results count: {len(result['results'])}")
    print(f"Relevance: {result['relevance']}")
    print(f"Retried: {result['retried']}")
    print(f"Cost: ${result['cost']:.4f}")


if __name__ == "__main__":
    test_pipeline()