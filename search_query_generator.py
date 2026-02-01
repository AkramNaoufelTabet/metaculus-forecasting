#!/usr/bin/env python3
"""
Search Query Generator
======================

Uses LLM to generate targeted, keyword-heavy search queries
for forecasting questions. Also extracts must-hit tokens for
relevance validation.

Author: Forecasting Team
Version: 1.0.1
"""

import re
from typing import Any, Dict, List

from openrouter_client import OpenRouterClient


DEFAULT_QUERY_MODEL = "anthropic/claude-sonnet-4.5"


QUERY_GENERATOR_PROMPT = '''You are a Search Query Engineer. Generate 4-5 precise search queries to find evidence for a forecasting question.

QUESTION: {question}

OPTIONS/CONTEXT: {options}

RESOLUTION CRITERIA: {fine_print}

CLOSE DATE: {close_date}

---

INSTRUCTIONS:

1. **Extract Key Entities:**
   - Proper nouns (people, companies, awards, events)
   - Specific dates, years, quarters
   - Official category/title names

2. **Time-Aware Queries:**
   - Close date is {close_date}
   - If FUTURE: use "forecast", "prediction", "odds", "expected", "projection"
   - If TODAY/PAST: use "results", "winner", "announced", "official", "confirmed"

3. **Domain-Specific Terms:**
   - Awards: "nominees", "nominations", "winner", "ceremony", publication names
   - Finance: ticker symbols, "Q4 2025", "earnings", "capex", "guidance"
   - Sports: "odds", "standings", "predictions", sport-specific terms
   - Politics: "polls", "legislation", "bill status"

4. **Query Format:**
   - 5-10 words each, keyword-heavy
   - Include year/date when relevant
   - Mix: 1-2 broad, 2-3 specific
   - One query targeting official/primary sources

---

OUTPUT FORMAT (exactly this structure, no extra text):

QUERY_1: [first search query]
QUERY_2: [second search query]
QUERY_3: [third search query]
QUERY_4: [fourth search query]
QUERY_5: [fifth search query - optional]

MUST_HIT_TOKENS: [comma-separated list of 5-8 key terms that MUST appear in relevant results]
'''


def generate_search_queries(
    client: OpenRouterClient,
    question: Dict[str, Any],
    model_id: str = DEFAULT_QUERY_MODEL,
) -> Dict[str, Any]:
    """
    Generate targeted search queries for a forecasting question.
    
    Args:
        client: OpenRouter client instance
        question: Question dict with keys: question, options, fine_print, close_date, background
        model_id: Model to use for generation
    
    Returns:
        {
            "queries": ["query1", "query2", ...],
            "must_hit_tokens": ["token1", "token2", ...],
            "raw_response": "...",
            "cost": 0.002,
            "success": True,
        }
    """
    # Build options string
    options = question.get("options", "")
    if isinstance(options, list):
        options = "\n".join(f"- {opt}" for opt in options)
    elif not options:
        options = question.get("background", "")[:500]
    
    # Format prompt
    prompt = QUERY_GENERATOR_PROMPT.format(
        question=question.get("question", ""),
        options=options,
        fine_print=question.get("fine_print", question.get("resolution_criteria", ""))[:500],
        close_date=question.get("close_date", "unknown"),
    )
    
    try:
        response = client.chat(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.2,  # Low temp for consistent output
        )
        
        content = response.get("content", "")
        usage = response.get("usage", {})
        cost = usage.get("cost", 0)
        
        # Parse queries
        queries = _parse_queries(content)
        must_hit_tokens = _parse_must_hit_tokens(content)
        
        # Fallback if parsing failed
        if not queries:
            queries = _fallback_queries(question)
        
        if not must_hit_tokens:
            must_hit_tokens = _extract_entities(question)
        
        return {
            "queries": queries,
            "must_hit_tokens": must_hit_tokens,
            "raw_response": content,
            "cost": cost,
            "success": len(queries) > 0,
        }
        
    except Exception as e:
        # Fallback on error
        return {
            "queries": _fallback_queries(question),
            "must_hit_tokens": _extract_entities(question),
            "raw_response": f"Error: {e}",
            "cost": 0,
            "success": False,
            "error": str(e),
        }


def _parse_queries(text: str) -> List[str]:
    """Extract QUERY_N: lines from response."""
    queries = []
    pattern = r'QUERY_\d+:\s*(.+)'
    
    for match in re.finditer(pattern, text, re.IGNORECASE):
        query = match.group(1).strip()
        # Clean up query
        query = query.strip('"\'[]')
        if query and len(query) > 5 and len(query) < 200:
            queries.append(query)
    
    return queries[:5]


def _parse_must_hit_tokens(text: str) -> List[str]:
    """Extract MUST_HIT_TOKENS from response."""
    pattern = r'MUST_HIT_TOKENS:\s*(.+)'
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        tokens_str = match.group(1).strip()
        # Split by comma or semicolon
        tokens = re.split(r'[,;]', tokens_str)
        tokens = [t.strip().strip('"\'[]').lower() for t in tokens]
        tokens = [t for t in tokens if t and len(t) > 2]
        return tokens[:10]
    
    return []


def _fallback_queries(question: Dict[str, Any]) -> List[str]:
    """Generate basic queries when LLM fails."""
    q_text = question.get("question", "")
    close_date = question.get("close_date", "")
    
    # Clean question text
    q_clean = re.sub(r'\?+$', '', q_text)[:100]
    
    queries = [q_clean]
    
    # Add prediction-focused query
    if close_date and close_date > "2025-01-01":
        queries.append(f"{q_clean} prediction forecast")
    else:
        queries.append(f"{q_clean} results winner")
    
    return queries


def _extract_entities(question: Dict[str, Any]) -> List[str]:
    """Extract key entities from question as fallback must-hit tokens."""
    text = f"{question.get('question', '')} {question.get('options', '')}"
    
    # Common stopwords
    stopwords = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'will', 'be', 'been',
        'to', 'of', 'and', 'in', 'for', 'on', 'at', 'by', 'with', 'from',
        'this', 'that', 'what', 'which', 'who', 'when', 'where', 'how', 'why',
        'question', 'resolve', 'following', 'option', 'options',
    }
    
    # Extract words
    words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9]*\b', text.lower())
    
    # Filter and count
    filtered = [w for w in words if w not in stopwords and len(w) > 2]
    freq = {}
    for w in filtered:
        freq[w] = freq.get(w, 0) + 1
    
    # Sort by frequency
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    
    return [w for w, _ in sorted_words[:8]]


# =============================================================================
# TEST (only runs when executed directly, not when imported)
# =============================================================================

if __name__ == "__main__":
    def _test_query_generator():
        """Test query generation - only for development."""
        client = OpenRouterClient()
        
        question = {
            "question": "Which video game soundtrack will win Best Score Soundtrack for Video Games at the 68th Grammy Awards?",
            "options": "Avatar: Frontiers of Pandora, Helldivers 2, Indiana Jones And The Great Circle, Star Wars Outlaws, Sword of the Sea",
            "fine_print": "Resolution based on official Grammy winner announcement",
            "close_date": "2026-02-01",
        }
        
        print("Generating queries...")
        result = generate_search_queries(client, question)
        
        print(f"\nSuccess: {result['success']}")
        print(f"Cost: ${result['cost']:.4f}")
        print(f"\nQueries:")
        for i, q in enumerate(result['queries'], 1):
            print(f"  {i}. {q}")
        print(f"\nMust-hit tokens: {result['must_hit_tokens']}")
    
    _test_query_generator()