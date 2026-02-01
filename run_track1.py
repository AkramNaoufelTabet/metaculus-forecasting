#!/usr/bin/env python3
"""
Track-1 Shared Evidence Forecasting Runner
===========================================

Architecture:
    Curator (with OpenRouter web plugin) → Evidence Packet → [6 Forecasters]

All forecasters receive identical curated evidence packet.
Tests reasoning ability with controlled information.

Uses OpenRouter's built-in web search (powered by Exa) - no separate API key needed.

Author: Akram Naoufel Tabet
Date: 2026-02-01
Version: 1.0.0

Usage:
    python track1.py [--max-questions N] [--max-workers N] [--no-cache]
"""

import csv
import json
import hashlib
import re
import time
import sys
import os
import uuid
import argparse
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

from dotenv import load_dotenv
from openrouter_client import OpenRouterClient
from search_pipeline import SearchPipeline
from exa_client import ExaClient


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ProviderLimits:
    """Rate limiting configuration per provider."""
    max_concurrent: int
    retry_base_delay: float = 1.5
    retry_max_delay: float = 30.0
    retry_max_attempts: int = 3


PROVIDER_LIMITS: Dict[str, ProviderLimits] = {
    "anthropic": ProviderLimits(max_concurrent=3),
    "openai": ProviderLimits(max_concurrent=4),
    "google": ProviderLimits(max_concurrent=2, retry_max_attempts=4),
    "deepseek": ProviderLimits(max_concurrent=3),
    "x-ai": ProviderLimits(max_concurrent=3),
    "moonshotai": ProviderLimits(max_concurrent=2),
    "default": ProviderLimits(max_concurrent=2),
}

DEFAULT_PARAMS = {
    "max_tokens": 14000,
}



DISTRIBUTION_SUM_TOLERANCE = 0.03
PROBABILITY_BOUNDS = (0.001, 0.999)

CURATOR_PROMPT_FILES = {
    "binary": "track1_curator_binary.txt",
    "multiclass": "track1_curator_multiclass.txt",
    "numeric": "track1_curator_numeric.txt",
}

FORECASTER_PROMPT_FILES = {
    "binary": "track1_forecaster_binary.txt",
    "multiclass": "track1_forecaster_multiclass.txt",
    "numeric": "track1_forecaster_numeric.txt",
}


# =============================================================================
# THREAD-SAFE INFRASTRUCTURE
# =============================================================================

_print_lock = threading.Lock()
_cost_lock = threading.Lock()
_completed_lock = threading.Lock()
_semaphore_lock = threading.Lock()
_cache_lock = threading.Lock()

_total_cost: float = 0.0
_total_input_tokens: int = 0
_total_output_tokens: int = 0
_completed_jobs: Set[str] = set()
_provider_semaphores: Dict[str, threading.Semaphore] = {}
_evidence_cache: Dict[str, Dict[str, Any]] = {}

# Locks for curator calls (one per question)
_curator_locks: Dict[str, threading.Lock] = {}
_curator_locks_lock = threading.Lock()


def log(msg: str, level: str = "INFO") -> None:
    """Thread-safe logging with timestamp."""
    timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
    with _print_lock:
        print(f"[{timestamp}] [{level}] {msg}", flush=True)


def log_debug(msg: str) -> None:
    log(msg, "DEBUG")


def log_warn(msg: str) -> None:
    log(msg, "WARN")


def log_error(msg: str) -> None:
    log(msg, "ERROR")


def get_provider_semaphore(model_id: str) -> threading.Semaphore:
    """Get or create a semaphore for rate limiting a provider."""
    provider = model_id.split("/")[0] if "/" in model_id else "default"
    
    with _semaphore_lock:
        if provider not in _provider_semaphores:
            limits = PROVIDER_LIMITS.get(provider, PROVIDER_LIMITS["default"])
            _provider_semaphores[provider] = threading.Semaphore(limits.max_concurrent)
            log(f"Rate limiter: {provider} max_concurrent={limits.max_concurrent}")
        return _provider_semaphores[provider]


def get_provider_limits(model_id: str) -> ProviderLimits:
    """Get retry/limit configuration for a provider."""
    provider = model_id.split("/")[0] if "/" in model_id else "default"
    return PROVIDER_LIMITS.get(provider, PROVIDER_LIMITS["default"])


def track_usage(usage: Optional[Dict]) -> Tuple[float, int, int]:
    """Track API usage. Returns (cost, input_tokens, output_tokens)."""
    global _total_cost, _total_input_tokens, _total_output_tokens
    
    if not usage:
        return 0.0, 0, 0
    
    cost = float(usage.get("cost", 0) or 0)
    input_tokens = int(usage.get("prompt_tokens", 0) or 0)
    output_tokens = int(usage.get("completion_tokens", 0) or 0)
    
    with _cost_lock:
        _total_cost += cost
        _total_input_tokens += input_tokens
        _total_output_tokens += output_tokens
    
    return cost, input_tokens, output_tokens


def get_totals() -> Tuple[float, int, int]:
    """Get current totals (thread-safe)."""
    with _cost_lock:
        return _total_cost, _total_input_tokens, _total_output_tokens


def mark_completed(job_key: str, completed_file: Path) -> None:
    """Mark a job as completed for resume capability."""
    with _completed_lock:
        _completed_jobs.add(job_key)
        with open(completed_file, "a", encoding="utf-8") as f:
            f.write(job_key + "\n")


def load_completed(completed_file: Path) -> Set[str]:
    """Load completed jobs from previous run."""
    global _completed_jobs
    
    if not completed_file.exists():
        return set()
    
    with open(completed_file, "r", encoding="utf-8") as f:
        _completed_jobs = set(line.strip() for line in f if line.strip())
    
    return _completed_jobs


def is_completed(job_key: str) -> bool:
    """Check if a job was already completed."""
    with _completed_lock:
        return job_key in _completed_jobs


def get_cached_evidence(question_id: str) -> Optional[Dict[str, Any]]:
    """Get cached evidence packet for a question."""
    with _cache_lock:
        return _evidence_cache.get(question_id)


def set_cached_evidence(question_id: str, data: Dict[str, Any]) -> None:
    """Cache evidence packet for a question."""
    with _cache_lock:
        _evidence_cache[question_id] = data


def get_curator_lock(question_id: str) -> threading.Lock:
    """Get or create lock for curator call on a question."""
    with _curator_locks_lock:
        if question_id not in _curator_locks:
            _curator_locks[question_id] = threading.Lock()
        return _curator_locks[question_id]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def today_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_csv_rows(path: str, max_rows: Optional[int] = None) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            rows.append({k: (v or "") for k, v in row.items()})
            if max_rows is not None and (i + 1) >= max_rows:
                break
    return rows


def append_jsonl_safe(path: Path, record: Dict[str, Any], lock: threading.Lock) -> None:
    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)


def job_key(model_id: str, question_id: str) -> str:
    return f"{model_id}|{question_id}"


# =============================================================================
# QUESTION TYPE AND CATEGORIES
# =============================================================================

def get_question_type(row: Dict[str, str]) -> str:
    qtype = (row.get("question_type") or "").strip().lower()
    
    if qtype in ("binary", "multiclass", "numeric"):
        return qtype
    
    title = (row.get("question_title") or "").strip().lower()
    
    if title.startswith("will "):
        return "binary"
    elif title.startswith("which "):
        return "multiclass"
    elif title.startswith("what will") or title.startswith("how much") or title.startswith("how many"):
        return "numeric"
    
    return "binary"


def get_categories(row: Dict[str, str]) -> List[str]:
    raw = (row.get("categories") or "").strip()
    
    if not raw:
        return []
    
    if "|" in raw:
        return [c.strip() for c in raw.split("|") if c.strip()]
    
    if "," in raw:
        return [c.strip() for c in raw.split(",") if c.strip()]
    
    return [raw]


def format_categories_for_prompt(categories: List[str]) -> str:
    if not categories:
        return "(No categories specified)"
    
    return "\n".join(f"{i+1}. {cat}" for i, cat in enumerate(categories))


def detect_canonical_units(row: Dict[str, str]) -> str:
    full_text = (
        f"{row.get('question_title', '')} "
        f"{row.get('resolution_criteria', '')} "
        f"{row.get('fine_print', '')}"
    ).lower()
    
    unit_patterns = [
        (r"\$[\d,\.]+\s*billion|\bbillions?\s*of\s*dollars|\busd\s*billion", "USD billions"),
        (r"\$[\d,\.]+\s*million|\bmillions?\s*of\s*dollars|\busd\s*million", "USD millions"),
        (r"billion.*\$|\$.*billion", "USD billions"),
        (r"million.*\$|\$.*million", "USD millions"),
        (r"\b\d+\.?\d*\s*%|\bpercent(age)?", "%"),
        (r"\bbasis points?\b|\bbps\b", "basis points"),
        (r"¥|yen|jpy", "JPY"),
        (r"€|euro|eur\b", "EUR"),
        (r"£|pounds?\s*sterling|gbp", "GBP"),
        (r"yuan|cny|rmb", "CNY"),
        (r"pesos?|ars\b", "ARS"),
        (r"per\s*troy\s*ounce|\/troy\s*oz", "USD/troy oz"),
        (r"per\s*5\s*kg|\/5\s*kg", "per 5kg"),
        (r"\bdollars?\b|\busd\b|\$", "USD"),
    ]
    
    for pattern, unit in unit_patterns:
        if re.search(pattern, full_text):
            return unit
    
    return ""


# =============================================================================
# PROMPT HANDLING
# =============================================================================

PLACEHOLDER_RE = re.compile(r"\{\{([^}]+)\}\}")
UNREPLACED_RE = re.compile(r"\{\{[^}]+\}\}")


def render_placeholders(template: str, mapping: Dict[str, str]) -> str:
    def replacer(match: re.Match) -> str:
        key = match.group(1)
        return str(mapping.get(key, ""))
    return PLACEHOLDER_RE.sub(replacer, template)


def split_system_user(prompt_template: str) -> Tuple[str, str]:
    marker = "### USER ###"
    if marker in prompt_template:
        parts = prompt_template.split(marker, 1)
        sys_part = parts[0].replace("### SYSTEM ###", "").strip()
        user_part = parts[1].strip()
        return sys_part, user_part
    return "", prompt_template.strip()


def validate_rendered_prompt(rendered: str, context: str) -> List[str]:
    leftovers = UNREPLACED_RE.findall(rendered)
    if leftovers:
        return [f"Unreplaced placeholders in {context}: {leftovers[:10]}"]
    return []


def get_curator_prompt_template(question_type: str, prompt_dir: str) -> Tuple[str, str]:
    filename = CURATOR_PROMPT_FILES.get(question_type, CURATOR_PROMPT_FILES["binary"])
    path = Path(prompt_dir) / filename
    
    if not path.exists():
        raise FileNotFoundError(f"Curator prompt template not found: {path}")
    
    return load_text(str(path)), str(path)


def get_forecaster_prompt_template(question_type: str, prompt_dir: str) -> Tuple[str, str]:
    filename = FORECASTER_PROMPT_FILES.get(question_type, FORECASTER_PROMPT_FILES["binary"])
    path = Path(prompt_dir) / filename
    
    if not path.exists():
        raise FileNotFoundError(f"Forecaster prompt template not found: {path}")
    
    return load_text(str(path)), str(path)


def build_curator_mapping(
    row: Dict[str, Any],
    question_type: str,
    canonical_units: str,
    raw_search_results: str = "",
) -> Dict[str, str]:
    """
    Build placeholder mapping for curator prompts.
    
    Args:
        row: Question data dict
        question_type: "binary", "multiclass", or "numeric"
        canonical_units: Units string for numeric questions
        raw_search_results: Formatted search results from pipeline
    
    Returns:
        Dict mapping placeholder names to values
    """
    # Handle options - could be list or string
    options = row.get("options", "")
    if isinstance(options, list):
        options = "\n".join(f"- {opt}" for opt in options)
    
    return {
        "question": row.get("question", ""),
        "question_id": str(row.get("question_id", "")),
        "background": row.get("background", ""),
        "fine_print": row.get("fine_print", "") or row.get("resolution_criteria", ""),
        "resolution_criteria": row.get("resolution_criteria", "") or row.get("fine_print", ""),
        "close_date": row.get("close_date", ""),
        "today_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "options": options,
        "units": canonical_units,
        "lower_bound": str(row.get("lower_bound", "")),
        "upper_bound": str(row.get("upper_bound", "")),
        "raw_search_results": raw_search_results,
    }

def build_forecaster_mapping(
    row: Dict[str, str],
    question_type: str,
    evidence_packet: str,
    canonical_units: str
) -> Dict[str, str]:
    """Build placeholder mapping for forecaster prompt."""
    raw_title = (row.get("question_title") or "").strip()
    q_text = raw_title.split("\n")[0].strip() if "\n" in raw_title else raw_title
    
    rc = (row.get("resolution_criteria") or "").strip()
    fp = (row.get("fine_print") or "").strip()
    
    mapping: Dict[str, str] = {
        "TODAY_DATE": today_utc_str(),
        "CLOSE_DATE": (row.get("close_date_utc") or "").strip(),
        "QUESTION_ID": (row.get("question_id") or "").strip(),
        "QUESTION_TEXT": q_text,
        "BACKGROUND_TEXT": (row.get("background") or "No background provided.").strip(),
        "RESOLUTION_CRITERIA": rc or "No resolution criteria provided.",
        "FINE_PRINT": fp or "No fine print.",
        "EVIDENCE_PACKET": evidence_packet,
    }
    
    if question_type == "multiclass":
        categories = get_categories(row)
        mapping["CATEGORY_LIST"] = format_categories_for_prompt(categories)
    
    if question_type == "numeric":
        mapping["CANONICAL_UNITS"] = canonical_units
        mapping["UNITS"] = row.get("units") or canonical_units
        mapping["LOWER_BOUND"] = row.get("lower_bound") or "-∞"
        mapping["UPPER_BOUND"] = row.get("upper_bound") or "+∞"
    
    return mapping


def save_rendered_prompt(
    out_dir: Path,
    model_id: str,
    question_id: str,
    question_type: str,
    prompt_type: str,
    sys_text: str,
    user_text: str
) -> Path:
    prompt_dir = out_dir / "rendered_prompts" / prompt_type / model_id.replace("/", "__")
    ensure_dir(prompt_dir)
    
    prompt_file = prompt_dir / f"{question_id}__{question_type}.txt"
    content = f"=== SYSTEM ===\n{sys_text}\n\n=== USER ===\n{user_text}"
    prompt_file.write_text(content.strip(), encoding="utf-8")
    
    return prompt_file


# =============================================================================
# RESPONSE PARSING
# =============================================================================

BINARY_FINAL_RE = re.compile(
    r"final\s+probability\s*:\s*p?\s*(?:=\s*)?(0(?:\.\d+)?|1(?:\.0+)?)",
    re.IGNORECASE
)
BINARY_FALLBACK_RE = re.compile(r"(?<![.\d])(0\.\d+)(?![.\d])")

DISTRIBUTION_BLOCK_RE = re.compile(
    r"---DISTRIBUTION_START---(.*?)---DISTRIBUTION_END---",
    re.DOTALL | re.IGNORECASE
)
CATEGORY_PROB_RE = re.compile(
    r"CATEGORY:\s*(.+?)\s*\n\s*PROBABILITY:\s*(0(?:\.\d+)?|1(?:\.0+)?)",
    re.IGNORECASE
)

ESTIMATE_BLOCK_RE = re.compile(
    r"---ESTIMATE_START---(.*?)---ESTIMATE_END---",
    re.DOTALL | re.IGNORECASE
)
P10_RE = re.compile(r"P10:\s*([-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?)", re.IGNORECASE)
MEDIAN_RE = re.compile(r"MEDIAN:\s*([-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?)", re.IGNORECASE)
P90_RE = re.compile(r"P90:\s*([-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?)", re.IGNORECASE)
UNITS_RE = re.compile(r"UNITS:\s*(.+?)(?:\n|$)", re.IGNORECASE)


@dataclass
class ParseResult:
    value: Any
    method: str
    warnings: List[str]
    raw_extracted: Dict[str, Any]
    
    @property
    def success(self) -> bool:
        return self.value is not None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "meta": {
                "method": self.method,
                "warnings": self.warnings if self.warnings else None,
                "raw_extracted": self.raw_extracted if self.raw_extracted else None,
            }
        }


def parse_binary(text: str) -> ParseResult:
    if not text or not text.strip():
        return ParseResult(None, "none", ["empty_content"], {})
    
    text = text.strip()
    warnings: List[str] = []
    
    match = BINARY_FINAL_RE.search(text)
    if match:
        p = float(match.group(1))
        p = max(PROBABILITY_BOUNDS[0], min(PROBABILITY_BOUNDS[1], p))
        return ParseResult(p, "final_line", warnings, {"raw": match.group(0)})
    
    all_decimals = BINARY_FALLBACK_RE.findall(text)
    if all_decimals:
        p = float(all_decimals[-1])
        if 0.0 <= p <= 1.0:
            p = max(PROBABILITY_BOUNDS[0], min(PROBABILITY_BOUNDS[1], p))
            warnings.append("weak_parse_fallback")
            return ParseResult(p, "last_decimal", warnings, {"candidates": all_decimals[-5:]})
    
    return ParseResult(None, "none", ["parse_failed"], {})


def parse_multiclass(text: str, expected_categories: List[str]) -> ParseResult:
    if not text or not text.strip():
        return ParseResult(None, "none", ["empty_content"], {})
    
    text = text.strip()
    warnings: List[str] = []
    
    # Find ALL distribution blocks and take the LAST one
    all_blocks = DISTRIBUTION_BLOCK_RE.findall(text)
    if not all_blocks:
        return ParseResult(None, "none", ["no_distribution_block"], {})
    
    # Take the last block (most likely the final answer)
    block = all_blocks[-1]
    
    if len(all_blocks) > 1:
        warnings.append(f"multiple_distribution_blocks:{len(all_blocks)}_found_using_last")
    
    pairs = CATEGORY_PROB_RE.findall(block)
    if not pairs:
        return ParseResult(None, "none", ["no_category_probabilities"], {"block": block[:500]})
    
    dist: Dict[str, float] = {}
    for cat, prob in pairs:
        cat_clean = cat.strip()
        dist[cat_clean] = float(prob)
    
    if expected_categories:
        expected_set = set(c.lower().strip() for c in expected_categories)
        returned_set = set(c.lower().strip() for c in dist.keys())
        
        missing = expected_set - returned_set
        extra = returned_set - expected_set
        
        if missing:
            warnings.append(f"missing_categories:{list(missing)[:5]}")
        if extra:
            warnings.append(f"extra_categories:{list(extra)[:5]}")
    
    total = sum(dist.values())
    if abs(total - 1.0) > DISTRIBUTION_SUM_TOLERANCE:
        warnings.append(f"sum={total:.4f}_deviates_from_1.0")
    
    if 0.9 <= total <= 1.1 and abs(total - 1.0) > 0.001:
        dist = {k: v / total for k, v in dist.items()}
        warnings.append("renormalized")
    
    return ParseResult(
        dist, 
        "distribution_block", 
        warnings,
        {"raw_sum": round(total, 4), "n_categories": len(dist), "blocks_found": len(all_blocks)}
    )

def parse_numeric(text: str, canonical_units: str) -> ParseResult:
    if not text or not text.strip():
        return ParseResult(None, "none", ["empty_content"], {})
    
    text = text.strip()
    warnings: List[str] = []
    
    # Find ALL estimate blocks and take the LAST one
    all_blocks = ESTIMATE_BLOCK_RE.findall(text)
    if not all_blocks:
        return ParseResult(None, "none", ["no_estimate_block"], {})
    
    # Take the last block (most likely the final answer)
    block = all_blocks[-1].replace(",", "")
    
    # If multiple blocks found, note it
    if len(all_blocks) > 1:
        warnings.append(f"multiple_estimate_blocks:{len(all_blocks)}_found_using_last")
    
    p10_m = P10_RE.search(block)
    median_m = MEDIAN_RE.search(block)
    p90_m = P90_RE.search(block)
    units_m = UNITS_RE.search(block)
    
    if not (p10_m and median_m and p90_m):
        missing = []
        if not p10_m: missing.append("P10")
        if not median_m: missing.append("MEDIAN")
        if not p90_m: missing.append("P90")
        return ParseResult(None, "none", [f"missing_{'+'.join(missing)}"], {"block": block[:500]})
    
    try:
        p10 = float(p10_m.group(1).replace(",", ""))
        median = float(median_m.group(1).replace(",", ""))
        p90 = float(p90_m.group(1).replace(",", ""))
    except ValueError as e:
        return ParseResult(None, "none", [f"float_parse_error:{e}"], {})
    
    units = units_m.group(1).strip() if units_m else ""
    
    if not (p10 <= median <= p90):
        warnings.append("ordering_violation_p10<=median<=p90")
        sorted_vals = sorted([p10, median, p90])
        p10, median, p90 = sorted_vals[0], sorted_vals[1], sorted_vals[2]
        warnings.append("auto_corrected_ordering")
    
    value = {
        "p10": p10,
        "median": median,
        "p90": p90,
        "units": units or canonical_units,
    }
    
    return ParseResult(
        value,
        "estimate_block",
        warnings,
        {"raw_units": units, "blocks_found": len(all_blocks)}
    )


def parse_response(
    text: str,
    question_type: str,
    canonical_units: str = "",
    expected_categories: Optional[List[str]] = None
) -> ParseResult:
    if question_type == "binary":
        return parse_binary(text)
    elif question_type == "multiclass":
        return parse_multiclass(text, expected_categories or [])
    elif question_type == "numeric":
        return parse_numeric(text, canonical_units)
    else:
        return ParseResult(None, "none", [f"unknown_question_type:{question_type}"], {})


# =============================================================================
# API CALL WITH RETRY
# =============================================================================

def call_api_with_retry(
    client: OpenRouterClient,
    model_id: str,
    messages: List[Dict[str, str]],
    reasoning_cfg: Optional[Dict[str, Any]],
    model_params: Dict[str, Any],
    plugins: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Optional[str], Optional[str], Optional[Dict], Optional[List[Dict]], Optional[str], int, float]:
    """
    Call API with exponential backoff retry.
    Returns: (content, reasoning, usage, annotations, error, attempts, elapsed_seconds)
    """
    limits = get_provider_limits(model_id)
    
    content = None
    reasoning = None
    usage = None
    annotations = None
    error = None
    
    start_time = time.time()
    
    for attempt in range(1, limits.retry_max_attempts + 1):
        try:
            kwargs = dict(model_params)
            if plugins:
                kwargs["plugins"] = plugins
            
            resp = client.chat(
                model=model_id,
                messages=messages,
                reasoning=reasoning_cfg,
                **kwargs
            )
            
            content = resp.get("content") or ""
            reasoning = resp.get("reasoning")
            usage = resp.get("usage")
            annotations = resp.get("annotations", [])
            
            content_len = len(content)
            reasoning_tokens = 0
            if usage and isinstance(usage, dict):
                details = usage.get("completion_tokens_details", {})
                if isinstance(details, dict):
                    reasoning_tokens = details.get("reasoning_tokens", 0)
            
            if content_len == 0 and reasoning_tokens > 0:
                if attempt < limits.retry_max_attempts:
                    error = "empty_content_with_reasoning_tokens"
                    delay = min(
                        limits.retry_max_delay,
                        limits.retry_base_delay * (2 ** (attempt - 1))
                    ) + random.uniform(0, 1)
                    log_warn(f"Empty content but {reasoning_tokens} reasoning tokens, retry in {delay:.1f}s")
                    time.sleep(delay)
                    continue
            
            elapsed = time.time() - start_time
            return content, reasoning, usage, annotations, None, attempt, elapsed
            
        except Exception as e:
            error = str(e)
            
            if attempt < limits.retry_max_attempts:
                delay = min(
                    limits.retry_max_delay,
                    limits.retry_base_delay * (2 ** (attempt - 1))
                ) + random.uniform(0, 0.5)
                log_warn(f"API error: {error[:100]}... Retry in {delay:.1f}s")
                time.sleep(delay)
            else:
                log_error(f"API failed after {attempt} attempts: {error[:200]}")
    
    elapsed = time.time() - start_time
    return content, reasoning, usage, None, error, limits.retry_max_attempts, elapsed

# =============================================================================
# EVIDENCE CURATION (TRACK 1 SPECIFIC)
# =============================================================================

_search_pipeline: Optional[SearchPipeline] = None

def get_search_pipeline(client: OpenRouterClient, cache_dir: Optional[Path] = None) -> SearchPipeline:
    """Get or create search pipeline singleton."""
    global _search_pipeline
    if _search_pipeline is None:
        exa_client = ExaClient()
        _search_pipeline = SearchPipeline(
            openrouter_client=client,
            exa_client=exa_client,
            cache_dir=cache_dir,
            query_model="anthropic/claude-sonnet-4.5",
            results_per_query=8,
            max_total_results=25,
            relevance_threshold=2,
            enable_retry=True,
        )
    return _search_pipeline


def curate_evidence(
    client: OpenRouterClient,
    curator_config: Dict[str, Any],
    row: Dict[str, Any],
    question_type: str,
    canonical_units: str,
    prompt_dir: str,
    out_dir: Path,
) -> Tuple[str, Dict[str, Any]]:
    """
    Run curator with intelligent web search to create evidence packet.
    
    Pipeline:
    1. Generate search queries (LLM)
    2. Execute searches (Exa direct)
    3. Validate relevance (sanity gate)
    4. Retry if needed
    5. Pass results to curator (NO PLUGIN)
    
    Args:
        client: OpenRouter client
        curator_config: Curator model configuration
        row: Question data dict
        question_type: "binary", "multiclass", or "numeric"
        canonical_units: Units for numeric questions
        prompt_dir: Path to prompt templates
        out_dir: Output directory
    
    Returns:
        Tuple of (evidence_content, metadata_dict)
    """
    question_id = str(row.get("question_id", "unknown"))
    model_id = curator_config["openrouter_id"]
    model_label = curator_config.get("label", "Curator")
    model_params = curator_config.get("params", {})
    
    # =========================================================
    # STEP 1-4: Search Pipeline (Query Gen → Exa → Validate)
    # =========================================================
    search_cache_dir = out_dir / "search_cache"
    pipeline = get_search_pipeline(client, cache_dir=search_cache_dir)
    
    log(f"  [SEARCH] Running search pipeline for Q:{question_id}...")
    search_result = pipeline.search_for_question(row, use_cache=True)
    
    search_cost = search_result.get("cost", 0)
    search_success = search_result.get("success", False)
    
    if not search_success:
        log_warn(f"  [SEARCH] Low relevance for Q:{question_id}")
    
    # Log search details
    queries = search_result.get("queries", [])
    relevance = search_result.get("relevance", {})
    log(f"  [SEARCH] Generated {len(queries)} queries")
    for i, q in enumerate(queries[:3], 1):
        log(f"  [SEARCH]   {i}. {q[:60]}...")
    log(f"  [SEARCH] Relevance: {relevance.get('match_count', 0)}/{len(search_result.get('must_hit_tokens', []))} tokens matched")
    
    # =========================================================
    # STEP 5: Build curator prompt with search results
    # =========================================================
    prompt_template, template_path = get_curator_prompt_template(question_type, prompt_dir)
    sys_text, user_template = split_system_user(prompt_template)
    
    # Build mapping WITH search results
    mapping = build_curator_mapping(
        row,
        question_type,
        canonical_units,
        raw_search_results=search_result["formatted_context"],
    )
    
    # Render template (replaces {{raw_search_results}} placeholder)
    user_text = render_placeholders(user_template, mapping)
    
    # Save rendered prompt for debugging
    prompt_file = save_rendered_prompt(
        out_dir, model_id, question_id, question_type, "curator", sys_text, user_text
    )
    
    # Build messages
    messages: List[Dict[str, str]] = []
    if sys_text:
        messages.append({"role": "system", "content": sys_text})
    messages.append({"role": "user", "content": user_text})
    
    # =========================================================
    # STEP 6: Call curator (NO PLUGIN - search already done)
    # =========================================================
    log(f"  [CURATOR] Generating evidence packet for Q:{question_id}...")
    
    t0 = time.time()
    try:
        response = client.chat(
            model=model_id,
            messages=messages,
            **model_params,
        )
        elapsed = time.time() - t0
        
        content = response.get("content", "")
        usage = response.get("usage", {})
        cost = usage.get("cost", 0)
        error = None
        
    except Exception as e:
        elapsed = time.time() - t0
        content = ""
        usage = {}
        cost = 0
        error = str(e)
        log_error(f"  [CURATOR] API error: {e}")
    
    total_cost = cost + search_cost
    
    # Build citations from search results
    citations = [
        {"url": r.get("url"), "title": r.get("title")}
        for r in search_result.get("results", [])
    ]
    
    # Build metadata
    metadata = {
        "model_id": model_id,
        "model_label": model_label,
        "search": {
            "method": "exa_direct",
            "queries": search_result.get("queries", []),
            "must_hit_tokens": search_result.get("must_hit_tokens", []),
            "relevance": search_result.get("relevance", {}),
            "retried": search_result.get("retried", False),
            "results_count": len(search_result.get("results", [])),
            "search_cost": round(search_cost, 6),
            "cached": search_result.get("cached", False),
        },
        "prompt_file": str(prompt_file.relative_to(out_dir)) if prompt_file else None,
        "template_file": template_path,
        "elapsed_seconds": round(elapsed, 2),
        "error": error,
        "cost": round(total_cost, 6),
        "curator_cost": round(cost, 6),
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "content_length": len(content) if content else 0,
        "citations_count": len(citations),
        "citations": citations,
    }
    
    if error or not content:
        log_error(f"  [CURATOR] Failed for Q:{question_id}: {error}")
        return "", metadata
    
    log(f"  [CURATOR] Done: {len(content)} chars, {len(citations)} sources, ${total_cost:.4f}, {elapsed:.1f}s")
    
    return content, metadata

def get_or_create_evidence(
    client: OpenRouterClient,
    curator_config: Dict[str, Any],
    row: Dict[str, str],
    question_type: str,
    canonical_units: str,
    prompt_dir: str,
    out_dir: Path,
    cache_dir: Path,
    use_cache: bool,
) -> Tuple[str, Dict[str, Any]]:
    """
    Get evidence packet for question (from cache or generate).
    Thread-safe: only one curator call per question.
    """
    question_id = (row.get("question_id") or "").strip()
    
    # Check in-memory cache first
    cached = get_cached_evidence(question_id)
    if cached:
        log_debug(f"  [CACHE HIT] In-memory evidence for Q:{question_id}")
        return cached["evidence_packet"], cached["metadata"]
    
    # Check disk cache
    cache_file = cache_dir / f"{question_id}_evidence.json"
    if use_cache and cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            
            cached_date = cached.get("date", "")
            if cached_date == today_utc_str():
                log(f"  [CACHE HIT] Disk evidence for Q:{question_id}")
                set_cached_evidence(question_id, cached)
                return cached["evidence_packet"], cached["metadata"]
            else:
                log(f"  [CACHE STALE] Evidence from {cached_date}, regenerating")
        except Exception as e:
            log_warn(f"  [CACHE ERROR] {e}")
    
    # Acquire lock for this question
    curator_lock = get_curator_lock(question_id)
    
    with curator_lock:
        # Double-check cache after acquiring lock
        cached = get_cached_evidence(question_id)
        if cached:
            return cached["evidence_packet"], cached["metadata"]
        
        # Run curator with web plugin
        evidence_packet, curator_metadata = curate_evidence(
            client=client,
            curator_config=curator_config,
            row=row,
            question_type=question_type,
            canonical_units=canonical_units,
            prompt_dir=prompt_dir,
            out_dir=out_dir,
        )
        
        metadata = {
            "curator": curator_metadata,
        }
        
        # Cache result
        cache_data = {
            "question_id": question_id,
            "date": today_utc_str(),
            "evidence_packet": evidence_packet,
            "metadata": metadata,
        }
        
        set_cached_evidence(question_id, cache_data)
        
        # Save to disk
        ensure_dir(cache_dir)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        
        return evidence_packet, metadata


# =============================================================================
# FORECASTER JOB EXECUTION
# =============================================================================

def execute_forecast_job(
    *,
    client: OpenRouterClient,
    curator_config: Dict[str, Any],
    run_id: str,
    out_dir: Path,
    cache_dir: Path,
    out_path: Path,
    out_lock: threading.Lock,
    completed_file: Path,
    model_id: str,
    model_label: str,
    reasoning_cfg: Optional[Dict[str, Any]],
    model_params: Dict[str, Any],
    row: Dict[str, str],
    prompt_dir: str,
    use_cache: bool,
) -> Optional[Dict[str, Any]]:
    """Execute a single forecasting job (one model × one question)."""
    question_id = (row.get("question_id") or "").strip()
    if not question_id:
        log_warn(f"[{model_id}] Missing question_id, skipping")
        return None
    
    jkey = job_key(model_id, question_id)
    
    if is_completed(jkey):
        log_debug(f"[{model_id}] Q:{question_id} already completed, skipping")
        return None
    
    question_type = get_question_type(row)
    categories = get_categories(row) if question_type == "multiclass" else []
    canonical_units = detect_canonical_units(row) if question_type == "numeric" else ""
    
    log(f"[{model_label}] Q:{question_id} type={question_type}")
    
    # Get or create evidence packet (shared across all models)
    evidence_packet, evidence_metadata = get_or_create_evidence(
        client=client,
        curator_config=curator_config,
        row=row,
        question_type=question_type,
        canonical_units=canonical_units,
        prompt_dir=prompt_dir,
        out_dir=out_dir,
        cache_dir=cache_dir,
        use_cache=use_cache,
    )
    
    if not evidence_packet:
        log_error(f"[{model_label}] Q:{question_id} No evidence packet, skipping forecast")
        return None
    
    # Load forecaster prompt
    try:
        prompt_template, template_path = get_forecaster_prompt_template(question_type, prompt_dir)
    except FileNotFoundError as e:
        log_error(f"[{model_label}] Q:{question_id} {e}")
        return None
    
    # Build and render prompt
    sys_text, user_template = split_system_user(prompt_template)
    mapping = build_forecaster_mapping(row, question_type, evidence_packet, canonical_units)
    user_text = render_placeholders(user_template, mapping)
    
    # Validate
    prompt_warnings = validate_rendered_prompt(user_text, f"Q:{question_id}")
    if prompt_warnings:
        log_error(f"[{model_label}] Q:{question_id} {prompt_warnings}")
        return None
    
    # Save rendered prompt
    prompt_file = save_rendered_prompt(
        out_dir, model_id, question_id, question_type, "forecaster", sys_text, user_text
    )
    prompt_hash = sha256_text((sys_text + "\n\n" + user_text).strip())
    
    # Build messages
    messages: List[Dict[str, str]] = []
    if sys_text:
        messages.append({"role": "system", "content": sys_text})
    messages.append({"role": "user", "content": user_text})
    
    # Execute API call (no web plugin for forecasters)
    content, reasoning, usage, _, error, attempts, elapsed = call_api_with_retry(
        client=client,
        model_id=model_id,
        messages=messages,
        reasoning_cfg=reasoning_cfg,
        model_params=model_params,
        plugins=None,
    )
    
    cost, input_tokens, output_tokens = track_usage(usage)
    
    # Parse response
    if content is not None:
        parse_result = parse_response(
            content, 
            question_type, 
            canonical_units,
            expected_categories=categories
        )
    else:
        parse_result = ParseResult(None, "none", ["api_error"], {})
    
    # Determine status
    if error and content is None:
        status = "api_error"
    elif not content:
        status = "empty_content"
    elif parse_result.success:
        status = "ok"
    else:
        status = "parse_failed"
    
    reasoning_tokens = 0
    if usage and isinstance(usage, dict):
        details = usage.get("completion_tokens_details", {})
        if isinstance(details, dict):
            reasoning_tokens = details.get("reasoning_tokens", 0)
    
    # Build result record
    record = {
        "run_id": run_id,
        "track": 1,
        "model": model_id,
        "model_label": model_label,
        "question_id": question_id,
        "question_type": question_type,
        "timestamp_utc": utc_now_iso(),
        "elapsed_seconds": round(elapsed, 2),
        "input": {
            "params": model_params,
            "reasoning_config": reasoning_cfg,
            "prompt_sha256": prompt_hash,
            "prompt_file": str(prompt_file.relative_to(out_dir)),
            "template_file": template_path,
            "canonical_units": canonical_units if question_type == "numeric" else None,
            "categories": categories if question_type == "multiclass" else None,
            "evidence_packet_length": len(evidence_packet),
        },
        "evidence": evidence_metadata,
        "output": {
            "content_length": len(content) if content else 0,
            "reasoning_length": len(reasoning) if reasoning else 0,
            "reasoning_tokens": reasoning_tokens,
            "content": content,
            "reasoning": reasoning,
            "usage": usage,
            "error": error,
            "attempts": attempts,
        },
        "parsed": parse_result.to_dict(),
        "status": status,
        "cost": {
            "this_call": round(cost, 6),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }
    
    # Write result
    append_jsonl_safe(out_path, record, out_lock)
    mark_completed(jkey, completed_file)
    
    # Log summary
    if question_type == "binary":
        log(f"  → {status} p={parse_result.value} cost=${cost:.4f} ({elapsed:.1f}s)")
    elif question_type == "multiclass":
        if parse_result.value:
            top = max(parse_result.value.items(), key=lambda x: x[1])
            log(f"  → {status} top='{top[0][:25]}'={top[1]:.3f} cost=${cost:.4f} ({elapsed:.1f}s)")
        else:
            log(f"  → {status} dist=None cost=${cost:.4f} ({elapsed:.1f}s)")
    elif question_type == "numeric":
        if parse_result.value:
            v = parse_result.value
            log(f"  → {status} median={v['median']} [{v['p10']}, {v['p90']}] {v['units']} cost=${cost:.4f} ({elapsed:.1f}s)")
        else:
            log(f"  → {status} value=None cost=${cost:.4f} ({elapsed:.1f}s)")
    
    return record


def worker_thread(
    job_tuple: Tuple,
    curator_config: Dict[str, Any],
    out_dir: Path,
    cache_dir: Path,
    completed_file: Path,
    prompt_dir: str,
    run_id: str,
    use_cache: bool,
) -> Optional[Dict[str, Any]]:
    """Worker function for thread pool."""
    (
        model_label, model_id, reasoning_cfg, model_params,
        out_path, out_lock, row
    ) = job_tuple
    
    semaphore = get_provider_semaphore(model_id)
    
    with semaphore:
        client = OpenRouterClient(debug=True)
        
        return execute_forecast_job(
            client=client,
            curator_config=curator_config,
            run_id=run_id,
            out_dir=out_dir,
            cache_dir=cache_dir,
            out_path=out_path,
            out_lock=out_lock,
            completed_file=completed_file,
            model_id=model_id,
            model_label=model_label,
            reasoning_cfg=reasoning_cfg,
            model_params=model_params,
            row=row,
            prompt_dir=prompt_dir,
            use_cache=use_cache,
        )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def create_manifest(
    run_id: str,
    out_dir: Path,
    config: Dict[str, Any],
    curator_config: Dict[str, Any],
    models: List[Dict],
    questions: List[Dict],
    type_counts: Dict[str, int],
) -> Dict[str, Any]:
    """Create experiment manifest for reproducibility."""
    manifest = {
        "run_id": run_id,
        "track": 1,
        "track_name": "shared-evidence",
        "started_at_utc": utc_now_iso(),
        "config": config,
        "inputs": {
            "n_questions": len(questions),
            "n_forecasters": len(models),
            "question_types": type_counts,
            "curator": {
                "label": curator_config.get("label"),
                "openrouter_id": curator_config.get("openrouter_id"),
                                "search_method": "exa_direct",
                "params": curator_config.get("params"),
            },
            "forecasters": [
                {
                    "label": m.get("label"),
                    "openrouter_id": m.get("openrouter_id"),
                    "reasoning": m.get("reasoning"),
                    "params": m.get("params"),
                }
                for m in models
            ],
        },
        "provider_limits": {
            k: asdict(v) for k, v in PROVIDER_LIMITS.items()
        },
        "curator_prompt_files": CURATOR_PROMPT_FILES,
        "forecaster_prompt_files": FORECASTER_PROMPT_FILES,
        "version": "1.0.0",
    }
    
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    
    return manifest


def finalize_run(
    out_dir: Path, 
    start_time: float, 
    jobs_total: int, 
    jobs_ok: int, 
    jobs_failed: int,
    questions_processed: int,
) -> None:
    """Write final summary and statistics."""
    elapsed = time.time() - start_time
    total_cost, total_in, total_out = get_totals()
    
    summary = {
        "completed_at_utc": utc_now_iso(),
        "elapsed_seconds": round(elapsed, 2),
        "elapsed_human": f"{elapsed/60:.1f} minutes",
        "jobs": {
            "total": jobs_total,
            "completed_ok": jobs_ok,
            "failed": jobs_failed,
            "success_rate": round(jobs_ok / jobs_total, 4) if jobs_total > 0 else 0,
        },
        "questions_processed": questions_processed,
        "costs": {
            "total_usd": round(total_cost, 4),
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "avg_cost_per_job": round(total_cost / jobs_ok, 6) if jobs_ok > 0 else 0,
        },
    }
    
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    
    log("=" * 60)
    log(f"RUN COMPLETE: {out_dir}")
    log(f"  Duration: {elapsed/60:.1f} minutes")
    log(f"  Questions processed: {questions_processed}")
    log(f"  Forecast jobs: {jobs_ok}/{jobs_total} OK ({jobs_failed} failed)")
    log(f"  Total cost: ${total_cost:.4f}")
    log(f"  Tokens: {total_in:,} in / {total_out:,} out")
    log("=" * 60)


def main():
    """Main entry point."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
    
    parser = argparse.ArgumentParser(
        description="Track-1 Shared Evidence Forecasting Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--questions", default="data/questions.csv", help="Questions CSV path")
    parser.add_argument("--models", default="data/models_track1.json", help="Models JSON path")
    parser.add_argument("--prompts", default="data/prompts", help="Prompts directory")
    parser.add_argument("--output", default="run/track1", help="Output base directory")
    parser.add_argument("--max-questions", type=int, default=None, help="Limit questions")
    parser.add_argument("--max-workers", type=int, default=6, help="Thread pool size")
    parser.add_argument("--resume", type=str, default=None, help="Resume from run_id")
    parser.add_argument("--no-cache", action="store_true", help="Disable evidence caching")
    args = parser.parse_args()
    
    load_dotenv()
    
    # Validate API key
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    
    if not openrouter_key:
        log_error("Missing OPENROUTER_API_KEY environment variable")
        sys.exit(1)
    
    # Determine run ID and output directory
    if args.resume:
        run_id = args.resume
        out_dir = Path(args.output) / run_id
        if not out_dir.exists():
            log_error(f"Resume directory not found: {out_dir}")
            sys.exit(1)
        log(f"RESUMING run_id={run_id}")
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        out_dir = Path(args.output) / run_id
    
    ensure_dir(out_dir)
    cache_dir = out_dir / "evidence_cache"
    ensure_dir(cache_dir)
    
    # Load completed jobs (for resume)
    completed_file = out_dir / "completed.txt"
    if args.resume:
        completed = load_completed(completed_file)
        log(f"Loaded {len(completed)} completed jobs from previous run")
    
    # Load inputs
    questions = load_csv_rows(args.questions, max_rows=args.max_questions)
    model_config = load_json(args.models)
    
    curator_config = model_config.get("curator", {})
    forecasters = model_config.get("forecasters", [])
    
    if not curator_config.get("openrouter_id"):
        log_error("Missing curator configuration in models JSON")
        sys.exit(1)
    
    if not forecasters:
        log_error("No forecasters defined in models JSON")
        sys.exit(1)
    
    # Count question types
    type_counts: Dict[str, int] = {"binary": 0, "multiclass": 0, "numeric": 0}
    for row in questions:
        qtype = get_question_type(row)
        type_counts[qtype] = type_counts.get(qtype, 0) + 1
    
    # Log configuration
    log("=" * 60)
    log(f"TRACK-1 SHARED EVIDENCE FORECASTING")
    log(f"  run_id: {run_id}")
    log(f"  output: {out_dir}")
    log(f"  questions: {len(questions)}")
    log(f"    - binary:     {type_counts['binary']}")
    log(f"    - multiclass: {type_counts['multiclass']}")
    log(f"    - numeric:    {type_counts['numeric']}")
    log(f"  curator: {curator_config.get('label')} [{curator_config.get('openrouter_id')}]")
    log(f"    search: Exa direct (query gen → retrieval → sanity gate)")
    log(f"  forecasters: {len(forecasters)}")
    for i, m in enumerate(forecasters, 1):
        log(f"    {i}. {m.get('label')} [{m.get('openrouter_id')}]")
    log(f"  evidence caching: {'disabled' if args.no_cache else 'enabled'}")
    log(f"  max_workers: {args.max_workers}")
    log("=" * 60)
    
    # Create manifest
    config = {
        "questions_csv": args.questions,
        "models_json": args.models,
        "prompts_dir": args.prompts,
        "max_questions": args.max_questions,
        "max_workers": args.max_workers,
        "use_cache": not args.no_cache,
        "default_params": DEFAULT_PARAMS,
    }
    
    if not args.resume:
        create_manifest(run_id, out_dir, config, curator_config, forecasters, questions, type_counts)
    
    # Prepare file locks
    out_locks: Dict[str, threading.Lock] = {}
    
    def get_out_lock(path: Path) -> threading.Lock:
        key = str(path)
        if key not in out_locks:
            out_locks[key] = threading.Lock()
        return out_locks[key]
    
    # Build job queue
    jobs: List[Tuple] = []
    
    for m in forecasters:
        model_label = m.get("label", "unknown")
        model_id = (m.get("openrouter_id") or "").strip()
        
        if not model_id:
            log_warn(f"Skipping model with missing openrouter_id: {model_label}")
            continue
        
        reasoning_cfg = m.get("reasoning")
        model_params = dict(DEFAULT_PARAMS)
        if isinstance(m.get("params"), dict):
            model_params.update(m["params"])
        
        out_path = out_dir / f"{model_id.replace('/', '__')}.jsonl"
        out_lock = get_out_lock(out_path)
        
        for row in questions:
            jobs.append((
                model_label, model_id, reasoning_cfg, model_params,
                out_path, out_lock, row
            ))
    
    log(f"Total forecast jobs: {len(jobs)} ({len(forecasters)} models × {len(questions)} questions)")
    
    # Execute with thread pool
    start_time = time.time()
    jobs_ok = 0
    jobs_failed = 0
    jobs_skipped = 0
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                worker_thread, job, curator_config, out_dir, cache_dir,
                completed_file, args.prompts, run_id, not args.no_cache
            ): job
            for job in jobs
        }
        
        for future in as_completed(futures):
            try:
                result = future.result()
                if result is None:
                    jobs_skipped += 1
                elif result.get("status") == "ok":
                    jobs_ok += 1
                else:
                    jobs_failed += 1
            except Exception as e:
                jobs_failed += 1
                log_error(f"Job exception: {e}")
            
            done = jobs_ok + jobs_failed + jobs_skipped
            if done % 5 == 0 or done == len(jobs):
                elapsed = time.time() - start_time
                total_cost, _, _ = get_totals()
                log(f"Progress: {done}/{len(jobs)} ({jobs_ok} ok, {jobs_failed} failed, {jobs_skipped} skipped) "
                    f"elapsed={elapsed:.0f}s cost=${total_cost:.4f}")
    
    finalize_run(out_dir, start_time, len(jobs) - jobs_skipped, jobs_ok, jobs_failed, len(questions))


if __name__ == "__main__":
    main()