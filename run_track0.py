#!/usr/bin/env python3
"""
Track-0 Closed-Book Forecasting Runner
=======================================

Academic-grade implementation for Metaculus forecasting experiments.

Features:
- Multi-model parallel execution with provider-aware rate limiting
- Uses pre-computed question types and categories from CSV
- Machine-parseable output formats
- Full audit trail (prompts, responses, costs, timings)
- Robust retry logic with exponential backoff
- Resume capability for interrupted runs
- Thread-safe logging and file writes

Author: Akram Naoufel Tabet
Date: 2026-01-28
Version: 1.1.0

Usage:
    python track0.py [--max-questions N] [--max-workers N]


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


# Provider-specific concurrency limits (tune based on your API tier)
PROVIDER_LIMITS: Dict[str, ProviderLimits] = {
    "anthropic": ProviderLimits(max_concurrent=3),
    "openai": ProviderLimits(max_concurrent=4),
    "google": ProviderLimits(max_concurrent=2, retry_max_attempts=4),
    "deepseek": ProviderLimits(max_concurrent=3),
    "x-ai": ProviderLimits(max_concurrent=3),
    "moonshotai": ProviderLimits(max_concurrent=2),
    "default": ProviderLimits(max_concurrent=2),
}

# Default model parameters
DEFAULT_PARAMS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 14000,
    "seed": 42,
}

# Parsing tolerances
DISTRIBUTION_SUM_TOLERANCE = 0.03
PROBABILITY_BOUNDS = (0.001, 0.999)

# Prompt template filenames
PROMPT_FILES = {
    "binary": "track0_binary.txt",
    "multiclass": "track0_multiclass.txt",
    "numeric": "track0_numeric.txt",
}


# =============================================================================
# THREAD-SAFE INFRASTRUCTURE
# =============================================================================

_print_lock = threading.Lock()
_cost_lock = threading.Lock()
_completed_lock = threading.Lock()
_semaphore_lock = threading.Lock()

_total_cost: float = 0.0
_total_input_tokens: int = 0
_total_output_tokens: int = 0
_completed_jobs: Set[str] = set()
_provider_semaphores: Dict[str, threading.Semaphore] = {}


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
    """
    Track API usage. Returns (cost, input_tokens, output_tokens).
    Thread-safe accumulation.
    """
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


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def utc_now_iso() -> str:
    """Current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def today_utc_str() -> str:
    """Current UTC date as YYYY-MM-DD."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def sha256_text(s: str) -> str:
    """SHA256 hash of text for fingerprinting."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def ensure_dir(p: Path) -> None:
    """Create directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)


def load_json(path: str) -> Any:
    """Load JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_text(path: str) -> str:
    """Load text file."""
    return Path(path).read_text(encoding="utf-8")


def load_csv_rows(path: str, max_rows: Optional[int] = None) -> List[Dict[str, str]]:
    """Load CSV as list of dicts, optionally limited."""
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            rows.append({k: (v or "") for k, v in row.items()})
            if max_rows is not None and (i + 1) >= max_rows:
                break
    return rows


def append_jsonl_safe(path: Path, record: Dict[str, Any], lock: threading.Lock) -> None:
    """Thread-safe JSONL append."""
    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)


def job_key(model_id: str, question_id: str) -> str:
    """Create unique job identifier."""
    return f"{model_id}|{question_id}"


# =============================================================================
# QUESTION TYPE AND CATEGORIES
# =============================================================================

def get_question_type(row: Dict[str, str]) -> str:
    """
    Get question type from pre-computed column.
    Falls back to simple title-based detection if column is missing.
    
    Expected column: 'question_type' with values: binary, multiclass, numeric
    """
    # Primary: read from column
    qtype = (row.get("question_type") or "").strip().lower()
    
    if qtype in ("binary", "multiclass", "numeric"):
        return qtype
    
    # Fallback: simple title-based detection
    title = (row.get("question_title") or "").strip().lower()
    
    if title.startswith("will "):
        return "binary"
    elif title.startswith("which "):
        return "multiclass"
    elif title.startswith("what will") or title.startswith("how much") or title.startswith("how many"):
        return "numeric"
    
    # Default
    return "binary"


def get_categories(row: Dict[str, str]) -> List[str]:
    """
    Get categories from pre-computed column.
    
    Expected column: 'categories' with pipe-separated values
    Example: "Norway|Germany|United States|Italy|Canada|Other"
    
    Returns: List of category strings
    """
    raw = (row.get("categories") or "").strip()
    
    if not raw:
        return []
    
    # Parse pipe-separated
    if "|" in raw:
        return [c.strip() for c in raw.split("|") if c.strip()]
    
    # Fallback: comma-separated (if no pipes)
    if "," in raw:
        return [c.strip() for c in raw.split(",") if c.strip()]
    
    # Single category
    return [raw]


def format_categories_for_prompt(categories: List[str]) -> str:
    """
    Format categories as numbered list for prompt.
    
    Input: ["Norway", "Germany", "United States"]
    Output:
        1. Norway
        2. Germany
        3. United States
    """
    if not categories:
        return "(No categories specified)"
    
    return "\n".join(f"{i+1}. {cat}" for i, cat in enumerate(categories))


def detect_canonical_units(row: Dict[str, str]) -> str:
    """
    Detect canonical units for numeric questions.
    Used for normalizing model outputs to consistent units.
    """
    full_text = (
        f"{row.get('question_title', '')} "
        f"{row.get('resolution_criteria', '')} "
        f"{row.get('fine_print', '')}"
    ).lower()
    
    # Order matters: check more specific patterns first
    unit_patterns = [
        # Currency with scale
        (r"\$[\d,\.]+\s*billion|\bbillions?\s*of\s*dollars|\busd\s*billion", "USD billions"),
        (r"\$[\d,\.]+\s*million|\bmillions?\s*of\s*dollars|\busd\s*million", "USD millions"),
        (r"billion.*\$|\$.*billion", "USD billions"),
        (r"million.*\$|\$.*million", "USD millions"),
        
        # Percentages
        (r"\b\d+\.?\d*\s*%|\bpercent(age)?", "%"),
        (r"\bbasis points?\b|\bbps\b", "basis points"),
        
        # Specific currencies
        (r"¥|yen|jpy", "JPY"),
        (r"€|euro|eur\b", "EUR"),
        (r"£|pounds?\s*sterling|gbp", "GBP"),
        (r"yuan|cny|rmb", "CNY"),
        (r"pesos?|ars\b", "ARS"),
        
        # Commodities
        (r"per\s*troy\s*ounce|\/troy\s*oz", "USD/troy oz"),
        (r"per\s*5\s*kg|\/5\s*kg", "per 5kg"),
        
        # Generic
        (r"\bdollars?\b|\busd\b|\$", "USD"),
    ]
    
    for pattern, unit in unit_patterns:
        if re.search(pattern, full_text):
            return unit
    
    return ""


def extract_numeric_units_hint(row: Dict[str, str]) -> str:
    """Extract units hint from question for prompt context."""
    full_text = f"{row.get('question_title', '')} {row.get('resolution_criteria', '')}".lower()
    
    unit_patterns = [
        (r"\$.*billion", "USD billions"),
        (r"\$.*million", "USD millions"),
        (r"percent(age)?|%", "%"),
        (r"basis points|bps", "basis points"),
        (r"troy ounce", "USD per troy ounce"),
        (r"¥|yen|jpy", "JPY"),
        (r"€|euro|eur", "EUR"),
        (r"£|pounds?|gbp", "GBP"),
        (r"yuan|cny|rmb", "CNY"),
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
    """Replace {{KEY}} placeholders with values from mapping."""
    def replacer(match: re.Match) -> str:
        key = match.group(1)
        return str(mapping.get(key, ""))
    return PLACEHOLDER_RE.sub(replacer, template)


def split_system_user(prompt_template: str) -> Tuple[str, str]:
    """Split prompt template into system and user parts."""
    marker = "### USER ###"
    if marker in prompt_template:
        parts = prompt_template.split(marker, 1)
        sys_part = parts[0].replace("### SYSTEM ###", "").strip()
        user_part = parts[1].strip()
        return sys_part, user_part
    return "", prompt_template.strip()


def validate_rendered_prompt(rendered: str, context: str) -> List[str]:
    """
    Validate rendered prompt has no unreplaced placeholders.
    Returns list of warnings (empty if valid).
    """
    leftovers = UNREPLACED_RE.findall(rendered)
    if leftovers:
        return [f"Unreplaced placeholders in {context}: {leftovers[:10]}"]
    return []


def get_prompt_template(question_type: str, prompt_dir: str) -> Tuple[str, str]:
    """
    Load prompt template for question type.
    Returns (template_content, template_path).
    """
    filename = PROMPT_FILES.get(question_type, PROMPT_FILES["binary"])
    path = Path(prompt_dir) / filename
    
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    
    return load_text(str(path)), str(path)


def build_prompt_mapping(
    row: Dict[str, str],
    question_type: str,
    canonical_units: str
) -> Dict[str, str]:
    """
    Build placeholder mapping for prompt rendering.
    
    Uses pre-computed categories from CSV for multiclass questions.
    """
    raw_title = (row.get("question_title") or "").strip()
    
    # Extract clean question text (first line if multiline)
    q_text = raw_title.split("\n")[0].strip() if "\n" in raw_title else raw_title
    q_url = (row.get("question_url") or "").strip()
    question_block = q_text + (f"\nURL: {q_url}" if q_url else "")
    
    # Resolution criteria + fine print
    rc = (row.get("resolution_criteria") or "").strip()
    fp = (row.get("fine_print") or "").strip()
    rc_plus_fp = rc + ("\n\nFine Print:\n" + fp if fp else "")
    
    mapping: Dict[str, str] = {
        "TODAY_DATE": today_utc_str(),
        "CLOSE_DATE": (row.get("close_date_utc") or "").strip(),
        "RESOLUTION_DATE": (row.get("resolution_date_utc") or row.get("close_date_utc") or "").strip(),
        "QUESTION_TEXT": question_block,
        "BACKGROUND_TEXT": (row.get("background") or "").strip(),
        "RESOLUTION_CRITERIA + FINE_PRINT": rc_plus_fp,
    }
    
    # Multiclass: format categories as numbered list
    if question_type == "multiclass":
        categories = get_categories(row)
        mapping["CATEGORY_LIST"] = format_categories_for_prompt(categories)
    
    # Numeric: add units and bounds
    if question_type == "numeric":
        mapping["CANONICAL_UNITS"] = canonical_units
        mapping["UNITS"] = row.get("units") or extract_numeric_units_hint(row) or canonical_units
        mapping["LOWER_BOUND"] = row.get("lower_bound") or ""
        mapping["UPPER_BOUND"] = row.get("upper_bound") or ""
        mapping["VARIABLE_NAME"] = row.get("variable_name") or q_text
    
    return mapping


def save_rendered_prompt(
    out_dir: Path,
    model_id: str,
    question_id: str,
    question_type: str,
    sys_text: str,
    user_text: str
) -> Path:
    """Save rendered prompt for audit trail."""
    prompt_dir = out_dir / "rendered_prompts" / model_id.replace("/", "__")
    ensure_dir(prompt_dir)
    
    prompt_file = prompt_dir / f"{question_id}__{question_type}.txt"
    content = f"=== SYSTEM ===\n{sys_text}\n\n=== USER ===\n{user_text}"
    prompt_file.write_text(content.strip(), encoding="utf-8")
    
    return prompt_file


# =============================================================================
# RESPONSE PARSING
# =============================================================================

# Binary patterns
BINARY_FINAL_RE = re.compile(
    r"final\s+probability\s*:\s*p?\s*(?:=\s*)?(0(?:\.\d+)?|1(?:\.0+)?)",
    re.IGNORECASE
)
BINARY_FALLBACK_RE = re.compile(r"(?<![.\d])(0\.\d+)(?![.\d])")

# Multiclass patterns
DISTRIBUTION_BLOCK_RE = re.compile(
    r"---DISTRIBUTION_START---(.*?)---DISTRIBUTION_END---",
    re.DOTALL | re.IGNORECASE
)
CATEGORY_PROB_RE = re.compile(
    r"CATEGORY:\s*(.+?)\s*\n\s*PROBABILITY:\s*(0(?:\.\d+)?|1(?:\.0+)?)",
    re.IGNORECASE
)

# Numeric patterns
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
    """Structured parse result."""
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
    """Parse binary probability from model output."""
    if not text or not text.strip():
        return ParseResult(None, "none", ["empty_content"], {})
    
    text = text.strip()
    warnings: List[str] = []
    
    # Primary: Look for "Final probability: p 0.xx"
    match = BINARY_FINAL_RE.search(text)
    if match:
        p = float(match.group(1))
        p = max(PROBABILITY_BOUNDS[0], min(PROBABILITY_BOUNDS[1], p))
        return ParseResult(p, "final_line", warnings, {"raw": match.group(0)})
    
    # Fallback: Last decimal in [0,1] range
    all_decimals = BINARY_FALLBACK_RE.findall(text)
    if all_decimals:
        p = float(all_decimals[-1])
        if 0.0 <= p <= 1.0:
            p = max(PROBABILITY_BOUNDS[0], min(PROBABILITY_BOUNDS[1], p))
            warnings.append("weak_parse_fallback")
            return ParseResult(p, "last_decimal", warnings, {"candidates": all_decimals[-5:]})
    
    return ParseResult(None, "none", ["parse_failed"], {})


def parse_multiclass(text: str, expected_categories: List[str]) -> ParseResult:
    """
    Parse multiclass distribution from model output.
    
    Args:
        text: Model response text
        expected_categories: List of expected category names from CSV
    """
    if not text or not text.strip():
        return ParseResult(None, "none", ["empty_content"], {})
    
    text = text.strip()
    warnings: List[str] = []
    
    # Find distribution block
    block_match = DISTRIBUTION_BLOCK_RE.search(text)
    if not block_match:
        return ParseResult(None, "none", ["no_distribution_block"], {})
    
    block = block_match.group(1)
    
    # Extract category-probability pairs
    pairs = CATEGORY_PROB_RE.findall(block)
    if not pairs:
        return ParseResult(None, "none", ["no_category_probabilities"], {"block": block[:500]})
    
    # Build distribution
    dist: Dict[str, float] = {}
    for cat, prob in pairs:
        cat_clean = cat.strip()
        dist[cat_clean] = float(prob)
    
    # Check if model returned expected categories
    if expected_categories:
        expected_set = set(c.lower().strip() for c in expected_categories)
        returned_set = set(c.lower().strip() for c in dist.keys())
        
        missing = expected_set - returned_set
        extra = returned_set - expected_set
        
        if missing:
            warnings.append(f"missing_categories:{list(missing)[:5]}")
        if extra:
            warnings.append(f"extra_categories:{list(extra)[:5]}")
    
    # Validate sum
    total = sum(dist.values())
    if abs(total - 1.0) > DISTRIBUTION_SUM_TOLERANCE:
        warnings.append(f"sum={total:.4f}_deviates_from_1.0")
    
    # Normalize if close
    if 0.9 <= total <= 1.1 and abs(total - 1.0) > 0.001:
        dist = {k: v / total for k, v in dist.items()}
        warnings.append("renormalized")
    
    return ParseResult(
        dist, 
        "distribution_block", 
        warnings,
        {"raw_sum": round(total, 4), "n_categories": len(dist)}
    )


def normalize_numeric_value(
    value: Dict[str, Any], 
    canonical_units: str
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Normalize numeric estimate to canonical units.
    Handles common unit mismatches (millions vs billions).
    """
    warnings: List[str] = []
    reported_units = (value.get("units") or "").strip().lower()
    canonical_lower = canonical_units.lower()
    
    # No units reported
    if not reported_units:
        value["units"] = canonical_units
        warnings.append("units_missing_assumed_canonical")
        return value, warnings
    
    # Units match (various forms)
    billion_forms = {"usd billions", "billions usd", "billion usd", "usd bn", "bn", "b", "$b", "usd b"}
    million_forms = {"usd millions", "millions usd", "million usd", "usd m", "m", "$m", "usd million"}
    
    if canonical_lower in {"usd billions", "usd billion", "billions", "billion"}:
        if reported_units in billion_forms or "billion" in reported_units:
            value["units"] = canonical_units
            return value, warnings
        
        if reported_units in million_forms or "million" in reported_units:
            # Convert millions to billions
            value["p10"] = value["p10"] / 1000.0
            value["median"] = value["median"] / 1000.0
            value["p90"] = value["p90"] / 1000.0
            value["units"] = canonical_units
            warnings.append("converted_millions_to_billions")
            return value, warnings
    
    # For other unit types, keep as-is but note if unexpected
    if reported_units != canonical_lower and canonical_units:
        warnings.append(f"units_mismatch_reported:{reported_units}_expected:{canonical_units}")
    
    value["units"] = canonical_units if canonical_units else reported_units
    return value, warnings


def parse_numeric(text: str, canonical_units: str) -> ParseResult:
    """Parse numeric estimate (P10, median, P90) from model output."""
    if not text or not text.strip():
        return ParseResult(None, "none", ["empty_content"], {})
    
    text = text.strip()
    warnings: List[str] = []
    
    # Find estimate block
    block_match = ESTIMATE_BLOCK_RE.search(text)
    if not block_match:
        return ParseResult(None, "none", ["no_estimate_block"], {})
    
    # Remove commas for number parsing
    block = block_match.group(1).replace(",", "")
    
    # Extract values
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
    
    # Validate ordering
    if not (p10 <= median <= p90):
        warnings.append("ordering_violation_p10<=median<=p90")
        # Attempt to fix obvious swaps
        sorted_vals = sorted([p10, median, p90])
        p10, median, p90 = sorted_vals[0], sorted_vals[1], sorted_vals[2]
        warnings.append("auto_corrected_ordering")
    
    value = {
        "p10": p10,
        "median": median,
        "p90": p90,
        "units": units,
    }
    
    # Normalize units
    value, unit_warnings = normalize_numeric_value(value, canonical_units)
    warnings.extend(unit_warnings)
    
    return ParseResult(
        value,
        "estimate_block",
        warnings,
        {"raw_units": units}
    )


def parse_response(
    text: str,
    question_type: str,
    canonical_units: str = "",
    expected_categories: Optional[List[str]] = None
) -> ParseResult:
    """Route to appropriate parser based on question type."""
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
) -> Tuple[Optional[str], Optional[str], Optional[Dict], Optional[str], int, float]:
    """
    Call API with exponential backoff retry.
    
    Returns: (content, reasoning, usage, error, attempts, elapsed_seconds)
    """
    limits = get_provider_limits(model_id)
    
    content = None
    reasoning = None
    usage = None
    error = None
    
    start_time = time.time()
    
    for attempt in range(1, limits.retry_max_attempts + 1):
        try:
            log_debug(f"API call attempt {attempt}/{limits.retry_max_attempts}")
            
            resp = client.chat(
                model=model_id,
                messages=messages,
                reasoning=reasoning_cfg,
                **model_params
            )
            
            content = resp.get("content") or ""
            reasoning = resp.get("reasoning")
            usage = resp.get("usage")
            
            # Check for empty content with reasoning tokens (common failure mode)
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
            
            # Success
            elapsed = time.time() - start_time
            return content, reasoning, usage, None, attempt, elapsed
            
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
    return content, reasoning, usage, error, limits.retry_max_attempts, elapsed


# =============================================================================
# JOB EXECUTION
# =============================================================================

def execute_forecast_job(
    *,
    client: OpenRouterClient,
    run_id: str,
    out_dir: Path,
    out_path: Path,
    out_lock: threading.Lock,
    completed_file: Path,
    model_id: str,
    model_label: str,
    reasoning_cfg: Optional[Dict[str, Any]],
    model_params: Dict[str, Any],
    row: Dict[str, str],
    prompt_dir: str,
) -> Optional[Dict[str, Any]]:
    """
    Execute a single forecasting job (one model × one question).
    
    Returns the result record or None if skipped.
    """
    question_id = (row.get("question_id") or "").strip()
    if not question_id:
        log_warn(f"[{model_id}] Missing question_id, skipping")
        return None
    
    jkey = job_key(model_id, question_id)
    
    # Check if already completed (for resume)
    if is_completed(jkey):
        log_debug(f"[{model_id}] Q:{question_id} already completed, skipping")
        return None
    
    # Get question type from pre-computed column
    question_type = get_question_type(row)
    
    # Get categories for multiclass
    categories = get_categories(row) if question_type == "multiclass" else []
    
    # Detect canonical units for numeric
    canonical_units = ""
    if question_type == "numeric":
        canonical_units = detect_canonical_units(row)
    
    log(f"[{model_id}] Q:{question_id} type={question_type}" + 
        (f" cats={len(categories)}" if categories else ""))
    
    # Load prompt template
    try:
        prompt_template, template_path = get_prompt_template(question_type, prompt_dir)
    except FileNotFoundError as e:
        log_error(f"[{model_id}] Q:{question_id} {e}")
        return None
    
    # Build and render prompt
    sys_text, user_template = split_system_user(prompt_template)
    mapping = build_prompt_mapping(row, question_type, canonical_units)
    user_text = render_placeholders(user_template, mapping)
    
    # Validate no unreplaced placeholders
    prompt_warnings = validate_rendered_prompt(
        user_text, f"Q:{question_id} type={question_type}"
    )
    if prompt_warnings:
        log_error(f"[{model_id}] Q:{question_id} {prompt_warnings}")
        # Save bad prompt for debugging
        bad_dir = out_dir / "bad_prompts" / model_id.replace("/", "__")
        ensure_dir(bad_dir)
        (bad_dir / f"{question_id}__{question_type}.txt").write_text(
            f"SYSTEM:\n{sys_text}\n\nUSER:\n{user_text}", encoding="utf-8"
        )
        return None
    
    # Save rendered prompt
    prompt_file = save_rendered_prompt(
        out_dir, model_id, question_id, question_type, sys_text, user_text
    )
    prompt_hash = sha256_text((sys_text + "\n\n" + user_text).strip())
    
    # Build messages
    messages: List[Dict[str, str]] = []
    if sys_text:
        messages.append({"role": "system", "content": sys_text})
    messages.append({"role": "user", "content": user_text})
    
    # Execute API call
    content, reasoning, usage, error, attempts, elapsed = call_api_with_retry(
        client=client,
        model_id=model_id,
        messages=messages,
        reasoning_cfg=reasoning_cfg,
        model_params=model_params,
    )
    
    # Track costs
    cost, input_tokens, output_tokens = track_usage(usage)
    total_cost, total_in, total_out = get_totals()
    
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
    
    # Compute reasoning token count for logging
    reasoning_tokens = 0
    if usage and isinstance(usage, dict):
        details = usage.get("completion_tokens_details", {})
        if isinstance(details, dict):
            reasoning_tokens = details.get("reasoning_tokens", 0)
    
    # Build result record
    record = {
        "run_id": run_id,
        "track": 0,
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
            "mapping_preview": {
                k: v[:100] + "..." if len(v) > 100 else v
                for k, v in mapping.items()
                if k in ["QUESTION_TEXT", "CATEGORY_LIST", "UNITS"]
            },
        },
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
    
    # Mark completed for resume
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
    out_dir: Path,
    completed_file: Path,
    prompt_dir: str,
    run_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Worker function for thread pool.
    Creates its own client and acquires provider semaphore.
    """
    (
        model_label, model_id, reasoning_cfg, model_params,
        out_path, out_lock, row
    ) = job_tuple
    
    # Acquire provider semaphore for rate limiting
    semaphore = get_provider_semaphore(model_id)
    
    with semaphore:
        # Create fresh client per call (thread-safe)
        client = OpenRouterClient()
        
        return execute_forecast_job(
            client=client,
            run_id=run_id,
            out_dir=out_dir,
            out_path=out_path,
            out_lock=out_lock,
            completed_file=completed_file,
            model_id=model_id,
            model_label=model_label,
            reasoning_cfg=reasoning_cfg,
            model_params=model_params,
            row=row,
            prompt_dir=prompt_dir,
        )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def create_manifest(
    run_id: str,
    out_dir: Path,
    config: Dict[str, Any],
    models: List[Dict],
    questions: List[Dict],
    type_counts: Dict[str, int],
) -> Dict[str, Any]:
    """Create experiment manifest for reproducibility."""
    manifest = {
        "run_id": run_id,
        "track": 0,
        "track_name": "closed-book",
        "started_at_utc": utc_now_iso(),
        "config": config,
        "inputs": {
            "n_questions": len(questions),
            "n_models": len(models),
            "question_types": type_counts,
            "models": [
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
        "prompt_files": PROMPT_FILES,
        "version": "1.1.0",
    }
    
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    
    return manifest


def finalize_run(out_dir: Path, start_time: float, jobs_total: int, jobs_ok: int, jobs_failed: int) -> None:
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
    log(f"  Jobs: {jobs_ok}/{jobs_total} OK ({jobs_failed} failed)")
    log(f"  Total cost: ${total_cost:.4f}")
    log(f"  Tokens: {total_in:,} in / {total_out:,} out")
    log("=" * 60)


def validate_csv_columns(questions: List[Dict[str, str]]) -> List[str]:
    """
    Validate that required columns exist in CSV.
    Returns list of warnings/errors.
    """
    warnings = []
    
    if not questions:
        return ["No questions loaded"]
    
    first_row = questions[0]
    required_cols = ["question_id", "question_title"]
    recommended_cols = ["question_type", "categories", "resolution_criteria", "background"]
    
    for col in required_cols:
        if col not in first_row:
            warnings.append(f"MISSING REQUIRED COLUMN: {col}")
    
    for col in recommended_cols:
        if col not in first_row:
            warnings.append(f"Missing recommended column: {col}")
    
    # Check question_type values
    type_counts = {"binary": 0, "multiclass": 0, "numeric": 0, "unknown": 0}
    missing_type = 0
    
    for row in questions:
        qtype = (row.get("question_type") or "").strip().lower()
        if qtype in type_counts:
            type_counts[qtype] += 1
        elif qtype:
            type_counts["unknown"] += 1
        else:
            missing_type += 1
    
    if missing_type > 0:
        warnings.append(f"{missing_type} questions missing 'question_type' - will use fallback detection")
    
    if type_counts["unknown"] > 0:
        warnings.append(f"{type_counts['unknown']} questions have unrecognized question_type values")
    
    # Check multiclass have categories
    multiclass_without_cats = 0
    for row in questions:
        if (row.get("question_type") or "").lower() == "multiclass":
            if not (row.get("categories") or "").strip():
                multiclass_without_cats += 1
    
    if multiclass_without_cats > 0:
        warnings.append(f"{multiclass_without_cats} multiclass questions missing 'categories' column")
    
    return warnings


def main():
    """Main entry point."""
    # Force line buffering for cleaner output
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Track-0 Closed-Book Forecasting Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--questions", default="data/questions.csv", help="Questions CSV path")
    parser.add_argument("--models", default="data/models.json", help="Models JSON path")
    parser.add_argument("--prompts", default="data/prompts", help="Prompts directory")
    parser.add_argument("--output", default="run/track0", help="Output base directory")
    parser.add_argument("--max-questions", type=int, default=None, help="Limit questions (for testing)")
    parser.add_argument("--max-workers", type=int, default=6, help="Thread pool size")
    parser.add_argument("--resume", type=str, default=None, help="Resume from run_id")
    args = parser.parse_args()
    
    # Load environment
    load_dotenv()
    
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
    
    # Load completed jobs (for resume)
    completed_file = out_dir / "completed.txt"
    if args.resume:
        completed = load_completed(completed_file)
        log(f"Loaded {len(completed)} completed jobs from previous run")
    
    # Load inputs
    questions = load_csv_rows(args.questions, max_rows=args.max_questions)
    models = load_json(args.models)
    
    if not isinstance(models, list) or not models:
        log_error("models.json must be a non-empty list")
        sys.exit(1)
    
    # Validate CSV columns
    csv_warnings = validate_csv_columns(questions)
    if csv_warnings:
        log("=" * 60)
        log("CSV VALIDATION WARNINGS:")
        for w in csv_warnings:
            log(f"  ⚠️  {w}")
        log("=" * 60)
        
        # Check for critical errors
        critical = [w for w in csv_warnings if "MISSING REQUIRED" in w]
        if critical:
            log_error("Cannot proceed with missing required columns. Run preprocess_questions.py first.")
            sys.exit(1)
    
    # Count question types (from pre-computed column)
    type_counts: Dict[str, int] = {"binary": 0, "multiclass": 0, "numeric": 0}
    for row in questions:
        qtype = get_question_type(row)
        type_counts[qtype] = type_counts.get(qtype, 0) + 1
    
    # Log configuration
    log("=" * 60)
    log(f"TRACK-0 CLOSED-BOOK FORECASTING")
    log(f"  run_id: {run_id}")
    log(f"  output: {out_dir}")
    log(f"  questions: {len(questions)}")
    log(f"    - binary:     {type_counts['binary']}")
    log(f"    - multiclass: {type_counts['multiclass']}")
    log(f"    - numeric:    {type_counts['numeric']}")
    log(f"  models: {len(models)}")
    for i, m in enumerate(models, 1):
        log(f"    {i}. {m.get('label')} [{m.get('openrouter_id')}]")
    log(f"  max_workers: {args.max_workers}")
    log("=" * 60)
    
    # Create manifest
    config = {
        "questions_csv": args.questions,
        "models_json": args.models,
        "prompts_dir": args.prompts,
        "max_questions": args.max_questions,
        "max_workers": args.max_workers,
        "default_params": DEFAULT_PARAMS,
    }
    
    if not args.resume:
        create_manifest(run_id, out_dir, config, models, questions, type_counts)
    
    # Prepare file locks per output file
    out_locks: Dict[str, threading.Lock] = {}
    
    def get_out_lock(path: Path) -> threading.Lock:
        key = str(path)
        if key not in out_locks:
            out_locks[key] = threading.Lock()
        return out_locks[key]
    
    # Build job queue
    jobs: List[Tuple] = []
    
    for m in models:
        model_label = m.get("label", "unknown")
        model_id = (m.get("openrouter_id") or "").strip()
        
        if not model_id:
            log_warn(f"Skipping model with missing openrouter_id: {model_label}")
            continue
        
        reasoning_cfg = m.get("reasoning")
        
        # Merge default params with model-specific overrides
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
    
    log(f"Total jobs: {len(jobs)} ({len(models)} models × {len(questions)} questions)")
    
    # Execute with thread pool
    start_time = time.time()
    jobs_ok = 0
    jobs_failed = 0
    jobs_skipped = 0
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                worker_thread, job, out_dir, completed_file, args.prompts, run_id
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
            
            # Progress update
            done = jobs_ok + jobs_failed + jobs_skipped
            if done % 5 == 0 or done == len(jobs):
                elapsed = time.time() - start_time
                total_cost, _, _ = get_totals()
                log(f"Progress: {done}/{len(jobs)} ({jobs_ok} ok, {jobs_failed} failed, {jobs_skipped} skipped) "
                    f"elapsed={elapsed:.0f}s cost=${total_cost:.4f}")
    
    # Finalize
    finalize_run(out_dir, start_time, len(jobs) - jobs_skipped, jobs_ok, jobs_failed)


if __name__ == "__main__":
    main()