#!/usr/bin/env python3
"""
Preprocess questions CSV: add question_type and categories columns.
Uses manual classifications provided by user.

Usage:
    python preprocess_questions.py
"""

import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_CSV = "data/questions.csv"
OUTPUT_CSV = "data/questions.csv"  # Overwrites (backup created first)

# =============================================================================
# MANUAL QUESTION TYPE CLASSIFICATIONS
# =============================================================================

QUESTION_TYPES: Dict[str, str] = {
    # Multiclass (6)
    "41331": "multiclass",  # Grammy game soundtrack
    "41323": "multiclass",  # Winter Olympics medal table
    "41465": "multiclass",  # Currency vs USD
    "41342": "multiclass",  # Russian oil refineries
    "41303": "multiclass",  # Colombian election
    "41681": "multiclass",  # US CPI inflation bins
    
    # Binary (13)
    "40927": "binary",   # Portugal presidential election
    "40726": "binary",   # Nvidia stock record
    "41358": "binary",   # German bond yield 20bps
    "41361": "binary",   # Motor gasoline final week
    "41501": "binary",   # Japan visitors from China
    "41213": "binary",   # China US Treasuries below $675B
    "41500": "binary",   # ChatGPT Atlas Windows release
    "41355": "binary",   # Rice price Japan below ¥4200
    "41302": "binary",   # Layoffs.fyi 100 AI layoffs
    "41336": "binary",   # OpenAI API token prices fall
    "41356": "binary",   # US electric utility $5B capex
    "41316": "binary",   # US sanctions on Russia
    "41343": "binary",   # Nvidia GPUs export to China
    
    # Numeric (17)
    "41333": "numeric",  # Combined capex MSFT/GOOG/AMZN
    "41235": "numeric",  # Super Bowl LX viewership %
    "41236": "numeric",  # India WPI inflation
    "41301": "numeric",  # Gold price highest
    "41560": "numeric",  # UK retail sales MoM
    "41583": "numeric",  # USD to Argentine Peso
    "41528": "numeric",  # China New Year Gala viewership
    "41341": "numeric",  # AI companies S&P 500 weight %
    "41683": "numeric",  # Redbook retail sales YoY growth
    "41596": "numeric",  # US-China trade balance
    "41603": "numeric",  # US goods export
    "41586": "numeric",  # Shanghai-LA vs Shanghai-Rotterdam freight
    "41328": "numeric",  # S&P Global Manufacturing PMI
    "41442": "numeric",  # Italian teams Champions League
    "41511": "numeric",  # Google vs ChatGPT search ratio
    "41362": "numeric",  # Wuthering Heights box office
    "41585": "numeric",  # EFAMA European fund inflows
}

# =============================================================================
# MANUAL CATEGORIES FOR MULTICLASS QUESTIONS
# =============================================================================

CATEGORIES: Dict[str, List[str]] = {
    "41331": [
        "Avatar: Frontiers of Pandora - Secrets of the Spires",
        "Helldivers 2",
        "Indiana Jones And The Great Circle",
        "Star Wars Outlaws: Wild Card & A Pirate's Fortune",
        "Sword of the Sea",
    ],
    "41323": [
        "Norway",
        "Germany",
        "United States",
        "Italy",
        "Canada",
        "Other",
    ],
    "41465": [
        "EUR",
        "JPY",
        "GBP",
        "CNY",
        "CHF",
        "AUD",
        "CAD",
        "MXN",
    ],
    "41342": [
        "Ryazan",
        "Kstovo (Lukoil)",
        "Omsk",
        "None",
        "Multiple",
    ],
    "41303": [
        "Historic Pact",
        "Liberal",
        "Conservative",
        "Democratic Center",
        "Party of the U",
        "Radical Change",
        "Green Alliance",
        "Other",
    ],
    "41681": [
        "≤0.0%",
        "0.1%",
        "0.2%",
        "0.3%",
        "0.4%",
        "≥0.5%",
    ],
}


# =============================================================================
# PROCESSING FUNCTIONS
# =============================================================================

def get_question_type(qid: str) -> str:
    """Get question type from manual classification."""
    return QUESTION_TYPES.get(qid, "binary")  # Default to binary if unknown


def get_categories(qid: str) -> str:
    """Get pipe-separated categories for multiclass questions."""
    cats = CATEGORIES.get(qid, [])
    return "|".join(cats) if cats else ""


def process_csv():
    """Main processing function."""
    
    input_path = Path(INPUT_CSV)
    output_path = Path(OUTPUT_CSV)
    
    # Verify input exists
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return
    
    # Read input CSV
    rows = []
    with open(input_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        original_headers = list(reader.fieldnames or [])
        for row in reader:
            rows.append(row)
    
    print(f"Loaded {len(rows)} questions from {input_path}")
    print(f"Original columns: {original_headers}\n")
    
    # Create backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = input_path.with_suffix(f".backup_{timestamp}.csv")
    with open(backup_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=original_headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Backup saved: {backup_path}\n")
    
    # Process each row
    stats = {"binary": 0, "multiclass": 0, "numeric": 0}
    unclassified = []
    
    print("=" * 80)
    print("PROCESSING QUESTIONS")
    print("=" * 80 + "\n")
    
    for i, row in enumerate(rows, 1):
        qid = row.get("question_id", "").strip()
        title = row.get("question_title", "")[:55]
        
        # Get type
        qtype = get_question_type(qid)
        row["question_type"] = qtype
        stats[qtype] += 1
        
        # Get categories (only for multiclass)
        if qtype == "multiclass":
            cats = get_categories(qid)
            row["categories"] = cats
            cat_count = len(cats.split("|")) if cats else 0
            cat_info = f"[{cat_count} categories]"
        else:
            row["categories"] = ""
            cat_info = ""
        
        # Track if not in our manual list
        if qid not in QUESTION_TYPES:
            unclassified.append(qid)
            status = "⚠️ "
        else:
            status = "✅"
        
        print(f"{i:2}. {status} Q:{qid:5} | {qtype:10} | {title}... {cat_info}")
    
    # Ensure new columns in headers
    output_headers = original_headers.copy()
    for col in ["question_type", "categories"]:
        if col not in output_headers:
            output_headers.append(col)
    
    # Write output
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n  Question Types:")
    print(f"    Binary:     {stats['binary']:3}")
    print(f"    Multiclass: {stats['multiclass']:3}")
    print(f"    Numeric:    {stats['numeric']:3}")
    print(f"    ─────────────────")
    print(f"    Total:      {sum(stats.values()):3}")
    
    if unclassified:
        print(f"\n  ⚠️  Unclassified questions (defaulted to binary):")
        for qid in unclassified:
            print(f"      - Q:{qid}")
        print(f"\n  Add these to QUESTION_TYPES dict if needed.")
    
    print(f"\n  Output saved: {output_path}")
    
    # Show multiclass details
    print("\n" + "=" * 80)
    print("MULTICLASS QUESTION DETAILS")
    print("=" * 80)
    
    for qid, cats in CATEGORIES.items():
        row = next((r for r in rows if r.get("question_id") == qid), None)
        if row:
            title = row.get("question_title", "")[:60]
            print(f"\n  Q:{qid} — {title}...")
            for j, cat in enumerate(cats, 1):
                print(f"    {j:2}. {cat}")
    
    print("\n" + "=" * 80)
    print("DONE! You can now run: python track0.py")
    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    process_csv()