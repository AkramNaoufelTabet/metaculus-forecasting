#!/usr/bin/env python3
"""Diagnose CSV data issues."""

import csv
from pathlib import Path
from collections import defaultdict

csv_path = "data/questions.csv"

# Read all rows
rows = []
with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    headers = reader.fieldnames
    for row in reader:
        rows.append(row)

print(f"Total rows: {len(rows)}")
print(f"Headers: {headers}\n")

# Check for empty fields
key_fields = ["question_id", "question_title", "background", "resolution_criteria"]

print("=" * 60)
print("EMPTY FIELD ANALYSIS")
print("=" * 60)

for field in key_fields:
    empty_count = sum(1 for r in rows if not (r.get(field) or "").strip())
    print(f"  {field}: {empty_count}/{len(rows)} empty")

print("\n" + "=" * 60)
print("ROW-BY-ROW CHECK (first 10 rows)")
print("=" * 60)

for i, row in enumerate(rows[:10], 1):
    qid = row.get("question_id", "?")
    title = (row.get("question_title") or "")[:50]
    bg = (row.get("background") or "")[:30]
    rc = (row.get("resolution_criteria") or "")[:30]
    
    issues = []
    if not title.strip():
        issues.append("NO_TITLE")
    if not bg.strip():
        issues.append("NO_BG")
    if not rc.strip():
        issues.append("NO_RC")
    
    status = " | ".join(issues) if issues else "OK"
    print(f"\n  Row {i} (Q:{qid}): {status}")
    print(f"    title: '{title}...'")
    print(f"    background: '{bg}...'")
    print(f"    resolution: '{rc}...'")

# Check for question_type column
print("\n" + "=" * 60)
print("QUESTION TYPE COLUMN")
print("=" * 60)

if "question_type" in headers:
    types = defaultdict(int)
    for r in rows:
        t = (r.get("question_type") or "auto-detect").strip().lower()
        types[t] += 1
    print(f"  Found question_type column: {dict(types)}")
else:
    print("  NO question_type column - will auto-detect")

# Sample a binary-looking question
print("\n" + "=" * 60)
print("SAMPLE BINARY-LOOKING QUESTION")
print("=" * 60)

for row in rows:
    title = (row.get("question_title") or "").lower()
    if title.startswith("will "):
        print(f"  Q:{row.get('question_id')}")
        print(f"  Title: {row.get('question_title', '')[:100]}")
        print(f"  Background length: {len(row.get('background', ''))}")
        print(f"  Resolution length: {len(row.get('resolution_criteria', ''))}")
        print(f"  Fine print length: {len(row.get('fine_print', ''))}")
        break