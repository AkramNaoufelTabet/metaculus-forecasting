import csv
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timezone

def load_questions(csv_path: str, max_questions: Optional[int] = None) -> List[Dict[str, str]]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            rows.append({k: (v or "") for k, v in row.items()})
            if max_questions is not None and (i + 1) >= max_questions:
                break
    return rows

def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def append_jsonl(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
