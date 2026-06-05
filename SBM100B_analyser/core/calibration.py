
import json
from pathlib import Path

def load_calibration(path: Path):
    if not path.exists():
        return {"offset_db": None}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {"offset_db": data.get("offset_db")}
    except Exception:
        return {"offset_db": None}

def save_calibration(path: Path, offset_db):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"offset_db": offset_db}, f, indent=2)

def dbfs_to_spl(dbfs, offset_db):
    if offset_db is None:
        return None
    return dbfs + offset_db
