import json
from datetime import datetime

LOG_FILE = "core/logging/confidence_logs.jsonl"

def log_confidence(data: dict):
    data["timestamp"] = str(datetime.utcnow())
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(data) + "\n")
