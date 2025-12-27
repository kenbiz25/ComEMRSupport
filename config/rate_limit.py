
import time

WINDOW = 10  # seconds
MAX_MSG = 6  # per window

_buckets = {}

def allow(number: str) -> bool:
    now = time.time()
    bucket = _buckets.get(number, [])
    bucket = [t for t in bucket if now - t <= WINDOW]
    if len(bucket) >= MAX_MSG:
        _buckets[number] = bucket
        return False
    bucket.append(now)
    _buckets[number] = bucket
    return True
