# src/utils.py
from datetime import datetime

def utcnow_iso():
    return datetime.utcnow().isoformat() + "Z"

def truncate(text: str, n: int = 400):
    if not text:
        return ""
    if len(text) <= n:
        return text
    return text[:n].rstrip() + "..."
