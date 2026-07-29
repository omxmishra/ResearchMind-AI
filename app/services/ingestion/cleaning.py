import re
import pandas as pd
from datetime import date
from app.core.logger import get_logger

logger = get_logger(__name__)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\$[^$]*\$', '', text)
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def parse_authors(raw: str) -> list[str]:
    if not isinstance(raw, str) or not raw:
        return []
    return [a.strip() for a in raw.split(";") if a.strip()]


def parse_categories(raw: str) -> list[str]:
    if not isinstance(raw, str) or not raw:
        return []
    return [c.strip() for c in raw.split(";") if c.strip()]


def parse_date(raw) -> date | None:
    try:
        return pd.to_datetime(raw).date()
    except Exception:
        return None


def truncate_text(text: str, max_chars: int = 500) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(' ', 1)[0] + "..."