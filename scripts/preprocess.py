import re
import pickle
import pandas as pd
from datetime import date
from app.core.logger import get_logger
from app.core.constants import PROCESSED_DATA_DIR, PAPERS_PKL_PATH
from app.models.documents import Paper

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


def row_to_paper(row: dict) -> Paper | None:
    try:
        title = clean_text(str(row.get("title", "")))
        abstract = clean_text(str(row.get("abstract", "")))

        if not title or not abstract or len(abstract) < 50:
            return None

        return Paper(
            paper_id=str(row.get("paper_id", "")),
            title=title,
            abstract=abstract,
            authors=parse_authors(str(row.get("authors", ""))),
            categories=parse_categories(str(row.get("all_categories", ""))),
            primary_category=str(row.get("primary_category", "")),
            published_date=parse_date(row.get("submitted_date")),
            updated_date=parse_date(row.get("updated_date")),
            arxiv_url=str(row.get("arxiv_url", "")),
            pdf_url=str(row.get("pdf_url", "")),
            word_count=int(row.get("word_count", 0)),
            abstract_length=int(row.get("abstract_length", 0)),
        )
    except Exception as e:
        logger.warning(f"Skipping row due to error: {e}")
        return None


def preprocess(df: pd.DataFrame) -> list[Paper]:
    logger.info(f"Preprocessing {len(df)} rows")
    papers = []
    skipped = 0

    for _, row in df.iterrows():
        paper = row_to_paper(row.to_dict())
        if paper:
            papers.append(paper)
        else:
            skipped += 1

    logger.info(f"Preprocessed {len(papers)} valid papers, skipped {skipped}")
    return papers


def save_papers(papers: list[Paper]):
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PAPERS_PKL_PATH, "wb") as f:
        pickle.dump(papers, f)
    logger.info(f"Saved {len(papers)} papers to {PAPERS_PKL_PATH}")


if __name__ == "__main__":
    snapshot_path = PROCESSED_DATA_DIR / "raw_snapshot.pkl"
    with open(snapshot_path, "rb") as f:
        df = pickle.load(f)

    papers = preprocess(df)

    print(f"\nSample paper:")
    print(f"  ID: {papers[0].paper_id}")
    print(f"  Title: {papers[0].title[:60]}")
    print(f"  Authors: {papers[0].authors[:3]}")
    print(f"  Categories: {papers[0].categories}")
    print(f"  Date: {papers[0].published_date}")

    save_papers(papers)
    logger.info("Preprocessing complete")