import pandas as pd
from datetime import date
from app.core.logger import get_logger
from app.models.documents import Paper
from app.services.ingestion.cleaning import clean_text, parse_authors, parse_categories, parse_date

logger = get_logger(__name__)


class Preprocessor:
    def process(self, df: pd.DataFrame) -> list[Paper]:
        logger.info(f"Processing {len(df)} rows")
        papers = []
        skipped = 0

        for _, row in df.iterrows():
            paper = self._row_to_paper(row.to_dict())
            if paper:
                papers.append(paper)
            else:
                skipped += 1

        logger.info(f"Processed {len(papers)} papers, skipped {skipped}")
        return papers

    def _row_to_paper(self, row: dict) -> Paper | None:
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
            logger.warning(f"Skipping row: {e}")
            return None