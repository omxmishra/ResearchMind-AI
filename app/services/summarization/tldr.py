import anthropic
from app.core.logger import get_logger
from app.core.config import get_settings
from app.models.documents import Paper

logger = get_logger(__name__)
settings = get_settings()


class TLDRSummarizer:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.model = "claude-3-haiku-20240307"

    def summarize_one(self, paper: Paper) -> str:
        logger.info(f"TLDR for: '{paper.title[:50]}'")

        response = self.client.messages.create(
            model=self.model,
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": (
                    f"Summarize this AI research paper in exactly 1-2 sentences. "
                    f"Be direct, no preamble.\n\n"
                    f"Title: {paper.title}\n"
                    f"Abstract: {paper.abstract[:400]}"
                )
            }]
        )

        return response.content[0].text.strip()

    def summarize_batch(self, papers: list[Paper]) -> dict[str, str]:
        logger.info(f"Batch TLDR for {len(papers)} papers")
        results = {}
        for paper in papers:
            try:
                results[paper.paper_id] = self.summarize_one(paper)
            except Exception as e:
                logger.warning(f"TLDR failed for {paper.paper_id}: {e}")
                results[paper.paper_id] = paper.abstract[:150] + "..."
        return results