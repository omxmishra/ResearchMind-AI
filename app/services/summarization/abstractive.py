from pathlib import Path
import anthropic
from app.core.logger import get_logger
from app.core.config import get_settings
from app.models.documents import Paper
from app.models.schemas import SummarizeRequest, SummarizeResponse

logger = get_logger(__name__)
settings = get_settings()

PROMPT_PATH = Path("app/prompts/summarization.txt")


def load_prompt() -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


class AbstractiveSummarizer:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.model = "claude-3-haiku-20240307"

    def summarize(self, paper: Paper, style: str = "tldr") -> SummarizeResponse:
        logger.info(f"Summarizing paper '{paper.title[:50]}' in style '{style}'")

        template = load_prompt()
        prompt = (
            template
            .replace("{title}", paper.title)
            .replace("{abstract}", paper.abstract)
            .replace("{style}", style)
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )

        summary = response.content[0].text.strip()
        logger.info(f"Summary generated — {len(summary)} characters")

        return SummarizeResponse(
            paper_id=paper.paper_id,
            title=paper.title,
            summary=summary,
            style=style,
        )