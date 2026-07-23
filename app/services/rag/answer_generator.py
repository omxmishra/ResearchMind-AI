from pathlib import Path
import anthropic
from app.core.logger import get_logger
from app.core.config import get_settings

logger = get_logger(__name__)
settings = get_settings()

PROMPT_PATH = Path("app/prompts/research_chat.txt")


def load_prompt_template() -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def generate_answer(
    query: str,
    context: str,
    conversation_history: list[dict] | None = None,
) -> tuple[str, int]:
    logger.info(f"Generating RAG answer for query: '{query[:60]}'")

    template = load_prompt_template()
    system_prompt = template.replace("{context}", context).replace("{query}", "")

    messages = conversation_history or []
    messages = messages + [{"role": "user", "content": query}]

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        system=system_prompt,
        messages=messages,
    )

    answer = response.content[0].text
    tokens_used = response.usage.input_tokens + response.usage.output_tokens

    logger.info(f"Answer generated — {tokens_used} tokens used")
    return answer, tokens_used


def generate_follow_up_questions(query: str, answer: str) -> list[str]:
    logger.info("Generating follow-up questions")

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": (
                f"Given this research question: '{query}'\n"
                f"And this answer: '{answer[:300]}'\n\n"
                f"Generate exactly 3 short follow-up questions a researcher might ask next. "
                f"Return only a numbered list, nothing else."
            )
        }]
    )

    raw = response.content[0].text.strip()
    lines = [l.strip().lstrip("0123456789.)- ") for l in raw.split("\n") if l.strip()]
    return lines[:3]