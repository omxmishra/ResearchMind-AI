from app.core.logger import get_logger
from app.vectorstore.retriever import retrieve
from app.services.rag.context_builder import build_context
from app.services.rag.answer_generator import generate_answer, generate_follow_up_questions
from app.services.rag.source_attribution import attribute_sources
from app.models.schemas import ChatRequest, ChatResponse

logger = get_logger(__name__)


class RAGPipeline:
    def run(self, request: ChatRequest) -> ChatResponse:
        logger.info(f"RAG pipeline started for query: '{request.query[:60]}'")

        results, _ = retrieve(
            query=request.query,
            top_k=request.top_k_context,
        )
        papers = [paper for paper, _ in results]
        logger.info(f"Retrieved {len(papers)} papers")

        context = build_context(papers)

        answer, tokens_used = generate_answer(
            query=request.query,
            context=context,
            conversation_history=request.conversation_history,
        )

        attributed_sources = attribute_sources(answer, papers)

        follow_ups = generate_follow_up_questions(request.query, answer)

        logger.info(f"RAG pipeline complete — {tokens_used} tokens used")

        return ChatResponse(
            answer=answer,
            sources=attributed_sources,
            follow_up_questions=follow_ups,
            tokens_used=tokens_used,
        )