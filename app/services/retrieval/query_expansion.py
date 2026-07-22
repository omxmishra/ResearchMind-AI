from app.core.logger import get_logger

logger = get_logger(__name__)

EXPANSION_MAP = {
    "llm": ["large language model", "language model", "GPT", "transformer"],
    "rag": ["retrieval augmented generation", "retrieval augmented", "knowledge retrieval"],
    "fine tuning": ["finetuning", "instruction tuning", "PEFT", "LoRA", "QLoRA"],
    "diffusion": ["diffusion model", "stable diffusion", "DDPM", "score matching"],
    "transformer": ["attention mechanism", "self attention", "BERT", "GPT", "encoder decoder"],
    "reinforcement learning": ["RL", "RLHF", "reward model", "policy gradient", "PPO"],
    "image generation": ["text to image", "generative model", "GAN", "diffusion model"],
    "object detection": ["YOLO", "detection model", "bounding box", "anchor box"],
    "nlp": ["natural language processing", "text classification", "language model"],
    "cv": ["computer vision", "image recognition", "visual model"],
    "graph neural network": ["GNN", "graph learning", "node classification", "link prediction"],
    "multimodal": ["vision language model", "VLM", "CLIP", "image text"],
    "agent": ["autonomous agent", "LLM agent", "tool use", "agentic"],
    "embedding": ["vector representation", "sentence embedding", "semantic similarity"],
    "quantization": ["model compression", "int8", "int4", "weight quantization"],
}


def expand_query(query: str) -> str:
    query_lower = query.lower()
    expansions = []

    for term, related in EXPANSION_MAP.items():
        if term in query_lower:
            expansions.extend(related)

    if not expansions:
        return query

    expanded = query + " " + " ".join(expansions[:6])
    logger.info(f"Expanded query: '{query[:40]}' → added {len(expansions)} terms")
    return expanded


def should_expand(query: str) -> bool:
    return len(query.split()) <= 5