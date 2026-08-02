import streamlit as st
import requests

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="ResearchMind AI",
    page_icon="🔬",
    layout="wide",
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stTextInput > div > div > input { background-color: #1e2130; color: white; }
    .paper-card {
        background-color: #1e2130;
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 3px solid #4f8bf9;
    }
    .score-badge {
        background-color: #4f8bf9;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
    }
    .category-badge {
        background-color: #2d3748;
        color: #a0aec0;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        margin-right: 4px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔬 ResearchMind AI")
st.caption("Semantic search and RAG-powered Q&A over 7,700+ ArXiv AI/ML research papers (2025-2026)")

with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Number of results", 1, 20, 5)
    rerank = st.toggle("Enable reranking", value=True)

    st.markdown("---")
    st.subheader(" Category Filter")
    try:
        cats = requests.get(f"{API_BASE}/categories", timeout=3).json().get("categories", [])
        selected_cats = st.multiselect("Filter by category", cats)
    except Exception:
        selected_cats = []
        st.error("API offline — run: uvicorn app.main:app --reload --port 8000")

    st.markdown("---")
    st.subheader("📅 Date Filter")
    date_from = st.date_input("From", value=None)
    date_to = st.date_input("To", value=None)

    st.markdown("---")
    try:
        health = requests.get(f"{API_BASE}/health", timeout=3).json()
        st.success(f"✅ {health['status'].upper()}")
        st.metric("Papers Indexed", f"{health['total_papers']:,}")
        st.metric("Model", health['embedding_model'])
    except Exception:
        st.error("API offline")

tab_search, tab_recommend, tab_chat = st.tabs(["🔍 Search", "📚 Recommend", "💬 Chat"])
def render_paper_card(paper: dict, score: float = None, rank: int = None):
    with st.container():
        rank_str = f"#{rank} " if rank else ""
        authors = paper.get("authors", [])
        authors_str = ", ".join(authors[:3])
        if len(authors) > 3:
            authors_str += " et al."
        date_str = f"· {paper.get('published_date', '')}" if paper.get("published_date") else ""
        category = paper.get("primary_category") or ""
        score_str = f'<span class="score-badge">Score: {score}</span>' if score else ""
        abstract = paper.get("abstract", "")[:350]

        st.markdown(f"""
        <div class="paper-card">
            <strong style="font-size:1rem; color:#e2e8f0;">{rank_str}{paper['title']}</strong><br/>
            <span style="color:#718096; font-size:0.85rem;">{authors_str} {date_str}</span><br/>
            <span class="category-badge">{category}</span>{score_str}
            <p style="color:#a0aec0; font-size:0.88rem; margin-top:0.5rem;">{abstract}...</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            if paper.get("arxiv_url"):
                st.link_button("📄 Paper", paper["arxiv_url"])
        with col2:
            if paper.get("pdf_url"):
                st.link_button("📥 PDF", paper["pdf_url"])


with tab_search:
    query = st.text_input(
        "Search query",
        placeholder="e.g. attention mechanism in vision transformers",
        label_visibility="collapsed",
    )

    if st.button("Search", type="primary", use_container_width=True) and query:
        with st.spinner("Searching 7,701 papers..."):
            try:
                resp = requests.post(f"{API_BASE}/search", json={
                    "query": query,
                    "top_k": top_k,
                    "rerank": rerank,
                    "category_filter": selected_cats or None,
                    "date_from": str(date_from) if date_from else None,
                    "date_to": str(date_to) if date_to else None,
                }, timeout=15).json()

                col1, col2, col3 = st.columns(3)
                col1.metric("Results", resp["total_found"])
                col2.metric("Search Time", f"{resp['search_time_ms']:.0f}ms")
                col3.metric("Model", resp["model_used"])

                st.markdown("---")
                for r in resp["results"]:
                    render_paper_card(r["paper"], score=r["score"], rank=r["rank"])

            except Exception as e:
                st.error(f"Search failed: {e}")


with tab_recommend:
    paper_id = st.text_input(
        "Paper ID",
        placeholder="e.g. 2604.13368v1",
        label_visibility="collapsed",
    )

    if st.button("Find Similar Papers", type="primary", use_container_width=True) and paper_id:
        with st.spinner("Finding similar papers..."):
            try:
                resp = requests.post(f"{API_BASE}/recommend", json={
                    "paper_id": paper_id,
                    "top_k": top_k,
                }, timeout=15).json()

                st.subheader("Source Paper")
                render_paper_card(resp["source_paper"])

                st.markdown("---")
                st.subheader(f"Similar Papers ({resp['total_found']} found)")
                for r in resp["recommendations"]:
                    render_paper_card(r["paper"], score=r["score"], rank=r["rank"])

            except Exception as e:
                st.error(f"Recommendation failed: {e}")


with tab_chat:
    st.caption("Ask anything about AI/ML research. Answers are grounded in real papers.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Sources"):
                    for s in msg["sources"]:
                        st.markdown(f"- [{s['title']}]({s.get('arxiv_url', '#')}) — {s.get('primary_category', '')}")
            if msg.get("follow_ups"):
                st.markdown("**Suggested follow-ups:**")
                for q in msg["follow_ups"]:
                    if st.button(q, key=f"fu_{q[:20]}"):
                        st.session_state.messages.append({"role": "user", "content": q})
                        st.rerun()

    if prompt := st.chat_input("Ask about AI/ML research..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving papers and generating answer..."):
                try:
                    resp = requests.post(f"{API_BASE}/chat", json={
                        "query": prompt,
                        "conversation_history": [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages[:-1]
                        ],
                        "top_k_context": 5,
                    }, timeout=30).json()

                    answer = resp.get("answer", "No answer generated.")
                    sources = resp.get("sources", [])
                    follow_ups = resp.get("follow_up_questions", [])

                except Exception as e:
                    answer = f"Error: {e}"
                    sources, follow_ups = [], []

            st.write(answer)

            if sources:
                with st.expander("📚 Sources Used"):
                    for s in sources:
                        st.markdown(f"- [{s['title']}]({s.get('arxiv_url', '#')})")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "follow_ups": follow_ups,
        })