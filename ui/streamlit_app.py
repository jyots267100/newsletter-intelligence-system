"""Streamlit UI for the Newsletter Intelligence System.

Flow: upload PDF -> processing status -> document info -> executive
summary / key insights / actionable items / important information ->
chat with the newsletter, with citations shown under every answer.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Streamlit executes this file as a standalone script and only puts its own
# directory (ui/) on sys.path, not the project root - without this, `from
# ui import api_client` would fail with ModuleNotFoundError regardless of
# the working directory `streamlit run` is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from ui import api_client
from ui.api_client import APIError

st.set_page_config(page_title="Newsletter Intelligence System", page_icon="📰", layout="wide")


def _init_session_state() -> None:
    st.session_state.setdefault("active_document_id", None)
    st.session_state.setdefault("analysis", None)
    st.session_state.setdefault("chat_history", [])  # list[{"role": .., "content": ..}]


def _citation_line(citations: list[dict]) -> str:
    if not citations:
        return ""
    seen = set()
    parts = []
    for c in citations:
        key = (c["filename"], c["page_number"])
        if key in seen:
            continue
        seen.add(key)
        parts.append(f"{c['filename']} — Page {c['page_number']}")
    return "Source: " + "; ".join(parts)


def _render_sidebar() -> None:
    st.sidebar.title("📰 Documents")

    if not api_client.check_backend_health():
        st.sidebar.error("Backend API is unreachable. Is it running?")
        st.sidebar.code("uvicorn app.main:app --reload", language="bash")
        return

    uploaded_file = st.sidebar.file_uploader("Upload a newsletter PDF", type=["pdf"])
    if uploaded_file is not None and st.sidebar.button("Upload & Process", type="primary"):
        with st.spinner("Processing PDF (extracting, chunking, indexing)..."):
            try:
                result = api_client.upload_document(uploaded_file.name, uploaded_file.getvalue())
                st.session_state.active_document_id = result["document_id"]
                st.session_state.analysis = None
                st.session_state.chat_history = []
                st.sidebar.success(f"Indexed {result['page_count']} pages, {result['chunk_count']} chunks.")
            except APIError as exc:
                if exc.status_code == 409:
                    st.sidebar.warning(f"Duplicate document: {exc.detail}")
                else:
                    st.sidebar.error(f"Upload failed: {exc.detail}")

    st.sidebar.divider()
    st.sidebar.subheader("Your documents")
    try:
        documents = api_client.list_documents()["documents"]
    except APIError as exc:
        st.sidebar.error(f"Could not load documents: {exc.detail}")
        return

    if not documents:
        st.sidebar.info("No documents uploaded yet.")
        return

    for doc in documents:
        is_active = doc["document_id"] == st.session_state.active_document_id
        label = f"{'🟢 ' if is_active else ''}{doc['filename']} ({doc['status']})"
        col1, col2 = st.sidebar.columns([4, 1])
        if col1.button(label, key=f"select_{doc['document_id']}", use_container_width=True):
            st.session_state.active_document_id = doc["document_id"]
            st.session_state.analysis = None
            st.session_state.chat_history = []
            st.rerun()
        if col2.button("🗑️", key=f"delete_{doc['document_id']}"):
            api_client.delete_document(doc["document_id"])
            if is_active:
                st.session_state.active_document_id = None
                st.session_state.analysis = None
                st.session_state.chat_history = []
            st.rerun()


def _render_analysis(document: dict) -> None:
    st.subheader(f"📄 {document['filename']}")
    meta_cols = st.columns(4)
    meta_cols[0].metric("Pages", document["page_count"])
    meta_cols[1].metric("Chunks", document["chunk_count"])
    meta_cols[2].metric("Status", document["status"])
    meta_cols[3].metric("Size (KB)", round(document["file_size_bytes"] / 1024, 1))

    if document["status"] != "ready":
        st.warning("This document is not ready for analysis yet.")
        if document.get("error_message"):
            st.error(document["error_message"])
        return

    col_a, col_b = st.columns([1, 3])
    if col_a.button("Generate Analysis", type="primary"):
        with st.spinner("Analyzing document with Gemini..."):
            try:
                st.session_state.analysis = api_client.analyze_document(document["document_id"])
            except APIError as exc:
                st.error(f"Analysis failed: {exc.detail}")
                if col_b.button("Retry analysis"):
                    st.rerun()
                return

    analysis = st.session_state.analysis
    if not analysis:
        st.info("Click **Generate Analysis** to produce an executive summary, key insights, and action items.")
        return

    st.markdown("### Executive Summary")
    st.write(analysis["executive_summary"]["summary"])
    st.caption(_citation_line(analysis["executive_summary"]["citations"]))

    st.markdown("### Key Insights")
    if not analysis["key_insights"]:
        st.caption("No key insights were extracted.")
    for insight in analysis["key_insights"]:
        badge = "📌 Explicit" if insight["source_type"] == "explicit" else "🤖 AI-inferred"
        st.markdown(f"- **[{insight['category'].title()}] {badge}:** {insight['text']}")
        caption = _citation_line(insight["citations"])
        if caption:
            st.caption(caption)

    st.markdown("### Actionable Items")
    if not analysis["action_items"]:
        st.caption("No explicit action items were found in this document.")
    for item in analysis["action_items"]:
        line = f"- **Action:** {item['action']}"
        if item.get("owner"):
            line += f" | **Owner:** {item['owner']}"
        if item.get("deadline"):
            line += f" | **Deadline:** {item['deadline']}"
        st.markdown(line)
        caption = _citation_line(item["citations"])
        if caption:
            st.caption(caption)

    st.markdown("### Important Information")
    entities = analysis["entities"]
    info_cols = st.columns(5)
    for col, (label, key) in zip(
        info_cols,
        [
            ("Dates", "dates"), ("Organizations", "organizations"),
            ("Products", "products"), ("People", "people"),
            ("Numbers/Stats", "numerical_facts"),
        ],
    ):
        with col:
            st.markdown(f"**{label}**")
            values = entities.get(key) or []
            if not values:
                st.caption("None found")
            for v in values:
                st.caption(f"• {v}")


def _render_chat(document: dict) -> None:
    st.markdown("### 💬 Chat with this newsletter")
    st.caption(f"Active document: **{document['filename']}**")

    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.rerun()

    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])
            if turn.get("citations"):
                st.caption(_citation_line(turn["citations"]))

    question = st.chat_input("Ask a question about this newsletter...")
    if not question:
        return

    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    history_payload = [
        {"role": t["role"], "content": t["content"]} for t in st.session_state.chat_history[:-1]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = api_client.send_chat_message(
                    document["document_id"], question, history_payload
                )
                st.write(response["answer"])
                caption = _citation_line(response["citations"])
                if caption:
                    st.caption(caption)
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": response["answer"],
                        "citations": response["citations"],
                    }
                )
            except APIError as exc:
                st.error(f"Chat failed: {exc.detail}")


def main() -> None:
    _init_session_state()
    st.title("📰 Newsletter Intelligence System")
    st.caption("Upload a newsletter PDF to get a grounded summary, insights, and a chatbot over its content.")

    _render_sidebar()

    if not st.session_state.active_document_id:
        st.info("👈 Upload a PDF or select a document from the sidebar to get started.")
        return

    try:
        document = api_client.get_document(st.session_state.active_document_id)
    except APIError as exc:
        st.error(f"Could not load the active document: {exc.detail}")
        return

    tab_analysis, tab_chat = st.tabs(["📊 Analysis", "💬 Chat"])
    with tab_analysis:
        _render_analysis(document)
    with tab_chat:
        if document["status"] == "ready":
            _render_chat(document)
        else:
            st.info("Chat is available once the document finishes processing.")


if __name__ == "__main__":
    main()
