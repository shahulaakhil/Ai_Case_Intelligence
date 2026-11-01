import os
import streamlit as st
import pandas as pd
from typing import Dict, Any, List

# Streamlit page setup
st.set_page_config(
    page_title="AI Case Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for legal dashboard styling
st.markdown("""
<style>
    .main-header {
        font-family: 'Georgia', serif;
        font-size: 2.3rem;
        font-weight: 700;
        color: #1A365D;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 1.1rem;
        color: #4A5568;
        margin-bottom: 1.5rem;
    }
    .evidence-card {
        background-color: #F7FAFC;
        border-left: 4px solid #2B6CB0;
        padding: 12px 16px;
        margin-bottom: 12px;
        border-radius: 4px;
        font-size: 0.95rem;
    }
    .evidence-meta {
        font-size: 0.82rem;
        font-weight: 600;
        color: #2B6CB0;
        margin-bottom: 6px;
    }
    .status-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-success { background-color: #C6F6D5; color: #22543D; }
    .badge-warning { background-color: #FEFCBF; color: #744210; }
    .badge-info { background-color: #EBF8FF; color: #2B6CB0; }
    .disclaimer-box {
        font-size: 0.82rem;
        color: #718096;
        background-color: #EDF2F7;
        padding: 10px 14px;
        border-radius: 6px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Imports from src package
from src.config import (
    is_gemini_configured,
    get_masked_api_key,
    GEMINI_MODEL,
    EMBEDDING_MODEL_NAME,
    RERANKER_MODEL_NAME,
    EVALUATION_DATASET_PATH
)
from src.pipeline import LegalRAGPipeline
from src.evaluator import LegalDatasetEvaluator

# Initialize session state for RAG pipeline
@st.cache_resource(show_spinner=False)
def get_pipeline() -> LegalRAGPipeline:
    return LegalRAGPipeline()

@st.cache_resource(show_spinner=False)
def get_evaluator() -> LegalDatasetEvaluator:
    return LegalDatasetEvaluator(EVALUATION_DATASET_PATH)

pipeline = get_pipeline()
evaluator = get_evaluator()

if "processed_files" not in st.session_state:
    st.session_state.processed_files = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = {}

# --- SIDEBAR: Configuration & Architecture Details ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/scale.png", width=64)
    st.title("System Status")
    
    st.markdown("### Architecture Pipeline")
    st.markdown(f"""
    - **Embeddings:** `{EMBEDDING_MODEL_NAME.split('/')[-1]}` (Local)
    - **Vector Store:** `FAISS IndexFlatIP` (Cosine)
    - **Reranker:** `{RERANKER_MODEL_NAME.split('/')[-1]}` (Local)
    - **LLM Generator:** `{pipeline.generator.model_name}`
    """)

    st.markdown("---")
    st.markdown("### Gemini API Status")
    configured = is_gemini_configured()
    if configured:
        st.success(f"Configured: {get_masked_api_key()}")
    else:
        st.warning("API Key not set in `.env`")
        custom_key = st.text_input("Enter Gemini API Key (Session only):", type="password")
        if custom_key:
            pipeline.generator.api_key = custom_key
            st.success("API Key updated for this session!")

    st.markdown("---")
    st.markdown("### Production Index Stats")
    stats = pipeline.get_status()
    col_s1, col_s2 = st.columns(2)
    col_s1.metric("Judgments", stats["total_documents"])
    col_s2.metric("Chunks", stats["total_chunks"])

    st.markdown("---")
    st.markdown("""
    <div class="disclaimer-box">
    <b>Evidence Grounding Policy:</b><br>
    Selected evidence passages are sent to Google's Gemini API for synthesis. All claims must correlate to retrieved document page references. No training or fine-tuning is performed.
    </div>
    """, unsafe_allow_html=True)


# --- MAIN HEADER ---
st.markdown('<div class="main-header">AI Case Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Evidence-Grounded Analysis of Indian Legal Judgments</div>', unsafe_allow_html=True)

# Main Navigation Tabs
tab_upload, tab_qa, tab_analysis, tab_manage, tab_eval = st.tabs([
    "📁 Document Ingestion",
    "💬 Ask Questions",
    "📑 Case Analysis",
    "⚙️ Document Management",
    "📊 Evaluation Dataset"
])

# =========================================================================
# TAB 1: UPLOAD DOCUMENTS (Section 3.A)
# =========================================================================
with tab_upload:
    st.subheader("Upload Legal Judgment Documents")
    st.write(
        "Upload Indian Supreme Court or High Court judgment PDFs. "
        "The system extracts text page-by-page, produces legal-aware chunks (~500-800 tokens), "
        "and builds a local FAISS index with local BGE embeddings."
    )

    uploaded_files = st.file_uploader(
        "Select Judgment PDF(s):",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload standard digital PDFs. Scanned PDFs without text layers will be detected and flagged."
    )

    if uploaded_files:
        st.write(f"**Selected Files ({len(uploaded_files)}):**")
        for f in uploaded_files:
            file_status = "Processed" if f.name in pipeline.vector_store.document_registry else "Ready to process"
            badge_class = "badge-success" if file_status == "Processed" else "badge-info"
            st.markdown(
                f"- **{f.name}** ({round(f.size / 1024, 1)} KB) — "
                f"<span class='status-badge {badge_class}'>{file_status}</span>",
                unsafe_allow_html=True
            )

        if st.button("⚡ Process & Index Documents", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, file_obj in enumerate(uploaded_files):
                filename = file_obj.name
                status_text.text(f"Processing '{filename}'...")
                file_bytes = file_obj.read()

                # Execute pipeline step 1-5
                res = pipeline.process_and_index_pdf(file_bytes, filename)

                if res["success"]:
                    st.success(
                        f"✅ **{filename}**: Extracted **{res['total_pages']} pages**, "
                        f"generated and indexed **{res['chunk_count']} legal chunks**."
                    )
                    st.session_state.processed_files[filename] = res
                else:
                    st.error(f"❌ **{filename}**: {res['error']}")

                progress_bar.progress((i + 1) / len(uploaded_files))

            status_text.text("Ingestion process complete!")
            st.rerun()

# =========================================================================
# TAB 2: ASK QUESTIONS (Section 3.C, 3.D, 4.6-4.10)
# =========================================================================
with tab_qa:
    st.subheader("Evidence-Grounded Legal Q&A")
    st.write(
        "Ask questions about the uploaded judgments. The system retrieves relevant passages using local "
        "FAISS + BGE Cross-Encoder reranking, and synthesizes an answer cited with exact source and page numbers."
    )

    if pipeline.vector_store.index.ntotal == 0:
        st.info("ℹ️ No documents are indexed yet. Please upload and process a judgment PDF in the 'Document Ingestion' tab first.")
    else:
        sample_questions = [
            "What was the core constitutional issue considered in the appeal?",
            "What were the primary arguments submitted by the Appellant?",
            "What findings did the Court make regarding the lower court's decree?",
            "What was the final outcome or order passed by the Court?"
        ]
        
        selected_sample = st.selectbox(
            "Or select a sample legal query:",
            [""] + sample_questions,
            index=0
        )

        user_query = st.text_input(
            "Enter your legal question:",
            value=selected_sample if selected_sample else "",
            placeholder="e.g., Did the Court grant special leave to appeal under Article 136?"
        )

        col_q1, col_q2 = st.columns([1, 4])
        with col_q1:
            run_query = st.button("🔍 Search & Answer", type="primary")

        if run_query and user_query.strip():
            with st.spinner("Retrieving evidence and synthesizing verified answer..."):
                response = pipeline.answer_legal_query(user_query.strip())

            # Display Answer
            st.markdown("### Answer")
            st.markdown(response["answer"])

            # Citation Verification Badge
            st.markdown("---")
            st.markdown("#### Citation & Grounding Verification")
            if response["is_grounded"]:
                st.markdown(
                    "<span class='status-badge badge-success'>✓ Grounded & Verified</span> "
                    "All cited documents and page numbers exist in retrieved evidence.",
                    unsafe_allow_html=True
                )
            else:
                if response.get("citations"):
                    st.markdown(
                        "<span class='status-badge badge-warning'>⚠️ Partially Verified</span> "
                        "Some cited details require manual review.",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        "<span class='status-badge badge-info'>ℹ️ No Inline Citations Detected</span>",
                        unsafe_allow_html=True
                    )

            if response.get("citations"):
                with st.expander(f"View Verified Citations ({len(response['citations'])})"):
                    for c in response["citations"]:
                        st.markdown(f"- **Document:** `{c['document']}` | **Page:** `{c['page']}` | Status: `{c['status']}`")

            if response.get("unmatched_citations"):
                with st.expander(f"⚠️ Unmatched Citations ({len(response['unmatched_citations'])})"):
                    for uc in response["unmatched_citations"]:
                        st.markdown(f"- **Document:** `{uc['document']}` | **Page:** `{uc['page']}` | Status: `{uc['status']}`")

            st.caption(f"Verification Note: {response.get('verification_notes', '')}")

            # Evidence Display (Section 3.D)
            st.markdown("---")
            st.markdown("### Supporting Evidence Passages")
            evidence_items = response.get("evidence_used", [])
            if evidence_items:
                for i, ev in enumerate(evidence_items, start=1):
                    with st.container():
                        st.markdown(f"""
                        <div class="evidence-card">
                            <div class="evidence-meta">
                                Passage #{i} | Document: <b>{ev['doc_name']}</b> | Page: <b>{ev['page_number']}</b> of {ev.get('total_pages', 'N/A')}
                                | Rerank Score: <code>{round(ev['rerank_score'], 3)}</code>
                                | Dense Cosine Score: <code>{round(ev['faiss_score'], 3)}</code>
                            </div>
                            <div>{ev['text']}</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.write("No evidence passages were retrieved for this query.")

# =========================================================================
# TAB 3: GENERATE CASE ANALYSIS (Section 3.B)
# =========================================================================
with tab_analysis:
    st.subheader("Generate Structured Case Analysis")
    st.write(
        "Generate a standardized 10-section legal analysis of an uploaded judgment. "
        "Any element not explicitly covered in the judgment text is clearly flagged as "
        "'Not available in the supplied judgment passages'."
    )

    indexed_docs = list(pipeline.vector_store.document_registry.keys())
    if not indexed_docs:
        st.info("ℹ️ No documents are indexed yet. Please upload a judgment PDF first.")
    else:
        selected_doc = st.selectbox("Select Judgment to Analyze:", indexed_docs)

        if st.button("📑 Generate 10-Section Case Analysis", type="primary"):
            with st.spinner(f"Analyzing substantive passages from '{selected_doc}'..."):
                analysis_res = pipeline.generate_case_analysis(selected_doc)
                st.session_state.last_analysis[selected_doc] = analysis_res

        if selected_doc in st.session_state.last_analysis:
            res = st.session_state.last_analysis[selected_doc]
            st.markdown("### Analysis Output")
            st.markdown(res["analysis"])

            with st.expander("Supporting Evidence Used for Analysis"):
                for i, ev in enumerate(res.get("evidence_used", []), start=1):
                    st.markdown(f"**Passage {i} [Page {ev['page_number']}]:**")
                    st.write(ev['text'])
                    st.markdown("---")

# =========================================================================
# TAB 4: DOCUMENT MANAGEMENT (Section 3.E)
# =========================================================================
with tab_manage:
    st.subheader("Active Document Management")
    st.write(
        "Review and manage documents currently loaded in the active FAISS retrieval index. "
        "Removing a document immediately deletes its vectors and chunks from active retrieval."
    )

    registry = pipeline.vector_store.document_registry
    if not registry:
        st.info("No documents currently active in the index.")
    else:
        doc_table = []
        for name, meta in registry.items():
            doc_table.append({
                "Document Name": name,
                "Total Pages": meta["total_pages"],
                "Indexed Chunks": meta["chunk_count"],
            })
        st.dataframe(pd.DataFrame(doc_table), use_container_width=True)

        st.markdown("---")
        st.markdown("### Remove Document")
        doc_to_remove = st.selectbox("Select document to remove:", list(registry.keys()))
        if st.button("🗑️ Remove Document & Rebuild Index", type="secondary"):
            with st.spinner(f"Removing '{doc_to_remove}' and rebuilding vector index..."):
                removed = pipeline.remove_document(doc_to_remove)
                if removed:
                    st.success(f"Successfully removed '{doc_to_remove}' from index.")
                    st.rerun()
                else:
                    st.error("Failed to remove document.")

        if st.button("⚠️ Clear Entire Index"):
            pipeline.vector_store.clear()
            st.success("Entire retrieval index cleared.")
            st.rerun()

# =========================================================================
# TAB 5: EVALUATION DATASET INSPECTION (Section 6)
# =========================================================================
with tab_eval:
    st.subheader("Evaluation Dataset Inspection & Methodology")
    st.markdown("""
    **Dataset Information:**
    - **Repository:** `dedol-hf/india-case-legal-rag`
    - **File:** `data/evaluation_dataset/clean_dataset.jsonl`
    - **Isolation Guarantee:** Strictly isolated from the production FAISS retrieval store. No evaluation data is indexed for user queries.
    """)

    with st.spinner("Loading evaluation dataset metrics..."):
        summary = evaluator.get_dataset_summary()

    if not summary.get("exists", False):
        st.error(f"Evaluation dataset not found: {summary.get('error')}")
    else:
        col_e1, col_e2, col_e3, col_e4 = st.columns(4)
        col_e1.metric("Total Records", f"{summary['total_records']:,}")
        col_e2.metric("File Size", f"{summary['file_size_mb']} MB")
        col_e3.metric("Source Judgments", summary["unique_source_judgments"])
        col_e4.metric("Avg Chunk Tokens", int(summary["token_estimates"]["mean"]))

        st.markdown("---")
        st.markdown("### Structural Metrics & Empirical Distributions")

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("#### Chunk Character Lengths")
            char_stats = summary["character_statistics"]
            df_char = pd.DataFrame({
                "Metric": ["Mean", "Std Dev", "Min", "25th Percentile", "Median (50th)", "75th Percentile", "Max"],
                "Value": [
                    round(char_stats["mean"], 1),
                    round(char_stats["std"], 1),
                    char_stats["min"],
                    round(char_stats["p25"], 1),
                    round(char_stats["median"], 1),
                    round(char_stats["p75"], 1),
                    char_stats["max"]
                ]
            })
            st.dataframe(df_char, hide_index=True, use_container_width=True)

        with col_m2:
            st.markdown("#### Top Source Documents by Chunk Count")
            top_src = pd.DataFrame(summary["top_sources_by_chunks"])
            st.dataframe(top_src, hide_index=True, use_container_width=True)

        st.markdown("---")
        st.markdown("### Verifiable Sample Records (From Dataset)")
        for i, s in enumerate(summary.get("samples", []), start=1):
            with st.expander(f"Sample Record #{i} (ID: {s['id']}) - Source: {s['metadata'].get('source')}"):
                st.json(s)

        st.markdown("---")
        st.markdown("### Scientific Evaluation Methodology Notice")
        st.info(summary["methodology_notice"])
