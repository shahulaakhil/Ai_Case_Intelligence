# AI Case Intelligence
### Evidence-Grounded Analysis of Indian Legal Judgments

**AI Case Intelligence** is an evidence-grounded legal judgment analysis system built with Retrieval-Augmented Generation (RAG). It enables legal researchers, advocates, and judicial scholars to query Indian Supreme Court judgments with strict verification against retrieved source text and page references.

---

## Architecture & Technology Stack

| Stage | Component | Specification |
| :--- | :--- | :--- |
| **Frontend** | Streamlit | Clean, professional UI with document upload, Q&A, structured case analysis, document management, and evaluation inspector |
| **PDF Extraction & Cleaning** | PyMuPDF (`pymupdf`) | Page-by-page extraction, scanned PDF detection (OCR required notice), citation/section-preserving cleaner |
| **Legal Chunking** | Rule-Based Legal Chunker | Semantic chunks of ~500–800 tokens (~2200-2600 chars) with ~100-150 tokens overlap |
| **Dense Embeddings** | `SentenceTransformers` | `BAAI/bge-base-en-v1.5` (768-dim, locally cached) |
| **Vector Retrieval** | FAISS (`faiss-cpu`) | `IndexFlatIP` (Cosine similarity via normalized embeddings) |
| **Reranking** | BGE Cross-Encoder | `BAAI/bge-reranker-base` (local cross-attention scoring, selects top 4-6 passages) |
| **LLM Generation** | Google GenAI SDK | Free-tier Gemini model (`gemini-2.5-flash` / `gemini-1.5-flash`), strictly conditioned on retrieved evidence |
| **Verification** | Lexical & Metadata Verifier | Validates all cited `[Source: ..., Page: ...]` against retrieved passages, flags ungrounded statements |
| **Evaluation Dataset** | Standalone JSONL | `data/evaluation_dataset/clean_dataset.jsonl` (9,375 chunks, strictly isolated from production FAISS) |

---

## Key Features

1. **Document Ingestion & Scanned PDF Detection:**
   - Upload multiple Indian Supreme Court judgments in PDF format.
   - Extracts page-by-page text while tracking source filenames and page numbers.
   - Detects image-only/scanned PDFs and explicitly informs the user that OCR is required, refusing to hallucinate text.

2. **Strict Vector Index Isolation:**
   - The production FAISS index contains **only** user-uploaded judgments.
   - The evaluation dataset (`data/evaluation_dataset/clean_dataset.jsonl`) is strictly isolated and never indexed into the production database.

3. **Two-Stage Local Retrieval (Dense + Cross-Encoder):**
   - Retrieves top 15 candidate passages using local FAISS and `BAAI/bge-base-en-v1.5`.
   - Reranks candidates using local `BAAI/bge-reranker-base` cross-encoder to select the 4–6 most relevant passages.

4. **10-Section Structured Case Analysis:**
   - Synthesizes comprehensive analysis covering:
     1. Case Summary
     2. Parties and Key Persons
     3. Timeline
     4. Legal Issues
     5. Relevant Laws & Constitutional Provisions
     6. Arguments of the Parties
     7. Evidence & Court Findings
     8. Court Reasoning
     9. Final Judgment / Outcome
     10. Supporting Sources & Page References
   - Explicitly flags any section or detail not present in the judgment text as *"Not available in the supplied judgment passages"*.

5. **Citation & Grounding Verification:**
   - Parses generated responses for inline citations (`[Source: <filename>, Page: <page>]`).
   - Cross-references each citation against retrieved evidence and highlights verification status.

6. **Dynamic Document Management:**
   - Add or remove judgments anytime.
   - Removing a document immediately purges its chunks from the FAISS vector index.

7. **Empirical Evaluation Dataset Inspection:**
   - Explores the 9,375 records in `clean_dataset.jsonl`.
   - Computes empirical length distributions and source document frequencies.
   - Clarifies that reference-free grounding is supported, while reference-based Q&A metrics require gold question-answer pairs not natively contained in the corpus.

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10 to 3.14
- Virtual environment (`uv` or `python -m venv`)

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Open `.env` and set your Google Gemini API Key:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

### 3. Install Dependencies
```bash
uv pip install -r requirements.txt
# OR
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
Or using the virtual environment:
```bash
.venv\Scripts\streamlit run app.py
```

---

## Directory Structure
```
AI_CASE/
├── app.py                         # Streamlit user interface
├── requirements.txt               # Compatible dependencies
├── .env.example                   # Template environment configuration
├── .env                           # Local environment secrets (in .gitignore)
├── .gitignore                     # Git ignore rules
├── README.md                      # Documentation
├── src/
│   ├── __init__.py
│   ├── config.py                  # Configurations and environment loader
│   ├── pdf_processor.py           # PyMuPDF extractor, cleaner, legal chunker
│   ├── vector_store.py            # Local BGE embeddings & FAISS index
│   ├── retriever.py               # Top-K retrieval & BGE Cross-Encoder reranker
│   ├── generator.py               # Gemini API generation & citation verifier
│   ├── pipeline.py                # End-to-end RAG orchestrator
│   └── evaluator.py               # Evaluation dataset parser & metrics
└── data/
    └── evaluation_dataset/
        └── clean_dataset.jsonl    # 9,375 legal chunks (Hugging Face corpus)
```

---

## Legal & Compliance Disclaimer
This software is intended for informational, academic, and legal research purposes only. AI Case Intelligence relies on retrieved judgment text and automated verification, but does not constitute professional legal advice. Users must independently verify findings against official law reports.
