"""
app.py
======
The interactive web interface (Streamlit). Run it with:

    streamlit run app.py

What the screen does, top to bottom:
    1. Sidebar: paste your Anthropic API key + tune retrieval settings.
    2. Upload up to 3 PDFs and click "Process documents" to build the index.
    3. Type a math/stats question and click "Ask".
         - Answer found  -> shows a structured answer (intuition / technical /
                            applications) with rendered equations + sources.
         - Not found     -> shows your "Let me search!" message plus a clickable
                            Google button for the same question.
    4. Every Q&A is kept in this session and can be downloaded as a Word file.

This file is intentionally the only place that knows about Streamlit and about
your API key; all the real work lives in rag.py and docx_export.py.
"""

import os
from urllib.parse import quote_plus

import streamlit as st

import rag
from docx_export import build_qa_document

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Math & Stats Q&A", page_icon="📐", layout="wide")
st.title("📐 Math & Statistics Q&A Assistant")
st.caption("Ask questions answered from your uploaded PDFs — with a professor's structure and real equations.")
st.caption("Developed by: Al Yazdani")

MAX_FILES = 3


# ---------------------------------------------------------------------------
# Cached heavy objects
# Streamlit reruns this whole script on every click, so we cache the things
# that are expensive to create. @st.cache_resource keeps ONE shared instance.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the embedding model (first run downloads it)…")
def get_embedder():
    return rag.load_embedder()


@st.cache_resource(show_spinner=False)
def get_llm_client(api_key: str):
    # Imported here so the app still loads if the package isn't installed yet.
    import anthropic
    return anthropic.Anthropic(api_key=api_key)


# ---------------------------------------------------------------------------
# Session state (persists across reruns within one browser session)
# ---------------------------------------------------------------------------
st.session_state.setdefault("chunks", None)        # list[rag.Chunk]
st.session_state.setdefault("index", None)         # np.ndarray of embeddings
st.session_state.setdefault("fingerprint", None)   # detects when uploads change
st.session_state.setdefault("history", [])         # list of {question, answer, sources}


# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    # API key: prefer the environment variable; otherwise let the user paste it.
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    api_key = st.text_input(
        "Anthropic API key",
        value=api_key,
        type="password",
        help="Used only to generate answers. Or set the ANTHROPIC_API_KEY env var.",
    )

    st.divider()
    top_k = st.slider("Chunks sent to the model (top-k)", 3, 12, rag.TOP_K)
    min_sim = st.slider(
        "‘Found’ threshold (similarity)", 0.10, 0.60, rag.MIN_SIMILARITY, 0.01,
        help="Higher = stricter. If good questions wrongly say 'not found', lower this.",
    )


# ---------------------------------------------------------------------------
# Document upload + indexing
# ---------------------------------------------------------------------------
st.subheader("1 · Upload your PDFs")
uploaded = st.file_uploader(
    f"Up to {MAX_FILES} PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)

# Enforce the 3-file limit clearly.
if uploaded and len(uploaded) > MAX_FILES:
    st.error(f"Please upload at most {MAX_FILES} files. Using the first {MAX_FILES}.")
    uploaded = uploaded[:MAX_FILES]

# A lightweight fingerprint lets us notice when the selected files change.
def files_fingerprint(files):
    return tuple((f.name, f.size) for f in files) if files else None

if st.button("📑 Process documents", disabled=not uploaded, type="primary"):
    embedder = get_embedder()
    all_chunks = []
    progress = st.progress(0.0, text="Reading PDFs…")
    for n, f in enumerate(uploaded, start=1):
        pages = rag.extract_pages(f.getvalue(), f.name)   # STEP 1: extract text
        all_chunks.extend(rag.chunk_pages(pages, f.name))  # STEP 2: chunk it
        progress.progress(n / len(uploaded), text=f"Read {f.name} ({len(pages)} pages)")

    if not all_chunks:
        st.error("No extractable text found. Are these scanned/image-only PDFs? "
                 "Those need OCR first (see README).")
    else:
        with st.spinner(f"Embedding {len(all_chunks):,} chunks… (one-time per upload)"):
            index = rag.build_index(embedder, all_chunks)  # STEP 3: embed
        st.session_state.chunks = all_chunks
        st.session_state.index = index
        st.session_state.fingerprint = files_fingerprint(uploaded)
        st.success(f"Indexed {len(all_chunks):,} chunks from {len(uploaded)} file(s). Ask away!")

# Warn if the user changed the file selection but hasn't re-processed.
if (uploaded and st.session_state.fingerprint
        and files_fingerprint(uploaded) != st.session_state.fingerprint):
    st.info("Your file selection changed — click **Process documents** to re-index.")


# ---------------------------------------------------------------------------
# Ask a question
# ---------------------------------------------------------------------------
st.subheader("2 · Ask a question")
question = st.text_input("Your question", placeholder="e.g. What is the intuition behind the Central Limit Theorem?")
ask_clicked = st.button("💬 Ask", type="primary", disabled=not question)

if ask_clicked:
    if st.session_state.index is None:
        st.warning("Please upload and process at least one PDF first.")
    elif not api_key:
        st.warning("Please add your Anthropic API key in the sidebar.")
    else:
        embedder = get_embedder()
        client = get_llm_client(api_key)
        with st.spinner("Searching your documents and composing an answer…"):
            try:
                result = rag.ask(
                    client, embedder, question,
                    st.session_state.chunks, st.session_state.index,
                    top_k=top_k, min_similarity=min_sim,
                )
            except Exception as e:
                result = None
                st.error(f"Something went wrong while answering: {e}")

        if result is not None:
            # Save to history so it appears below and can be exported.
            st.session_state.history.append({
                "question": question,
                "answer": result["answer"],
                "sources": result["sources"],
                "found": result["found"],
            })


# ---------------------------------------------------------------------------
# Show the conversation (newest first)
# ---------------------------------------------------------------------------
if st.session_state.history:
    st.subheader("3 · Answers")

    for item in reversed(st.session_state.history):
        with st.container(border=True):
            st.markdown(f"**Q: {item['question']}**")

            if item["found"]:
                # st.markdown renders both the structure AND the $…$ / $$…$$
                # math (Streamlit uses KaTeX under the hood).
                st.markdown(item["answer"])
                if item["sources"]:
                    st.caption(item["sources"])
            else:
                # Not found: show the message + a clickable Google search button.
                st.info(item["answer"])
                google_url = "https://www.google.com/search?q=" + quote_plus(item["question"])
                st.link_button("🔎 Search this question on Google", google_url)

    st.divider()

    # -----------------------------------------------------------------------
    # Download the whole session as a Word document
    # -----------------------------------------------------------------------
    st.subheader("4 · Download")
    docx_bytes = build_qa_document([
        {"question": h["question"], "answer": h["answer"], "sources": h["sources"]}
        for h in st.session_state.history
    ])
    st.download_button(
        "⬇️ Download Q&A as Word (.docx)",
        data=docx_bytes,
        file_name="math_qa_session.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    if st.button("🗑️ Clear all answers"):
        st.session_state.history = []
        st.rerun()
