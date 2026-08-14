"""
rag.py
======
The "brains" of the Q&A app. This module is deliberately framework-agnostic
(it does NOT import Streamlit) so you can unit-test it or reuse it elsewhere.

Pipeline implemented here:

    PDF bytes ──▶ extract text per page ──▶ split into overlapping chunks
              ──▶ embed every chunk (sentence-transformers, runs locally)
              ──▶ at question time: embed the question, find the most similar
                  chunks (cosine similarity), and:
                    • if nothing is similar enough  -> signal "not found"
                    • otherwise -> ask the LLM (Claude) to answer *only* from
                      those chunks, in the structured "professor" format.

Everything that you might want to tune lives in the CONFIG section right below.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# CONFIG  —  the knobs you are most likely to change
# ---------------------------------------------------------------------------

# Local embedding model. all-MiniLM-L6-v2 is small (~80 MB), fast on CPU,
# and good enough for retrieval. It downloads automatically the first time.
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# How we cut the documents into pieces ("chunks") for retrieval.
# Bigger chunks = more context per piece but coarser matching.
CHUNK_WORDS = 220          # approx words per chunk
CHUNK_OVERLAP_WORDS = 40   # words shared between consecutive chunks (keeps context)

# Retrieval / "is the answer in the docs?" gate.
TOP_K = 6                  # how many chunks to feed the LLM
MIN_SIMILARITY = 0.30      # below this top-score we assume the answer is NOT in the PDFs.
                           # 0..1 scale (cosine). Tune for your documents:
                           #   too many false "not found"  -> lower it (e.g. 0.25)
                           #   irrelevant answers slip through -> raise it (e.g. 0.40)

# LLM (answer generator). These are Anthropic Claude settings.
LLM_MODEL = "claude-sonnet-4-6"   # change to another Claude model string if you prefer
LLM_MAX_TOKENS = 900              # ~ one page. Keep modest to honor the length limit.

# Exact message the user asked for when the answer is not in the PDFs.
# (Added the missing word "find" so the on-screen text reads cleanly — edit freely.)
NOT_FOUND_MESSAGE = "I couldn't find an answer based on the provided info. Let me search!"

# Sentinel the LLM is told to return if the retrieved text doesn't actually
# contain the answer. We detect it and fall back to the "not found" path.
NO_ANSWER_SENTINEL = "NO_ANSWER_IN_CONTEXT"

# The "college math professor" instructions. This single prompt controls the
# persona, the structure, the length, and the math formatting of every answer.
SYSTEM_PROMPT = f"""You are a patient, precise college mathematics and statistics professor.
You answer a student's question using ONLY the excerpts provided in the CONTEXT block.
Do not use outside knowledge and do not invent facts. If the CONTEXT does not contain
enough information to answer, reply with exactly this token and nothing else:
{NO_ANSWER_SENTINEL}

When you CAN answer, follow these rules strictly:
1. Keep the whole answer under ~750 words (it must fit on one or two pages).
2. Structure the answer with these four markdown headings, in this order:
   ### Intuition
   ### Technical details
   ### One simple example
   ### Applications & further study
   Under each heading, write 2-4 tight sentences (bullets are fine where natural).
3. Mathematics formatting (important — the answer is later exported to Word):
   - Inline math uses single dollar signs:  $f(x) = x^2$
   - A displayed/standalone equation goes on ITS OWN LINE wrapped in double
     dollar signs on that same line, e.g.:
     $$ \\int_a^b f(x)\\,dx = F(b) - F(a) $$
   - Do not use \\[ \\], \\( \\), or LaTeX environments like \\begin{{align}}.
4. Be faithful to the source. If the context only partially answers, answer the
   part you can and say briefly what is missing.
"""

# ---------------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """One retrievable piece of a document, with enough metadata to cite it."""
    text: str        # the chunk's text
    source: str      # original PDF filename
    page: int        # 1-based page number the chunk came from


# ---------------------------------------------------------------------------
# STEP 1 — read text out of a PDF
# ---------------------------------------------------------------------------

def extract_pages(pdf_bytes: bytes, filename: str) -> list[tuple[int, str]]:
    """
    Return a list of (page_number, page_text) for one PDF.

    Uses PyMuPDF (imported as `fitz`), which is fast even on 1000-page files.
    We import it lazily so this module can be imported without the dependency
    installed (handy for reading/reviewing the code).
    """
    import fitz  # PyMuPDF

    pages: list[tuple[int, str]] = []
    # Open the PDF directly from the in-memory bytes (no temp file needed).
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text") or ""
            # Light cleanup: collapse runs of whitespace/newlines.
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if text:
                pages.append((i + 1, text))  # +1 so page numbers are human-friendly
    return pages


# ---------------------------------------------------------------------------
# STEP 2 — split pages into overlapping chunks
# ---------------------------------------------------------------------------

def chunk_pages(pages: list[tuple[int, str]], source: str) -> list[Chunk]:
    """
    Turn (page, text) pairs into a flat list of Chunk objects.

    We chunk *within* each page so every chunk keeps an accurate page number
    for citations. Consecutive chunks overlap by CHUNK_OVERLAP_WORDS words so a
    sentence split across a boundary is still findable.
    """
    chunks: list[Chunk] = []
    step = max(1, CHUNK_WORDS - CHUNK_OVERLAP_WORDS)  # how far we advance each time

    for page_num, text in pages:
        words = text.split()
        if not words:
            continue
        # Slide a CHUNK_WORDS-wide window across the page's words.
        for start in range(0, len(words), step):
            window = words[start:start + CHUNK_WORDS]
            if not window:
                break
            chunk_text = " ".join(window).strip()
            # Skip tiny trailing fragments that carry little meaning.
            if len(chunk_text) >= 40:
                chunks.append(Chunk(text=chunk_text, source=source, page=page_num))
            if start + CHUNK_WORDS >= len(words):
                break  # we've covered the whole page
    return chunks


# ---------------------------------------------------------------------------
# STEP 3 — embeddings (turn text into vectors)
# ---------------------------------------------------------------------------

def load_embedder():
    """
    Load the sentence-transformers model. Call this once and reuse the result
    (in the Streamlit app we cache it so it isn't reloaded on every interaction).
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL_NAME)


def embed_texts(embedder, texts: list[str]) -> np.ndarray:
    """
    Embed a list of strings and return an (N, dim) float32 array whose rows are
    L2-normalized. Normalizing means a simple dot product == cosine similarity.
    """
    vectors = embedder.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,   # <-- gives us unit vectors directly
        convert_to_numpy=True,
    )
    return vectors.astype("float32")


def build_index(embedder, chunks: list[Chunk]) -> np.ndarray:
    """Embed every chunk's text once, up front. Returns the matrix of vectors."""
    return embed_texts(embedder, [c.text for c in chunks])


# ---------------------------------------------------------------------------
# STEP 4 — retrieval (find the chunks most similar to the question)
# ---------------------------------------------------------------------------

def retrieve(embedder, question: str, chunks: list[Chunk],
             index: np.ndarray, top_k: int = TOP_K) -> list[tuple[Chunk, float]]:
    """
    Return the top_k (chunk, similarity_score) pairs, best first.

    Because both the question vector and the chunk vectors are normalized,
    the dot product is the cosine similarity in [-1, 1] (here effectively 0..1).
    For a few thousand chunks this brute-force search is instant; if you ever
    scale to millions of chunks, swap this for FAISS (see README).
    """
    q_vec = embed_texts(embedder, [question])[0]   # shape (dim,)
    scores = index @ q_vec                          # shape (N,) — one score per chunk
    # Indices of the top_k highest scores, sorted high -> low.
    top_idx = np.argsort(-scores)[:top_k]
    return [(chunks[i], float(scores[i])) for i in top_idx]


# ---------------------------------------------------------------------------
# STEP 5 — generate the grounded answer with the LLM
# ---------------------------------------------------------------------------

def _build_context_block(retrieved: list[tuple[Chunk, float]]) -> str:
    """Format the retrieved chunks into a labelled CONTEXT block for the LLM."""
    parts = []
    for chunk, _score in retrieved:
        parts.append(f"[Source: {chunk.source}, page {chunk.page}]\n{chunk.text}")
    return "\n\n---\n\n".join(parts)


def sources_caption(retrieved: list[tuple[Chunk, float]]) -> str:
    """A short, de-duplicated 'Sources: file.pdf p.3, p.7' string for display."""
    seen: dict[str, list[int]] = {}
    for chunk, _ in retrieved:
        seen.setdefault(chunk.source, [])
        if chunk.page not in seen[chunk.source]:
            seen[chunk.source].append(chunk.page)
    bits = [f"{src} p.{', '.join(map(str, sorted(pages)))}" for src, pages in seen.items()]
    return "Sources: " + "; ".join(bits)


def answer_question(client, question: str,
                    retrieved: list[tuple[Chunk, float]]) -> tuple[bool, str]:
    """
    Ask Claude to answer the question using only the retrieved context.

    `client` is an anthropic.Anthropic() instance (created in the app so the
    API key stays out of this module).

    Returns (found, text):
        found == False  -> answer is not in the documents (text is "").
        found == True   -> text is the structured answer (markdown + LaTeX).
    """
    context = _build_context_block(retrieved)
    user_prompt = (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"Answer using only the CONTEXT above, following all formatting rules."
    )

    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    # Concatenate any text blocks Claude returned.
    text = "".join(block.text for block in response.content
                   if getattr(block, "type", None) == "text").strip()

    # The model signals "answer not present" with our sentinel token.
    if NO_ANSWER_SENTINEL in text:
        return False, ""
    return True, text


# ---------------------------------------------------------------------------
# STEP 5b — the top-level "ask" used by the app (ties the gate + LLM together)
# ---------------------------------------------------------------------------

def ask(client, embedder, question: str, chunks: list[Chunk],
        index: np.ndarray, top_k: int = TOP_K,
        min_similarity: float = MIN_SIMILARITY) -> dict:
    """
    Full question -> answer flow with the 'not found' gate.

    Returns a dict the UI can render directly:
        {
          "found": bool,
          "answer": str,          # structured answer, or NOT_FOUND_MESSAGE
          "sources": str | None,  # "Sources: ..." caption when found
        }
    """
    retrieved = retrieve(embedder, question, chunks, index, top_k=top_k)
    best_score = retrieved[0][1] if retrieved else 0.0

    # Gate 1 (cheap): nothing in the docs is even close -> don't bother the LLM.
    if best_score < min_similarity:
        return {"found": False, "answer": NOT_FOUND_MESSAGE, "sources": None}

    # Gate 2: the LLM reads the retrieved text and decides if it truly answers.
    found, text = answer_question(client, question, retrieved)
    if not found:
        return {"found": False, "answer": NOT_FOUND_MESSAGE, "sources": None}

    return {"found": True, "answer": text, "sources": sources_caption(retrieved)}
