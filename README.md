# 📐 Math & Statistics Q&A Assistant (PDF RAG)

A small Retrieval-Augmented Generation (RAG) app. Upload up to **3 PDFs**, ask a
math/stats question, and get a one-page, professor-style answer drawn **only**
from your documents — with rendered equations. If the answer isn't in the PDFs,
the app says so and offers a one-click Google search. Every Q&A can be exported
to a **Word document**.

---

## How it works

```
PDF(s) ─▶ extract text (PyMuPDF) ─▶ chunk into overlapping pieces
       ─▶ embed each chunk locally (sentence-transformers)
                                   │
question ─▶ embed ─▶ cosine-similarity search ─▶ top-k chunks
                                   │
        best score < threshold ? ──┬─ yes ─▶ "Let me search!" + Google link
                                   └─ no  ─▶ Claude answers from those chunks
                                            ├─ context insufficient ─▶ "Let me search!"
                                            └─ answers ─▶ Intuition / Technical /
                                                          Applications  (+ sources)
```

Two independent gates decide "not found": a cheap **similarity threshold**, and
the **LLM itself** (it returns a sentinel when the retrieved text doesn't
actually contain the answer). This keeps the app honest about what's in your PDFs.

## Project structure

| File | What it does |
|------|--------------|
| `app.py` | Streamlit UI — upload, ask, render answers/math, download Word. The only file that touches Streamlit or your API key. |
| `rag.py` | The engine — PDF parsing, chunking, embeddings, retrieval, the "not found" gate, and the Claude call. Framework-free and easy to test. |
| `docx_export.py` | Turns the Q&A (markdown + LaTeX) into a `.docx`, rendering equations to images so they display in Word. |
| `requirements.txt` | Dependencies. |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The first run downloads the embedding model (`all-MiniLM-L6-v2`, ~80 MB) once.

## Configure the answer model (Anthropic Claude)

Set your key as an environment variable (recommended) or paste it into the
sidebar at runtime:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."     # Windows: setx ANTHROPIC_API_KEY "sk-ant-..."
```

The default model is `claude-sonnet-4-6` (see `LLM_MODEL` in `rag.py`).

## Run

```bash
streamlit run app.py
```

Then in the browser: **upload PDFs → Process documents → ask → download**.

## Tuning (sidebar + `rag.py`)

- **‘Found’ threshold** — raise it if irrelevant answers slip through; lower it
  if good questions wrongly report "not found". Defaults to `0.30`.
- **Top-k** — how many chunks the model sees (default 6).
- **Chunk size / overlap** — `CHUNK_WORDS`, `CHUNK_OVERLAP_WORDS` in `rag.py`.
- **Answer length** — `LLM_MAX_TOKENS` (~900 ≈ one page) and the word limit in
  `SYSTEM_PROMPT`.
- **Wording of the "not found" message** — `NOT_FOUND_MESSAGE` in `rag.py`.

## Notes & limitations

- **Scanned/image-only PDFs** have no extractable text. OCR them first (e.g.
  `ocrmypdf in.pdf out.pdf`) before uploading.
- **Equations in Word** are rendered with matplotlib's *mathtext*, which covers
  most common notation. A few exotic commands are auto-rewritten; anything it
  still can't render falls back to showing the LaTeX source. On-screen rendering
  uses KaTeX and is more complete. (For publication-grade Word equations you'd
  swap in a MathML→OMML conversion — not needed for study notes.)
- **Large files**: a ~1000-page PDF produces a few thousand chunks; the one-time
  embedding step runs on CPU and may take a couple of minutes. Re-asking is
  instant. For very large corpora, replace the brute-force search in
  `rag.retrieve()` with [FAISS](https://github.com/facebookresearch/faiss).
- This tool reports what your documents say; it is not a substitute for a
  textbook or instructor, and it does not browse the web itself (it links out).

## Deployment options

- **Local / shared server:** `streamlit run app.py --server.port 8501 --server.address 0.0.0.0`
- **Docker** (sketch):
  ```dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt
  COPY . .
  EXPOSE 8501
  CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
  ```
- Pass `ANTHROPIC_API_KEY` as a secret/env var rather than baking it into the image.
