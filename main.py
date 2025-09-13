# main.py — Lore Leaf Knowledge Bot (FastAPI + FAISS + OpenAI)
# Robust .env loading + readable errors

import os, re, glob, textwrap
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import faiss
import tiktoken
from markdown import markdown

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# -------------------- ENV LOADING (robust) --------------------
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)

def _mask(s: str, show: int = 6) -> str:
    if not s:
        return "None"
    return s[:show] + "…" + s[-4:]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
print(f"[env] .env path       : {ENV_PATH}")
print(f"[env] API key present : {bool(OPENAI_API_KEY)} ({_mask(OPENAI_API_KEY)})")

# -------------------- CONFIG --------------------
DOC_DIR = str(ROOT / "notes")
STATIC_DIR = str(ROOT / "static")   # put index.html + styles.css here

MODEL = os.getenv("MODEL", "gpt-4.1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHUNK_TOKENS = 400
CHUNK_OVERLAP = 120
TOP_K = 4

Path(DOC_DIR).mkdir(parents=True, exist_ok=True)
Path(STATIC_DIR).mkdir(parents=True, exist_ok=True)

# -------------------- OPENAI CLIENT --------------------
from openai import OpenAI

def get_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing. Put it in .env or set it in the shell.")
    return OpenAI(api_key=key)

# -------------------- UTILITIES --------------------
def md_to_text(md: str) -> str:
    """Convert Markdown to plain text-ish for embedding."""
    html = markdown(md)
    return re.sub("<[^<]+?>", "", html)

def token_chunks(text: str, chunk_tokens=CHUNK_TOKENS, overlap=CHUNK_OVERLAP) -> List[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    toks = enc.encode(text)
    out = []
    start = 0
    while start < len(toks):
        end = min(start + chunk_tokens, len(toks))
        out.append(enc.decode(toks[start:end]))
        if end == len(toks):
            break
        start = end - overlap
    return out

@dataclass
class DocChunk:
    text: str
    source: str
    order: int

# -------------------- INDEX STATE --------------------
INDEX = None
CHUNKS: List[DocChunk] = []
VECS = None

def _embed_texts(texts: List[str]) -> np.ndarray:
    client = get_client()
    embeds = []
    BATCH = 128
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        embeds.extend([e.embedding for e in resp.data])
    arr = np.array(embeds, dtype="float32")
    faiss.normalize_L2(arr)  # cosine via inner product
    return arr

def build_index():
    """Read notes/, chunk, embed, and build FAISS index."""
    global INDEX, CHUNKS, VECS
    files = sorted(
        glob.glob(os.path.join(DOC_DIR, "*.md"))
        + glob.glob(os.path.join(DOC_DIR, "*.txt"))
        + glob.glob(os.path.join(DOC_DIR, "*.pdf"))
    )
    if not files:
        INDEX, CHUNKS, VECS = None, [], None
        app.state.index_ntotal = 0
        return

    chunks: List[DocChunk] = []
    for fp in files:
        if fp.lower().endswith(".pdf"):
            # We don't embed binary PDFs here—skip or add your own PDF->text
            continue
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        body = md_to_text(raw) if fp.endswith(".md") else raw
        for i, ch in enumerate(token_chunks(body)):
            txt = ch.strip()
            if txt:
                chunks.append(DocChunk(text=txt, source=os.path.basename(fp), order=i))

    if not chunks:
        INDEX, CHUNKS, VECS = None, [], None
        app.state.index_ntotal = 0
        return

    texts = [c.text for c in chunks]
    vecs = _embed_texts(texts)
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    INDEX, CHUNKS, VECS = index, chunks, vecs
    app.state.index_ntotal = int(index.ntotal)
    print(f"[index] built with {len(CHUNKS)} chunks from {len(files)} files")

def search(query: str, k=TOP_K) -> List[DocChunk]:
    if INDEX is None or not CHUNKS:
        return []
    client = get_client()
    q = client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
    q = np.array([q], dtype="float32")
    faiss.normalize_L2(q)
    D, I = INDEX.search(q, k)
    hits = []
    for idx in I[0]:
        if 0 <= idx < len(CHUNKS):
            hits.append(CHUNKS[idx])
    return hits

# -------------------- GENERIC PROMPTS (story-agnostic) --------------------
SYSTEM_BASE = (
    "You are a helpful knowledge assistant. Answer using ONLY the provided context. "
    "Be clear and concise. If the context is insufficient, say so and suggest what to add. "
    "Cite like (Source: filename#chunk)."
)

PERSONAS = {
    "neutral":  "Speak neutrally and directly.",
    "tutor":    "Use brief Socratic questions before answering. Teach, don't lecture.",
    "analyst":  "Be crisp. Use bullets. Call out risks, assumptions, unknowns.",
    "friendly": "Warm but concise. Offer next steps.",
    "skeptic":  "Challenge assumptions and point out gaps.",
}
DEFAULT_PERSONA = "neutral"

def make_prompt(query: str, hits: List[DocChunk], persona: str) -> str:
    ctx_lines = []
    for h in hits:
        pv = re.sub(r"\s+", " ", h.text.replace("\n", " ").strip())
        ctx_lines.append(f"[{h.source}#{h.order}] {pv}")
    ctx = "\n".join(ctx_lines)
    persona_id = (persona or DEFAULT_PERSONA).lower()
    persona_instr = PERSONAS.get(persona_id, PERSONAS[DEFAULT_PERSONA])
    return f"""Persona: {persona_instr}
Context:
{ctx}

User question: {query}

Instructions:
- Answer only from the context.
- If context is insufficient, say you don't know and suggest what notes to add.
- Append citations like (Source: filename#chunk)."""

# -------------------- FASTAPI APP --------------------
app = FastAPI(title="Lore Leaf Knowledge Bot")

# Static mounts
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/notes", StaticFiles(directory=DOC_DIR), name="notes")  # for PDF links

class AskPayload(BaseModel):
    query: str
    persona: Optional[str] = DEFAULT_PERSONA

# ----- Root / index.html -----
@app.get("/", response_class=HTMLResponse)
def home():
    index_path = Path(STATIC_DIR) / "index.html"
    if not index_path.exists():
        return HTMLResponse(
            "<h3>Missing static/index.html</h3><p>Put your built UI in <code>/static/index.html</code>.</p>",
            status_code=500,
        )
    return index_path.read_text(encoding="utf-8")

# ----- Favicon -----
@app.get("/favicon.ico")
def favicon():
    ico = Path(STATIC_DIR) / "favicon.ico"
    if ico.exists():
        return FileResponse(str(ico), media_type="image/x-icon")
    return PlainTextResponse("", media_type="image/x-icon")

# ----- Upload (dropzone) -----
@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    Path(DOC_DIR).mkdir(parents=True, exist_ok=True)
    saved = 0
    for f in files:
        dest = Path(DOC_DIR) / Path(f.filename).name  # strip any paths
        with open(dest, "wb") as out:
            out.write(await f.read())
        saved += 1
    return {"saved": saved}

# ----- Reindex -----
@app.post("/reindex")
def reindex():
    try:
        Path(DOC_DIR).mkdir(parents=True, exist_ok=True)
        build_index()
        return JSONResponse({"status": "ok", "chunks": len(CHUNKS)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

# ----- Ask -----
@app.post("/ask")
def ask(payload: AskPayload):
    try:
        if INDEX is None:
            raise HTTPException(status_code=400, detail="Index empty. Add notes and click Rebuild Index.")
        q = (payload.query or "").trim() if hasattr(str, "trim") else (payload.query or "").strip()
        if not q:
            raise HTTPException(status_code=400, detail="Empty query.")
        hits = search(q, k=TOP_K)
        prompt = make_prompt(q, hits, payload.persona or DEFAULT_PERSONA)
        client = get_client()
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_BASE},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        answer = resp.choices[0].message.content.strip()
        previews = [
            {"ref": f"{h.source}#{h.order}",
             "preview": textwrap.shorten(h.text, width=180, placeholder="…")}
            for h in hits
        ]
        return JSONResponse({"answer": answer, "sources": previews})
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

# ----- Stats (sidebar metrics) -----
@app.get("/stats")
def stats():
    files = []
    for ext in ("*.md", "*.txt", "*.pdf"):
        files += glob.glob(str(Path(DOC_DIR) / "**" / ext), recursive=True)
    chunks = int(getattr(app.state, "index_ntotal", 0))
    return {"files": len(files), "chunks": chunks, "model": MODEL}

# ----- Source preview (safe) -----
def _safe_notes_path(ref: str) -> Path:
    p = (Path(DOC_DIR) / ref).resolve()
    notes_root = Path(DOC_DIR).resolve()
    if notes_root not in p.parents and p != notes_root:
        raise HTTPException(status_code=404, detail="Not found")
    return p

@app.get("/preview", response_class=HTMLResponse)
def preview(ref: str = Query(..., description="Relative path/filename under /notes")):
    # Support "filename#chunk" or plain "filename"
    if "#" in ref:
        ref, _ = ref.split("#", 1)
    path = _safe_notes_path(ref)
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "Not found")

    ext = path.suffix.lower()
    if ext == ".pdf":
        # Serve a link; actual file is available at /notes/<filename>
        return f"<p><a href='/notes/{path.name}' target='_blank' rel='noreferrer'>Open PDF</a></p>"

    text = path.read_text(encoding="utf-8", errors="ignore")
    html = markdown(text)
    return f"<article class='doc'>{html}</article>"

# -------------------- DEV ENTRY --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
