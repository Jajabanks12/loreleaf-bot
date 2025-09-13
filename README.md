# Lore Leaf — Knowledge Bot

Bring-your-own-notes RAG bot (FastAPI + FAISS + OpenAI).  
Drop .md/.txt/.pdf into \
otes/\, click **Rebuild Index**, ask questions.

## Quickstart
\\\ash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload
\\\

## Environment
Create a \.env\ (NOT committed):
\\\
OPENAI_API_KEY=sk-...
MODEL=gpt-4.1
EMBED_MODEL=text-embedding-3-small
\\\

## Endpoints
- \POST /upload\  — upload files to \
otes/\
- \POST /reindex\ — rebuild FAISS index
- \POST /ask\     — ask with {query, persona}
- \GET  /preview\ — render source snippet
- \GET  /stats\   — files/chunks/model

## License
MIT
