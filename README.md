# 📚 Lore Leaf – Knowledge Bot

**Lore Leaf** is a lightweight, self-hosted knowledge bot powered by **FastAPI**, **FAISS**, and **OpenAI**.  
Drop your notes (`.md`, `.txt`, `.pdf`) into the `notes/` folder, rebuild the index, and query your own knowledge base through a modern web interface.

---

## ✨ Features

- **Bring-Your-Own Notes** – ingest Markdown, text, and PDF files  
- **Retrieval-Augmented Generation (RAG)** with FAISS + OpenAI embeddings  
- **Modern Web UI**  
  - Dropzone file uploads  
  - Tabs (Answer / Sources / Chunks)  
  - Source preview rail  
  - History of recent queries  
  - Persona modes (Neutral, Tutor, Analyst, Friendly, Skeptic)  
  - Sidebar metrics & settings drawer  
- **Safe Defaults** – `.env` support, `.gitignore` excludes secrets and local data  
- **Extensible** – easy to add new personas, endpoints, or storage backends  

---

## 🚀 Quickstart

### 1. Clone and enter the repo
```bash
git clone https://github.com/<your-username>/loreleaf-bot.git
cd loreleaf-bot
---
2. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Mac/Linux
source .venv/bin/activate
---
3. Install dependencies
pip install -r requirements.txt
---
4. Add environment variables
OPENAI_API_KEY=sk-...
MODEL=gpt-4.1
EMBED_MODEL=text-embedding-3-small
---
5. Run the app
uvicorn main:app --reload
---
📂 Project Structure
loreleaf-bot/
├── main.py            # FastAPI app
├── requirements.txt   # Dependencies
├── .env.example       # Example env vars
├── static/            # index.html, styles.css
├── notes/             # Your uploaded documents
└── README.md

🔧 API Endpoints
| Method | Endpoint   | Description                                |
| ------ | ---------- | ------------------------------------------ |
| POST   | `/upload`  | Upload one or more files to `notes/`       |
| POST   | `/reindex` | Rebuild FAISS index from documents         |
| POST   | `/ask`     | Query with `{query, persona}`              |
| GET    | `/preview` | Render a source document preview           |
| GET    | `/stats`   | Return file count, chunk count, model name |

🛠 Requirements

Python 3.9+

OpenAI API key

Recommended: virtual environment (venv or conda)

📝 Development Notes

The index is rebuilt manually by hitting Rebuild Index in the UI or calling /reindex.

PDF ingestion is currently limited to preview (not chunked/embedded by default).

Add more personas by editing the PERSONAS dict in main.py.

📜 License

MIT License © 2025 Jade Bomar-Mitchell

🙋 Contributing

Issues and pull requests are welcome. Feel free to fork and adapt for your own knowledge workflows.

