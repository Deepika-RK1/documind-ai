# DocuMind AI 🧠

> AI-powered document Q&A platform with production-grade RAG pipeline

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-orange.svg)](https://trychroma.com)

## What It Does

Upload any PDF, DOCX, or TXT file and ask questions about it in natural language. DocuMind AI uses Retrieval Augmented Generation (RAG) to find relevant content and generate accurate, grounded answers — with zero hallucination.

## Live Demo

> Coming soon — deploying to Railway.app

## My Enhancements (forked from [Anand-1216/documind-ai](https://github.com/Anand-1216/documind-ai))

- ✅ Built complete frontend UI — login, register, dashboard with real-time chat
- ✅ Added drag-and-drop PDF upload with live processing status
- ✅ Extended file support from PDF-only to PDF, DOCX, and TXT
- ✅ Fixed Windows path bug in PyMuPDF file handling
- ✅ Fixed chat history bug — clear chat now deletes from DB permanently
- ✅ Source chunk viewer — click any source to see retrieved text
- ✅ Chat history loads automatically when switching documents

## Architecture

```
File Upload → PyMuPDF/python-docx extraction → Chunking (800 chars, 100 overlap)
→ SentenceTransformer embeddings (384-dim, runs locally — no API cost)
→ ChromaDB vector storage (per-user collections)
→ Question embedding → Cosine similarity retrieval (top 4 chunks)
→ Groq Llama 3.1 answer generation with strict hallucination prevention
→ Answer + source chunks + retrieval score returned
→ All Q&A saved to SQLite chat history
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, FastAPI, SQLAlchemy |
| Vector DB | ChromaDB |
| Embeddings | SentenceTransformers (all-MiniLM-L6-v2) |
| LLM | Groq — Llama 3.1 8B |
| Auth | JWT (python-jose + bcrypt) |
| Database | SQLite |
| Frontend | HTML, CSS, Vanilla JS |
| PDF parsing | PyMuPDF, python-docx |

## Features

- 🔐 JWT secured multi-user authentication
- 📄 Multi-format support — PDF, DOCX, TXT
- ⚡ Async background document processing
- 🧠 Local embeddings — no embedding API cost
- 💬 Real-time Q&A with source attribution
- 📊 Retrieval score shown per answer
- 🗂️ Chat history saved and restored per document
- 🛡️ 0% hallucination via strict prompt engineering
- 🖥️ Clean responsive frontend — no framework needed

## Project Structure

```
documind-ai/
├── main.py              ← FastAPI app + middleware + routes
├── database.py          ← SQLAlchemy engine + session
├── models.py            ← User, Document, ChatHistory tables
├── schemas.py           ← Pydantic request/response models
├── auth.py              ← JWT token creation + verification
├── dependencies.py      ← get_current_user dependency
├── rag_pipeline.py      ← Complete RAG pipeline (8 functions)
├── exceptions.py        ← Custom HTTP exceptions
├── config.py            ← Settings from .env
├── routers/
│   ├── auth.py          ← /users/register + /users/login
│   ├── documents.py     ← /documents/ CRUD endpoints
│   └── chat.py          ← /chat/ Q&A endpoints
├── frontend/
│   ├── index.html       ← Login page
│   ├── register.html    ← Register page
│   ├── dashboard.html   ← Main app — upload + chat
│   └── style.css        ← Shared styles
├── requirements.txt
├── .env.example
└── README.md
```

## API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | /users/register | ❌ | Create account |
| POST | /users/login | ❌ | Get JWT token |
| POST | /documents/upload | ✅ | Upload PDF/DOCX/TXT |
| GET | /documents/ | ✅ | List your documents |
| GET | /documents/{id} | ✅ | Document status |
| DELETE | /documents/{id} | ✅ | Delete document |
| POST | /chat/{doc_id} | ✅ | Ask a question |
| GET | /chat/{doc_id}/history | ✅ | Chat history |
| DELETE | /chat/{chat_id} | ✅ | Delete chat entry |
| GET | /health | ❌ | Health check |

## Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/documind-ai.git
cd documind-ai

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env and add your API keys

# Create folders
mkdir uploads chroma_db

# Run
uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000/frontend/index.html`

## Environment Variables

```env
SECRET_KEY=your-secret-key-minimum-32-characters
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
GEMINI_API_KEY=your-gemini-api-key
GROQ_API_KEY=your-groq-api-key
DATABASE_URL=sqlite:///./documind.db
UPLOAD_DIR=uploads
CHROMA_DIR=chroma_db
MAX_FILE_SIZE_MB=10
```

Get free API keys:
- Groq (LLM): [console.groq.com](https://console.groq.com) — free, no card needed
- Gemini (optional): [aistudio.google.com](https://aistudio.google.com) — free tier

## RAG Pipeline Details

```
1. extract_text_from_file()  → PyMuPDF/docx/txt extraction
2. chunk_text()              → 800 char chunks, 100 overlap
3. create_embeddings()       → SentenceTransformer local model
4. store_in_chromadb()       → per-user isolated collections
5. process_document()        → background task, updates DB status
6. retrieve_relevant_chunks()→ top-4 cosine similarity search
7. generate_answer()         → Groq LLM with strict prompt
8. answer_question()         → orchestrates retrieve + generate
```

## What I Learned Building This

- How RAG pipelines work end to end — chunking, embedding, retrieval
- Vector databases — ChromaDB with cosine similarity search
- JWT authentication in FastAPI with proper security patterns
- Async background task processing for long-running operations
- Production debugging — fixed Windows path bugs, rate limit handling
- Frontend development without frameworks — pure HTML/CSS/JS
- How strict prompt engineering eliminates hallucination

## Upcoming Enhancements

- [ ] Conversation memory — follow-up questions work naturally
- [ ] Answer feedback — thumbs up/down to track quality
- [ ] Document summarisation — auto-summary on upload
- [ ] Export chat history as PDF
- [ ] Deploy to Railway.app — live demo URL
- [ ] Multi-document chat — query across multiple files

## Contributing

Built collaboratively with [@Anand-1216](https://github.com/Anand-1216).
Pull requests welcome!
