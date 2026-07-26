# main.py — FastAPI application entry point
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import logging
from database import engine, Base
from routers import auth, documents, chat
from config import settings
from fastapi.staticfiles import StaticFiles
import os

# Create all DB tables on startup
Base.metadata.create_all(bind=engine)

# Logging
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("documind")

# ── App ──────────────────────────────────────────────────
app = FastAPI(
    title       = "DocuMind AI",
    description = "AI-powered document Q&A platform using RAG",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc"
)

# ── Middleware ───────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"]
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request with timing"""
    start    = time.time()
    response = await call_next(request)
    duration = round(time.time() - start, 3)

    logger.info(
        f"{request.method} {request.url.path} "
        f"→ {response.status_code} ({duration}s)"
    )
    return response

# ── Global exception handler ─────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code = 500,
        content     = {
            "error":  "Internal server error",
            "detail": "Something went wrong. Please try again."
        }
    )

# ── Routers ──────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(chat.router)

# ── Root endpoints ───────────────────────────────────────
@app.get("/")
def root():
    return {
        "app":     "DocuMind AI",
        "status":  "healthy",
        "version": "1.0.0",
        "docs":    "/docs"
    }

@app.get("/health")
def health_check():
    """Health check for monitoring"""
    import chromadb as _chromadb
    from sqlalchemy import text
    from database import SessionLocal

    # Check database
    db_status = "disconnected"
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db_status = "connected"
        db.close()
    except:
        pass

    # Check ChromaDB
    chroma_status = "disconnected"
    try:
        client = _chromadb.PersistentClient(path=settings.CHROMA_DIR)
        chroma_status = "connected"
    except:
        pass

    return {
        "status":   "healthy",
        "database": db_status,
        "chromadb": chroma_status
    }
if os.path.exists("frontend"):
    app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")