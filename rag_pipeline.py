# rag_pipeline.py — top imports (replace google import)
import os
import json
import time
import uuid
import fitz
import chromadb
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from groq import Groq                    # ← changed
from sqlalchemy.orm import Session
from models import Document
from config import settings

# Models
print("🔧 Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Embedding model loaded")

groq_client   = Groq(api_key=settings.GROQ_API_KEY)   # ← changed
chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DIR) 

# ── FUNCTION 1: Extract text from PDF ───────────────────
def extract_text_from_file(file_path: str, file_ext: str):
    """Support PDF, DOCX and TXT files"""

    if file_ext == "pdf":
        return extract_text_from_pdf(file_path)

    elif file_ext == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return text, 1

    elif file_ext == "docx":
        import docx
        doc   = docx.Document(file_path)
        text  = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        return text, 1

    else:
        raise ValueError(f"Unsupported file type: {file_ext}")
def extract_text_from_pdf(file_path: str) -> tuple:
    # Fix Windows backslash path issues
    file_path = os.path.normpath(file_path)

    doc        = fitz.open(file_path)
    page_count = len(doc)
    full_text  = ""

    for page_num in range(page_count):
        page      = doc[page_num]
        page_text = page.get_text()
        if page_text.strip():
            full_text += f"\n[Page {page_num + 1}]\n{page_text}"

    doc.close()

    if not full_text.strip():
        raise ValueError("No text could be extracted from this PDF. It may be scanned or image-based.")

    return full_text.strip(), page_count


# ── FUNCTION 2: Chunk text ───────────────────────────────
def chunk_text(
    text:       str,
    chunk_size: int = 800,
    overlap:    int = 100
) -> List[str]:
    """
    Split text into overlapping chunks.
    chunk_size=800 chars — optimised from Day 10 testing
    overlap=100 — preserves context at boundaries
    """
    chunks = []
    start  = 0

    while start < len(text):
        end   = start + chunk_size
        chunk = text[start:end]

        # Skip chunks that are too small to be meaningful
        if len(chunk.strip()) >= 50:
            chunks.append(chunk.strip())

        start += chunk_size - overlap

    return chunks


# ── FUNCTION 3: Create embeddings ───────────────────────
def create_embeddings(chunks: List[str]) -> List[List[float]]:
    """
    Embed chunks using SentenceTransformer.
    Local model — no API calls, no rate limits, no cost.
    Returns list of embedding vectors.
    """
    embeddings = embedding_model.encode(
        chunks,
        show_progress_bar=False,
        convert_to_numpy=True
    )
    return embeddings.tolist()


# ── FUNCTION 4: Store in ChromaDB ───────────────────────
def store_in_chromadb(
    document_id: int,
    chunks:      List[str],
    embeddings:  List[List[float]],
    username:    str
) -> None:
    """
    Store chunks and embeddings in ChromaDB.
    Each document gets its own collection: doc_{id}_{username}
    This ensures per-user document isolation.
    """
    collection_name = f"doc_{document_id}_{username}"

    # Delete old collection if exists (for re-processing)
    try:
        chroma_client.delete_collection(collection_name)
    except:
        pass

    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    # Store in batches of 100
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_embeds = embeddings[i:i+batch_size]
        batch_ids    = [
            f"doc_{document_id}_chunk_{i+j}"
            for j in range(len(batch_chunks))
        ]
        batch_metas  = [
            {
                "chunk_index": i + j,
                "document_id": document_id
            }
            for j in range(len(batch_chunks))
        ]

        collection.add(
            documents=batch_chunks,
            embeddings=batch_embeds,
            ids=batch_ids,
            metadatas=batch_metas
        )


# ── FUNCTION 5: Process document (background task) ──────
def process_document(
    document_id: int,
    file_path:   str,
    username:    str,
    db:          Session
) -> None:
    """
    Background task — full document processing pipeline.
    Called after PDF upload. Updates DB status when done.

    Flow:
    extract text → chunk → embed → store → update status
    """
    file_path = os.path.normpath(file_path)
    
    doc = db.query(Document).filter(Document.id == document_id).first()

    try:
        # Step 1: Extract text
        print(f"📄 [{document_id}] Extracting text...")
        file_ext = file_path.split(".")[-1].lower()
        full_text, page_count = extract_text_from_file(file_path, file_ext)
        print(f"   Extracted {len(full_text)} chars from {page_count} pages")

        # Step 2: Chunk
        print(f"✂️  [{document_id}] Chunking...")
        chunks      = chunk_text(full_text)
        chunk_count = len(chunks)
        print(f"   Created {chunk_count} chunks")

        # Step 3: Create embeddings
        print(f"🧠 [{document_id}] Creating embeddings...")
        embeddings = create_embeddings(chunks)
        print(f"   Created {len(embeddings)} embeddings")

        # Step 4: Store in ChromaDB
        print(f"💾 [{document_id}] Storing in ChromaDB...")
        store_in_chromadb(document_id, chunks, embeddings, username)
        print(f"   Stored successfully")

        # Step 5: Update DB — mark as ready
        doc.status      = "ready"
        doc.page_count  = page_count
        doc.chunk_count = chunk_count
        db.commit()
        print(f"✅ [{document_id}] Processing complete!")

    except Exception as e:
        error_msg = str(e)

        # Give friendlier messages for common errors
        if "Failed to open" in error_msg:
            error_msg = "Could not open the file. Please try uploading again."
        elif "No text could be extracted" in error_msg:
            error_msg = "This PDF appears to be scanned or image-based. Please use a text-based PDF."
        elif "No such file" in error_msg:
            error_msg = "File not found. Please try uploading again."

        print(f"❌ [{document_id}] Processing failed: {error_msg}")
        if doc:
            doc.status        = "failed"
            doc.error_message = error_msg[:500]
            db.commit()
    
    
    
    
    
    


# ── FUNCTION 6: Retrieve relevant chunks ────────────────
def retrieve_relevant_chunks(
    document_id: int,
    question:    str,
    username:    str,
    n_results:   int = 4
) -> Tuple[List[str], List[float]]:
    """
    Embed question and retrieve top n_results chunks.
    Returns: (chunks, distances)
    """
    collection_name = f"doc_{document_id}_{username}"

    try:
        collection = chroma_client.get_collection(collection_name)
    except Exception:
        raise ValueError(f"No embeddings found for document {document_id}. Was it processed successfully?")

    # Embed the question using same model
    question_embedding = embedding_model.encode([question])[0].tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(n_results, collection.count())
    )

    chunks    = results["documents"][0]
    distances = results["distances"][0]

    return chunks, distances


# ── FUNCTION 7: Generate answer ──────────────────────────
# generate_answer function — only LLM call section changes
def generate_answer(
    question:       str,
    context_chunks: List[str],
    distances:      List[float],
    document_name:  str
) -> Dict:

    context_text = ""
    for i, chunk in enumerate(context_chunks):
        context_text += f"\n[Source {i+1}]:\n{chunk}\n"

    similarities    = [1 - d for d in distances]
    retrieval_score = float(np.mean(similarities)) if similarities else 0.0

    prompt = f"""You are DocuMind AI, an intelligent document assistant.

STRICT RULES:
1. Answer ONLY using the provided document excerpts below
2. If the answer is not in the excerpts, say exactly: "I don't have that information in this document."
3. Never use outside knowledge or make assumptions
4. Always be specific and cite which source you're drawing from
5. Keep your answer clear and concise

Document: {document_name}

Excerpts from the document:
{context_text}

Question: {question}

Answer (based strictly on the document excerpts above):"""

    # ── Groq LLM call with retry ─────────────────────────
    max_retries = 3
    answer      = "I encountered an error generating the answer. Please try again."

    for attempt in range(max_retries):
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1000
            )
            answer = response.choices[0].message.content.strip()
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait = (attempt + 1) * 10
                print(f"⚠️ Groq error (attempt {attempt+1}): {str(e)[:50]}. Waiting {wait}s...")
                time.sleep(wait)

    sources = [
        {"content": chunk[:300], "chunk_index": i}
        for i, chunk in enumerate(context_chunks)
    ]

    return {
        "answer":          answer,
        "sources":         sources,
        "retrieval_score": round(retrieval_score, 4)
    }

# ── FUNCTION 8: Orchestrator ─────────────────────────────
def answer_question(
    document_id: int,
    question:    str,
    username:    str,
    document_name: str
) -> Dict:
    """
    Main orchestrator — ties retrieve + generate together.
    Called by the chat router.
    """
    # Step 1: Retrieve relevant chunks
    chunks, distances = retrieve_relevant_chunks(
        document_id=document_id,
        question=question,
        username=username,
        n_results=4
    )

    if not chunks:
        return {
            "answer":          "I couldn't find any relevant content in this document.",
            "sources":         [],
            "retrieval_score": 0.0
        }

    # Step 2: Generate answer
    result = generate_answer(
        question=question,
        context_chunks=chunks,
        distances=distances,
        document_name=document_name
    )

    return result