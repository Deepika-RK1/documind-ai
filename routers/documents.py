# routers/documents.py — document upload + management
from fastapi import APIRouter, Depends, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
import os
import uuid
from database import get_db
from models import User, Document
from schemas import DocumentResponse, UploadResponse
from dependencies import get_current_user
from exceptions import (
    DocumentNotFoundException,
    FileTooLargeException,
    InvalidFileTypeException
)
from rag_pipeline import process_document
from config import settings
import chromadb

router = APIRouter(prefix="/documents", tags=["Documents"])

@router.post("/upload", response_model=UploadResponse, status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    file:             UploadFile = File(...),
    db:               Session    = Depends(get_db),
    current_user:     User       = Depends(get_current_user)
):
    """Upload a PDF document for processing"""

    # Validate file type
    allowed = [".pdf", ".txt", ".docx"]
    if not any(file.filename.lower().endswith(ext) for ext in allowed):
        raise InvalidFileTypeException(file.filename.split(".")[-1])
    
    
    

    # Read file and check size
    file_bytes = await file.read()
    size_mb    = len(file_bytes) / (1024 * 1024)

    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise FileTooLargeException(size_mb, settings.MAX_FILE_SIZE_MB)

    # Save file with UUID name
    unique_filename = f"{uuid.uuid4()}.pdf"
    file_path = os.path.join(settings.UPLOAD_DIR, unique_filename).replace("\\", "/")

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file_bytes)

    # Create DB record
    doc = Document(
        filename       = unique_filename,
        original_name  = file.filename,
        file_path      = file_path,
        status         = "processing",
        owner_username = current_user.username
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # Trigger background processing
    background_tasks.add_task(
        process_document,
        document_id = doc.id,
        file_path   = file_path,
        username    = current_user.username,
        db          = db
    )

    return UploadResponse(
        document_id   = doc.id,
        original_name = doc.original_name,
        status        = "processing",
        message       = "Document uploaded successfully. Processing in background."
    )

@router.get("/", response_model=List[DocumentResponse])
def list_documents(
    db:           Session = Depends(get_db),
    current_user: User    = Depends(get_current_user)
):
    """List all documents for current user"""
    docs = db.query(Document)\
             .filter(Document.owner_username == current_user.username)\
             .order_by(Document.uploaded_at.desc())\
             .all()
    return docs

@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id:  int,
    db:           Session = Depends(get_db),
    current_user: User    = Depends(get_current_user)
):
    """Get single document details"""
    doc = db.query(Document).filter(
        Document.id            == document_id,
        Document.owner_username == current_user.username
    ).first()

    if not doc:
        raise DocumentNotFoundException(document_id)

    return doc

@router.delete("/{document_id}")
def delete_document(
    document_id:  int,
    db:           Session = Depends(get_db),
    current_user: User    = Depends(get_current_user)
):
    """Delete document + embeddings + chat history"""
    doc = db.query(Document).filter(
        Document.id            == document_id,
        Document.owner_username == current_user.username
    ).first()

    if not doc:
        raise DocumentNotFoundException(document_id)

    # Delete ChromaDB collection
    try:
        chroma = chromadb.PersistentClient(path=settings.CHROMA_DIR)
        chroma.delete_collection(f"doc_{document_id}_{current_user.username}")
    except:
        pass   # collection may not exist if processing failed

    # Delete file from disk
    try:
        if os.path.exists(doc.file_path):
            os.remove(doc.file_path)
    except:
        pass

    # Delete DB record (cascade deletes chat history)
    db.delete(doc)
    db.commit()

    return {"message": f"Document '{doc.original_name}' deleted successfully"}