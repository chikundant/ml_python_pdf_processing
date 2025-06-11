from fastapi import APIRouter
import os
from fastapi import FastAPI, UploadFile, File, Form
from models.document import SessionLocal, Document
from utils.extract_test import extract_text_from_pdf

router = APIRouter()


@router.post("/api/files/")
async def upload_file(file: UploadFile = File(...)):
    path = f"documents/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())

    text = extract_text_from_pdf(path)
    db = SessionLocal()
    doc = Document(filename=file.filename, content=text)
    db.add(doc)
    db.commit()
    db.close()
    return {"status": "uploaded"}

@router.get("/api/files/")
def list_files():
    db = SessionLocal()
    files = db.query(Document).all()
    return [{"id": d.id, "filename": d.filename} for d in files]