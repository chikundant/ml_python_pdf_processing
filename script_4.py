#!/usr/bin/env python3
"""
Simple AI Knowledge Base - Single File Implementation
Run with: python knowledge_base.py
Then visit: http://localhost:8000/docs for API documentation
"""

import os
import re
import json
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import sqlite3
from io import BytesIO
import tempfile

# FastAPI and web dependencies
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# ML dependencies
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# PDF processing
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("Warning: PyPDF2 not installed. PDF support disabled.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
DB_PATH = "knowledge_base.db"

def init_database():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create documents table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            content TEXT NOT NULL,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create document chunks table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            FOREIGN KEY (document_id) REFERENCES documents (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Pydantic models
class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    confidence: float
    source_documents: List[str]
    total_relevant_chunks: int

class DocumentInfo(BaseModel):
    id: int
    filename: str
    upload_date: str
    chunks_count: int

# ML Engine
class SimpleMLEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.8,
            min_df=1  # Lower for small datasets
        )
        self.is_fitted = False
        
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not text:
            return ""
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        # Keep only letters, numbers and basic punctuation
        text = re.sub(r'[^\w\s.,!?;:\-]', ' ', text)
        return text.strip().lower()
    
    def chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        if len(words) <= chunk_size:
            return [text]
        
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.strip()) > 20:  # Skip very short chunks
                chunks.append(chunk)
            if i + chunk_size >= len(words):
                break
        
        return chunks
    
    def fit_vectorizer(self, all_chunks: List[str]):
        """Fit TF-IDF vectorizer on all chunks"""
        if not all_chunks:
            logger.warning("No chunks to fit vectorizer")
            return
        
        preprocessed_chunks = [self.preprocess_text(chunk) for chunk in all_chunks]
        preprocessed_chunks = [chunk for chunk in preprocessed_chunks if chunk]
        
        if preprocessed_chunks:
            self.vectorizer.fit(preprocessed_chunks)
            self.is_fitted = True
            logger.info(f"Fitted vectorizer on {len(preprocessed_chunks)} chunks")
    
    def find_relevant_chunks(self, query: str, chunks_data: List[Dict], top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Find most relevant chunks for the query"""
        if not chunks_data or not self.is_fitted:
            return []
        
        # Preprocess query
        query_clean = self.preprocess_text(query)
        if not query_clean:
            return []
        
        try:
            # Vectorize query
            query_vector = self.vectorizer.transform([query_clean])
            
            # Vectorize all chunks
            chunk_texts = [self.preprocess_text(chunk['chunk_text']) for chunk in chunks_data]
            chunk_texts = [text for text in chunk_texts if text]  # Filter empty
            
            if not chunk_texts:
                return []
            
            chunk_vectors = self.vectorizer.transform(chunk_texts)
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_vector, chunk_vectors)[0]
            
            # Get top k most similar chunks
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if idx < len(chunks_data) and similarities[idx] > 0.05:  # Relevance threshold
                    results.append((chunks_data[idx], float(similarities[idx])))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in find_relevant_chunks: {str(e)}")
            return []
    
    def generate_answer(self, query: str, relevant_chunks: List[Tuple[Dict, float]]) -> Dict:
        """Generate answer from relevant chunks"""
        if not relevant_chunks:
            return {
                "answer": "I couldn't find relevant information to answer your question. Please try rephrasing or check if documents are uploaded.",
                "confidence": 0.0,
                "source_documents": [],
                "total_relevant_chunks": 0
            }
        
        # Use the most relevant chunk as the primary answer
        best_chunk, best_score = relevant_chunks[0]
        answer_text = best_chunk['chunk_text']
        
        # Combine information from top chunks if multiple are very relevant
        if len(relevant_chunks) > 1 and relevant_chunks[1][1] > 0.3:
            # Add information from second best chunk if it's also highly relevant
            second_chunk = relevant_chunks[1][0]['chunk_text']
            if len(answer_text + second_chunk) < 1000:  # Keep answer reasonable length
                answer_text = f"{answer_text}\n\nAdditional context: {second_chunk[:200]}..."
        
        # Truncate if too long
        if len(answer_text) > 800:
            sentences = answer_text.split('.')
            answer_text = '. '.join(sentences[:4]) + '.'
        
        # Get source documents
        source_docs = list(set([
            chunk[0].get('document_filename', 'Unknown') 
            for chunk in relevant_chunks[:3]  # Top 3 sources
        ]))
        
        return {
            "answer": answer_text,
            "confidence": float(best_score),
            "source_documents": source_docs,
            "total_relevant_chunks": len(relevant_chunks)
        }

# File processing utilities
class FileProcessor:
    @staticmethod
    def extract_text_from_file(file_content: bytes, filename: str) -> str:
        """Extract text from uploaded file"""
        if filename.lower().endswith('.pdf'):
            if not PDF_SUPPORT:
                raise ValueError("PDF support not available. Please install PyPDF2.")
            return FileProcessor._extract_from_pdf(file_content)
        elif filename.lower().endswith(('.txt', '.md')):
            try:
                return file_content.decode('utf-8')
            except UnicodeDecodeError:
                return file_content.decode('utf-8', errors='ignore')
        else:
            raise ValueError(f"Unsupported file type: {filename}")
    
    @staticmethod
    def _extract_from_pdf(pdf_content: bytes) -> str:
        """Extract text from PDF content"""
        pdf_file = BytesIO(pdf_content)
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting PDF: {str(e)}")
            raise ValueError(f"Could not extract text from PDF: {str(e)}")

# Database operations
class DatabaseManager:
    @staticmethod
    def save_document(filename: str, content: str) -> int:
        """Save document and return its ID"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO documents (filename, content) VALUES (?, ?)",
            (filename, content)
        )
        doc_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return doc_id
    
    @staticmethod
    def save_chunks(document_id: int, chunks: List[str]):
        """Save document chunks"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        for i, chunk in enumerate(chunks):
            cursor.execute(
                "INSERT INTO document_chunks (document_id, chunk_text, chunk_index) VALUES (?, ?, ?)",
                (document_id, chunk, i)
            )
        
        conn.commit()
        conn.close()
    
    @staticmethod
    def get_all_documents() -> List[Dict]:
        """Get all documents info"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT d.id, d.filename, d.upload_date, COUNT(c.id) as chunks_count
            FROM documents d
            LEFT JOIN document_chunks c ON d.id = c.document_id
            GROUP BY d.id, d.filename, d.upload_date
            ORDER BY d.upload_date DESC
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                "id": row[0],
                "filename": row[1],
                "upload_date": row[2],
                "chunks_count": row[3]
            }
            for row in results
        ]
    
    @staticmethod
    def get_all_chunks() -> List[Dict]:
        """Get all chunks with document info"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT c.chunk_text, c.chunk_index, d.filename, d.id
            FROM document_chunks c
            JOIN documents d ON c.document_id = d.id
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                "chunk_text": row[0],
                "chunk_index": row[1],
                "document_filename": row[2],
                "document_id": row[3]
            }
            for row in results
        ]
    
    @staticmethod
    def delete_document(doc_id: int) -> bool:
        """Delete document and its chunks"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Delete chunks first
        cursor.execute("DELETE FROM document_chunks WHERE document_id = ?", (doc_id,))
        # Delete document
        cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted

# Initialize components
init_database()
ml_engine = SimpleMLEngine()

# FastAPI app
app = FastAPI(title="Simple AI Knowledge Base", version="1.0.0")

# Web interface HTML
WEB_INTERFACE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Knowledge Base</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        .container { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .section { border: 1px solid #ddd; padding: 20px; border-radius: 8px; }
        .upload-area { border: 2px dashed #ccc; padding: 20px; text-align: center; border-radius: 8px; }
        .file-list { max-height: 300px; overflow-y: auto; }
        .file-item { padding: 10px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; }
        .chat-area { height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; }
        .message { margin-bottom: 15px; padding: 10px; border-radius: 8px; }
        .user-message { background: #e3f2fd; text-align: right; }
        .ai-message { background: #f5f5f5; }
        .confidence { font-size: 0.9em; color: #666; }
        .sources { font-size: 0.8em; color: #888; margin-top: 5px; }
        input, textarea, button { padding: 10px; margin: 5px 0; }
        button { background: #2196F3; color: white; border: none; cursor: pointer; border-radius: 4px; }
        button:hover { background: #1976D2; }
        .delete-btn { background: #f44336; padding: 5px 10px; font-size: 0.8em; }
        .delete-btn:hover { background: #d32f2f; }
    </style>
</head>
<body>
    <h1>🤖 AI Knowledge Base</h1>
    
    <div class="container">
        <div class="section">
            <h2>📁 Document Management</h2>
            
            <div class="upload-area">
                <input type="file" id="fileInput" accept=".txt,.pdf,.md" multiple>
                <br><br>
                <button onclick="uploadFiles()">Upload Files</button>
            </div>
            
            <h3>Uploaded Documents:</h3>
            <div class="file-list" id="fileList">
                Loading...
            </div>
        </div>
        
        <div class="section">
            <h2>💬 Ask Questions</h2>
            
            <div class="chat-area" id="chatArea">
                <div class="message ai-message">
                    👋 Hello! Upload some documents and ask me questions about them.
                </div>
            </div>
            
            <div>
                <textarea id="questionInput" placeholder="Ask a question about your documents..." 
                         style="width: 100%; height: 60px; resize: vertical;"></textarea>
                <br>
                <button onclick="askQuestion()" style="width: 100%;">Ask Question</button>
            </div>
        </div>
    </div>

    <script>
        // Load files on page load
        loadFiles();
        
        async function uploadFiles() {
            const fileInput = document.getElementById('fileInput');
            const files = fileInput.files;
            
            if (files.length === 0) {
                alert('Please select files to upload');
                return;
            }
            
            for (let file of files) {
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const response = await fetch('/api/files/', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (response.ok) {
                        console.log(`Uploaded: ${file.name}`);
                    } else {
                        const error = await response.json();
                        alert(`Error uploading ${file.name}: ${error.detail}`);
                    }
                } catch (error) {
                    alert(`Error uploading ${file.name}: ${error.message}`);
                }
            }
            
            fileInput.value = '';
            loadFiles();
        }
        
        async function loadFiles() {
            try {
                const response = await fetch('/api/files/');
                const files = await response.json();
                
                const fileList = document.getElementById('fileList');
                if (files.length === 0) {
                    fileList.innerHTML = '<p>No documents uploaded yet.</p>';
                } else {
                    fileList.innerHTML = files.map(file => `
                        <div class="file-item">
                            <div>
                                <strong>${file.filename}</strong><br>
                                <small>${file.upload_date} • ${file.chunks_count} chunks</small>
                            </div>
                            <button class="delete-btn" onclick="deleteFile(${file.id})">Delete</button>
                        </div>
                    `).join('');
                }
            } catch (error) {
                document.getElementById('fileList').innerHTML = '<p>Error loading files</p>';
            }
        }
        
        async function deleteFile(fileId) {
            if (!confirm('Are you sure you want to delete this document?')) return;
            
            try {
                const response = await fetch(`/api/files/${fileId}/`, {
                    method: 'DELETE'
                });
                
                if (response.ok) {
                    loadFiles();
                } else {
                    alert('Error deleting file');
                }
            } catch (error) {
                alert('Error deleting file');
            }
        }
        
        async function askQuestion() {
            const questionInput = document.getElementById('questionInput');
            const question = questionInput.value.trim();
            
            if (!question) {
                alert('Please enter a question');
                return;
            }
            
            // Add user message to chat
            addMessage(question, 'user');
            questionInput.value = '';
            
            // Add loading message
            const loadingDiv = addMessage('Thinking...', 'ai');
            
            try {
                const response = await fetch('/api/ask/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ question: question })
                });
                
                const result = await response.json();
                
                // Remove loading message
                loadingDiv.remove();
                
                if (response.ok) {
                    const answerHtml = `
                        ${result.answer}
                        <div class="confidence">Confidence: ${(result.confidence * 100).toFixed(1)}%</div>
                        <div class="sources">Sources: ${result.source_documents.join(', ')}</div>
                    `;
                    addMessage(answerHtml, 'ai');
                } else {
                    addMessage(`Error: ${result.detail}`, 'ai');
                }
            } catch (error) {
                loadingDiv.remove();
                addMessage(`Error: ${error.message}`, 'ai');
            }
        }
        
        function addMessage(content, type) {
            const chatArea = document.getElementById('chatArea');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}-message`;
            messageDiv.innerHTML = content;
            chatArea.appendChild(messageDiv);
            chatArea.scrollTop = chatArea.scrollHeight;
            return messageDiv;
        }
        
        // Allow Enter to submit question
        document.getElementById('questionInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                askQuestion();
            }
        });
    </script>
</body>
</html>
"""

# API Routes
@app.get("/", response_class=HTMLResponse)
async def get_web_interface():
    """Serve the web interface"""
    return WEB_INTERFACE

@app.post("/api/files/")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a file"""
    try:
        # Read file content
        content = await file.read()
        
        # Extract text
        text_content = FileProcessor.extract_text_from_file(content, file.filename)
        
        if not text_content.strip():
            raise HTTPException(status_code=400, detail="File appears to be empty or unreadable")
        
        # Save document
        doc_id = DatabaseManager.save_document(file.filename, text_content)
        
        # Process into chunks
        chunks = ml_engine.chunk_text(text_content)
        DatabaseManager.save_chunks(doc_id, chunks)
        
        # Refit vectorizer with all documents
        await refit_vectorizer()
        
        logger.info(f"Processed file: {file.filename}, {len(chunks)} chunks")
        
        return {
            "message": "File uploaded successfully",
            "filename": file.filename,
            "document_id": doc_id,
            "chunks_created": len(chunks)
        }
        
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/files/")
async def list_files():
    """List all uploaded files"""
    try:
        documents = DatabaseManager.get_all_documents()
        return documents
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/files/{file_id}/")
async def delete_file(file_id: int):
    """Delete a file and its chunks"""
    try:
        deleted = DatabaseManager.delete_document(file_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Refit vectorizer after deletion
        await refit_vectorizer()
        
        return {"message": "File deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ask/", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question and get an AI-generated answer"""
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Get all chunks
        chunks_data = DatabaseManager.get_all_chunks()
        
        if not chunks_data:
            raise HTTPException(
                status_code=404, 
                detail="No documents found. Please upload some documents first."
            )
        
        # Ensure vectorizer is fitted
        if not ml_engine.is_fitted:
            await refit_vectorizer()
        
        # Find relevant chunks
        relevant_chunks = ml_engine.find_relevant_chunks(
            request.question, 
            chunks_data, 
            top_k=5
        )
        
        # Generate answer
        response = ml_engine.generate_answer(request.question, relevant_chunks)
        
        logger.info(f"Question: {request.question}, Confidence: {response['confidence']:.3f}")
        
        return QuestionResponse(**response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

async def refit_vectorizer():
    """Refit the vectorizer with all current chunks"""
    try:
        chunks_data = DatabaseManager.get_all_chunks()
        if chunks_data:
            all_chunks = [chunk['chunk_text'] for chunk in chunks_data]
            ml_engine.fit_vectorizer(all_chunks)
    except Exception as e:
        logger.error(f"Error refitting vectorizer: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "vectorizer_fitted": ml_engine.is_fitted,
        "documents_count": len(DatabaseManager.get_all_documents()),
        "pdf_support": PDF_SUPPORT
    }

if __name__ == "__main__":
    print("🚀 Starting AI Knowledge Base Server...")
    print("📝 Visit http://localhost:8000 for the web interface")
    print("📚 Visit http://localhost:8000/docs for API documentation")
    print("💡 Install PyPDF2 for PDF support: pip install PyPDF2")
    
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")