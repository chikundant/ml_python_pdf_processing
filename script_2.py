#!/usr/bin/env python3
"""
Intelligent AI Knowledge Base - Enhanced Version
Features: Context-aware responses, document understanding, human-like answers
Run with: python knowledge_base.py
"""

import re
import json
import logging
from typing import List, Dict, Tuple
import sqlite3
from io import BytesIO

# FastAPI and web dependencies
from fastapi import FastAPI, HTTPException, UploadFile, File
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
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            content TEXT NOT NULL,
            processed_content TEXT,
            document_type TEXT,
            metadata TEXT,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create document chunks table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            chunk_type TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            keywords TEXT,
            FOREIGN KEY (document_id) REFERENCES documents (id)
        )
    """)

    conn.commit()
    conn.close()


# Pydantic models
class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    answer: str
    confidence: float
    source_documents: List[str]
    context_used: List[str]


class DocumentInfo(BaseModel):
    id: int
    filename: str
    document_type: str
    upload_date: str
    chunks_count: int


# Enhanced ML Engine with context understanding
class IntelligentMLEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words="english",
            ngram_range=(1, 3),
            max_df=0.8,
            min_df=1,
        )
        self.is_fitted = False

        # Question type patterns
        self.question_patterns = {
            "what_is": [r"what is", r"what are", r"define", r"explain"],
            "how_to": [r"how to", r"how do", r"how can"],
            "when": [r"when", r"what time", r"what date"],
            "where": [r"where", r"what location", r"what place"],
            "who": [r"who", r"which person", r"what person"],
            "why": [r"why", r"what reason", r"for what reason"],
            "list": [r"list", r"what are the", r"show me", r"give me"],
            "compare": [r"compare", r"difference", r"versus", r"vs"],
            "summary": [r"summarize", r"summary", r"overview", r"brief"],
            "experience": [r"experience", r"worked", r"job", r"position"],
            "skills": [r"skills", r"abilities", r"competencies", r"expertise"],
            "education": [r"education", r"degree", r"university", r"study"],
            "contact": [r"contact", r"email", r"phone", r"address"],
        }

    def detect_question_type(self, question: str) -> str:
        """Detect the type of question being asked"""
        question_lower = question.lower()

        for q_type, patterns in self.question_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    return q_type

        return "general"

    def detect_document_type(self, content: str, filename: str) -> str:
        """Detect document type (CV, resume, report, etc.)"""
        content_lower = content.lower()
        filename_lower = filename.lower()

        # CV/Resume indicators
        cv_indicators = [
            "experience",
            "education",
            "skills",
            "resume",
            "cv",
            "curriculum",
            "employment",
            "qualifications",
            "achievements",
            "linkedin",
            "email",
        ]

        # Technical document indicators
        tech_indicators = [
            "api",
            "function",
            "class",
            "method",
            "algorithm",
            "code",
            "technical",
        ]

        # Report indicators
        report_indicators = [
            "analysis",
            "findings",
            "conclusion",
            "methodology",
            "results",
        ]

        cv_score = sum(
            1
            for indicator in cv_indicators
            if indicator in content_lower or indicator in filename_lower
        )
        tech_score = sum(
            1 for indicator in tech_indicators if indicator in content_lower
        )
        report_score = sum(
            1 for indicator in report_indicators if indicator in content_lower
        )

        if cv_score >= 3:
            return "cv"
        elif tech_score >= 2:
            return "technical"
        elif report_score >= 2:
            return "report"
        else:
            return "general"

    def extract_structured_info(self, content: str, doc_type: str) -> Dict:
        """Extract structured information based on document type"""
        info = {}

        if doc_type == "cv":
            info = self._extract_cv_info(content)
        elif doc_type == "technical":
            info = self._extract_tech_info(content)

        return info

    def _extract_cv_info(self, content: str) -> Dict:
        """Extract CV-specific information"""
        info = {
            "contact": [],
            "skills": [],
            "experience": [],
            "education": [],
            "achievements": [],
        }

        lines = content.split("\n")
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            line_lower = line.lower()

            # Detect sections
            if any(
                keyword in line_lower
                for keyword in ["experience", "employment", "work"]
            ):
                current_section = "experience"
            elif any(
                keyword in line_lower
                for keyword in ["education", "qualification", "degree"]
            ):
                current_section = "education"
            elif any(
                keyword in line_lower
                for keyword in ["skills", "competencies", "abilities"]
            ):
                current_section = "skills"
            elif any(keyword in line_lower for keyword in ["contact", "profile"]):
                current_section = "contact"

            # Extract contact info
            email_match = re.search(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", line
            )
            if email_match:
                info["contact"].append(f"Email: {email_match.group()}")

            phone_match = re.search(r"[\+]?[1-9]?[0-9]{7,14}", line)
            if phone_match and len(phone_match.group()) > 7:
                info["contact"].append(f"Phone: {phone_match.group()}")

            if "linkedin" in line_lower:
                info["contact"].append(f"LinkedIn: {line}")

            # Extract skills (look for common programming languages, tools, etc.)
            skills_keywords = [
                "python",
                "javascript",
                "java",
                "react",
                "node",
                "sql",
                "aws",
                "docker",
                "kubernetes",
                "fastapi",
                "django",
                "flask",
                "git",
                "linux",
                "machine learning",
            ]

            for skill in skills_keywords:
                if skill in line_lower and skill not in [
                    s.lower() for s in info["skills"]
                ]:
                    info["skills"].append(skill.title())

            # Add content to current section
            if current_section and line:
                if current_section == "experience" and any(
                    keyword in line_lower
                    for keyword in [
                        "company",
                        "role",
                        "position",
                        "developer",
                        "engineer",
                    ]
                ):
                    info["experience"].append(line)
                elif current_section == "education" and any(
                    keyword in line_lower
                    for keyword in [
                        "university",
                        "college",
                        "degree",
                        "bachelor",
                        "master",
                    ]
                ):
                    info["education"].append(line)

        return info

    def _extract_tech_info(self, content: str) -> Dict:
        """Extract technical document information"""
        info = {"technologies": [], "functions": [], "concepts": []}

        # Extract technical terms
        tech_terms = re.findall(
            r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b", content
        )  # CamelCase
        info["technologies"] = list(set(tech_terms[:10]))  # Limit to top 10

        return info

    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing"""
        if not text:
            return ""

        # Clean up common PDF artifacts
        text = re.sub(r"[^\w\s.,!?;:\-@()]", " ", text)
        text = re.sub(r"\s+", " ", text)

        # Remove very short words and numbers-only strings
        words = text.split()
        filtered_words = []
        for word in words:
            if len(word) > 2 and not word.isdigit():
                filtered_words.append(word)

        return " ".join(filtered_words).lower()

    def create_semantic_chunks(
        self, content: str, doc_type: str, structured_info: Dict
    ) -> List[Dict]:
        """Create semantically meaningful chunks"""
        chunks = []

        if doc_type == "cv":
            chunks = self._create_cv_chunks(content, structured_info)
        else:
            chunks = self._create_general_chunks(content)

        return chunks

    def _create_cv_chunks(self, content: str, structured_info: Dict) -> List[Dict]:
        """Create CV-specific chunks"""
        chunks = []

        # Contact information chunk
        if structured_info.get("contact"):
            contact_text = " ".join(structured_info["contact"])
            chunks.append(
                {
                    "text": contact_text,
                    "type": "contact",
                    "keywords": ["contact", "email", "phone", "linkedin"],
                }
            )

        # Skills chunk
        if structured_info.get("skills"):
            skills_text = (
                f"Skills and technologies: {', '.join(structured_info['skills'])}"
            )
            chunks.append(
                {
                    "text": skills_text,
                    "type": "skills",
                    "keywords": ["skills", "technologies", "programming", "expertise"],
                }
            )

        # Experience chunk
        if structured_info.get("experience"):
            experience_text = " ".join(structured_info["experience"])
            chunks.append(
                {
                    "text": experience_text,
                    "type": "experience",
                    "keywords": ["experience", "work", "job", "position", "role"],
                }
            )

        # Education chunk
        if structured_info.get("education"):
            education_text = " ".join(structured_info["education"])
            chunks.append(
                {
                    "text": education_text,
                    "type": "education",
                    "keywords": ["education", "degree", "university", "study"],
                }
            )

        # If no structured info, create general chunks
        if not chunks:
            chunks = self._create_general_chunks(content)

        return chunks

    def _create_general_chunks(self, content: str) -> List[Dict]:
        """Create general document chunks"""
        # Split by paragraphs first
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            if len(current_chunk + paragraph) > 400:  # Max chunk size
                if current_chunk:
                    chunks.append(
                        {
                            "text": current_chunk.strip(),
                            "type": "general",
                            "keywords": self._extract_keywords(current_chunk),
                        }
                    )
                current_chunk = paragraph
            else:
                current_chunk += " " + paragraph

        # Add the last chunk
        if current_chunk:
            chunks.append(
                {
                    "text": current_chunk.strip(),
                    "type": "general",
                    "keywords": self._extract_keywords(current_chunk),
                }
            )

        return chunks

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Simple keyword extraction
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        # Remove common words
        stop_words = {
            "the",
            "and",
            "for",
            "are",
            "but",
            "not",
            "you",
            "all",
            "can",
            "had",
            "her",
            "was",
            "one",
            "our",
            "out",
            "day",
            "get",
            "has",
            "him",
            "his",
            "how",
            "man",
            "new",
            "now",
            "old",
            "see",
            "two",
            "way",
            "who",
            "boy",
            "did",
            "its",
            "let",
            "put",
            "say",
            "she",
            "too",
            "use",
        }
        keywords = [word for word in set(words) if word not in stop_words]
        return keywords[:10]

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

    def find_relevant_chunks(
        self, query: str, chunks_data: List[Dict], top_k: int = 3
    ) -> List[Tuple[Dict, float]]:
        """Find relevant chunks with enhanced matching"""
        if not chunks_data or not self.is_fitted:
            return []

        question_type = self.detect_question_type(query)
        query_clean = self.preprocess_text(query)

        if not query_clean:
            return []

        try:
            # Vectorize query
            query_vector = self.vectorizer.transform([query_clean])

            # Prepare chunks with type-based boosting
            chunk_texts = []
            chunk_scores = []

            for chunk in chunks_data:
                chunk_text = self.preprocess_text(chunk["chunk_text"])
                chunk_texts.append(chunk_text)

                # Boost score based on question type and chunk type
                boost = 1.0
                chunk_type = chunk.get("chunk_type", "general")

                if question_type == "skills" and chunk_type == "skills":
                    boost = 2.0
                elif question_type == "experience" and chunk_type == "experience":
                    boost = 2.0
                elif question_type == "contact" and chunk_type == "contact":
                    boost = 2.0
                elif question_type == "education" and chunk_type == "education":
                    boost = 2.0

                chunk_scores.append(boost)

            # Vectorize chunks
            chunk_vectors = self.vectorizer.transform(chunk_texts)

            # Calculate cosine similarity
            similarities = cosine_similarity(query_vector, chunk_vectors)[0]

            # Apply boosts
            boosted_similarities = [
                sim * boost for sim, boost in zip(similarities, chunk_scores)
            ]

            # Get top k most similar chunks
            top_indices = np.argsort(boosted_similarities)[::-1][:top_k]

            results = []
            for idx in top_indices:
                if idx < len(chunks_data) and boosted_similarities[idx] > 0.05:
                    results.append((chunks_data[idx], float(boosted_similarities[idx])))

            return results

        except Exception as e:
            logger.error(f"Error in find_relevant_chunks: {str(e)}")
            return []

    def generate_human_response(
        self, query: str, relevant_chunks: List[Tuple[Dict, float]], doc_info: Dict
    ) -> Dict:
        """Generate human-like response based on context"""
        if not relevant_chunks:
            return {
                "answer": "I don't have enough information to answer that question. Could you try asking something else or upload more relevant documents?",
                "confidence": 0.0,
                "source_documents": [],
                "context_used": [],
            }

        question_type = self.detect_question_type(query)
        doc_type = doc_info.get("document_type", "general")

        # Generate context-aware response
        response = self._generate_contextual_answer(
            query, question_type, relevant_chunks, doc_type
        )

        # Get source documents
        source_docs = list(
            set(
                [
                    chunk[0].get("document_filename", "Unknown")
                    for chunk in relevant_chunks
                ]
            )
        )

        # Get context used
        context_used = [
            chunk[0].get("chunk_type", "general") for chunk in relevant_chunks
        ]

        return {
            "answer": response,
            "confidence": float(relevant_chunks[0][1]) if relevant_chunks else 0.0,
            "source_documents": source_docs,
            "context_used": context_used,
        }

    def _generate_contextual_answer(
        self,
        query: str,
        question_type: str,
        relevant_chunks: List[Tuple[Dict, float]],
        doc_type: str,
    ) -> str:
        """Generate contextual answer based on question type and document type"""

        # Get the best chunk
        best_chunk = relevant_chunks[0][0]
        chunk_type = best_chunk.get("chunk_type", "general")
        content = best_chunk["chunk_text"]

        # CV-specific responses
        if doc_type == "cv":
            if question_type == "skills" or "skills" in query.lower():
                return f"Based on the CV, here are the key skills and technologies: {content}"

            elif question_type == "experience" or any(
                word in query.lower()
                for word in ["work", "job", "experience", "position"]
            ):
                return f"Here's information about the work experience: {content}"

            elif question_type == "contact" or any(
                word in query.lower() for word in ["contact", "email", "phone"]
            ):
                return f"Here are the contact details: {content}"

            elif question_type == "education" or any(
                word in query.lower() for word in ["education", "degree", "university"]
            ):
                return f"Here's the educational background: {content}"

            elif question_type == "summary" or any(
                word in query.lower() for word in ["summary", "about", "overview"]
            ):
                # Combine multiple chunks for summary
                all_content = []
                for chunk, _ in relevant_chunks:
                    all_content.append(chunk["chunk_text"])

                combined_content = " ".join(all_content)
                return f"Here's a summary of the person's profile: {combined_content}"

        # General responses
        if question_type == "what_is":
            return f"Based on the document, here's what I found: {content}"

        elif question_type == "list":
            return f"Here's what I found: {content}"

        elif question_type == "summary":
            return f"Here's a summary: {content}"

        else:
            # Default response
            return f"Based on the document, {content}"


# File processing utilities
class FileProcessor:
    @staticmethod
    def extract_text_from_file(file_content: bytes, filename: str) -> str:
        """Extract text from uploaded file"""
        if filename.lower().endswith(".pdf"):
            if not PDF_SUPPORT:
                raise ValueError("PDF support not available. Please install PyPDF2.")
            return FileProcessor._extract_from_pdf(file_content)
        elif filename.lower().endswith((".txt", ".md")):
            try:
                return file_content.decode("utf-8")
            except UnicodeDecodeError:
                return file_content.decode("utf-8", errors="ignore")
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
    def save_document(
        filename: str,
        content: str,
        processed_content: str,
        doc_type: str,
        metadata: str,
    ) -> int:
        """Save document with enhanced information"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """INSERT INTO documents (filename, content, processed_content, document_type, metadata) 
               VALUES (?, ?, ?, ?, ?)""",
            (filename, content, processed_content, doc_type, metadata),
        )
        doc_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return doc_id

    @staticmethod
    def save_chunks(document_id: int, chunks: List[Dict]):
        """Save document chunks with type information"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        for i, chunk in enumerate(chunks):
            cursor.execute(
                """INSERT INTO document_chunks (document_id, chunk_text, chunk_type, chunk_index, keywords) 
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    document_id,
                    chunk["text"],
                    chunk["type"],
                    i,
                    json.dumps(chunk.get("keywords", [])),
                ),
            )

        conn.commit()
        conn.close()

    @staticmethod
    def get_all_documents() -> List[Dict]:
        """Get all documents info"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT d.id, d.filename, d.document_type, d.upload_date, COUNT(c.id) as chunks_count
            FROM documents d
            LEFT JOIN document_chunks c ON d.id = c.document_id
            GROUP BY d.id, d.filename, d.document_type, d.upload_date
            ORDER BY d.upload_date DESC
        """)

        results = cursor.fetchall()
        conn.close()

        return [
            {
                "id": row[0],
                "filename": row[1],
                "document_type": row[2],
                "upload_date": row[3],
                "chunks_count": row[4],
            }
            for row in results
        ]

    @staticmethod
    def get_all_chunks() -> List[Dict]:
        """Get all chunks with document info"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT c.chunk_text, c.chunk_type, c.chunk_index, c.keywords, d.filename, d.id, d.document_type
            FROM document_chunks c
            JOIN documents d ON c.document_id = d.id
        """)

        results = cursor.fetchall()
        conn.close()

        return [
            {
                "chunk_text": row[0],
                "chunk_type": row[1],
                "chunk_index": row[2],
                "keywords": json.loads(row[3]) if row[3] else [],
                "document_filename": row[4],
                "document_id": row[5],
                "document_type": row[6],
            }
            for row in results
        ]

    @staticmethod
    def get_document_info(doc_id: int) -> Dict:
        """Get document information"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT filename, document_type, metadata
            FROM documents
            WHERE id = ?
        """,
            (doc_id,),
        )

        result = cursor.fetchone()
        conn.close()

        if result:
            return {
                "filename": result[0],
                "document_type": result[1],
                "metadata": json.loads(result[2]) if result[2] else {},
            }
        return {}

    @staticmethod
    def delete_document(doc_id: int) -> bool:
        """Delete document and its chunks"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM document_chunks WHERE document_id = ?", (doc_id,))
        cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted


# Initialize components
init_database()
ml_engine = IntelligentMLEngine()

# FastAPI app
app = FastAPI(title="Intelligent AI Knowledge Base", version="2.0.0")

# Enhanced Web interface
WEB_INTERFACE = """
<!DOCTYPE html>
<html>
<head>
    <title>🤖 Intelligent AI Knowledge Base</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
        .container { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .section { background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .upload-area { border: 2px dashed #4CAF50; padding: 20px; text-align: center; border-radius: 8px; background: #f9f9f9; }
        .file-list { max-height: 300px; overflow-y: auto; }
        .file-item { padding: 15px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; align-items: center; }
        .file-info { flex-grow: 1; }
        .file-type { background: #2196F3; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; margin-left: 10px; }
        .chat-area { height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 15px; margin-bottom: 10px; background: #fafafa; border-radius: 8px; }
        .message { margin-bottom: 20px; padding: 15px; border-radius: 12px; }
        .user-message { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-align: right; margin-left: 20%; }
        .ai-message { background: white; border: 1px solid #e0e0e0; margin-right: 20%; }
        .confidence { font-size: 0.9em; color: #666; margin-top: 10px; }
        .sources { font-size: 0.8em; color: #888; margin-top: 5px; background: #f0f0f0; padding: 5px 10px; border-radius: 5px; }
        .context { font-size: 0.8em; color: #666; margin-top: 5px; }
        input, textarea, button { padding: 12px; margin: 5px 0; border: 1px solid #ddd; border-radius: 6px; }
        button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; cursor: pointer; font-weight: 500; }
        button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4); }
        .delete-btn { background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%); padding: 8px 15px; font-size: 0.8em; }
        .delete-btn:hover { background: linear-gradient(135deg, #ff5252 0%, #e53935 100%); }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .typing-indicator { display: none; color: #666; font-style: italic; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Intelligent AI Knowledge Base</h1>
        <p>Upload your documents and have natural conversations about their content</p>
    </div>
    
    <div class="container">
        <div class="section">
            <h2>📁 Document Management</h2>
            
            <div class="upload-area">
                <input type="file" id="fileInput" accept=".txt,.pdf,.md" multiple>
                <br><br>
                <button onclick="uploadFiles()">📤 Upload Files</button>
            </div>
            
            <h3>📚 Your Documents:</h3>
            <div class="file-list" id="fileList">
                <!-- File list will be populated by JavaScript -->
            </div>
        </div>
        
        <div class="section">
            <h2>💬 Ask a Question</h2>
            <div class="chat-area" id="chatArea">
                <!-- Chat messages will be displayed here -->
            </div>
            
            <input type="text" id="questionInput" placeholder="Type your question here..." />
            <button onclick="askQuestion()">❓ Ask</button>
        </div>
    </div>
    
    <div class="typing-indicator" id="typingIndicator">AI is typing...</div>
    
    <script>
        // JavaScript code for handling file uploads, question asking, and UI interactions
        
        async function uploadFiles() {
            const input = document.getElementById('fileInput');
            const files = input.files;
            
            if (files.length === 0) {
                alert("Please select a file to upload.");
                return;
            }
            
            const formData = new FormData();
            for (let i = 0; i < files.length; i++) {
                formData.append("files", files[i]);
            }
            
            try {
                const response = await fetch("/api/files/", {
                    method: "POST",
                    body: formData
                });
                
                const data = await response.json();
                if (response.ok) {
                    alert("Files uploaded successfully.");
                    input.value = "";
                    loadDocumentList();
                } else {
                    alert("Error uploading files: " + data.detail);
                }
            } catch (error) {
                alert("Error uploading files: " + error.message);
            }
        }
        
        async function loadDocumentList() {
            try {
                const response = await fetch("/api/files/");
                const data = await response.json();
                
                const fileList = document.getElementById('fileList');
                fileList.innerHTML = "";
                
                data.documents.forEach(doc => {
                    const item = document.createElement('div');
                    item.className = 'file-item';
                    item.innerHTML = `
                        <div class="file-info">
                            ${doc.filename} <span class="file-type">${doc.document_type}</span>
                        </div>
                        <button class="delete-btn" onclick="deleteDocument(${doc.id})">🗑️ Delete</button>
                    `;
                    fileList.appendChild(item);
                });
            } catch (error) {
                console.error("Error loading document list:", error);
            }
        }
        
        async function deleteDocument(docId) {
            if (!confirm("Are you sure you want to delete this document?")) {
                return;
            }
            
            try {
                const response = await fetch(`/api/files/${docId}`, {
                    method: "DELETE"
                });
                
                const data = await response.json();
                if (response.ok) {
                    alert("Document deleted successfully.");
                    loadDocumentList();
                } else {
                    alert("Error deleting document: " + data.detail);
                }
            } catch (error) {
                alert("Error deleting document: " + error.message);
            }
        }
        
        async function askQuestion() {
            const input = document.getElementById('questionInput');
            const question = input.value.trim();
            
            if (!question) {
                alert("Please enter a question.");
                return;
            }
            
            // Add user message to chat
            addMessageToChat("user", question);
            input.value = "";
            
            // Show typing indicator
            const typingIndicator = document.getElementById('typingIndicator');
            typingIndicator.style.display = "block";
            
            try {
                const response = await fetch("/api/ask/", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({ question })
                });
                
                const data = await response.json();
                if (response.ok) {
                    // Add AI response to chat
                    addMessageToChat("ai", data.answer);
                } else {
                    addMessageToChat("ai", "Error: " + data.detail);
                }
            } catch (error) {
                addMessageToChat("ai", "Error: " + error.message);
            } finally {
                // Hide typing indicator
                typingIndicator.style.display = "none";
            }
        }
        
        function addMessageToChat(sender, text) {
            const chatArea = document.getElementById('chatArea');
            const message = document.createElement('div');
            message.className = 'message ' + (sender === "user" ? "user-message" : "ai-message");
            message.innerText = text;
            chatArea.appendChild(message);
            
            // Scroll to bottom of chat
            chatArea.scrollTop = chatArea.scrollHeight;
        }
        
        // Load document list on page load
        window.onload = function() {
            loadDocumentList();
        };
    </script>
</body>
</html>
"""


@app.post("/api/files/")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file and process its content."""
    try:
        file_content = await file.read()
        extracted_text = FileProcessor.extract_text_from_file(
            file_content, file.filename
        )
        processed_content = ml_engine.preprocess_text(extracted_text)
        doc_type = ml_engine.detect_document_type(extracted_text, file.filename)
        structured_info = ml_engine.extract_structured_info(extracted_text, doc_type)
        chunks = ml_engine.create_semantic_chunks(
            extracted_text, doc_type, structured_info
        )

        # Save document and chunks to the database
        doc_id = DatabaseManager.save_document(
            file.filename,
            extracted_text,
            processed_content,
            doc_type,
            json.dumps(structured_info),
        )
        DatabaseManager.save_chunks(doc_id, chunks)

        return {
            "message": f"File '{file.filename}' uploaded successfully.",
            "document_id": doc_id,
        }
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to upload file.")


@app.get("/api/files/")
def list_files():
    """List all uploaded files."""
    try:
        documents = DatabaseManager.get_all_documents()
        return {"documents": documents}
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list files.")


@app.delete("/api/files/{doc_id}")
def delete_file(doc_id: int):
    """Delete a file and its chunks."""
    try:
        deleted = DatabaseManager.delete_document(doc_id)
        if deleted:
            return {"message": f"Document with ID {doc_id} deleted successfully."}
        else:
            raise HTTPException(status_code=404, detail="Document not found.")
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete file.")


@app.post("/api/ask/")
def ask_question(request: QuestionRequest):
    """Ask a question and get an answer based on the knowledge base."""
    try:
        chunks_data = DatabaseManager.get_all_chunks()
        relevant_chunks = ml_engine.find_relevant_chunks(request.question, chunks_data)
        if not relevant_chunks:
            return {"answer": "No relevant information found.", "confidence": 0.0}

        doc_info = DatabaseManager.get_document_info(
            relevant_chunks[0][0]["document_id"]
        )
        response = ml_engine.generate_human_response(
            request.question, relevant_chunks, doc_info
        )
        return response
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to answer question.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
