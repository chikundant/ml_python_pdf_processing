from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import pickle
import hashlib
from typing import List, Optional, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime
from sqlalchemy import Column, Integer, LargeBinary, String, DateTime, Text, select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import sessionmaker

from app.models.knowledge_base import KnowledgeBase

# class MLService:
#     def __init__(self, s3_bucket_service):
#         self._s3_bucket_service = s3_bucket_service
#         self._documents = []
#         self._vectorizer = TfidfVectorizer()
#         self._tfidf_matrix = None

#     async def answer_question(self, question: str) -> str:
#         # Load documents from S3
#         files = await self._s3_bucket_service.list_files()
#         for file in files:
#             if file["filename"].endswith(".pdf"):
#                 text = await self._s3_bucket_service.extract_text_from_s3_pdf(
#                     file["filename"]
#                 )
#                 self._documents.append(text)

#         if not self._documents:
#             raise ValueError("No documents loaded. Cannot answer questions.")

#         # Build knowledge base
#         self._tfidf_matrix = self._vectorizer.fit_transform(self._documents)

#         if self._tfidf_matrix is None:
#             raise ValueError(
#                 "Knowledge base is not built. Please load documents and build the knowledge base first."
#             )

#         # Process the question
#         question_vector = self._vectorizer.transform([question])
#         similarities = cosine_similarity(question_vector, self._tfidf_matrix).flatten()
#         best_match_index = similarities.argmax()
#         return self._documents[best_match_index]

class MLService:
    def __init__(self, s3_bucket_service, async_session: sessionmaker):
        self._s3_bucket_service = s3_bucket_service
        self._async_session = async_session
        self._documents = []
        self._document_metadata = []  # Store metadata for each document
        self._vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.8
        )
        self._tfidf_matrix = None
        self._knowledge_base_hash = None
        
    async def _get_documents_hash(self, files: List[Dict]) -> str:
        """Generate hash of all document filenames and their last modified dates"""
        content = ""
        for file in sorted(files, key=lambda x: x["filename"]):
            content += f"{file['filename']}_{file.get('last_modified', '')}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def _load_matrix_from_db(self) -> bool:
        """Load TF-IDF matrix and vectorizer from database"""
        try:
            async with self._async_session() as session:
                # Get the latest knowledge base entry
                stmt = select(KnowledgeBase).order_by(KnowledgeBase.created_at.desc()).limit(1)
                result = await session.execute(stmt)
                kb_entry = result.scalar_one_or_none()
                
                if kb_entry:
                    # Deserialize the vectorizer and matrix
                    self._vectorizer = pickle.loads(kb_entry.vectorizer_data)
                    self._tfidf_matrix = pickle.loads(kb_entry.tfidf_matrix)
                    self._document_metadata = kb_entry.documents_metadata
                    self._knowledge_base_hash = kb_entry.knowledge_base_hash
                    
                    # Reconstruct documents list from metadata
                    self._documents = [doc['content'] for doc in self._document_metadata]
                    
                    return True
                    
        except Exception as e:
            print(f"Error loading matrix from DB: {e}")
            return False
        
        return False
    
    async def _save_matrix_to_db(self, knowledge_base_hash: str):
        """Save TF-IDF matrix and vectorizer to database"""
        try:
            async with self._async_session() as session:
                # Serialize the vectorizer and matrix
                vectorizer_data = pickle.dumps(self._vectorizer)
                tfidf_matrix_data = pickle.dumps(self._tfidf_matrix)
                
                # Create new knowledge base entry
                kb_entry = KnowledgeBase(
                    vectorizer_data=vectorizer_data,
                    tfidf_matrix=tfidf_matrix_data,
                    documents_metadata=self._document_metadata,
                    knowledge_base_hash=knowledge_base_hash,
                    created_at=datetime.utcnow()
                )
                
                session.add(kb_entry)
                await session.commit()
                
                # Clean up old entries (keep only the latest 5)
                subquery = (
                    select(KnowledgeBase.id)
                    .order_by(KnowledgeBase.created_at.desc())
                    .limit(5)
                ).subquery()
                
                cleanup_stmt = delete(KnowledgeBase).where(
                    KnowledgeBase.id.notin_(select(subquery.c.id))
                )
                await session.execute(cleanup_stmt)
                await session.commit()
                
        except Exception as e:
            print(f"Error saving matrix to DB: {e}")
            raise
    
    async def _build_knowledge_base(self) -> bool:
        """Build the knowledge base from all documents"""
        try:
            # Load documents from S3
            files = await self._s3_bucket_service.list_files()
            
            if not files:
                return False
            
            # Generate hash for current document set
            current_hash = await self._get_documents_hash(files)
            
            # Check if we need to rebuild
            if (self._knowledge_base_hash == current_hash and 
                self._tfidf_matrix is not None):
                return True
            
            # Clear existing data
            self._documents = []
            self._document_metadata = []
            
            # Process each file
            for file in files:
                try:
                    if file["filename"].endswith(".pdf"):
                        text = await self._s3_bucket_service.extract_text_from_s3_pdf(
                            file["filename"]
                        )
                    elif file["filename"].endswith(".txt"):
                        text = await self._s3_bucket_service.get_text_file_content(
                            file["filename"]
                        )
                    else:
                        continue  # Skip unsupported file types
                    
                    if text and text.strip():  # Only add non-empty documents
                        self._documents.append(text)
                        self._document_metadata.append({
                            'filename': file["filename"],
                            'content': text,
                            'size': len(text),
                            'last_modified': file.get('last_modified', ''),
                            'file_type': file["filename"].split('.')[-1].lower()
                        })
                        
                except Exception as e:
                    print(f"Error processing file {file['filename']}: {e}")
                    continue
            
            if not self._documents:
                raise ValueError("No valid documents found to build knowledge base.")
            
            # Build TF-IDF matrix
            self._tfidf_matrix = self._vectorizer.fit_transform(self._documents)
            self._knowledge_base_hash = current_hash
            
            # Save to database
            await self._save_matrix_to_db(current_hash)
            
            return True
            
        except Exception as e:
            print(f"Error building knowledge base: {e}")
            raise
    
    async def initialize(self):
        """Initialize the ML service by loading or building the knowledge base"""
        try:
            # First, try to load from database
            if await self._load_matrix_from_db():
                # Check if the loaded knowledge base is still valid
                files = await self._s3_bucket_service.list_files()
                current_hash = await self._get_documents_hash(files)
                
                if self._knowledge_base_hash == current_hash:
                    print("Loaded knowledge base from database")
                    return True
                else:
                    print("Knowledge base outdated, rebuilding...")
            
            # Build new knowledge base
            return await self._build_knowledge_base()
            
        except Exception as e:
            print(f"Error initializing ML service: {e}")
            return False
    
    async def answer_question(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """Answer a question using the knowledge base"""
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        # Ensure knowledge base is initialized
        if self._tfidf_matrix is None:
            await self.initialize()
        
        if self._tfidf_matrix is None or not self._documents:
            raise ValueError("Knowledge base is empty. Please upload documents first.")
        
        try:
            # Process the question
            question_vector = self._vectorizer.transform([question.strip()])
            similarities = cosine_similarity(question_vector, self._tfidf_matrix).flatten()
            
            # Get top-k most similar documents
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Filter out very low similarity scores (threshold: 0.1)
            filtered_results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:
                    filtered_results.append({
                        'content': self._documents[idx],
                        'similarity': float(similarities[idx]),
                        'metadata': self._document_metadata[idx]
                    })
            
            if not filtered_results:
                return {
                    'answer': "I couldn't find relevant information to answer your question.",
                    'confidence': 0.0,
                    'sources': [],
                    'total_documents': len(self._documents)
                }
            
            # Return the best match as the primary answer
            best_result = filtered_results[0]
            
            return {
                'answer': best_result['content'][:1000] + "..." if len(best_result['content']) > 1000 else best_result['content'],
                'confidence': best_result['similarity'],
                'sources': [
                    {
                        'filename': result['metadata']['filename'],
                        'similarity': result['similarity'],
                        'file_type': result['metadata']['file_type']
                    }
                    for result in filtered_results
                ],
                'total_documents': len(self._documents)
            }
            
        except Exception as e:
            print(f"Error answering question: {e}")
            raise ValueError(f"Error processing question: {str(e)}")
    
    async def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get statistics about the current knowledge base"""
        if self._tfidf_matrix is None:
            await self.initialize()
        
        async with self._async_session() as session:
            # Get total number of knowledge base entries
            stmt = select(KnowledgeBase)
            result = await session.execute(stmt)
            total_kb_entries = len(result.fetchall())
        
        return {
            'total_documents': len(self._documents),
            'vocabulary_size': len(self._vectorizer.vocabulary_) if hasattr(self._vectorizer, 'vocabulary_') else 0,
            'matrix_shape': self._tfidf_matrix.shape if self._tfidf_matrix is not None else None,
            'knowledge_base_hash': self._knowledge_base_hash,
            'total_kb_entries': total_kb_entries,
            'documents_info': [
                {
                    'filename': doc['filename'],
                    'size': doc['size'],
                    'file_type': doc['file_type']
                }
                for doc in self._document_metadata
            ]
        }
    
    async def rebuild_knowledge_base(self) -> bool:
        """Force rebuild of the knowledge base"""
        self._knowledge_base_hash = None
        self._tfidf_matrix = None
        return await self._build_knowledge_base()
    
    async def get_all_knowledge_bases(self) -> List[Dict[str, Any]]:
        """Get all knowledge base entries from database"""
        async with self._async_session() as session:
            stmt = select(KnowledgeBase).order_by(KnowledgeBase.created_at.desc())
            result = await session.execute(stmt)
            kb_entries = result.scalars().all()
            
            return [
                {
                    'id': kb.id,
                    'knowledge_base_hash': kb.knowledge_base_hash,
                    'created_at': kb.created_at,
                    'documents_count': len(kb.documents_metadata),
                    'total_size': sum(doc.get('size', 0) for doc in kb.documents_metadata)
                }
                for kb in kb_entries
            ]
    
    async def delete_knowledge_base(self, kb_id: int) -> bool:
        """Delete a specific knowledge base entry"""
        try:
            async with self._async_session() as session:
                stmt = delete(KnowledgeBase).where(KnowledgeBase.id == kb_id)
                result = await session.execute(stmt)
                await session.commit()
                return result.rowcount > 0
        except Exception as e:
            print(f"Error deleting knowledge base {kb_id}: {e}")
            return False