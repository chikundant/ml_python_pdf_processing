from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import pickle
import hashlib
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from datetime import datetime
from sqlalchemy import Column, Integer, LargeBinary, String, DateTime, Text, select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import sessionmaker
from app.models.knowledge_base import KnowledgeBase
import re
from collections import Counter
from app.services.s3_bucket_service import S3BucketService
from dataclasses import dataclass, field
from enum import Enum

@dataclass
class DocumentChunk:
    """Represents a chunk of document with metadata"""
    content: str
    filename: str
    chunk_id: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'content': self.content,
            'filename': self.filename,
            'chunk_id': self.chunk_id,
            'start_char': self.start_char,
            'end_char': self.end_char,
            'metadata': self.metadata
        }

class MLService:
    def __init__(self, s3_bucket_service, async_session: AsyncSession):
        self._s3_bucket_service: S3BucketService = s3_bucket_service
        self._async_session = async_session
        self._documents = []
        self._document_chunks: List[DocumentChunk] = []
        self._document_metadata = []
        self._vectorizer = None
        self._tfidf_matrix = None
        self._knowledge_base_hash = None
        
        # Chunking parameters
        self._chunk_size = 500  # words
        self._chunk_overlap = 100  # words
        self._min_chunk_size = 50  # words
        
        # Non-pickleable attributes
        self._non_pickleable_attributes = ["_s3_bucket_service", "_async_session"]
    
    def __getstate__(self):
        """Exclude non-pickleable attributes during serialization"""
        state = self.__dict__.copy()
        for attr in self._non_pickleable_attributes:
            state.pop(attr, None)
        return state
    
    def __setstate__(self, state):
        """Restore state during deserialization"""
        self.__dict__.update(state)
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace while preserving structure
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
        text = re.sub(r'[ \t]+', ' ', text)  # Single spaces
        text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)  # Trim lines
        
        return text.strip()
    
    def _extract_document_structure(self, text: str) -> Dict[str, Any]:
        """Extract structural information from document"""
        structure = {
            'headings': [],
            'lists': [],
            'tables': [],
            'code_blocks': [],
            'paragraphs': [],
            'total_words': len(text.split()),
            'has_toc': False,
            'language': 'en'  # Default, could be enhanced with language detection
        }
        
        lines = text.split('\n')
        
        # Detect headings (various formats)
        heading_patterns = [
            (r'^#{1,6}\s+(.+)$', 'markdown'),  # Markdown headers
            (r'^([A-Z][A-Z\s]{2,})$', 'uppercase'),  # ALL CAPS headers
            (r'^(\d+\.(?:\d+\.)*)\s+(.+)$', 'numbered'),  # 1.2.3 Headers
            (r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,5}):?\s*$', 'title_case'),  # Title Case Headers
        ]
        
        for i, line in enumerate(lines):
            for pattern, heading_type in heading_patterns:
                match = re.match(pattern, line.strip())
                if match and len(line.strip()) < 100:  # Reasonable header length
                    structure['headings'].append({
                        'text': line.strip(),
                        'line': i,
                        'type': heading_type,
                        'level': self._get_heading_level(line, heading_type)
                    })
                    break
        
        # Detect lists
        list_patterns = [
            r'^\s*[\*\-\+]\s+',  # Unordered lists
            r'^\s*\d+[\.\)]\s+',  # Ordered lists
            r'^\s*[a-z][\.\)]\s+',  # Alphabetic lists
        ]
        
        current_list = []
        for i, line in enumerate(lines):
            if any(re.match(pattern, line) for pattern in list_patterns):
                current_list.append(line)
            elif current_list:
                structure['lists'].append({
                    'items': current_list,
                    'start_line': i - len(current_list),
                    'end_line': i - 1
                })
                current_list = []
        
        # Detect code blocks
        code_block_pattern = r'```[\w]*\n(.*?)```'
        code_blocks = re.finditer(code_block_pattern, text, re.DOTALL)
        for match in code_blocks:
            structure['code_blocks'].append({
                'content': match.group(1),
                'start': match.start(),
                'end': match.end()
            })
        
        # Check for table of contents
        toc_indicators = ['table of contents', 'contents', 'index', 'toc']
        for line in lines[:50]:  # Usually in the beginning
            if any(indicator in line.lower() for indicator in toc_indicators):
                structure['has_toc'] = True
                break
        
        return structure
    
    def _get_heading_level(self, line: str, heading_type: str) -> int:
        """Determine heading hierarchy level"""
        if heading_type == 'markdown':
            return len(re.match(r'^(#+)', line).group(1))
        elif heading_type == 'numbered':
            return len(re.match(r'^(\d+(?:\.\d+)*)', line).group(1).split('.'))
        else:
            # For other types, use heuristics
            return 1 if heading_type == 'uppercase' else 2
    
    def _smart_chunk_document(self, text: str, filename: str) -> List[DocumentChunk]:
        """Intelligently chunk document based on its structure"""
        chunks = []
        structure = self._extract_document_structure(text)
        
        # Identify logical sections based on headings
        sections = self._identify_sections(text, structure['headings'])
        
        chunk_id = 0
        for section in sections:
            # Determine optimal chunk size for this section
            section_length = len(section['content'].split())
            
            if section_length <= self._chunk_size:
                # Small section - keep as single chunk
                chunk = DocumentChunk(
                    content=section['content'],
                    filename=filename,
                    chunk_id=chunk_id,
                    start_char=section['start'],
                    end_char=section['end'],
                    metadata={
                        'section_title': section.get('title', ''),
                        'section_level': section.get('level', 0),
                        'is_complete_section': True
                    }
                )
                chunks.append(chunk)
                chunk_id += 1
            else:
                # Large section - create overlapping chunks
                sub_chunks = self._create_overlapping_chunks(
                    section['content'],
                    section['start']
                )
                
                for i, (chunk_text, start_pos, end_pos) in enumerate(sub_chunks):
                    chunk = DocumentChunk(
                        content=chunk_text,
                        filename=filename,
                        chunk_id=chunk_id,
                        start_char=start_pos,
                        end_char=end_pos,
                        metadata={
                            'section_title': section.get('title', ''),
                            'section_level': section.get('level', 0),
                            'is_complete_section': False,
                            'chunk_index_in_section': i,
                            'total_chunks_in_section': len(sub_chunks)
                        }
                    )
                    chunks.append(chunk)
                    chunk_id += 1
        
        # If no sections identified, fall back to simple chunking
        if not chunks:
            simple_chunks = self._create_overlapping_chunks(text, 0)
            for i, (chunk_text, start_pos, end_pos) in enumerate(simple_chunks):
                chunk = DocumentChunk(
                    content=chunk_text,
                    filename=filename,
                    chunk_id=i,
                    start_char=start_pos,
                    end_char=end_pos,
                    metadata={'is_fallback_chunk': True}
                )
                chunks.append(chunk)
        
        return chunks
    
    def _identify_sections(self, text: str, headings: List[Dict]) -> List[Dict]:
        """Identify document sections based on headings"""
        if not headings:
            return [{'content': text, 'start': 0, 'end': len(text), 'title': 'Document', 'level': 0}]
        
        sections = []
        lines = text.split('\n')
        
        # Sort headings by line number
        headings = sorted(headings, key=lambda h: h['line'])
        
        # Create sections between headings
        for i, heading in enumerate(headings):
            start_line = heading['line']
            end_line = headings[i + 1]['line'] if i + 1 < len(headings) else len(lines)
            
            # Find character positions
            start_char = sum(len(line) + 1 for line in lines[:start_line])
            end_char = sum(len(line) + 1 for line in lines[:end_line])
            
            section_content = '\n'.join(lines[start_line:end_line])
            
            sections.append({
                'content': section_content,
                'start': start_char,
                'end': end_char,
                'title': heading['text'],
                'level': heading['level']
            })
        
        return sections
    
    def _create_overlapping_chunks(self, text: str, base_position: int) -> List[Tuple[str, int, int]]:
        """Create overlapping chunks from text"""
        chunks = []
        words = text.split()
        
        if not words:
            return chunks
        
        # Calculate positions for each word
        word_positions = []
        current_pos = 0
        for word in words:
            word_positions.append(current_pos)
            current_pos += len(word) + 1  # +1 for space
        
        # Create chunks
        i = 0
        while i < len(words):
            # Determine chunk end
            chunk_end = min(i + self._chunk_size, len(words))
            
            # Extract chunk
            chunk_words = words[i:chunk_end]
            chunk_text = ' '.join(chunk_words)
            
            # Calculate character positions
            start_char = base_position + word_positions[i]
            end_char = base_position + word_positions[chunk_end - 1] + len(words[chunk_end - 1])
            
            if len(chunk_words) >= self._min_chunk_size:
                chunks.append((chunk_text, start_char, end_char))
            
            # Move to next chunk with overlap
            i += self._chunk_size - self._chunk_overlap
        
        return chunks
    
    def _extract_key_phrases(self, text: str, top_n: int = 20) -> List[str]:
        """Extract key phrases from text using TF-IDF"""
        # Simple n-gram extraction
        words = text.lower().split()
        
        # Extract 1-3 grams
        ngrams = []
        for n in range(1, 4):
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i+n])
                # Filter out common words and very short n-grams
                if len(ngram) > 3 and not ngram.startswith(('the', 'a', 'an', 'and', 'or', 'but')):
                    ngrams.append(ngram)
        
        # Count frequencies
        ngram_freq = Counter(ngrams)
        
        # Return most common
        return [ngram for ngram, _ in ngram_freq.most_common(top_n)]
    
    def _calculate_bm25_score(self, query_terms: List[str], document: str, k1: float = 1.5, b: float = 0.75) -> float:
        """Calculate BM25 score for ranking"""
        doc_terms = document.lower().split()
        doc_len = len(doc_terms)
        
        if not self._document_chunks:
            return 0.0
        
        # Calculate average document length
        avg_doc_len = np.mean([len(chunk.content.split()) for chunk in self._document_chunks])
        
        score = 0.0
        for term in query_terms:
            # Term frequency in document
            tf = doc_terms.count(term.lower())
            if tf == 0:
                continue
            
            # Document frequency
            df = sum(1 for chunk in self._document_chunks if term.lower() in chunk.content.lower())
            
            # IDF calculation
            N = len(self._document_chunks)
            idf = np.log((N - df + 0.5) / (df + 0.5) + 1)
            
            # BM25 formula
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))
        
        return score
    
    def _create_vectorizer(self) -> TfidfVectorizer:
        """Create TF-IDF vectorizer with optimized parameters"""
        return TfidfVectorizer(
            lowercase=True,
            max_features=10000,
            ngram_range=(1, 3),  # Unigrams, bigrams, and trigrams
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,  # Use log(TF) instead of raw TF
            use_idf=True,
            smooth_idf=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b',
            strip_accents='unicode',
            analyzer='word',
            stop_words='english'
        )
    
    async def _get_documents_hash(self, files: List[Dict]) -> str:
        """Generate hash of all document filenames and their last modified dates"""
        content = ""
        for file in sorted(files, key=lambda x: x["filename"]):
            content += f"{file['filename']}_{file.get('last_modified', '')}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def _load_matrix_from_db(self) -> bool:
        """Load TF-IDF matrix and vectorizer from database"""
        try:
            stmt = select(KnowledgeBase).order_by(KnowledgeBase.created_at.desc()).limit(1)
            result = await self._async_session.execute(stmt)
            kb_entry = result.scalar_one_or_none()
            
            if kb_entry:
                # Deserialize the vectorizer and matrix
                self._vectorizer = pickle.loads(kb_entry.vectorizer_data)
                self._tfidf_matrix = pickle.loads(kb_entry.tfidf_matrix)
                self._document_metadata = kb_entry.documents_metadata
                self._knowledge_base_hash = kb_entry.knowledge_base_hash
                
                # Reconstruct documents and chunks from metadata
                self._documents = [doc['content'] for doc in self._document_metadata]
                self._document_chunks = []
                
                for doc_meta in self._document_metadata:
                    # Recreate chunks from stored metadata
                    for chunk_data in doc_meta.get('chunks', []):
                        chunk = DocumentChunk(
                            content=chunk_data['content'],
                            filename=doc_meta['filename'],
                            chunk_id=chunk_data['chunk_id'],
                            start_char=chunk_data['start_char'],
                            end_char=chunk_data['end_char'],
                            metadata=chunk_data.get('metadata', {})
                        )
                        self._document_chunks.append(chunk)
                
                return True
                
        except Exception as e:
            print(f"Error loading matrix from DB: {e}")
            return False
        
        return False
    
    async def _save_matrix_to_db(self, knowledge_base_hash: str):
        """Save TF-IDF matrix and vectorizer to database"""
        try:
            # Prepare metadata with chunk information
            metadata_with_chunks = []
            for doc_meta in self._document_metadata:
                doc_chunks = [
                    chunk.to_dict() for chunk in self._document_chunks 
                    if chunk.filename == doc_meta['filename']
                ]
                doc_meta_copy = doc_meta.copy()
                doc_meta_copy['chunks'] = doc_chunks
                metadata_with_chunks.append(doc_meta_copy)
            
            # Serialize the vectorizer and matrix
            vectorizer_data = pickle.dumps(self._vectorizer)
            tfidf_matrix_data = pickle.dumps(self._tfidf_matrix)
            
            # Create new knowledge base entry
            kb_entry = KnowledgeBase(
                vectorizer_data=vectorizer_data,
                tfidf_matrix=tfidf_matrix_data,
                documents_metadata=metadata_with_chunks,
                knowledge_base_hash=knowledge_base_hash,
                created_at=datetime.utcnow(),
            )
            
            self._async_session.add(kb_entry)
            await self._async_session.commit()
            
            # Clean up old entries (keep only the latest 5)
            subquery = (
                select(KnowledgeBase.id)
                .order_by(KnowledgeBase.created_at.desc())
                .limit(5)
            ).subquery()
            
            cleanup_stmt = delete(KnowledgeBase).where(
                KnowledgeBase.id.notin_(select(subquery.c.id))
            )
            
            await self._async_session.execute(cleanup_stmt)
            await self._async_session.commit()
            
        except Exception as e:
            print(f"Error saving matrix to DB: {e}")
            raise
    
    async def _build_knowledge_base(self) -> bool:
        """Build the knowledge base from all documents"""
        # Load documents from S3
        files = await self._s3_bucket_service.list_files()
        print(f"Found {len(files)} files in S3 bucket.", flush=True)
        
        if not files:
            return False
        
        # Generate hash for current document set
        current_hash = await self._get_documents_hash(files)
        print(f"Current knowledge base hash: {current_hash}", flush=True)
        
        # Check if we need to rebuild
        if (self._knowledge_base_hash == current_hash and 
            self._tfidf_matrix is not None):
            return True
        
        # Clear existing data
        self._documents = []
        self._document_chunks = []
        self._document_metadata = []
        
        # Process each file
        for file in files:
            try:
                # Extract text based on file type
                if file["filename"].endswith(".pdf"):
                    text = await self._s3_bucket_service.extract_text_from_s3_pdf(file["filename"])
                elif file["filename"].endswith(".txt"):
                    text = await self._s3_bucket_service.get_text_file_content(file["filename"])
                else:
                    continue  # Skip unsupported file types
                
                if not text or not text.strip():
                    continue
                
                # Clean text
                text = self._preprocess_text(text)
                
                # Store full document
                self._documents.append(text)
                
                # Create chunks
                chunks = self._smart_chunk_document(text, file["filename"])
                
                # Store metadata
                self._document_metadata.append({
                    'filename': file["filename"],
                    'content': text,
                    'size': len(text),
                    'word_count': len(text.split()),
                    'last_modified': file.get('last_modified', ''),
                    'file_type': file["filename"].split('.')[-1].lower(),
                    'key_phrases': self._extract_key_phrases(text),
                    'chunk_count': len(chunks)
                })
                
                # Add chunks to collection
                self._document_chunks.extend(chunks)
                
            except Exception as e:
                print(f"Error processing file {file['filename']}: {e}")
                continue
        
        if not self._document_chunks:
            raise ValueError("No valid documents found to build knowledge base.")
        
        print(f"Processing {len(self._documents)} documents with {len(self._document_chunks)} chunks", flush=True)
        
        # Create vectorizer
        self._vectorizer = self._create_vectorizer()
        
        # Build TF-IDF matrix using chunks
        chunk_texts = [chunk.content for chunk in self._document_chunks]
        self._tfidf_matrix = self._vectorizer.fit_transform(chunk_texts)
        self._knowledge_base_hash = current_hash
        
        print(f"Built TF-IDF matrix with shape: {self._tfidf_matrix.shape}", flush=True)
        print(f"Vocabulary size: {len(self._vectorizer.vocabulary_)}", flush=True)
        
        # Save to database
        await self._save_matrix_to_db(current_hash)
        
        return True
    
    async def initialize(self):
        """Initialize the ML service by loading or building the knowledge base"""
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
    
    def _rerank_results(self, query: str, results: List[Tuple[DocumentChunk, float]]) -> List[Tuple[DocumentChunk, float]]:
        """Rerank results based on additional signals"""
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        reranked = []
        
        for chunk, base_score in results:
            boost = 1.0
            
            # Exact phrase match boost
            if query_lower in chunk.content.lower():
                boost *= 2.0
            
            # Term coverage boost
            chunk_terms = set(chunk.content.lower().split())
            coverage = len(query_terms.intersection(chunk_terms)) / len(query_terms) if query_terms else 0
            boost *= (1 + coverage * 0.5)
            
            # Position boost (chunks from beginning of sections might be more relevant)
            if chunk.metadata.get('chunk_index_in_section', 0) == 0:
                boost *= 1.1
            
            # Section title relevance
            section_title = chunk.metadata.get('section_title', '').lower()
            if section_title and any(term in section_title for term in query_terms):
                boost *= 1.3
            
            reranked.append((chunk, base_score * boost))
        
        # Sort by boosted score
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return reranked
    
    async def answer_question(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Answer a question using the knowledge base"""
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        # Ensure knowledge base is initialized
        if self._tfidf_matrix is None:
            await self.initialize()
        
        if self._tfidf_matrix is None or not self._document_chunks:
            raise ValueError("Knowledge base is empty. Please upload documents first.")
        
        try:
            # Process the question
            processed_question = question.strip().lower()
            question_terms = processed_question.split()
            
            # Get TF-IDF similarity scores
            question_vector = self._vectorizer.transform([processed_question])
            tfidf_similarities = cosine_similarity(question_vector, self._tfidf_matrix).flatten()
            
            # Calculate combined scores
            combined_scores = []
            
            for i, chunk in enumerate(self._document_chunks):
                # TF-IDF score (weight: 0.5)
                tfidf_score = tfidf_similarities[i]
                
                # BM25 score (weight: 0.5)
                bm25_score = self._calculate_bm25_score(question_terms, chunk.content)
                normalized_bm25 = min(1.0, bm25_score / 10)  # Normalize to [0, 1]
                
                # Combined score
                combined_score = 0.5 * tfidf_score + 0.5 * normalized_bm25
                
                combined_scores.append((chunk, combined_score))
            
            # Sort by score and get top results
            combined_scores.sort(key=lambda x: x[1], reverse=True)
            top_results = combined_scores[:top_k * 2]  # Get more for reranking
            
            # Rerank results
            reranked_results = self._rerank_results(question, top_results)
            
            # Filter by threshold
            similarity_threshold = 0.15
            filtered_results = [(chunk, score) for chunk, score in reranked_results[:top_k] 
                               if score > similarity_threshold]
            
            if not filtered_results:
                return {
                    'answer': "I couldn't find relevant information to answer your question. Try rephrasing or using different keywords.",
                    'confidence': 0.0,
                    'sources': [],
                    'total_documents': len(self._documents)
                }
            
            # Build comprehensive answer
            answer_parts = []
            seen_content = set()  # Avoid duplicate content
            
            for chunk, score in filtered_results[:3]:  # Use top 3 chunks
                # Skip if content is too similar to already included content
                chunk_hash = hash(chunk.content[:100])
                if chunk_hash in seen_content:
                    continue
                seen_content.add(chunk_hash)
                
                # Add chunk content with context
                if chunk.metadata.get('section_title'):
                    answer_parts.append(f"From section '{chunk.metadata['section_title']}':")
                answer_parts.append(chunk.content)
                answer_parts.append("")  # Empty line for readability
            
            # Combine answer
            answer = "\n".join(answer_parts).strip()
            
            # Limit answer length
            max_length = 2000
            if len(answer) > max_length:
                answer = answer[:max_length] + "..."
            
            return {
                'answer': answer,
                'confidence': float(filtered_results[0][1]),
                'sources': [
                    {
                        'filename': chunk.filename,
                        'similarity': float(score),
                        'chunk_id': chunk.chunk_id,
                        'section': chunk.metadata.get('section_title', 'General'),
                        'position': f"chars {chunk.start_char}-{chunk.end_char}"
                    }
                    for chunk, score in filtered_results
                ],
                'total_documents': len(self._documents),
                'total_chunks': len(self._document_chunks),
                'query_terms': question_terms
            }
            
        except Exception as e:
            print(f"Error answering question: {e}")
            raise ValueError(f"Error processing question: {str(e)}")
    
    async def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get statistics about the current knowledge base"""
        if self._tfidf_matrix is None:
            await self.initialize()
        
        # Get total number of knowledge base entries
        stmt = select(KnowledgeBase)
        result = await self._async_session.execute(stmt)
        total_kb_entries = len(result.fetchall())
        
        stats = {
            'total_documents': len(self._documents),
            'total_chunks': len(self._document_chunks),
            'vocabulary_size': len(self._vectorizer.vocabulary_) if self._vectorizer and hasattr(self._vectorizer, 'vocabulary_') else 0,
            'matrix_shape': self._tfidf_matrix.shape if self._tfidf_matrix is not None else None,
            'knowledge_base_hash': self._knowledge_base_hash,
            'total_kb_entries': total_kb_entries,
            'chunk_size_config': {
                'chunk_size': self._chunk_size,
                'overlap': self._chunk_overlap,
                'min_size': self._min_chunk_size
            },
            'documents_info': []
        }
        
        # Add document-specific information
        for doc_meta in self._document_metadata:
            doc_chunks = [chunk for chunk in self._document_chunks if chunk.filename == doc_meta['filename']]
            
            stats['documents_info'].append({
                'filename': doc_meta['filename'],
                'size_bytes': doc_meta['size'],
                'word_count': doc_meta['word_count'],
                'chunk_count': len(doc_chunks),
                'file_type': doc_meta['file_type'],
                'key_phrases': doc_meta.get('key_phrases', [])[:10],  # Top 10
                'avg_chunk_size': np.mean([len(chunk.content.split()) for chunk in doc_chunks]) if doc_chunks else 0
            })
        
        return stats
    
    async def rebuild_knowledge_base(self) -> bool:
        """Force rebuild of the knowledge base"""
        self._knowledge_base_hash = None
        self._tfidf_matrix = None
        self._vectorizer = None
        return await self._build_knowledge_base()
    
    async def search_similar_chunks(self, text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for chunks similar to the given text"""
        if not text or not text.strip():
            return []
        
        if self._tfidf_matrix is None:
            await self.initialize()
        
        if self._tfidf_matrix is None or not self._document_chunks:
            return []
        
        # Transform input text
        text_vector = self._vectorizer.transform([text.lower()])
        similarities = cosine_similarity(text_vector, self._tfidf_matrix).flatten()
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Threshold
                chunk = self._document_chunks[idx]
                results.append({
                    'chunk_id': chunk.chunk_id,
                    'filename': chunk.filename,
                    'content': chunk.content,
                    'similarity': float(similarities[idx]),
                    'metadata': chunk.metadata
                })
        
        return results
    
    async def get_all_knowledge_bases(self) -> List[Dict[str, Any]]:
        """Get all knowledge base entries from database"""
        stmt = select(KnowledgeBase).order_by(KnowledgeBase.created_at.desc())
        result = await self._async_session.execute(stmt)
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
            stmt = delete(KnowledgeBase).where(KnowledgeBase.id == kb_id)
            result = await self._async_session.execute(stmt)
            await self._async_session.commit()
            return result.rowcount > 0
        except Exception as e:
            print(f"Error deleting knowledge base {kb_id}: {e}")
            return False