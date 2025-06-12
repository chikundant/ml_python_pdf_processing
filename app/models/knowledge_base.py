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

Base = declarative_base()

class KnowledgeBase(Base):
    __tablename__ = 'knowledge_base'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    vectorizer_data = Column(LargeBinary, nullable=False)
    tfidf_matrix = Column(LargeBinary, nullable=False)
    documents_metadata = Column(JSONB, nullable=False)
    knowledge_base_hash = Column(String(32), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
