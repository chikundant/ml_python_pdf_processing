from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    LargeBinary,
    String,
    DateTime,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSONB

Base = declarative_base()


class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"

    id = Column(Integer, primary_key=True, autoincrement=True)
    vectorizer_data = Column(LargeBinary, nullable=False)
    tfidf_matrix = Column(LargeBinary, nullable=False)
    documents_metadata = Column(JSONB, nullable=False)
    knowledge_base_hash = Column(String(32), nullable=False)
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
