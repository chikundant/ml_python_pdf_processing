from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()
engine = create_engine("sqlite:///knowledge.db")
SessionLocal = sessionmaker(bind=engine)

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True)
    filename = Column(String)
    content = Column(Text)

Base.metadata.create_all(bind=engine)
