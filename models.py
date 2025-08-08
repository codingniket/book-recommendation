from sqlalchemy import Column, String, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Book(Base):
    __tablename__ = 'books'
    
    isbn13 = Column(String(13), primary_key=True, nullable=False)
    title = Column(String(500), nullable=False)
    authors = Column(String(300), nullable=True)
    categories = Column(String(200), nullable=False)
    thumbnail = Column(String(1000), nullable=False)
    description = Column(Text, nullable=True)
    tagged_description = Column(Text, nullable=False)
    
    def __repr__(self):
        return f"<Book(isbn13='{self.isbn13}', title='{self.title}')>"