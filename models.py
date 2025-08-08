from sqlalchemy import Column, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Book(Base):
    __tablename__ = 'books'
    isbn13 = Column(String, primary_key=True)
    title = Column(String)
    authors = Column(String)
    categories = Column(String)
    thumbnail = Column(String)
    description = Column(String)
    tagged_description = Column(String)
