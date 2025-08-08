import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Book
from dotenv import load_dotenv
import os

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

books_df = pd.read_csv("books_cleaned.csv")

with Session() as session:
    for _, row in books_df.iterrows():
        try:
            book = Book(
                isbn13=str(row['isbn13']),
                title=row['TITLE'],
                authors=row['AUTHORS'],
                categories=row['categories'],
                thumbnail=row['thumbnail'],
                description=row['description'],
                tagged_description=row['tagged_description']
            )
            session.add(book)
            session.commit() 
        except Exception as e:
            session.rollback()
            print(f"Error inserting ISBN {row['isbn13']}: {e}")
    print("Data migrated!")
