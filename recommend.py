import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Book


class Recommender:
    def __init__(self, database_url):
        self.engine = create_engine(database_url)
        self.Session = sessionmaker(bind=self.engine)

        # Load books from DB into DataFrame
        with self.Session() as session:
            books_list = session.query(Book).all()
            data = [{
                'isbn13': book.isbn13,
                'title': book.title,
                'authors': book.authors,
                'categories': book.categories,
                'thumbnail': book.thumbnail,
                'description': book.description,
                'tagged_description': book.tagged_description  # Keep it, but do not use
            } for book in books_list]

        self.books = pd.DataFrame(data)
        self.books['isbn13'] = self.books['isbn13'].astype(str)

        print(f"Loaded {len(self.books)} books from DB.")
        if self.books.empty:
            print("Warning: No books loaded from DB. Recommendations may fail.")

        # Combine fields (without tagged_description)
        documents = []
        for _, row in self.books.iterrows():
            content = f"{row['isbn13']} {row['title']} {row['categories']} {row['description']}"
            documents.append(Document(page_content=content))

        # Create vector store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
        self.vectorstore = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")
        print(f"Loaded {len(documents)} documents into vector store.")

    def get_recommendations(self, query, top_k=5, raw=False):
        if self.books.empty:
            return []

        print(f"User query: '{query}'")

        # Extract keywords from query
        keywords = set(query.lower().split())

        # Broad search
        results = self.vectorstore.similarity_search_with_score(query, k=30)

        print(f"Found {len(results)} raw results from vector store.")

        seen_isbns = set()
        scored_results = []

        for res, score in results:
            isbn_parts = res.page_content.split()
            if not isbn_parts:
                continue

            isbn = isbn_parts[0]
            if isbn in seen_isbns:
                continue

            seen_isbns.add(isbn)
            matching_books = self.books[self.books['isbn13'] == isbn]
            if matching_books.empty:
                continue

            book = matching_books.iloc[0]

            # Score boost
            boost = 0
            for kw in keywords:
                if pd.notna(book['title']) and kw in book['title'].lower():
                    boost += 1.5
                if pd.notna(book['categories']) and kw in book['categories'].lower():
                    boost += 1.0
                if pd.notna(book['description']) and kw in book['description'].lower():
                    boost += 0.5

            total_score = score - boost  # Lower is better
            scored_results.append((book, total_score))

        scored_results.sort(key=lambda x: x[1])
        top_results = scored_results[:top_k]

        if raw:
            return [{
                'title': book['title'],
                'author': book['authors'],
                'category': book['categories'],
                'description': book['description'],
                'thumbnail': book['thumbnail'],
                'score': score
            } for book, score in top_results]

        return [{
            'title': book['title'],
            'author': book['authors'],
            'category': book['categories'],
            'description': book['description'],
            'thumbnail': book['thumbnail']
        } for book, _ in top_results]

    def rebuild_vectorstore(self):
        try:
            with self.Session() as session:
                books_list = session.query(Book).all()
                data = [{
                    'isbn13': book.isbn13,
                    'title': book.title,
                    'authors': book.authors,
                    'categories': book.categories,
                    'thumbnail': book.thumbnail,
                    'description': book.description,
                    'tagged_description': book.tagged_description  # still included
                } for book in books_list]

            self.books = pd.DataFrame(data)
            self.books['isbn13'] = self.books['isbn13'].astype(str)

            print(f"Rebuilding with {len(self.books)} books.")

            documents = []
            for _, row in self.books.iterrows():
                content = f"{row['isbn13']} {row['title']} {row['categories']} {row['description']}"
                documents.append(Document(page_content=content))

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            self.vectorstore = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")
            print("Vector store rebuilt.")

        except Exception as e:
            print(f"Rebuild failed: {str(e)}")
            raise
