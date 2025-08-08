import pandas as pd
from langchain_text_splitters import CharacterTextSplitter
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
                'tagged_description': book.tagged_description
            } for book in books_list]
        self.books = pd.DataFrame(data)
        self.books['isbn13'] = self.books['isbn13'].astype(str)

        print(f"Loaded {len(self.books)} books from DB.")
        print("DataFrame columns:", self.books.columns.tolist())
        if self.books.empty:
            print("Warning: No books loaded from DB. Recommendations may fail.")

        # Create documents from tagged descriptions
        documents = [f"{row['tagged_description']}" for _, row in self.books.iterrows()]

        # Split documents with larger chunk_size to avoid tiny fragments
        text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0) 
        split_docs = text_splitter.create_documents(documents)

        # Create vector store with Hugging Face embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = Chroma.from_documents(split_docs, embeddings, persist_directory="./chroma_db")
        print(f"Loaded {len(split_docs)} documents into vector store.")

    def get_recommendations(self, query, top_k=5, raw=False):
        if self.books.empty:
            return []  # Handle empty case gracefully

        print("Current books DataFrame shape:", self.books.shape)
        print("Current columns:", self.books.columns.tolist())

        results = self.vectorstore.similarity_search_with_score(query, k=top_k * 3)  # Increased multiplier for more candidates
        print(f"Raw search results for query '{query}': {results}")

        seen_isbns = set()
        unique_results = []
        for res, score in results:
            isbn_parts = res.page_content.split()
            if isbn_parts:
                isbn = isbn_parts[0]
                if isbn not in seen_isbns:
                    seen_isbns.add(isbn)
                    unique_results.append((res, score))

        if raw:
            return [{"page_content": res.page_content, "metadata": res.metadata, "score": score} for res, score in unique_results[:top_k]]

        recommendations = []
        for res, score in unique_results[:top_k]:
            isbn_parts = res.page_content.split()
            if isbn_parts:
                isbn = isbn_parts[0]
                print(f"Extracted ISBN: {isbn} with score: {score}")
                matching_books = self.books[self.books['isbn13'] == isbn]
                if not matching_books.empty:
                    book = matching_books.iloc[0]
                    recommendations.append({
                        'title': book['title'],
                        'author': book['authors'],
                        'category': book['categories'],
                        'description': book['description'],
                        'thumbnail': book['thumbnail']
                    })
                else:
                    print(f"No book match for ISBN: {isbn}")
            else:
                print("Failed to extract ISBN from page_content")
        print(f"Returning {len(recommendations)} recommendations (requested top_k={top_k})")
        return recommendations

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
                    'tagged_description': book.tagged_description
                } for book in books_list]
            self.books = pd.DataFrame(data)
            self.books['isbn13'] = self.books['isbn13'].astype(str)

            print(f"Rebuilt with {len(self.books)} books from DB.")
            print("DataFrame columns:", self.books.columns.tolist())
            if self.books.empty:
                print("Warning: No books loaded during rebuild.")

            documents = [f"{row['tagged_description']}" for _, row in self.books.iterrows()]
            text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)  # Increased for full/meaningful chunks
            split_docs = text_splitter.create_documents(documents)

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            self.vectorstore = Chroma.from_documents(split_docs, embeddings, persist_directory="./chroma_db")
            print("Vector store rebuilt with latest data.")
        except Exception as e:
            print(f"Rebuild failed: {str(e)}")
            raise
