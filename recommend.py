# import pandas as pd
# from langchain_core.documents import Document
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker
# from models import Book


# class Recommender:
#     def __init__(self, database_url):
#         self.engine = create_engine(database_url)
#         self.Session = sessionmaker(bind=self.engine)

#         # Load books from DB into DataFrame
#         with self.Session() as session:
#             books_list = session.query(Book).all()
#             data = [{
#                 'isbn13': book.isbn13,
#                 'title': book.title,
#                 'authors': book.authors,
#                 'categories': book.categories,
#                 'thumbnail': book.thumbnail,
#                 'description': book.description,
#                 'tagged_description': book.tagged_description  # Keep it, but do not use
#             } for book in books_list]

#         self.books = pd.DataFrame(data)
#         self.books['isbn13'] = self.books['isbn13'].astype(str)

#         print(f"Loaded {len(self.books)} books from DB.")
#         if self.books.empty:
#             print("Warning: No books loaded from DB. Recommendations may fail.")

#         # Combine fields (without tagged_description)
#         documents = []
#         for _, row in self.books.iterrows():
#             content = f"{row['isbn13']} {row['title']} {row['categories']} {row['description']}"
#             documents.append(Document(page_content=content))

#         # Create vector store
#         embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#         self.vectorstore = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")
#         print(f"Loaded {len(documents)} documents into vector store.")

#     def get_recommendations(self, query, top_k=5, raw=False):
#         if self.books.empty:
#             return []

#         print(f"User query: '{query}'")

#         # Extract keywords from query
#         keywords = set(query.lower().split())

#         # Broad search
#         results = self.vectorstore.similarity_search_with_score(query, k=30)

#         print(f"Found {len(results)} raw results from vector store.")

#         seen_isbns = set()
#         scored_results = []

#         for res, score in results:
#             isbn_parts = res.page_content.split()
#             if not isbn_parts:
#                 continue

#             isbn = isbn_parts[0]
#             if isbn in seen_isbns:
#                 continue

#             seen_isbns.add(isbn)
#             matching_books = self.books[self.books['isbn13'] == isbn]
#             if matching_books.empty:
#                 continue

#             book = matching_books.iloc[0]

#             # Score boost
#             boost = 0
#             for kw in keywords:
#                 if pd.notna(book['title']) and kw in book['title'].lower():
#                     boost += 1.5
#                 if pd.notna(book['categories']) and kw in book['categories'].lower():
#                     boost += 1.0
#                 if pd.notna(book['description']) and kw in book['description'].lower():
#                     boost += 0.5

#             total_score = score - boost  # Lower is better
#             scored_results.append((book, total_score))

#         scored_results.sort(key=lambda x: x[1])
#         top_results = scored_results[:top_k]

#         if raw:
#             return [{
#                 'title': book['title'],
#                 'author': book['authors'],
#                 'category': book['categories'],
#                 'description': book['description'],
#                 'thumbnail': book['thumbnail'],
#                 'score': score
#             } for book, score in top_results]

#         return [{
#             'title': book['title'],
#             'author': book['authors'],
#             'category': book['categories'],
#             'description': book['description'],
#             'thumbnail': book['thumbnail']
#         } for book, _ in top_results]

#     def rebuild_vectorstore(self):
#         try:
#             with self.Session() as session:
#                 books_list = session.query(Book).all()
#                 data = [{
#                     'isbn13': book.isbn13,
#                     'title': book.title,
#                     'authors': book.authors,
#                     'categories': book.categories,
#                     'thumbnail': book.thumbnail,
#                     'description': book.description,
#                     'tagged_description': book.tagged_description  # still included
#                 } for book in books_list]

#             self.books = pd.DataFrame(data)
#             self.books['isbn13'] = self.books['isbn13'].astype(str)

#             print(f"Rebuilding with {len(self.books)} books.")

#             documents = []
#             for _, row in self.books.iterrows():
#                 content = f"{row['isbn13']} {row['title']} {row['categories']} {row['description']}"
#                 documents.append(Document(page_content=content))

#             embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#             self.vectorstore = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")
#             print("Vector store rebuilt.")

#         except Exception as e:
#             print(f"Rebuild failed: {str(e)}")
#             raise

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Book
import re


class Recommender:
    def __init__(self, database_url):
        self.engine = create_engine(database_url)
        self.Session = sessionmaker(bind=self.engine)
        
        # Initialize with lightweight components only
        self.books = None
        self.tfidf_matrix = None
        self.vectorizer = None
        
        print("Recommender initialized (lazy loading enabled)")

    def _load_data(self):
        """Load data only when needed"""
        if self.books is not None:
            return
            
        print("Loading books from database...")
        with self.Session() as session:
            books_list = session.query(Book).all()
            data = [{
                'isbn13': book.isbn13,
                'title': book.title or '',
                'authors': book.authors or '',
                'categories': book.categories or '',
                'thumbnail': book.thumbnail or '',
                'description': book.description or '',
            } for book in books_list]

        self.books = pd.DataFrame(data)
        if self.books.empty:
            print("Warning: No books loaded from DB")
            return
            
        self.books['isbn13'] = self.books['isbn13'].astype(str)
        print(f"Loaded {len(self.books)} books from DB")
        
        # Create search content and build TF-IDF matrix
        self._build_search_index()

    def _build_search_index(self):
        """Build lightweight TF-IDF search index"""
        if self.books is None or self.books.empty:
            return
            
        print("Building search index...")
        
        # Combine text fields for search
        search_content = []
        for _, row in self.books.iterrows():
            content = f"{row['title']} {row['categories']} {row['description']} {row['authors']}"
            # Clean and normalize text
            content = re.sub(r'[^\w\s]', ' ', content.lower())
            content = re.sub(r'\s+', ' ', content).strip()
            search_content.append(content)
        
        # Use TF-IDF instead of heavy embedding models
        self.vectorizer = TfidfVectorizer(
            max_features=1000,  # Limit features to reduce memory
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams
            min_df=1,
            max_df=0.8
        )
        
        self.tfidf_matrix = self.vectorizer.fit_transform(search_content)
        print(f"Search index built with {self.tfidf_matrix.shape} matrix")

    def get_recommendations(self, query, top_k=5, raw=False):
        """Get recommendations using TF-IDF similarity"""
        self._load_data()  # Lazy loading
        
        if self.books is None or self.books.empty:
            return []

        print(f"User query: '{query}'")
        
        # Clean query
        clean_query = re.sub(r'[^\w\s]', ' ', query.lower())
        clean_query = re.sub(r'\s+', ' ', clean_query).strip()
        
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([clean_query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top results with scores
        top_indices = similarities.argsort()[::-1][:min(top_k * 2, len(similarities))]
        
        # Apply keyword boosting for better results
        keywords = set(clean_query.split())
        scored_results = []
        
        for idx in top_indices:
            if similarities[idx] < 0.01:  # Skip very low similarity
                continue
                
            book = self.books.iloc[idx]
            score = similarities[idx]
            
            # Boost score for exact keyword matches
            boost = 0
            for keyword in keywords:
                if keyword in book['title'].lower():
                    boost += 0.3
                if keyword in book['categories'].lower():
                    boost += 0.2
                if keyword in book['authors'].lower():
                    boost += 0.2
            
            final_score = score + boost
            scored_results.append((book, final_score))
        
        # Sort by final score and get top results
        scored_results.sort(key=lambda x: x[1], reverse=True)
        top_results = scored_results[:top_k]
        
        print(f"Returning {len(top_results)} recommendations")
        
        if raw:
            return [{
                'title': book['title'],
                'author': book['authors'],
                'category': book['categories'],
                'description': book['description'],
                'thumbnail': book['thumbnail'],
                'score': float(score)  # Convert numpy float to Python float
            } for book, score in top_results]

        return [{
            'title': book['title'],
            'author': book['authors'],
            'category': book['categories'],
            'description': book['description'],
            'thumbnail': book['thumbnail']
        } for book, _ in top_results]

    def rebuild_vectorstore(self):
        """Rebuild the search index after adding new books"""
        try:
            print("Rebuilding search index...")
            # Reset data to force reload
            self.books = None
            self.tfidf_matrix = None
            self.vectorizer = None
            
            # Reload and rebuild
            self._load_data()
            print("Search index rebuilt successfully")
            
        except Exception as e:
            print(f"Rebuild failed: {str(e)}")
            raise

    def get_stats(self):
        """Get basic stats about the recommendation system"""
        self._load_data()
        return {
            'total_books': len(self.books) if self.books is not None else 0,
            'index_built': self.tfidf_matrix is not None
        }