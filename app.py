# from flask import Flask, request, jsonify
# from recommend import Recommender
# from models import Book
# import os
# from sqlalchemy.exc import IntegrityError
# from dotenv import load_dotenv

# load_dotenv()

# app = Flask(__name__)

# DATABASE_URL = os.getenv("DATABASE_URL")
# recommender = Recommender(DATABASE_URL)

# @app.route('/recommend', methods=['POST'])
# def recommend():
#     data = request.get_json()
#     if not data or 'query' not in data:
#         return jsonify({'error': 'Query is required in the JSON body'}), 400
#     query = data.get('query')
#     top_k = data.get('top_k', 5)
#     raw = data.get('raw', False)
#     recommendations = recommender.get_recommendations(query, top_k, raw)
#     return jsonify({
#         'query': query,
#         'recommendations': recommendations
#     })

# @app.route('/add_book', methods=['POST'])
# def add_book():
#     data = request.get_json()
#     required_fields = ['isbn13', 'title', 'authors', 'categories', 'thumbnail', 'description', 'tagged_description']
#     if not all(field in data for field in required_fields):
#         return jsonify({'error': 'Missing required fields'}), 400
#     with recommender.Session() as session:
#         try:
#             new_book = Book(
#                 isbn13=str(data['isbn13']),
#                 title=data['title'],
#                 authors=data['authors'],
#                 categories=data['categories'],
#                 thumbnail=data['thumbnail'],
#                 description=data['description'],
#                 tagged_description=data['tagged_description']
#             )
#             session.add(new_book)
#             session.commit()
#             recommender.rebuild_vectorstore()
#             return jsonify({'message': 'Book added successfully'}), 201
#         except IntegrityError:
#             session.rollback()
#             return jsonify({'error': 'ISBN13 already exists'}), 409
#         except Exception as e:
#             session.rollback()
#             return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     port = int(os.environ.get("PORT", 5000))
#     app.run(host="0.0.0.0", port=port)


from flask import Flask, request, jsonify
from recommend import Recommender
from models import Book
import os
from sqlalchemy.exc import IntegrityError
from dotenv import load_dotenv
import gc

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure Flask for low memory usage
app.config['JSON_SORT_KEYS'] = False

DATABASE_URL = os.getenv("DATABASE_URL")

# Initialize recommender (will use lazy loading)
recommender = None

def get_recommender():
    """Get recommender instance with lazy initialization"""
    global recommender
    if recommender is None:
        recommender = Recommender(DATABASE_URL)
    return recommender

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        rec = get_recommender()
        stats = rec.get_stats()
        return jsonify({
            'status': 'healthy',
            'stats': stats
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/recommend', methods=['POST'])
def recommend():
    """Get book recommendations"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required in the JSON body'}), 400
        
        query = data.get('query', '').strip()
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
            
        top_k = min(data.get('top_k', 5), 10)  # Limit max results
        raw = data.get('raw', False)
        
        rec = get_recommender()
        recommendations = rec.get_recommendations(query, top_k, raw)
        
        # Force garbage collection to free memory
        gc.collect()
        
        return jsonify({
            'query': query,
            'recommendations': recommendations,
            'count': len(recommendations)
        })
        
    except Exception as e:
        print(f"Error in recommend: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/add_book', methods=['POST'])
def add_book():
    """Add a new book to the database"""
    try:
        data = request.get_json()
        required_fields = ['isbn13', 'title', 'authors', 'categories', 'thumbnail', 'description']
        
        # Validate required fields
        missing_fields = [field for field in required_fields if not data.get(field)]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        rec = get_recommender()
        with rec.Session() as session:
            try:
                # Check if book already exists
                existing_book = session.query(Book).filter_by(isbn13=str(data['isbn13'])).first()
                if existing_book:
                    return jsonify({'error': 'Book with this ISBN already exists'}), 409
                
                # Create new book
                new_book = Book(
                    isbn13=str(data['isbn13']),
                    title=data['title'].strip(),
                    authors=data['authors'].strip(),
                    categories=data['categories'].strip(),
                    thumbnail=data['thumbnail'].strip(),
                    description=data['description'].strip(),
                    tagged_description=data.get('tagged_description', data['description']).strip()
                )
                
                session.add(new_book)
                session.commit()
                
                # Rebuild search index
                rec.rebuild_vectorstore()
                
                # Force garbage collection
                gc.collect()
                
                return jsonify({
                    'message': 'Book added successfully',
                    'isbn': new_book.isbn13
                }), 201
                
            except IntegrityError:
                session.rollback()
                return jsonify({'error': 'ISBN13 already exists'}), 409
            except Exception as e:
                session.rollback()
                print(f"Database error: {str(e)}")
                return jsonify({'error': 'Database error occurred'}), 500
                
    except Exception as e:
        print(f"Error in add_book: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/books', methods=['GET'])
def list_books():
    """List all books (for debugging)"""
    try:
        rec = get_recommender()
        with rec.Session() as session:
            books = session.query(Book).all()
            book_list = [{
                'isbn13': book.isbn13,
                'title': book.title,
                'authors': book.authors,
                'categories': book.categories
            } for book in books]
            
        return jsonify({
            'books': book_list,
            'count': len(book_list)
        })
        
    except Exception as e:
        print(f"Error in list_books: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    # Configure for production
    app.run(
        host="0.0.0.0", 
        port=port,
        debug=False,  # Disable debug mode in production
        threaded=True
    )