from flask import Flask, request, jsonify
from recommend import Recommender
from models import Book
import os
from sqlalchemy.exc import IntegrityError
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")
recommender = Recommender(DATABASE_URL)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({'error': 'Query is required in the JSON body'}), 400
    query = data.get('query')
    top_k = data.get('top_k', 5)
    raw = data.get('raw', False)
    recommendations = recommender.get_recommendations(query, top_k, raw)
    return jsonify({
        'query': query,
        'recommendations': recommendations
    })

@app.route('/add_book', methods=['POST'])
def add_book():
    data = request.get_json()
    required_fields = ['isbn13', 'title', 'authors', 'categories', 'thumbnail', 'description', 'tagged_description']
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400
    with recommender.Session() as session:
        try:
            new_book = Book(
                isbn13=str(data['isbn13']),
                title=data['title'],
                authors=data['authors'],
                categories=data['categories'],
                thumbnail=data['thumbnail'],
                description=data['description'],
                tagged_description=data['tagged_description']
            )
            session.add(new_book)
            session.commit()
            recommender.rebuild_vectorstore()
            return jsonify({'message': 'Book added successfully'}), 201
        except IntegrityError:
            session.rollback()
            return jsonify({'error': 'ISBN13 already exists'}), 409
        except Exception as e:
            session.rollback()
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)  
