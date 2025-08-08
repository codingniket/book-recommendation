import gradio as gr
import requests

API_URL = "https://book-recommendation-1tpc.onrender.com"

def add_book(isbn13, title, authors, categories, thumbnail, description, tagged_description):
    payload = {
        'isbn13': isbn13,
        'title': title,
        'authors': authors,
        'categories': categories,
        'thumbnail': thumbnail,
        'description': description,
        'tagged_description': tagged_description
    }
    response = requests.post(API_URL, json=payload)
    if response.status_code == 201:
        return "Book added!"
    else:
        return response.json().get('error', 'An unknown error occurred')

with gr.Blocks(title="Add New Book") as demo:
    gr.Markdown("# Add New Book")
    
    isbn13 = gr.Textbox(label="ISBN13")
    title = gr.Textbox(label="Title")
    authors = gr.Textbox(label="Authors")
    categories = gr.Textbox(label="Categories")
    thumbnail = gr.Textbox(label="Thumbnail URL")
    description = gr.TextArea(label="Description")
    tagged_description = gr.TextArea(label="Tagged Description")
    
    output = gr.Textbox(label="Result", interactive=False)
    
    btn = gr.Button("Add Book")
    btn.click(
        add_book,
        inputs=[isbn13, title, authors, categories, thumbnail, description, tagged_description],
        outputs=output
    )

demo.launch()
