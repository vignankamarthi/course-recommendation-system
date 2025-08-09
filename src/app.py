# app.py

#TODO: Add better comments. 

from gradio_interface import create_gradio_interface
from content_agent import AgenticRAG
from config import cohere_api_key

if __name__ == "__main__":
    content_agent = AgenticRAG(cohere_key = cohere_api_key)
    create_gradio_interface()
