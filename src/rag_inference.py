# ### RAG Inference
import os
import re
import requests
import numpy as np
import pandas as pd
import fitz
import docx2txt
import cohere
# LangChain core
from langchain.schema import Document
# LangChain core
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader, DirectoryLoader, CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.tools import tool
from langchain.schema.runnable import RunnablePassthrough

# LangChain provider integrations
from langchain_openai import OpenAI
from langchain_cohere import ChatCohere

# LangChain-Community extensions
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS as CommunityFAISS

# LangGraph
import langgraph
from langgraph.graph import StateGraph
from recommender_langgraph import RecommendationSystem
from config import DATA_DIR, IMPEL_CSV, PAPERS_DIR, VECTOR_DB, EMBED_MODEL, cohere_api_key
from helper_functions import load_impel_courses

def run_rag_inference(user_id, education, age_group, profession, user_query, uploaded_files):
    recommender = RecommendationSystem()
    response, similar_user_courses = recommender.handle_user_query(
        user_id=user_id,
        education=education,
        age_group=age_group,
        profession=profession,
        query=user_query
    )

    # If database agent handled the query
    if similar_user_courses is None:
        return {"error": response}, None

    return response, similar_user_courses