import os
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# Set up relative paths for Hugging Face Spaces
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMPEL_CSV = os.path.join(DATA_DIR, "Course_Module_New.xlsx")
PAPERS_DIR = os.path.join(DATA_DIR, "papers")
VECTOR_DB = os.path.join(DATA_DIR, "vector_db")

neo4j_uri = "neo4j+s://2b228b65.databases.neo4j.io"
neo4j_user = "neo4j"
neo4j_password = "<neo4j-db-password>"
cohere_api_key = "<cohere-api-key>"
cohere_model = "embed-english-v3.0"
tavily_api_key = "<tavily-api-key>"

# Embedding & Splitter
EMBED_MODEL   = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
SPLITTER      = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

COURSE_VS = FAISS.from_documents(
    [
        Document(
            page_content=f"Course Title: {r['Courses']}\nModule: {r['Modules']}\nSummary: {r['Summary']}",
            metadata={"source": "impel"}
        ) for r in pd.read_excel(IMPEL_CSV).to_dict(orient='records')
    ],
    EMBED_MODEL
)
