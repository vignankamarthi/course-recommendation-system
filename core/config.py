import os
import sys
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from database.mysql_connector import MySQLConnector

# Set up relative paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PAPERS_DIR = os.path.join(DATA_DIR, "papers")

# Database configuration from environment variables
MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', 'course_recommendation')
MYSQL_USER = os.getenv('MYSQL_USER', 'root')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', 'rootpassword')
MYSQL_POOL_SIZE = int(os.getenv('MYSQL_POOL_SIZE', '5'))

# Neo4j configuration
neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
neo4j_password = os.getenv('NEO4J_PASSWORD', 'neo4jpassword')

# API Keys - fail fast if not provided
cohere_api_key = os.getenv('COHERE_API_KEY')
if not cohere_api_key:
    print("ERROR: COHERE_API_KEY not set in environment variables")
    sys.exit(1)

tavily_api_key = os.getenv('TAVILY_API_KEY')
if not tavily_api_key:
    print("ERROR: TAVILY_API_KEY not set in environment variables")
    sys.exit(1)

# Model configuration (now configurable via env)
COHERE_EMBED_MODEL = os.getenv('COHERE_EMBED_MODEL', 'embed-english-v3.0')
COHERE_GENERATE_MODEL = os.getenv('COHERE_GENERATE_MODEL', 'command-r-plus')
COHERE_CHAT_MODEL = os.getenv('COHERE_CHAT_MODEL', 'command-r')
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2')

# Embedding & Splitter (now configurable)
EMBED_MODEL = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
SPLITTER = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Database connection singletons
_mysql_connector = None
_neo4j_connector = None

def get_mysql_connection():
    """Get singleton MySQL connector with connection pooling."""
    global _mysql_connector
    if _mysql_connector is None:
        _mysql_connector = MySQLConnector(
            host=MYSQL_HOST,
            database=MYSQL_DATABASE,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            pool_size=MYSQL_POOL_SIZE
        )
    return _mysql_connector

def get_neo4j_connection():
    """Get singleton Neo4j connector."""
    global _neo4j_connector
    if _neo4j_connector is None:
        from database.neo4j_connector import Neo4jConnector
        _neo4j_connector = Neo4jConnector()
    return _neo4j_connector

def _load_course_vector_store():
    """Load course data from MySQL and create FAISS vector store."""
    try:
        mysql_connector = get_mysql_connection()
        
        # Get all courses and modules from MySQL
        courses_data = mysql_connector.get_courses()
        
        # Convert to Document format for FAISS
        documents = []
        for row in courses_data:
            page_content = f"Course Title: {row['course_name']}\nModule: {row['module_name']}\nSummary: {row['module_summary']}"
            documents.append(Document(
                page_content=page_content,
                metadata={"source": "impel_mysql", "course": row['course_name'], "module": row['module_name']}
            ))
        
        # Create and return FAISS vector store
        return FAISS.from_documents(documents, EMBED_MODEL)
    
    except Exception as e:
        # Fallback to empty vector store with warning
        print(f"Warning: Could not load courses from MySQL: {e}")
        print("Using empty vector store. Make sure MySQL is running and properly configured.")
        return FAISS.from_documents([
            Document(page_content="No courses available", metadata={"source": "fallback"})
        ], EMBED_MODEL)

# Initialize course vector store
COURSE_VS = _load_course_vector_store()
