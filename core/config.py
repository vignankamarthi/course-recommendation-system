import os
import sys
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from database.mysql_connector import MySQLConnector
from utils.logger import SystemLogger
from utils.exceptions import (
    ConfigurationError, DatabaseConnectionError, VectorStoreError
)

# Set up relative paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PAPERS_DIR = os.path.join(DATA_DIR, "papers")

# Database configuration from environment variables
try:
    MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
    MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', 'course_recommendation')
    MYSQL_USER = os.getenv('MYSQL_USER', 'root')
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', 'rootpassword')
    
    # Validate pool size configuration
    pool_size_str = os.getenv('MYSQL_POOL_SIZE', '5')
    try:
        MYSQL_POOL_SIZE = int(pool_size_str)
        if MYSQL_POOL_SIZE <= 0 or MYSQL_POOL_SIZE > 100:
            SystemLogger.error(
                "Invalid MySQL pool size - Must be between 1 and 100",
                context={'pool_size': MYSQL_POOL_SIZE}
            )
            raise ConfigurationError(f"Invalid MySQL pool size: {MYSQL_POOL_SIZE}")
    except ValueError as ve:
        SystemLogger.error(
            "MySQL pool size must be a valid integer",
            exception=ve,
            context={'pool_size_str': pool_size_str}
        )
        raise ConfigurationError(f"Invalid MySQL pool size format: {pool_size_str}")
        
    SystemLogger.info("MySQL configuration loaded successfully", {
        'host': MYSQL_HOST,
        'database': MYSQL_DATABASE,
        'user': MYSQL_USER,
        'pool_size': MYSQL_POOL_SIZE
    })
    
except Exception as e:
    SystemLogger.error(
        "Failed to load MySQL configuration - Check environment variables",
        exception=e,
        context={'available_vars': list(os.environ.keys())}
    )
    raise ConfigurationError(f"MySQL configuration failed: {e}")

# Neo4j configuration
try:
    neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
    neo4j_password = os.getenv('NEO4J_PASSWORD', 'neo4jpassword')
    
    # Validate Neo4j URI format
    if not neo4j_uri.startswith(('bolt://', 'neo4j://', 'neo4j+s://')):
        SystemLogger.error(
            "Invalid Neo4j URI format - Must start with bolt://, neo4j://, or neo4j+s://",
            context={'provided_uri': neo4j_uri}
        )
        raise ConfigurationError(f"Invalid Neo4j URI format: {neo4j_uri}")
    
    SystemLogger.info("Neo4j configuration loaded successfully", {
        'uri': neo4j_uri,
        'user': neo4j_user
    })
    
except Exception as e:
    SystemLogger.error(
        "Failed to load Neo4j configuration - Check environment variables",
        exception=e,
        context={'neo4j_uri': neo4j_uri}
    )
    raise ConfigurationError(f"Neo4j configuration failed: {e}")

# API Keys - fail fast if not provided
try:
    cohere_api_key = os.getenv('COHERE_API_KEY')
    if not cohere_api_key or not cohere_api_key.strip():
        SystemLogger.error(
            "Cohere API key not configured - Check COHERE_API_KEY environment variable",
            context={'api_key_provided': bool(cohere_api_key)}
        )
        raise ConfigurationError("COHERE_API_KEY not set in environment variables")
        
    # Basic validation - Cohere API keys should be reasonable length
    if len(cohere_api_key.strip()) < 20:
        SystemLogger.error(
            "Cohere API key appears too short - Check key format",
            context={'key_length': len(cohere_api_key.strip())}
        )
        raise ConfigurationError("Invalid Cohere API key format")
    
    tavily_api_key = os.getenv('TAVILY_API_KEY')
    if not tavily_api_key or not tavily_api_key.strip():
        SystemLogger.error(
            "Tavily API key not configured - Check TAVILY_API_KEY environment variable",
            context={'api_key_provided': bool(tavily_api_key)}
        )
        raise ConfigurationError("TAVILY_API_KEY not set in environment variables")
        
    # Basic validation - Tavily API keys should be reasonable length
    if len(tavily_api_key.strip()) < 20:
        SystemLogger.error(
            "Tavily API key appears too short - Check key format",
            context={'key_length': len(tavily_api_key.strip())}
        )
        raise ConfigurationError("Invalid Tavily API key format")
    
    SystemLogger.info("API keys validated successfully", {
        'cohere_key_length': len(cohere_api_key.strip()),
        'tavily_key_length': len(tavily_api_key.strip())
    })
    
except ConfigurationError as e:
    SystemLogger.error(
        "API key configuration failed - Application cannot start",
        exception=e,
        context={'initialization_step': 'api_key_validation'}
    )
    print(f"ERROR: {e}")
    sys.exit(1)
except Exception as e:
    SystemLogger.error(
        "Unexpected error during API key validation",
        exception=e,
        context={'initialization_step': 'api_key_validation'}
    )
    print(f"ERROR: Unexpected API key validation failure: {e}")
    sys.exit(1)

# Model configuration (now configurable via env)
try:
    COHERE_EMBED_MODEL = os.getenv('COHERE_EMBED_MODEL', 'embed-english-v3.0')
    COHERE_GENERATE_MODEL = os.getenv('COHERE_GENERATE_MODEL', 'command-r-plus')
    COHERE_CHAT_MODEL = os.getenv('COHERE_CHAT_MODEL', 'command-r')
    EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2')
    
    # Validate model names are not empty
    model_configs = {
        'COHERE_EMBED_MODEL': COHERE_EMBED_MODEL,
        'COHERE_GENERATE_MODEL': COHERE_GENERATE_MODEL,
        'COHERE_CHAT_MODEL': COHERE_CHAT_MODEL,
        'EMBEDDING_MODEL_NAME': EMBEDDING_MODEL_NAME
    }
    
    for model_name, model_value in model_configs.items():
        if not model_value or not model_value.strip():
            SystemLogger.error(
                f"Model configuration {model_name} is empty - Check environment variables",
                context={'model_name': model_name, 'model_value': model_value}
            )
            raise ConfigurationError(f"{model_name} cannot be empty")
    
    SystemLogger.info("Model configurations loaded successfully", {
        'cohere_embed_model': COHERE_EMBED_MODEL,
        'cohere_generate_model': COHERE_GENERATE_MODEL,
        'cohere_chat_model': COHERE_CHAT_MODEL,
        'embedding_model_name': EMBEDDING_MODEL_NAME
    })
    
except Exception as e:
    SystemLogger.error(
        "Failed to load model configurations - Check environment variables",
        exception=e,
        context={'initialization_step': 'model_configuration'}
    )
    raise ConfigurationError(f"Model configuration failed: {e}")

# Embedding & Splitter (now configurable)
try:
    SystemLogger.debug("Initializing HuggingFace embeddings model")
    EMBED_MODEL = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    SystemLogger.debug("Initializing text splitter")
    chunk_size = int(os.getenv('CHUNK_SIZE', '500'))
    chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '50'))
    
    if chunk_size <= 0 or chunk_size > 10000:
        SystemLogger.error(
            "Invalid chunk size - Must be between 1 and 10000",
            context={'chunk_size': chunk_size}
        )
        raise ConfigurationError(f"Invalid chunk size: {chunk_size}")
        
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        SystemLogger.error(
            "Invalid chunk overlap - Must be between 0 and chunk_size",
            context={'chunk_overlap': chunk_overlap, 'chunk_size': chunk_size}
        )
        raise ConfigurationError(f"Invalid chunk overlap: {chunk_overlap}")
    
    SPLITTER = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    SystemLogger.info("Embedding model and text splitter initialized successfully", {
        'embedding_model': EMBEDDING_MODEL_NAME,
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap
    })
    
except Exception as e:
    SystemLogger.error(
        "Failed to initialize embedding model or text splitter",
        exception=e,
        context={'embedding_model_name': EMBEDDING_MODEL_NAME}
    )
    raise ConfigurationError(f"Embedding/splitter initialization failed: {e}")

# Database connection singletons
_mysql_connector = None
_neo4j_connector = None

def get_mysql_connection():
    """Get singleton MySQL connector with connection pooling."""
    SystemLogger.debug("Acquiring MySQL connection singleton")
    
    global _mysql_connector
    if _mysql_connector is None:
        try:
            SystemLogger.debug("Creating new MySQL connector instance")
            _mysql_connector = MySQLConnector(
                host=MYSQL_HOST,
                database=MYSQL_DATABASE,
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                pool_size=MYSQL_POOL_SIZE
            )
            SystemLogger.info("MySQL connector singleton created successfully", {
                'host': MYSQL_HOST,
                'database': MYSQL_DATABASE,
                'pool_size': MYSQL_POOL_SIZE
            })
        except Exception as e:
            SystemLogger.error(
                "Failed to create MySQL connector singleton - Check database configuration",
                exception=e,
                context={
                    'host': MYSQL_HOST,
                    'database': MYSQL_DATABASE,
                    'user': MYSQL_USER,
                    'pool_size': MYSQL_POOL_SIZE
                }
            )
            raise DatabaseConnectionError(f"MySQL connector creation failed: {e}")
    
    SystemLogger.debug("Returning existing MySQL connector singleton")
    return _mysql_connector

def get_neo4j_connection():
    """Get singleton Neo4j connector."""
    SystemLogger.debug("Acquiring Neo4j connection singleton")
    
    global _neo4j_connector
    if _neo4j_connector is None:
        try:
            SystemLogger.debug("Creating new Neo4j connector instance")
            from database.neo4j_connector import Neo4jConnector
            _neo4j_connector = Neo4jConnector()
            
            SystemLogger.info("Neo4j connector singleton created successfully", {
                'uri': neo4j_uri,
                'user': neo4j_user
            })
        except Exception as e:
            SystemLogger.error(
                "Failed to create Neo4j connector singleton - Check database configuration",
                exception=e,
                context={
                    'uri': neo4j_uri,
                    'user': neo4j_user
                }
            )
            raise DatabaseConnectionError(f"Neo4j connector creation failed: {e}")
    
    SystemLogger.debug("Returning existing Neo4j connector singleton")
    return _neo4j_connector

def _load_course_vector_store():
    """Load course data from MySQL and create FAISS vector store."""
    SystemLogger.debug("Loading course data from MySQL to create FAISS vector store")
    
    try:
        # Get MySQL connection
        mysql_connector = get_mysql_connection()
        
        if not mysql_connector:
            SystemLogger.error(
                "MySQL connector not available for course vector store loading",
                context={'mysql_connector_available': mysql_connector is not None}
            )
            raise DatabaseConnectionError("MySQL connector not available")
        
        SystemLogger.debug("Fetching course data from MySQL")
        courses_data = mysql_connector.get_courses()
        
        if not courses_data:
            SystemLogger.info("No course data found in MySQL - creating empty vector store", {
                'courses_returned': len(courses_data) if courses_data else 0
            })
            # Create empty vector store for graceful degradation
            return FAISS.from_documents([
                Document(page_content="No courses available", metadata={"source": "empty_fallback"})
            ], EMBED_MODEL)
        
        SystemLogger.debug("Converting course data to Document format", {
            'total_courses': len(courses_data)
        })
        
        # Convert to Document format for FAISS
        documents = []
        processed_count = 0
        error_count = 0
        
        for i, row in enumerate(courses_data):
            try:
                if not row or not isinstance(row, dict):
                    SystemLogger.debug(f"Skipping invalid course data row {i}", {
                        'row_type': type(row), 'row_data': row
                    })
                    error_count += 1
                    continue
                
                # Validate required fields
                required_fields = ['course_name', 'module_name', 'module_summary']
                missing_fields = [field for field in required_fields if field not in row or not row[field]]
                if missing_fields:
                    SystemLogger.debug(f"Skipping course row {i} with missing fields", {
                        'missing_fields': missing_fields, 'row_data': row
                    })
                    error_count += 1
                    continue
                
                page_content = f"Course Title: {row['course_name']}\nModule: {row['module_name']}\nSummary: {row['module_summary']}"
                documents.append(Document(
                    page_content=page_content,
                    metadata={"source": "impel_mysql", "course": row['course_name'], "module": row['module_name']}
                ))
                processed_count += 1
                
            except Exception as row_error:
                SystemLogger.error(
                    f"Error processing course data row {i}",
                    exception=row_error,
                    context={'row_index': i, 'row_data': row}
                )
                error_count += 1
                continue
        
        if not documents:
            SystemLogger.error(
                "No valid documents created from course data - All rows had errors",
                context={'total_rows': len(courses_data), 'error_count': error_count}
            )
            # Create empty vector store as fallback
            return FAISS.from_documents([
                Document(page_content="No valid courses available", metadata={"source": "error_fallback"})
            ], EMBED_MODEL)
        
        SystemLogger.debug("Creating FAISS vector store from documents")
        
        # Create and return FAISS vector store
        vector_store = FAISS.from_documents(documents, EMBED_MODEL)
        
        SystemLogger.info("Course vector store created successfully", {
            'total_documents': len(documents),
            'processed_successfully': processed_count,
            'processing_errors': error_count,
            'embedding_model': EMBEDDING_MODEL_NAME
        })
        
        return vector_store
    
    except (DatabaseConnectionError, VectorStoreError) as e:
        SystemLogger.error(
            "Database or vector store error loading course data",
            exception=e,
            context={'initialization_step': 'course_vector_store'}
        )
        # Create fallback vector store to prevent total system failure
        return FAISS.from_documents([
            Document(page_content="Course data unavailable due to database error", metadata={"source": "db_error_fallback"})
        ], EMBED_MODEL)
    except Exception as e:
        SystemLogger.error(
            "Unexpected error loading course vector store - Using fallback",
            exception=e,
            context={'initialization_step': 'course_vector_store'}
        )
        # Create fallback vector store to prevent total system failure
        return FAISS.from_documents([
            Document(page_content="Course data unavailable due to system error", metadata={"source": "system_error_fallback"})
        ], EMBED_MODEL)

# Initialize course vector store
try:
    SystemLogger.debug("Initializing global course vector store")
    COURSE_VS = _load_course_vector_store()
    SystemLogger.info("Course vector store initialized successfully")
except Exception as e:
    SystemLogger.error(
        "Failed to initialize course vector store - System may not function properly",
        exception=e,
        context={'initialization_step': 'global_course_vs'}
    )
    # Create absolute fallback to prevent import errors
    COURSE_VS = FAISS.from_documents([
        Document(page_content="System initialization error", metadata={"source": "init_error_fallback"})
    ], EMBED_MODEL)
