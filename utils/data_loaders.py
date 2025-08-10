from core.config import SPLITTER
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
import os
from typing import List
from utils.logger import SystemLogger
from utils.exceptions import FileProcessingError, ConfigurationError

def load_research_papers(path: str) -> List[Document]:
    """
    Load and split research papers from PDF files in specified directory.
    
    Args:
        path: Directory path containing PDF files
    
    Returns:
        List of Document objects with chunked content
        
    Raises:
        FileProcessingError: If directory access or PDF processing fails
        ConfigurationError: If text splitter is not configured
    """
    SystemLogger.info("Loading research papers from directory", {
        'path': path
    })
    
    # Validate input path
    if not path:
        SystemLogger.error(
            "Research papers directory path is empty - Cannot load papers without valid path",
            context={'path': repr(path)}
        )
        raise FileProcessingError("Path cannot be empty")
    
    # Check if directory exists
    if not os.path.exists(path):
        SystemLogger.error(
            f"Research papers directory does not exist - Check if path is correct: {path}",
            context={'path': path, 'absolute_path': os.path.abspath(path)}
        )
        raise FileProcessingError(f"Directory not found: {path}")
    
    if not os.path.isdir(path):
        SystemLogger.error(
            f"Path is not a directory - Expected directory path, got file: {path}",
            context={'path': path, 'is_file': os.path.isfile(path)}
        )
        raise FileProcessingError(f"Path is not a directory: {path}")
    
    # Check directory permissions
    if not os.access(path, os.R_OK):
        SystemLogger.error(
            f"No read permission for research papers directory - Check directory permissions: {path}",
            context={'path': path}
        )
        raise FileProcessingError(f"No read permission for directory: {path}")
    
    # Validate text splitter configuration
    if not SPLITTER:
        SystemLogger.error(
            "Text splitter is not configured - Check SPLITTER initialization in config.py",
            context={'splitter_configured': SPLITTER is not None}
        )
        raise ConfigurationError("Text splitter not configured")
    
    try:
        # Get list of files
        SystemLogger.debug(f"Scanning directory for PDF files", {'path': path})
        all_files = os.listdir(path)
        pdf_files = [fn for fn in all_files if fn.lower().endswith('.pdf')]
        
        if not pdf_files:
            SystemLogger.info(f"No PDF files found in research papers directory", {
                'path': path,
                'total_files': len(all_files),
                'pdf_files': 0
            })
            return []
        
        SystemLogger.info(f"Found PDF files for processing", {
            'path': path,
            'pdf_count': len(pdf_files),
            'total_files': len(all_files)
        })
        
        docs = []
        failed_files = []
        
        for fn in pdf_files:
            file_path = os.path.join(path, fn)
            SystemLogger.debug(f"Processing PDF file", {
                'filename': fn,
                'file_path': file_path,
                'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0
            })
            
            try:
                # Check if file is readable
                if not os.access(file_path, os.R_OK):
                    SystemLogger.error(
                        f"No read permission for PDF file: {fn}",
                        context={'filename': fn, 'file_path': file_path}
                    )
                    failed_files.append(fn)
                    continue
                
                # Check file size
                file_size = os.path.getsize(file_path)
                if file_size == 0:
                    SystemLogger.error(
                        f"PDF file is empty: {fn}",
                        context={'filename': fn, 'file_size': file_size}
                    )
                    failed_files.append(fn)
                    continue
                
                # Load PDF
                loader = PyPDFLoader(file_path)
                file_docs = loader.load()
                
                if not file_docs:
                    SystemLogger.error(
                        f"PDF loader returned no documents for file: {fn}",
                        context={'filename': fn, 'file_path': file_path}
                    )
                    failed_files.append(fn)
                    continue
                
                # Add metadata and collect documents
                for d in file_docs:
                    d.metadata = {**d.metadata, 'source': 'research_paper', 'filename': fn}
                    docs.append(d)
                
                SystemLogger.debug(f"Successfully processed PDF file", {
                    'filename': fn,
                    'pages_loaded': len(file_docs),
                    'total_docs': len(docs)
                })
                
            except Exception as pdf_error:
                SystemLogger.error(
                    f"Failed to process PDF file: {fn}",
                    exception=pdf_error,
                    context={
                        'filename': fn,
                        'file_path': file_path,
                        'error_type': type(pdf_error).__name__
                    }
                )
                failed_files.append(fn)
                continue
        
        if failed_files:
            SystemLogger.info(f"Some PDF files failed to process", {
                'successful_files': len(pdf_files) - len(failed_files),
                'failed_files': len(failed_files),
                'failed_file_names': failed_files
            })
        
        if not docs:
            SystemLogger.error(
                "No documents loaded from PDF files - All PDF processing failed",
                context={
                    'path': path,
                    'pdf_files_found': len(pdf_files),
                    'failed_files': len(failed_files)
                }
            )
            raise FileProcessingError("Failed to load any PDF documents")
        
        # Split documents
        SystemLogger.debug(f"Splitting documents with text splitter", {
            'total_docs': len(docs),
            'splitter_type': type(SPLITTER).__name__
        })
        
        split_docs = SPLITTER.split_documents(docs)
        
        SystemLogger.info("Research papers loaded and split successfully", {
            'path': path,
            'pdf_files_processed': len(pdf_files) - len(failed_files),
            'failed_files': len(failed_files),
            'original_docs': len(docs),
            'split_docs': len(split_docs)
        })
        
        return split_docs
        
    except OSError as os_error:
        SystemLogger.error(
            f"Operating system error while accessing research papers directory",
            exception=os_error,
            context={'path': path, 'operation': 'directory_listing'}
        )
        raise FileProcessingError(f"OS error accessing directory: {os_error}")
        
    except Exception as e:
        SystemLogger.error(
            f"Unexpected error loading research papers",
            exception=e,
            context={
                'path': path,
                'error_type': type(e).__name__
            }
        )
        raise FileProcessingError(f"Failed to load research papers: {e}")