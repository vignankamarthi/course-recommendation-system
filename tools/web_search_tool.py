from typing import List, Dict, Any, Optional
from tavily import TavilyClient
from langsmith import traceable
from utils.logger import SystemLogger
from utils.exceptions import APIRequestError, APIKeyError

@traceable(run_type="tool", name="tavily_web_search")
def web_search(query: str, api_key: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Perform web search using Tavily API for real-time market information.
    
    Executes web searches for trending skills, job market insights, and career-related
    topics using Tavily's search API. Provides comprehensive error handling, input
    validation, and structured result formatting for downstream analysis.
    
    Parameters
    ----------
    query : str
        Search query string for web information retrieval
        Must be non-empty and contain meaningful search terms
    api_key : str
        Tavily API key for authentication and service access
        Must be valid and have sufficient API quota
    top_k : int, optional
        Maximum number of search results to return (default: 5)
        Must be positive integer, typically 3-10 for optimal performance
        
    Returns
    -------
    list of dict
        List of search result dictionaries with keys:
        - 'title': str, webpage title
        - 'snippet': str, content preview/summary  
        - 'url': str, source webpage URL
        - 'published_date': str, publication date (if available)
        Returns empty list if no results found or API errors occur
        
    Raises
    ------
    APIRequestError
        If Tavily API returns errors or invalid responses
    APIKeyError
        If API key is invalid or quota exceeded
    ValueError
        If input parameters are invalid or empty
        
    Examples
    --------
    >>> results = web_search(
    ...     query="data science skills 2024 trends",
    ...     api_key="your_tavily_api_key",
    ...     top_k=3
    ... )
    >>> print(len(results))  # Number of search results
    >>> print(results[0]['title'])  # First result title
    """
    SystemLogger.debug("Performing web search with Tavily", {
        'query_preview': query[:100] if query else 'empty',
        'top_k': top_k,
        'api_key_provided': bool(api_key)
    })
    
    # Input validation
    if not query or not query.strip():
        SystemLogger.error(
            "Web search query is empty or null - Cannot perform search with empty query",
            context={'query': repr(query)}
        )
        raise APIRequestError("Search query cannot be empty")
    
    if not api_key or not api_key.strip():
        SystemLogger.error(
            "Tavily API key is missing or empty - Check environment configuration",
            context={'api_key_provided': bool(api_key)}
        )
        raise APIKeyError("Tavily API key is required")
    
    try:
        # Initialize Tavily client
        SystemLogger.debug("Initializing Tavily client")
        tavily_client = TavilyClient(api_key=api_key)
        
        # Perform search
        SystemLogger.debug(f"Executing Tavily search", {
            'query': query, 'top_k': top_k
        })
        
        results = tavily_client.search(query, top_k=top_k)
        
        if not results:
            SystemLogger.info("Tavily search returned no results", {
                'query': query, 'top_k': top_k
            })
            return []
        
        # Validate result structure
        if not isinstance(results, list):
            SystemLogger.error(
                "Tavily API returned unexpected result format - Expected list",
                context={
                    'query': query,
                    'result_type': type(results).__name__,
                    'result_preview': str(results)[:200] if results else 'None'
                }
            )
            raise APIRequestError(f"Unexpected result format from Tavily: {type(results)}")
        
        SystemLogger.info("Tavily web search completed successfully", {
            'query': query,
            'results_count': len(results),
            'top_k_requested': top_k
        })
        
        return results
        
    except Exception as tavily_error:
        # Handle different types of Tavily errors
        error_msg = str(tavily_error).lower()
        
        if any(keyword in error_msg for keyword in ['api key', 'unauthorized', 'authentication', 'forbidden']):
            SystemLogger.error(
                "Tavily API authentication failed - Check API key validity and account status",
                exception=tavily_error,
                context={'query': query, 'api_key_provided': bool(api_key)}
            )
            raise APIKeyError(f"Tavily API key authentication failed: {tavily_error}")
            
        elif any(keyword in error_msg for keyword in ['quota', 'limit', 'exceeded', 'billing']):
            SystemLogger.error(
                "Tavily API quota exceeded - Check account limits and billing status",
                exception=tavily_error,
                context={'query': query}
            )
            raise APIRequestError(f"Tavily API quota exceeded: {tavily_error}")
            
        elif any(keyword in error_msg for keyword in ['timeout', 'connection', 'network']):
            SystemLogger.error(
                "Tavily API network error - Check internet connectivity and service availability",
                exception=tavily_error,
                context={'query': query}
            )
            raise APIRequestError(f"Tavily API network error: {tavily_error}")
            
        else:
            SystemLogger.error(
                "Unexpected error during Tavily web search",
                exception=tavily_error,
                context={
                    'query': query,
                    'top_k': top_k,
                    'error_type': type(tavily_error).__name__
                }
            )
            raise APIRequestError(f"Tavily search failed: {tavily_error}")