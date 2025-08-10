from typing import List, Dict, Any, Optional
from tavily import TavilyClient
from utils.logger import SystemLogger
from utils.exceptions import APIRequestError, APIKeyError

#TODO: Better docstring and comments all around

def web_search(query: str, api_key: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Performs web search using Tavily API for real-time information.
    Used for current job market trends, tools, and industry insights.
    
    Args:
        query: Search query string
        api_key: Tavily API key
        top_k: Number of results to return (default 5)
    
    Returns:
        List of search results with title, snippet, and URL
        
    Raises:
        APIKeyError: If API key is invalid or missing
        APIRequestError: If search request fails
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