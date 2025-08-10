from typing import List, Dict, Any
from tavily import TavilyClient

#TODO: Better docstring and comments all around

def web_search(query: str, api_key) -> List[Dict[str, Any]]:
    """
    Performs web search using Tavily API for real-time information.
    Used for current job market trends, tools, and industry insights.
    """
    tavily_client = TavilyClient(api_key=api_key)
    return tavily_client.search(query, top_k=5)