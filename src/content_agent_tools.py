from typing import List, Dict, Any, Union
import requests
from langchain.tools import tool
from tavily import TavilyClient

#TODO: Better docstring and commetns all around

def web_search(query: str, api_key) -> List[Dict[str, Any]]:
    tavily_client = TavilyClient(api_key=api_key)
    return tavily_client.search(query, top_k=5)

@tool
def get_job_details(job_title: str = "") -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """Fetch job details from static Gist. If job_title is empty, return full jobs list."""
    url = (
        "https://gist.githubusercontent.com/sanidhya-karnik/"
        "190f353794b931b64d972b9c0e5cb49d492b221ac71b632c667f/"
        "raw/ffe8608f635051eda28f492b221ac71b632c667f/gistfile2.txt"
    )
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return {"error": "Unable to fetch job details."}

    jobs = data.get("jobs", [])
    if not job_title:
        return jobs

    for item in jobs:
        if job_title.lower() == item.get("title", "").lower():
            return item
    return {"error": "Job title not found"}

@tool
def get_trending_tools(domain: str) -> Dict[str, Any]:
    """Fetch trending tools/tech stack for a given domain, safely parsing JSON."""
    url = (
        "https://gist.githubusercontent.com/sanidhya-karnik/"
        "20e2b549a171b4adae179c75c494700c/raw/fa12c77ac7f8513a1c2e6f7f0ea3d6ae82d89862/gistfile1.txt"
    )
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {"error": "Unable to fetch trending tools."}