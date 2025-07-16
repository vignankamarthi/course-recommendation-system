from config import SPLITTER
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
import os
from typing import List, Dict

def load_impel_courses(path: str) -> List[Document]:
    df = __import__('pandas').read_excel(path)
    docs = []
    for _, r in df.iterrows():
        content = (
            f"Course Title: {r['Courses']}\n"
            f"Module: {r['Modules']}\n"
            f"Summary: {r['Summary']}"
        )
        docs.append(Document(page_content=content, metadata={"source": "impel"}))
    return docs

def load_research_papers(path: str) -> List[Document]:
    docs = []
    for fn in os.listdir(path):
        if fn.lower().endswith('.pdf'):
            loader = PyPDFLoader(os.path.join(path, fn))
            for d in loader.load():
                d.metadata = {**d.metadata, 'source': 'research_paper', 'filename': fn}
                docs.append(d)
    return SPLITTER.split_documents(docs)