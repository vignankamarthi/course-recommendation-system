from core.config import SPLITTER
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
import os
from typing import List

def load_research_papers(path: str) -> List[Document]:
    docs = []
    for fn in os.listdir(path):
        if fn.lower().endswith('.pdf'):
            loader = PyPDFLoader(os.path.join(path, fn))
            for d in loader.load():
                d.metadata = {**d.metadata, 'source': 'research_paper', 'filename': fn}
                docs.append(d)
    return SPLITTER.split_documents(docs)