import re
import pandas as pd
from tabulate import tabulate
from langchain.schema import Document

# -------- Structured Paper Formatter --------
def format_additional_recommendations(paper_sources):
    print(" Additional Recommendations (from research papers):\n")
    for doc in paper_sources:
        content = doc.page_content.strip()
        source = doc.metadata['source']
        print(f"📄 From: {source}")

        # Try to extract tables (e.g., skills + % or category listings)
        table_match = re.findall(r"([A-Za-z0-9\s\+\-\#\.]+?)\s+(\d+\s?%)\s*", content)
        if len(table_match) >= 3:
            table_data = []
            for row in table_match:
                table_data.append([cell.strip() for cell in row])
            print(tabulate(table_data, headers=["Skill/Category", "Usage %"], tablefmt="github"))
        else:
            # Print 5 key lines as bullet points
            lines = content.split("\n")
            print("📝 Insight:")
            for line in lines[:5]:
                print(f" - {line.strip()}")
        print("\n" + "-" * 80 + "\n")

def load_impel_courses(excel_path):
    df = pd.read_excel(excel_path)
    docs = []
    for _, row in df.iterrows():
        content = f"Course Title: {row['Courses']}\nModule: {row['Modules']}\nDetails: {row['Summary']}"
        docs.append(Document(page_content=content, metadata={"source": "impel_course"}))
    return docs
