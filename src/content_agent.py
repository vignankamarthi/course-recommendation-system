import re
import json
from typing import List, Dict, Any
import fitz
import docx2txt

from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain_cohere import ChatCohere

from config import COURSE_VS, PAPERS_DIR, EMBED_MODEL, tavily_api_key
from content_agent_tools import web_search
from content_agent_helper_functions import load_research_papers

# TODO: Add better comments and docstrings

class AgenticRAG:
    def __init__(self, cohere_key: str):
        # Initialize LLM and paper retrieverlaslty what are the three
        self.llm = ChatCohere(
            cohere_api_key=cohere_key,
            model="command-r",
            temperature=0
        )
        docs = load_research_papers(PAPERS_DIR)
        self.paper_vs = FAISS.from_documents(docs, EMBED_MODEL).as_retriever(
            search_kwargs={"k": 5}
        )

    def classify_query(self, query: str) -> Dict[str, Any]:
        # TODOL: Maybe improve initiliation prompt?
        prompt = (
            f"You are a parsing assistant. Given the user query:\n  '{query}'\n"
            "Return ONLY a JSON object with keys:\n"
            "  - is_relevant: boolean (true if about courses, job roles, or trending skills)\n"
            "  - intents: list from ['learn_courses','job_info','trending_skills']\n"
            "  - target_role: string\n"
            "  - domain: string\n"
            "Important: include all intents that apply.\n"
            "If user asks about switching roles or career paths, include both 'job_info' and 'learn_courses' in the intent.\n"
            "If user asks for trending skills, include 'trending_skills'.\n"
            "Example:\n"
            "{'is_relevant': true, 'intents':['learn_courses','job_info'], 'target_role':'Data Analyst', 'domain':''}"
        )
        resp = self.llm.invoke(prompt)
        text = resp.get('content') if isinstance(resp, dict) else getattr(resp, 'content', '')
        cleaned_text = re.sub(r"^```json\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned_text.strip())
        except Exception:
            ql = query.lower().strip()
            intents: List[str] = []
            target_role = ""

            if any(w in ql for w in ["skill", "trend", "latest"]):
                intents.append("trending_skills")
            if any(w in ql for w in ["job", "role", "career", "path"]):
                intents.append("job_info")
            if any(w in ql for w in ["course", "learn", "career", "path"]):
                intents.append("learn_courses")

            intents = list(dict.fromkeys(intents))
            return {
                "is_relevant": bool(intents),
                "intents": intents,
                "target_role": target_role,
                "domain": ""
            }

    def run(self, query: str, uploaded_files: List[str] = []) -> str:
        # Build resume text
        resume = ''
        for f in uploaded_files:
            if f.lower().endswith('.pdf'):
                pdf = fitz.open(f)
                resume += '\n'.join(page.get_text() for page in pdf)
            elif f.lower().endswith('.docx'):
                resume += docx2txt.process(f)

        # Classify query
        meta = self.classify_query(query)
        if not meta.get('is_relevant'):
            return (
                "Sorry—I’m optimized for course recommendations, job summaries, "
                "and trending skills. Please ask a related question."
            )

        sections: List[str] = []

        # Trending skills via Tavily + LLM
        if 'trending_skills' in meta['intents']:
            results = web_search(query, tavily_api_key)
            context = '\n'.join(
                f"- {r['title']}: {r.get('snippet','')}" if isinstance(r, dict) else f"- {r}"
                for r in results
            )
            # TODO: Improve prompt to be stronger?
            trend_prompt = (
                f"You are an industry analyst. User query: '{query}'.\n"
                "Based only on these search results (title and snippet):\n"
                f"{context}\n\n"
                "Summarize the top hard and soft skills as bullet points."
            )
            raw = self.llm.invoke(trend_prompt)
            content = raw.get('content') if isinstance(raw, dict) else raw.content
            sections.append(f"## 🔥 Trending Skills\n{content.strip()}")

        # Job info via Tavily + LLM
        if 'job_info' in meta['intents']:
            results = web_search(query, tavily_api_key)
            context = '\n'.join(
                f"- {r['title']}: {r.get('snippet','')}" if isinstance(r, dict) else f"- {r}"
                for r in results
            )
            job_prompt = (
                # TODO: Improve prompt to be more specific?
                f"You are a career advisor. User query: '{query}'.\n"
                "Based only on these search results (title and snippet):\n"
                f"{context}\n\n"
                "Summarize skills, salary, responsibilities in 3 bullet points."
            )
            raw = self.llm.invoke(job_prompt)
            content = raw.get('content') if isinstance(raw, dict) else raw.content
            role = meta.get('target_role') or query
            sections.append(f"## 📋 Job Role ({role})\n{content.strip()}")

        # Course recommendations
        if 'learn_courses' in meta['intents']:
            sections.append(self._build_course_section(resume, query))

        # Final assembly
        assemble_prompt = (
            #TODO: Improve this prompt to be more specific and clear, reducing incorrect outputs
            "Combine these sections exactly, preserving titles and bullets. "
            "Do NOT add intros or follow-ups and DO NOT make any content changes to the respective sections"
            "(If there are summaries for each research paper recommendation if any, do not exclude them. "
            "They're very important.)"
            "But do keep/add relevant icons to respective sub sections titles.\n\n" + "\n---\n".join(sections)
        )
        raw = self.llm.invoke(assemble_prompt)
        final = raw.content if hasattr(raw, 'content') else raw.get('content', str(raw))
        return final.strip()

    def _build_course_section(self, resume: str, query: str) -> str:
        q = resume + '\n' + query
        recs = COURSE_VS.similarity_search(q, k=3)
        lines = ['## 🎓 Top 3 IMPEL Course Reccommendations for You']
        for i, d in enumerate(recs, start=1):
            content = d.page_content or ''
            parts = content.splitlines()
            if ':' in parts[0]:
                title = parts[0].split(':', 1)[1].strip()
            else:
                title = parts[0][:60]
            if len(parts) > 1 and ':' in parts[1]:
                module = parts[1].split(':', 1)[1].strip()
            else:
                module = ''
            lines.append(f"{i}. {title} - Module: {module}")

        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.paper_vs,
            return_source_documents=True
        )
        res = qa.invoke({'query': f"Resume:\n{resume}\nQuery:\n{query}"})

        seen = set()
        papers = ['## 📄 Related Research Papers']
        for doc in res['source_documents']:
            fn = doc.metadata.get('filename', 'paper.pdf')
            if fn in seen:
                continue
            seen.add(fn)
            snippet = doc.page_content[:200].replace('\n', ' ')
            summ_raw = self.llm.invoke(f"Summarize this paper in 1-2 sentences: {snippet}")
            summ = summ_raw.get('content') if isinstance(summ_raw, dict) else summ_raw.content
            papers.append(f"- **{fn}**: {summ.strip()}")

        return '\n'.join(lines + papers)
