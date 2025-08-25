import re
import json
from typing import List, Dict, Any
import fitz
import docx2txt

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_cohere import ChatCohere
from langsmith import traceable

from core.config import COURSE_VS, PAPERS_DIR, EMBED_MODEL, tavily_api_key, COHERE_CHAT_MODEL
from tools.web_search_tool import web_search
from utils.data_loaders import load_research_papers
from utils.logger import SystemLogger
from utils.exceptions import (
    FileProcessingError, APIRequestError, VectorStoreError, 
    AgentExecutionError, ConfigurationError
)

class ContentAgent:
    """
    AI agent for content-based recommendations and market analysis.
    
    Specializes in processing user queries about trending skills, job market 
    insights, and content-based course recommendations. Integrates web search,
    document processing, and research paper analysis to provide comprehensive
    responses about career paths and skill development.
    
    Attributes
    ----------
    llm : ChatCohere
        Cohere chat model for natural language processing
    paper_vs : FAISS.as_retriever
        Vector store retriever for research papers
        
    Raises
    ------
    ConfigurationError
        If API keys or configuration parameters are invalid
    FileProcessingError
        If research paper loading fails
    AgentExecutionError
        If agent initialization fails
    """
    
    def __init__(self, cohere_key: str):
        SystemLogger.info("Initializing ContentAgent")
        
        try:
            # Validate API key
            if not cohere_key or not cohere_key.strip():
                SystemLogger.error(
                    "Cohere API key not provided to ContentAgent - Check initialization parameters",
                    context={'api_key_provided': bool(cohere_key)}
                )
                raise ConfigurationError("Cohere API key is required")
            
            # Initialize LLM
            SystemLogger.debug("Initializing ChatCohere LLM for ContentAgent")
            self.llm = ChatCohere(
                cohere_api_key=cohere_key,
                model=COHERE_CHAT_MODEL,
                temperature=0
            )
            
            # Load research papers
            SystemLogger.debug("Loading research papers for ContentAgent")
            if not PAPERS_DIR:
                SystemLogger.error(
                    "Papers directory not configured - Check PAPERS_DIR in config",
                    context={'papers_dir': PAPERS_DIR}
                )
                raise ConfigurationError("Papers directory not configured")
                
            docs = load_research_papers(PAPERS_DIR)
            
            if not docs:
                SystemLogger.error(
                    "No research papers loaded - Check papers directory and file permissions",
                    context={'papers_dir': PAPERS_DIR, 'docs_count': len(docs)}
                )
                raise FileProcessingError("No research papers loaded")
            
            # Initialize vector store
            SystemLogger.debug("Creating FAISS vector store for research papers")
            if not EMBED_MODEL:
                SystemLogger.error(
                    "Embedding model not configured - Check EMBED_MODEL in config",
                    context={'embed_model_configured': EMBED_MODEL is not None}
                )
                raise ConfigurationError("Embedding model not configured")
                
            self.paper_vs = FAISS.from_documents(docs, EMBED_MODEL).as_retriever(
                search_kwargs={"k": 5}
            )
            
            SystemLogger.info("ContentAgent initialized successfully", {
                'llm_model': COHERE_CHAT_MODEL,
                'papers_loaded': len(docs),
                'papers_dir': PAPERS_DIR,
                'embedding_model': type(EMBED_MODEL).__name__
            })
            
        except (ConfigurationError, FileProcessingError) as e:
            SystemLogger.error(
                "Configuration or file processing error initializing ContentAgent",
                exception=e,
                context={'cohere_key_provided': bool(cohere_key)}
            )
            raise
        except Exception as e:
            SystemLogger.error(
                "Unexpected error initializing ContentAgent",
                exception=e,
                context={'cohere_key_provided': bool(cohere_key)}
            )
            raise AgentExecutionError(f"Failed to initialize ContentAgent: {e}")

    @traceable(run_type="llm", name="content_agent_query_classification")
    def classify_query(self, query: str) -> Dict[str, Any]:
        """Classify user query to determine content agent intents."""
        SystemLogger.debug("Classifying query for ContentAgent", {
            'query_preview': query[:100] if query else 'empty'
        })
        
        if not query or not query.strip():
            SystemLogger.error(
                "Query classification failed - Empty query provided",
                context={'query': repr(query)}
            )
            raise AgentExecutionError("Query cannot be empty")
        
        try:
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
            
            SystemLogger.debug("Invoking LLM for query classification")
            resp = self.llm.invoke(prompt)
            
            if not resp:
                SystemLogger.error(
                    "LLM returned empty response for query classification",
                    context={'query': query, 'prompt_length': len(prompt)}
                )
                raise APIRequestError("LLM returned empty response")
                
            text = resp.get('content') if isinstance(resp, dict) else getattr(resp, 'content', '')
            
            if not text:
                SystemLogger.error(
                    "LLM returned empty text content for query classification",
                    context={'query': query, 'response_structure': str(resp)[:200]}
                )
                raise APIRequestError("LLM returned empty text")
            
            cleaned_text = re.sub(r"^```json\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
            
            try:
                result = json.loads(cleaned_text.strip())
                SystemLogger.debug("Query classification successful via LLM", {
                    'is_relevant': result.get('is_relevant', False),
                    'intents': result.get('intents', []),
                    'target_role': result.get('target_role', '')
                })
                return result
            except json.JSONDecodeError as json_error:
                SystemLogger.debug(
                    "JSON parsing failed for LLM classification - falling back to keyword matching",
                    context={'json_error': str(json_error), 'cleaned_text': cleaned_text[:200]}
                )
                
        except Exception as llm_error:
            SystemLogger.error(
                "Error during LLM query classification - falling back to keyword matching",
                exception=llm_error,
                context={'query': query}
            )
        
        # Fallback: keyword-based classification
        SystemLogger.debug("Using fallback keyword-based query classification")
        try:
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
            result = {
                "is_relevant": bool(intents),
                "intents": intents,
                "target_role": target_role,
                "domain": ""
            }
            
            SystemLogger.debug("Fallback keyword classification completed", {
                'is_relevant': result['is_relevant'],
                'intents': result['intents']
            })
            
            return result
            
        except Exception as fallback_error:
            SystemLogger.error(
                "Fallback keyword classification failed",
                exception=fallback_error,
                context={'query': query}
            )
            raise AgentExecutionError(f"Query classification failed: {fallback_error}")

    @traceable(run_type="agent", name="content_agent_analysis")
    @traceable(run_type="agent", name="content_agent_execution")
    def run(self, query: str, uploaded_files: List[str] = []) -> str:
        """
        Execute comprehensive content analysis and recommendation generation.
        
        Processes user queries through multi-modal analysis including file processing,
        web search for trending information, and research paper recommendations.
        Handles intent classification and routes to appropriate content generation
        workflows (trending skills, job information, course recommendations).
        
        Parameters
        ----------
        query : str
            User's natural language query for content analysis
        uploaded_files : list of str, optional
            File paths for uploaded resumes/documents (default: empty list)
            Supports PDF and DOCX formats for resume parsing
            
        Returns
        -------
        str
            Formatted response containing relevant content sections:
            - Trending skills analysis (if applicable)
            - Job market information (if applicable) 
            - Course recommendations with research papers (if applicable)
            - Error message for irrelevant queries
            
        Raises
        ------
        AgentExecutionError
            If query is empty or classification fails
        FileProcessingError
            If uploaded file processing encounters errors
        APIRequestError
            If external API calls (Cohere, Tavily) fail
        VectorStoreError
            If vector store operations fail
            
        Examples
        --------
        >>> agent = ContentAgent(cohere_key="your_key")
        >>> result = agent.run(
        ...     query="What are trending data science skills in 2024?",
        ...     uploaded_files=["/path/to/resume.pdf"]
        ... )
        """
        SystemLogger.info("ContentAgent executing main run method", {
            'query_preview': query[:100] if query else 'empty',
            'uploaded_files_count': len(uploaded_files) if uploaded_files else 0
        })
        
        # Input validation
        if not query or not query.strip():
            SystemLogger.error(
                "ContentAgent run method called with empty query",
                context={'query': repr(query)}
            )
            raise AgentExecutionError("Query cannot be empty")
        
        # Build resume text from uploaded files
        resume = ''
        if uploaded_files:
            SystemLogger.debug("Processing uploaded files for resume text", {
                'files_count': len(uploaded_files),
                'file_types': [f.split('.')[-1].lower() if '.' in f else 'unknown' for f in uploaded_files]
            })
            
            for f in uploaded_files:
                try:
                    if not f or not isinstance(f, str):
                        SystemLogger.debug("Skipping invalid file path", {'file': f})
                        continue
                        
                    if not f.strip():
                        SystemLogger.debug("Skipping empty file path")
                        continue
                    
                    SystemLogger.debug(f"Processing file", {'filename': f, 'file_type': f.split('.')[-1].lower()})
                    
                    if f.lower().endswith('.pdf'):
                        try:
                            pdf = fitz.open(f)
                            pdf_text = '\n'.join(page.get_text() for page in pdf)
                            pdf.close()
                            resume += pdf_text + '\n'
                            SystemLogger.debug("PDF file processed successfully", {
                                'filename': f, 'text_length': len(pdf_text)
                            })
                        except Exception as pdf_error:
                            SystemLogger.error(
                                f"Failed to process PDF file: {f}",
                                exception=pdf_error,
                                context={'filename': f}
                            )
                            # Continue with other files instead of failing completely
                            continue
                            
                    elif f.lower().endswith('.docx'):
                        try:
                            docx_text = docx2txt.process(f)
                            resume += docx_text + '\n'
                            SystemLogger.debug("DOCX file processed successfully", {
                                'filename': f, 'text_length': len(docx_text)
                            })
                        except Exception as docx_error:
                            SystemLogger.error(
                                f"Failed to process DOCX file: {f}",
                                exception=docx_error,
                                context={'filename': f}
                            )
                            # Continue with other files instead of failing completely
                            continue
                    else:
                        SystemLogger.debug("Skipping unsupported file type", {
                            'filename': f, 'file_type': f.split('.')[-1].lower() if '.' in f else 'unknown'
                        })
                        
                except Exception as file_error:
                    SystemLogger.error(
                        f"Unexpected error processing file: {f}",
                        exception=file_error,
                        context={'filename': f}
                    )
                    continue
            
            SystemLogger.info("File processing completed", {
                'files_processed': len([f for f in uploaded_files if f and f.strip()]),
                'resume_length': len(resume)
            })

        # Classify query to determine intents
        try:
            SystemLogger.debug("Classifying query to determine intents")
            meta = self.classify_query(query)
            
            if not meta or not isinstance(meta, dict):
                SystemLogger.error(
                    "Query classification returned invalid result",
                    context={'meta_type': type(meta), 'meta': meta}
                )
                raise AgentExecutionError("Query classification failed")
                
            SystemLogger.debug("Query classification completed", {
                'is_relevant': meta.get('is_relevant', False),
                'intents': meta.get('intents', [])
            })
            
        except Exception as classify_error:
            SystemLogger.error(
                "Error during query classification",
                exception=classify_error,
                context={'query': query}
            )
            raise AgentExecutionError(f"Query classification failed: {classify_error}")
        
        # Check if query is relevant
        if not meta.get('is_relevant'):
            SystemLogger.info("Query classified as irrelevant - returning standard message")
            return (
                "Sorry—I'm optimized for course recommendations, job summaries, "
                "and trending skills. Please ask a related question."
            )

        sections: List[str] = []
        intents = meta.get('intents', [])
        SystemLogger.info("Processing intents", {'intents': intents})

        # Trending skills via Tavily + LLM
        if 'trending_skills' in intents:
            try:
                SystemLogger.debug("Processing trending skills intent")
                
                if not tavily_api_key:
                    SystemLogger.error(
                        "Tavily API key not available for trending skills search",
                        context={'tavily_key_provided': bool(tavily_api_key)}
                    )
                    sections.append("## Trending Skills\nUnable to fetch trending skills - API key not configured.")
                else:
                    results = web_search(query, tavily_api_key)
                    
                    if not results:
                        SystemLogger.info("No web search results for trending skills")
                        sections.append("## Trending Skills\nNo current trending skills data available.")
                    else:
                        context = '\n'.join(
                            f"- {r.get('title', 'No title')}: {r.get('snippet', 'No snippet')}" 
                            if isinstance(r, dict) else f"- {str(r)}"
                            for r in results
                        )
                        
                        trend_prompt = (
                            f"You are an industry analyst. User query: '{query}'.\n"
                            "Based only on these search results (title and snippet):\n"
                            f"{context}\n\n"
                            "Summarize the top hard and soft skills as bullet points."
                        )
                        
                        raw = self.llm.invoke(trend_prompt)
                        if not raw:
                            SystemLogger.error("LLM returned empty response for trending skills")
                            sections.append("## Trending Skills\nUnable to analyze trending skills data.")
                        else:
                            content = raw.get('content') if isinstance(raw, dict) else getattr(raw, 'content', str(raw))
                            sections.append(f"## Trending Skills\n{content.strip()}")
                            SystemLogger.debug("Trending skills section generated successfully")
                        
            except Exception as trend_error:
                SystemLogger.error(
                    "Error processing trending skills intent",
                    exception=trend_error,
                    context={'query': query}
                )
                sections.append("## Trending Skills\nUnable to fetch trending skills due to system error.")

        # Job info via Tavily + LLM
        if 'job_info' in intents:
            try:
                SystemLogger.debug("Processing job info intent")
                
                if not tavily_api_key:
                    SystemLogger.error(
                        "Tavily API key not available for job info search",
                        context={'tavily_key_provided': bool(tavily_api_key)}
                    )
                    sections.append("## Job Information\nUnable to fetch job information - API key not configured.")
                else:
                    results = web_search(query, tavily_api_key)
                    
                    if not results:
                        SystemLogger.info("No web search results for job info")
                        sections.append("## Job Information\nNo current job information available.")
                    else:
                        context = '\n'.join(
                            f"- {r.get('title', 'No title')}: {r.get('snippet', 'No snippet')}" 
                            if isinstance(r, dict) else f"- {str(r)}"
                            for r in results
                        )
                        
                        job_prompt = (
                            f"You are a career advisor. User query: '{query}'.\n"
                            "Based only on these search results (title and snippet):\n"
                            f"{context}\n\n"
                            "Summarize skills, salary, responsibilities in 3 bullet points."
                        )
                        
                        raw = self.llm.invoke(job_prompt)
                        if not raw:
                            SystemLogger.error("LLM returned empty response for job info")
                            sections.append("## Job Information\nUnable to analyze job information data.")
                        else:
                            content = raw.get('content') if isinstance(raw, dict) else getattr(raw, 'content', str(raw))
                            role = meta.get('target_role') or query
                            sections.append(f"## Job Role ({role})\n{content.strip()}")
                            SystemLogger.debug("Job info section generated successfully")
                        
            except Exception as job_error:
                SystemLogger.error(
                    "Error processing job info intent",
                    exception=job_error,
                    context={'query': query}
                )
                sections.append("## Job Information\nUnable to fetch job information due to system error.")

        # Course recommendations
        if 'learn_courses' in intents:
            try:
                SystemLogger.debug("Processing learn courses intent")
                course_section = self._build_course_section(resume, query)
                sections.append(course_section)
                SystemLogger.debug("Course recommendations section generated successfully")
                
            except Exception as course_error:
                SystemLogger.error(
                    "Error processing learn courses intent",
                    exception=course_error,
                    context={'query': query, 'resume_length': len(resume)}
                )
                sections.append("## Course Recommendations\nUnable to generate course recommendations due to system error.")

        # Check if any sections were generated
        if not sections:
            SystemLogger.info("No sections generated for any intents")
            return "I wasn't able to process your request. Please try rephrasing your question about courses, jobs, or trending skills."

        try:
            # Final assembly
            SystemLogger.debug("Assembling final response from sections")
            assemble_prompt = (
                "Combine these sections exactly, preserving titles and bullets. "
                "Do NOT add intros or follow-ups and DO NOT make any content changes to the respective sections"
                "(If there are summaries for each research paper recommendation if any, do not exclude them. "
                "They're very important.)\n\n" + "\n---\n".join(sections)
            )
            
            raw = self.llm.invoke(assemble_prompt)
            if not raw:
                SystemLogger.error("LLM returned empty response for final assembly")
                # Fallback: return sections joined with separators
                final_response = "\n\n---\n\n".join(sections)
            else:
                final = raw.content if hasattr(raw, 'content') else raw.get('content', str(raw))
                final_response = final.strip() if final else "\n\n---\n\n".join(sections)
            
            SystemLogger.info("ContentAgent run completed successfully", {
                'query_preview': query[:100],
                'sections_generated': len(sections),
                'response_length': len(final_response),
                'intents_processed': intents
            })
            
            return final_response
            
        except Exception as assembly_error:
            SystemLogger.error(
                "Error during final response assembly",
                exception=assembly_error,
                context={'sections_count': len(sections)}
            )
            # Fallback: return sections joined with separators
            return "\n\n---\n\n".join(sections)

    @traceable(run_type="chain", name="build_course_recommendations_section")
    def _build_course_section(self, resume: str, query: str) -> str:
        """Build course recommendations section with IMPEL courses and research papers."""
        SystemLogger.debug("Building course recommendations section", {
            'resume_length': len(resume) if resume else 0,
            'query_preview': query[:50] if query else 'empty'
        })
        
        try:
            # Validate inputs
            if not query or not query.strip():
                SystemLogger.error(
                    "Empty query provided to course section builder",
                    context={'query': repr(query)}
                )
                return "## Course Recommendations\nUnable to generate recommendations - no query provided."
            
            # Prepare search query
            q = (resume + '\n' + query).strip() if resume else query.strip()
            
            SystemLogger.debug("Searching for similar courses", {
                'search_query_length': len(q),
                'vector_store_available': COURSE_VS is not None
            })
            
            # Course similarity search
            if not COURSE_VS:
                SystemLogger.error(
                    "Course vector store not available for similarity search",
                    context={'course_vs_configured': COURSE_VS is not None}
                )
                return "## Course Recommendations\nCourse database not available."
            
            try:
                recs = COURSE_VS.similarity_search(q, k=3)
                SystemLogger.debug("Course similarity search completed", {
                    'recommendations_found': len(recs) if recs else 0
                })
            except Exception as search_error:
                SystemLogger.error(
                    "Error during course similarity search",
                    exception=search_error,
                    context={'search_query_length': len(q)}
                )
                return "## Course Recommendations\nUnable to search course database."
            
            # Build course recommendations
            lines = ['## Top 3 IMPEL Course Recommendations for You']
            if not recs:
                lines.append("No matching courses found for your query.")
            else:
                for i, d in enumerate(recs, start=1):
                    try:
                        content = d.page_content or ''
                        if not content:
                            SystemLogger.debug(f"Empty content for recommendation {i}")
                            lines.append(f"{i}. Course information not available")
                            continue
                            
                        parts = content.splitlines()
                        if not parts:
                            SystemLogger.debug(f"No content parts for recommendation {i}")
                            lines.append(f"{i}. Course details not available")
                            continue
                        
                        # Extract title
                        if parts and ':' in parts[0]:
                            title = parts[0].split(':', 1)[1].strip()
                        elif parts:
                            title = parts[0][:60]
                        else:
                            title = "Course title not available"
                        
                        # Extract module
                        if len(parts) > 1 and ':' in parts[1]:
                            module = parts[1].split(':', 1)[1].strip()
                        else:
                            module = "Module information not available"
                            
                        lines.append(f"{i}. {title} - Module: {module}")
                        
                    except Exception as rec_error:
                        SystemLogger.error(
                            f"Error processing course recommendation {i}",
                            exception=rec_error,
                            context={'recommendation_index': i}
                        )
                        lines.append(f"{i}. Course processing error")

            # Research papers section
            SystemLogger.debug("Building research papers section")
            papers = ['## Related Research Papers']
            
            try:
                if not self.paper_vs:
                    SystemLogger.error(
                        "Paper vector store not available for research recommendations",
                        context={'paper_vs_configured': self.paper_vs is not None}
                    )
                    papers.append("Research papers database not available.")
                else:
                    qa = RetrievalQA.from_chain_type(
                        llm=self.llm,
                        retriever=self.paper_vs,
                        return_source_documents=True
                    )
                    
                    qa_query = f"Resume:\n{resume}\nQuery:\n{query}" if resume else query
                    SystemLogger.debug("Executing research paper retrieval", {
                        'qa_query_length': len(qa_query)
                    })
                    
                    res = qa.invoke({'query': qa_query})
                    
                    if not res or 'source_documents' not in res:
                        SystemLogger.error(
                            "Research paper QA returned invalid result",
                            context={'result_keys': list(res.keys()) if res else 'None'}
                        )
                        papers.append("Unable to retrieve research papers.")
                    else:
                        source_docs = res['source_documents']
                        if not source_docs:
                            SystemLogger.info("No research papers found for query")
                            papers.append("No relevant research papers found.")
                        else:
                            seen = set()
                            for doc in source_docs:
                                try:
                                    fn = doc.metadata.get('filename', 'paper.pdf')
                                    if fn in seen:
                                        continue
                                    seen.add(fn)
                                    
                                    snippet = doc.page_content[:200].replace('\n', ' ') if doc.page_content else "No content available"
                                    
                                    # Summarize paper
                                    try:
                                        summ_raw = self.llm.invoke(f"Summarize this paper in 1-2 sentences: {snippet}")
                                        if summ_raw:
                                            summ = summ_raw.get('content') if isinstance(summ_raw, dict) else getattr(summ_raw, 'content', 'Summary not available')
                                            papers.append(f"- **{fn}**: {summ.strip()}")
                                        else:
                                            papers.append(f"- **{fn}**: Summary not available")
                                    except Exception as summ_error:
                                        SystemLogger.error(
                                            f"Error summarizing paper: {fn}",
                                            exception=summ_error,
                                            context={'filename': fn}
                                        )
                                        papers.append(f"- **{fn}**: Unable to generate summary")
                                        
                                except Exception as doc_error:
                                    SystemLogger.error(
                                        "Error processing research paper document",
                                        exception=doc_error,
                                        context={'document_available': doc is not None}
                                    )
                                    continue
                            
                            SystemLogger.debug("Research papers section completed", {
                                'papers_processed': len(seen)
                            })
                            
            except Exception as papers_error:
                SystemLogger.error(
                    "Error building research papers section",
                    exception=papers_error,
                    context={'query': query, 'resume_length': len(resume)}
                )
                papers.append("Unable to retrieve research papers due to system error.")

            result = '\n'.join(lines + papers)
            SystemLogger.debug("Course section built successfully", {
                'total_lines': len(lines + papers),
                'result_length': len(result)
            })
            
            return result
            
        except Exception as section_error:
            SystemLogger.error(
                "Unexpected error building course section",
                exception=section_error,
                context={'query': query, 'resume_length': len(resume) if resume else 0}
            )
            return "## Course Recommendations\nUnable to generate course recommendations due to system error."
