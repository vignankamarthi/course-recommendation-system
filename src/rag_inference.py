#TODO: WAYYYYY TOOO MANY UNUSED IMPORTS HOLY

#TODO: This file is misleadingly named - it's not actually performing RAG (Retrieval-Augmented Generation).
# Currently just a pass-through wrapper to RecommendationSystem. The actual RAG logic lives in:
# - content_agent.py: FAISS vector retrieval + RetrievalQA chains
# - recommender_langgraph.py: Prompt augmentation with course data
#
# REFACTOR PLAN - Transform this into a centralized RAG tool:
# 1. Convert this file into a proper RAG tool that agents can call
# 2. Extract and centralize all RAG operations here:
#    - FAISS vector search from content_agent.py
#    - Paper retrieval and summarization from content_agent.py
#    - Course data augmentation from recommender_langgraph.py
# 3. Implement as a @tool decorated function that agents can invoke:
#    @tool
#    def rag_course_recommendations(query, user_context):
#        # Retrieve relevant courses from FAISS
#        # Retrieve related papers
#        # Augment prompt with context
#        # Generate recommendations via LLM
#        return structured_recommendations
# 4. Agents become lightweight routers that choose tools (RAG, web search, DB lookup)
#    rather than implementing retrieval logic themselves
#
# BENEFITS:
# - Single source of truth for all RAG operations (SRP)
# - Agents focus on orchestration, not implementation
# - Easy to test, modify, and extend RAG pipeline
# - Clear separation: Agents = decision makers, Tools = executors (better for error handling)

# ### RAG Inference
from recommender_langgraph import RecommendationSystem

def run_rag_inference(user_id, education, age_group, profession, user_query, uploaded_files):
    recommender = RecommendationSystem()
    response, similar_user_courses = recommender.handle_user_query(
        user_id=user_id,
        education=education,
        age_group=age_group,
        profession=profession,
        query=user_query
    )

    # If database agent handled the query
    if similar_user_courses is None:
        return {"error": response}, None

    return response, similar_user_courses