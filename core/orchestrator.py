# LangGraph Workflow
import cohere
from langgraph.graph import StateGraph
from core.config import cohere_api_key, COHERE_GENERATE_MODEL, get_neo4j_connection
from agents.database_agent import DatabaseAgent
from agents.collaborative_agent import CollaborativeAgent
from agents.content_agent import ContentAgent


# TODO: Need better docstring and comments throughout.
class RecommendationSystem:
    """Orchestrates different agents based on user query intent."""
    
    def __init__(self):
        self.workflow = StateGraph(dict)
        self.neo4j = get_neo4j_connection()
        self.cohere_client = cohere.Client(cohere_api_key)
        self.database_agent = DatabaseAgent()
        self.collaborative_agent = CollaborativeAgent()
        self.content_agent = ContentAgent(cohere_key=cohere_api_key)


# TODO: Explore improving this initilization prompt
    def classify_intent(self, query):
        prompt = f"""
You are an intent classification assistant. Categorize the user's query as one of the following:
- "database_lookup": if they want to list or explore specific IMPEL courses/modules or descriptions.
- "recommendation": if they are asking what course suits their goal, background, or if they are exploring learning paths, skills or roles in the broad spectrum of Data Science or AI (e.g., how to become a data scientist, what an ML Engineer does, data scientist average salary, etc.).
- "content_analysis": if they are asking about trending skills, job market insights, or want content-based recommendations with research papers.
- "irrelevant": if the query is clearly unrelated to Data Science, AI, Information Technology or education, such as questions about movies, cooking, weather, jokes, casual greetings or personal life.


User query: "{query}"
Only reply with one of the following words: "database_lookup", "recommendation", "content_analysis", or "irrelevant".
"""
        response = self.cohere_client.generate(
            model=COHERE_GENERATE_MODEL,
            prompt=prompt,
            max_tokens=5,
            temperature=0
        )
        return response.generations[0].text.strip().lower()

    def collect_user_data(self, state):
        vector = self.neo4j.get_user_vector(
            state["education"], state["age_group"], state["profession"], state["query"]
        )
        state["user_vector"] = vector
        #TODO: Demographics is misleading THROUGHOUT THE CODE BASE, from general understanding, demographics means ethinic, socioeconomic, etc., but here, demogrpahics is just education, age_group, and profession. We need to be clear on that.
        state["demographics"] = {
            "education": state["education"],
            "age_group": state["age_group"],
            "profession": state["profession"]
        }
        return state

    def generate_recommendations(self, state):
        """Delegate to CollaborativeAgent for recommendations."""
        user_context = {
            "education": state["education"],
            "age_group": state["age_group"],
            "profession": state["profession"]
        }
        
        result = self.collaborative_agent.generate_recommendations(
            query=state["query"], 
            user_context=user_context
        )
        
        state["response"] = result["response"]
        state["similar_user_courses"] = result["similar_user_courses"]
        state["user_vector"] = result["user_vector"]
        return state

    def run_database_agent(self, state):
        """Delegate to DatabaseAgent for course lookups."""
        user_context = {
            "education": state["education"],
            "age_group": state["age_group"],
            "profession": state["profession"]
        }
        
        result = self.database_agent.lookup_courses(
            query=state["query"],
            user_context=user_context
        )
        
        state["response"] = result["response"]
        state["similar_user_courses"] = result["similar_user_courses"]
        state["user_vector"] = result["user_vector"]
        state["demographics"] = user_context
        return state

    def run_content_agent(self, state):
        """Delegate to ContentAgent for content-based recommendations and market analysis."""
        uploaded_files = state.get("uploaded_files", [])
        
        # ContentAgent handles its own logic and returns formatted response
        result = self.content_agent.run(
            query=state["query"],
            uploaded_files=uploaded_files
        )
        
        state["response"] = result
        state["similar_user_courses"] = ""  # ContentAgent doesn't use collaborative data
        return state

    def store_result(self, state):
        demographics = state.get("demographics", {})
        self.neo4j.store_interaction(
            user_id=state["user_id"],
            education=demographics.get("education", ""),
            age_group=demographics.get("age_group", ""),
            profession=demographics.get("profession", ""),
            user_query=state["query"],
            response=state["response"],
            user_vector=state.get("user_vector", [])
        )
        return state

    def build_workflow(self):
        graph = StateGraph(dict)
        graph.add_node("collect_data", self.collect_user_data)
        graph.add_node("generate_recs", self.generate_recommendations)
        graph.add_node("store_result", self.store_result)

        graph.set_entry_point("collect_data")
        graph.add_edge("collect_data", "generate_recs")
        graph.add_edge("generate_recs", "store_result")

        return graph.compile()

    def build_database_lookup_workflow(self):
        graph = StateGraph(dict)
        graph.add_node("collect_data", self.collect_user_data)
        graph.add_node("run_database_agent", self.run_database_agent)
        graph.add_node("store_result", self.store_result)

        graph.set_entry_point("collect_data")
        graph.add_edge("collect_data", "run_database_agent")
        graph.add_edge("run_database_agent", "store_result")

        return graph.compile()

    def build_content_workflow(self):
        graph = StateGraph(dict)
        graph.add_node("collect_data", self.collect_user_data)
        graph.add_node("run_content_agent", self.run_content_agent)
        graph.add_node("store_result", self.store_result)

        graph.set_entry_point("collect_data")
        graph.add_edge("collect_data", "run_content_agent")
        graph.add_edge("run_content_agent", "store_result")

        return graph.compile()

    def handle_user_query(self, user_id, education, age_group, profession, query, uploaded_files=None):
        intent = self.classify_intent(query)

        state = {
            "user_id": user_id,
            "education": education,
            "age_group": age_group,
            "profession": profession,
            "query": query,
            "uploaded_files": uploaded_files or []
        }

        if intent == "recommendation":
            app = self.build_workflow()
            final_state = app.invoke(state)
            return final_state["response"], final_state["similar_user_courses"]

        elif intent == "database_lookup":
            app = self.build_database_lookup_workflow()
            final_state = app.invoke(state)
            return final_state["response"], final_state["similar_user_courses"]

        elif intent == "content_analysis":
            app = self.build_content_workflow()
            final_state = app.invoke(state)
            return final_state["response"], final_state.get("similar_user_courses", "")

        elif intent == "irrelevant":
            return (
                "Sorry, your query seems unrelated to our course offerings. "
                "Please ask about IMPEL courses or Data Science learning goals.",
                None
            )

        else:
            return (
                "Sorry, I couldn't understand your request. Please try again.",
                None
            )
