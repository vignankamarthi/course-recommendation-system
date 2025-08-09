# LangGraph Workflow
import os
import pandas as pd
import cohere
from langchain_cohere import ChatCohere
from langgraph.graph import StateGraph
from config import cohere_api_key
from neo4j_connector import Neo4jConnector
from config import DATA_DIR


# TODO: Need better docstring and comments throughout.

# TODO: Need better strucutre, as we have three flows: irrelevant, database_lookup, and recommendation, and that is not clear. 
class RecommendationSystem:
    def __init__(self):
        self.workflow = StateGraph(dict)
        self.neo4j = Neo4jConnector()
        self.cohere_client = cohere.Client(cohere_api_key)
        self.impel_data = self.load_impel_courses_and_modules(
            os.path.join(DATA_DIR, "Course_Module_New.xlsx")
        )


# TODO: Figure out how to acheive same eng goal with MySQL transition
    def load_impel_courses_and_modules(self, filepath):
        df = pd.read_excel(filepath)
        grouped = df.groupby("Courses")
        formatted = ""
        for course, group in grouped:
            formatted += f"**Course: {course}**\nModules:\n"
            for _, row in group.iterrows():
                module = row["Modules"].strip()
                summary = row["Summary"].strip()
                formatted += f"- {module}: {summary}\n"
            formatted += "\n"
        return formatted.strip()

# TODO: Explore improving this initilization prompt
    def classify_intent(self, query):
        prompt = f"""
You are an intent classification assistant. Categorize the user's query as one of the following:
- "database_lookup": if they want to list or explore specific IMPEL courses/modules or descriptions.
- "recommendation": if they are asking what course suits their goal, background, or if they are exploring learning paths, skills or roles in the broad spectrum of Data Science or AI (e.g., how to become a data scientist, what an ML Engineer does, data scientist average salary, etc.).
- "irrelevant": if the query is clearly unrelated to Data Science, AI, Information Technology or education, such as questions about movies, cooking, weather, jokes, casual greetings or personal life.


User query: "{query}"
Only reply with one of the following words: "database_lookup", "recommendation", or "irrelevant".
"""
        response = self.cohere_client.generate(
            model="command-r-plus",
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
        #TODO: Demogrpahics is misleading THORUGHOUT THE CODE BASE, from general understanding, demogrpahics means ethinic, socioeconomic, etc., but here, demogrpahics is just education, age_group, and profession. We need to be clear on that.
        state["demographics"] = {
            "education": state["education"],
            "age_group": state["age_group"],
            "profession": state["profession"]
        }
        return state

    def generate_recommendations(self, state):
        user_vector = state["user_vector"]
        similar_users = self.neo4j.get_similar_users(user_vector)

        if not similar_users:
            state["response"] = "No similar users found. Here are some suggested IMPEL courses and modules:\n\n"
            prompt = f"""
You are a course recommendation assistant. Below are courses and their modules from the IMPEL database:
{self.impel_data}
User query: '{state['query']}'
Suggest relevant courses and their modules.
Format:
**Course: <Course Name>**
- <Module 1>
- <Module 2>
"""
        else:
            state["response"] = "Recommended based on similar users’ interests:\n\n"
            most_similar_query = similar_users[0]["query"]
            similar_user_recs = self.neo4j.get_recommendations_for_user(most_similar_query)
            prompt = f"""
You are a course recommendation assistant. Below are courses and their modules from the IMPEL database:
{self.impel_data}
A similar user was interested in: {similar_user_recs}
Current user query: '{state['query']}'
Suggest relevant courses and modules for this user.
Format:
**Course: <Course Name>**
- <Module 1>
- <Module 2>
"""

        response = self.cohere_client.generate(
            model='command-r-plus',
            prompt=prompt,
            max_tokens=400
        )
        state["response"] += response.generations[0].text.strip()

        similar_user_ids = [user["user_id"] for user in similar_users]
        enrolled_courses = self.neo4j.get_enrolled_courses_from_similar_users(similar_user_ids)

        if enrolled_courses:
            state["similar_user_courses"] = "\n".join(f"- {name}" for name in sorted(set(enrolled_courses)))
        else:
            state["similar_user_courses"] = "No similar users who enrolled for IMPEL courses found."

        return state

    def run_database_agent(self, state):
        query = state["query"]
        demographics = {
            "education": state["education"],
            "age_group": state["age_group"],
            "profession": state["profession"]
        }

        vector = self.neo4j.get_user_vector(
            demographics["education"],
            demographics["age_group"],
            demographics["profession"],
            query
        )
        state["user_vector"] = vector
        state["demographics"] = demographics

        prompt = f"""
You are an educational assistant helping users explore a database of courses and modules from the IMPEL program.
Here is the available data:
{self.impel_data}
User Query: "{query}"
Task:
1. If the user asks for all courses and modules, return all course names and their modules' names. 
2. If asking for modules under a course, return only those.
3. If keyword/topic search, return matching courses.
Format:
**Course: <Course>**
- <Module>: <Summary>
"""
        response = self.cohere_client.generate(
            model="command-r-plus",
            prompt=prompt,
            max_tokens=2000,
            temperature=0.3
        )

        state["response"] = response.generations[0].text.strip()

        similar_users = self.neo4j.get_similar_users(vector)
        if similar_users:
            user_ids = [user["user_id"] for user in similar_users]
            courses = self.neo4j.get_enrolled_courses_from_similar_users(user_ids)
            state["similar_user_courses"] = "\n".join(f"- {c}" for c in sorted(set(courses))) if courses else "No similar users who enrolled for IMPEL courses found."
        else:
            state["similar_user_courses"] = "No similar users who enrolled for IMPEL courses found."

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

    def handle_user_query(self, user_id, education, age_group, profession, query):
        intent = self.classify_intent(query)

        state = {
            "user_id": user_id,
            "education": education,
            "age_group": age_group,
            "profession": profession,
            "query": query,
        }

        if intent == "recommendation":
            app = self.build_workflow()
            final_state = app.invoke(state)
            return final_state["response"], final_state["similar_user_courses"]

        elif intent == "database_lookup":
            app = self.build_database_lookup_workflow()
            final_state = app.invoke(state)
            return final_state["response"], final_state["similar_user_courses"]

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
