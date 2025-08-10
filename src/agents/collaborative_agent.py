import os
import pandas as pd
import cohere
from src.core.config import cohere_api_key, DATA_DIR
from src.database.neo4j_connector import Neo4jConnector


class CollaborativeAgent:
    """Agent for collaborative filtering recommendations based on similar users."""
    
    def __init__(self):
        self.neo4j = Neo4jConnector()
        self.cohere_client = cohere.Client(cohere_api_key)
        self.impel_data = self._load_impel_courses_and_modules(
            os.path.join(DATA_DIR, "Course_Module_New.xlsx")
        )
    
    def _load_impel_courses_and_modules(self, filepath):
        """Load and format course data for recommendations."""
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

    def generate_recommendations(self, query: str, user_context: dict) -> dict:
        """
        Generate collaborative filtering recommendations based on similar users.
        """
        # Get user vector for similarity matching
        user_vector = self.neo4j.get_user_vector(
            user_context["education"], 
            user_context["age_group"], 
            user_context["profession"], 
            query
        )
        
        # Find similar users
        similar_users = self.neo4j.get_similar_users(user_vector)

        if not similar_users:
            response = "No similar users found. Here are some suggested IMPEL courses and modules:\n\n"
            prompt = f"""
You are a course recommendation assistant. Below are courses and their modules from the IMPEL database:
{self.impel_data}
User query: '{query}'
Suggest relevant courses and their modules.
Format:
**Course: <Course Name>**
- <Module 1>
- <Module 2>
"""
        else:
            response = "Recommended based on similar users' interests:\n\n"
            most_similar_query = similar_users[0]["query"]
            similar_user_recs = self.neo4j.get_recommendations_for_user(most_similar_query)
            prompt = f"""
You are a course recommendation assistant. Below are courses and their modules from the IMPEL database:
{self.impel_data}
A similar user was interested in: {similar_user_recs}
Current user query: '{query}'
Suggest relevant courses and modules for this user.
Format:
**Course: <Course Name>**
- <Module 1>
- <Module 2>
"""

        # Generate recommendation using LLM
        llm_response = self.cohere_client.generate(
            model='command-r-plus',
            prompt=prompt,
            max_tokens=400
        )
        response += llm_response.generations[0].text.strip()

        # Get courses similar users enrolled in
        similar_user_courses = ""
        if similar_users:
            similar_user_ids = [user["user_id"] for user in similar_users]
            enrolled_courses = self.neo4j.get_enrolled_courses_from_similar_users(similar_user_ids)
            if enrolled_courses:
                similar_user_courses = "\n".join(f"- {name}" for name in sorted(set(enrolled_courses)))
            else:
                similar_user_courses = "No similar users who enrolled for IMPEL courses found."
        else:
            similar_user_courses = "No similar users who enrolled for IMPEL courses found."

        return {
            "response": response,
            "similar_user_courses": similar_user_courses,
            "user_vector": user_vector
        }