import os
import pandas as pd
import cohere
from src.core.config import cohere_api_key, DATA_DIR
from src.database.neo4j_connector import Neo4jConnector


class DatabaseAgent:
    """Agent for direct database queries and course catalog lookup."""
    
    def __init__(self):
        self.neo4j = Neo4jConnector()
        self.cohere_client = cohere.Client(cohere_api_key)
        self.impel_data = self._load_impel_courses_and_modules(
            os.path.join(DATA_DIR, "Course_Module_New.xlsx")
        )
    
    def _load_impel_courses_and_modules(self, filepath):
        """Load and format course data for database queries."""
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

    def lookup_courses(self, query: str, user_context: dict) -> dict:
        """
        Handle direct course/module lookup queries.
        Returns course information without recommendation logic.
        """
        # Extract demographics for user vector
        vector = self.neo4j.get_user_vector(
            user_context["education"],
            user_context["age_group"],
            user_context["profession"],
            query
        )
        
        # Generate database lookup response
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

        # Get similar users' courses
        similar_users = self.neo4j.get_similar_users(vector)
        similar_user_courses = ""
        if similar_users:
            user_ids = [user["user_id"] for user in similar_users]
            courses = self.neo4j.get_enrolled_courses_from_similar_users(user_ids)
            similar_user_courses = "\n".join(f"- {c}" for c in sorted(set(courses))) if courses else "No similar users who enrolled for IMPEL courses found."
        else:
            similar_user_courses = "No similar users who enrolled for IMPEL courses found."

        return {
            "response": response.generations[0].text.strip(),
            "similar_user_courses": similar_user_courses,
            "user_vector": vector
        }