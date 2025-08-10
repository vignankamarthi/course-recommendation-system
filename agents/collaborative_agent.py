import cohere
from core.config import (
    cohere_api_key, COHERE_GENERATE_MODEL,
    get_mysql_connection, get_neo4j_connection
)


class CollaborativeAgent:
    """Agent for collaborative filtering recommendations based on similar users."""
    
    def __init__(self):
        self.neo4j = get_neo4j_connection()
        self.mysql = get_mysql_connection()
        self.cohere_client = cohere.Client(cohere_api_key)
        self.impel_data = self._load_impel_courses_and_modules()
    
    def _load_impel_courses_and_modules(self):
        """Load and format course data from MySQL for recommendations."""
        try:
            courses_data = self.mysql.get_courses()
            
            # Group by course name
            courses_dict = {}
            for row in courses_data:
                course_name = row['course_name']
                if course_name not in courses_dict:
                    courses_dict[course_name] = []
                courses_dict[course_name].append({
                    'module': row['module_name'],
                    'summary': row['module_summary']
                })
            
            # Format for LLM
            formatted = ""
            for course_name, modules in courses_dict.items():
                formatted += f"**Course: {course_name}**\nModules:\n"
                for module_info in modules:
                    formatted += f"- {module_info['module']}: {module_info['summary']}\n"
                formatted += "\n"
            
            return formatted.strip()
            
        except Exception as e:
            print(f"Warning: Could not load courses from MySQL: {e}")
            return "No course data available. Please check database connection."

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
            model=COHERE_GENERATE_MODEL,
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