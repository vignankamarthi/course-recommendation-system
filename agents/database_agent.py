import cohere
from core.config import (
    cohere_api_key, COHERE_GENERATE_MODEL, 
    get_mysql_connection, get_neo4j_connection
)


class DatabaseAgent:
    """Agent for direct database queries and course catalog lookup."""
    
    def __init__(self):
        self.neo4j = get_neo4j_connection()
        self.mysql = get_mysql_connection()
        self.cohere_client = cohere.Client(cohere_api_key)
        self.impel_data = self._load_impel_courses_and_modules()
    
    def _load_impel_courses_and_modules(self):
        """Load and format course data from MySQL for database queries."""
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
            model=COHERE_GENERATE_MODEL,
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