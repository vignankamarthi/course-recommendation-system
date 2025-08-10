import cohere
from core.config import (
    cohere_api_key, COHERE_GENERATE_MODEL, 
    get_mysql_connection, get_neo4j_connection
)
from utils.logger import SystemLogger
from utils.exceptions import (
    DatabaseConnectionError, DatabaseQueryError, APIRequestError, 
    AgentExecutionError, ConfigurationError
)


class DatabaseAgent:
    """
    AI agent specialized in direct database queries and course catalog exploration.
    
    Handles user queries for specific IMPEL course and module information through
    direct database lookups. Provides detailed course descriptions, module summaries,
    and similar user enrollment patterns for educational guidance.
    
    Attributes
    ----------
    neo4j : Neo4jConnector
        Neo4j database connection for user similarity and interactions
    mysql : MySQLConnector  
        MySQL database connection for course catalog data
    cohere_client : cohere.Client
        Cohere API client for natural language response generation
    impel_data : str
        Formatted string of course and module information from database
        
    Raises
    ------
    ConfigurationError
        If required API keys or database configurations are invalid
    DatabaseConnectionError
        If database connections cannot be established
    AgentExecutionError
        If agent initialization fails
    """
    
    def __init__(self):
        SystemLogger.info("Initializing DatabaseAgent")
        
        try:
            # Initialize database connections
            SystemLogger.debug("Acquiring database connections for DatabaseAgent")
            self.neo4j = get_neo4j_connection()
            self.mysql = get_mysql_connection()
            
            # Initialize Cohere client
            SystemLogger.debug("Initializing Cohere client for DatabaseAgent")
            if not cohere_api_key:
                SystemLogger.error(
                    "Cohere API key not configured for DatabaseAgent - Check environment variables",
                    context={'api_key_provided': bool(cohere_api_key)}
                )
                raise ConfigurationError("Cohere API key not configured")
            
            self.cohere_client = cohere.Client(cohere_api_key)
            
            # Load course data
            SystemLogger.debug("Loading IMPEL course data from MySQL")
            self.impel_data = self._load_impel_courses_and_modules()
            
            SystemLogger.info("DatabaseAgent initialized successfully", {
                'mysql_connected': self.mysql is not None,
                'neo4j_connected': self.neo4j is not None,
                'cohere_configured': bool(cohere_api_key),
                'course_data_loaded': len(self.impel_data) > 0
            })
            
        except (DatabaseConnectionError, ConfigurationError) as e:
            SystemLogger.error(
                "Failed to initialize DatabaseAgent - Database or configuration error",
                exception=e,
                context={'initialization_step': 'database_connections'}
            )
            raise
        except Exception as e:
            SystemLogger.error(
                "Unexpected error initializing DatabaseAgent",
                exception=e,
                context={'initialization_step': 'unknown'}
            )
            raise AgentExecutionError(f"Failed to initialize DatabaseAgent: {e}")
    
    def _load_impel_courses_and_modules(self):
        """Load and format course data from MySQL for database queries."""
        SystemLogger.debug("Loading and formatting IMPEL course data from MySQL")
        
        try:
            if not self.mysql:
                SystemLogger.error(
                    "MySQL connection not available for course loading - Check database initialization",
                    context={'mysql_connector_available': self.mysql is not None}
                )
                raise DatabaseConnectionError("MySQL connection not available")
            
            courses_data = self.mysql.get_courses()
            
            if not courses_data:
                SystemLogger.info("No course data found in MySQL database", {
                    'courses_returned': len(courses_data) if courses_data else 0
                })
                return "No course data available in database."
            
            SystemLogger.debug(f"Retrieved course data from MySQL", {
                'total_entries': len(courses_data)
            })
            
            # Group by course name
            courses_dict = {}
            for row in courses_data:
                if not row or 'course_name' not in row:
                    SystemLogger.debug("Skipping invalid course data row", {
                        'row_data': row
                    })
                    continue
                    
                course_name = row['course_name']
                if course_name not in courses_dict:
                    courses_dict[course_name] = []
                courses_dict[course_name].append({
                    'module': row.get('module_name', 'Unknown Module'),
                    'summary': row.get('module_summary', 'No summary available')
                })
            
            if not courses_dict:
                SystemLogger.error(
                    "No valid course data found after processing - Check database schema and data integrity",
                    context={'raw_entries': len(courses_data)}
                )
                raise DatabaseQueryError("No valid course data found")
            
            # Format for LLM
            formatted = ""
            for course_name, modules in courses_dict.items():
                formatted += f"**Course: {course_name}**\nModules:\n"
                for module_info in modules:
                    formatted += f"- {module_info['module']}: {module_info['summary']}\n"
                formatted += "\n"
            
            formatted_data = formatted.strip()
            
            SystemLogger.info("IMPEL course data loaded and formatted successfully", {
                'unique_courses': len(courses_dict),
                'total_modules': sum(len(modules) for modules in courses_dict.values()),
                'formatted_length': len(formatted_data)
            })
            
            return formatted_data
            
        except (DatabaseConnectionError, DatabaseQueryError) as e:
            SystemLogger.error(
                "Database error loading IMPEL course data",
                exception=e,
                context={'mysql_available': self.mysql is not None}
            )
            # Re-raise database errors - they should be handled upstream
            raise
        except Exception as e:
            SystemLogger.error(
                "Unexpected error loading IMPEL course data",
                exception=e,
                context={'mysql_available': self.mysql is not None}
            )
            raise DatabaseQueryError(f"Failed to load course data: {e}")

    def lookup_courses(self, query: str, user_context: dict) -> dict:
        """
        Handle direct course/module lookup queries.
        Returns course information without recommendation logic.
        """
        SystemLogger.info("DatabaseAgent processing course lookup query", {
            'query_preview': query[:100] if query else 'empty',
            'user_context_keys': list(user_context.keys()) if user_context else []
        })
        
        # Input validation
        if not query or not query.strip():
            SystemLogger.error(
                "Course lookup query is empty - Cannot process empty query",
                context={'query': repr(query)}
            )
            raise AgentExecutionError("Query cannot be empty")
        
        if not user_context or not isinstance(user_context, dict):
            SystemLogger.error(
                "User context is missing or invalid for course lookup",
                context={'user_context_type': type(user_context), 'user_context': user_context}
            )
            raise AgentExecutionError("Valid user context is required")
        
        # Validate required user context fields
        required_fields = ["education", "age_group", "profession"]
        missing_fields = [field for field in required_fields if field not in user_context or not user_context[field]]
        if missing_fields:
            SystemLogger.error(
                f"Missing required user context fields for course lookup: {missing_fields}",
                context={'user_context': user_context, 'missing_fields': missing_fields}
            )
            raise AgentExecutionError(f"Missing required user context fields: {missing_fields}")
        
        try:
            # Generate user vector
            SystemLogger.debug("Generating user vector for course lookup")
            vector = self.neo4j.get_user_vector(
                user_context["education"],
                user_context["age_group"],
                user_context["profession"],
                query
            )
            
            if not vector:
                SystemLogger.error(
                    "Failed to generate user vector - Neo4j returned empty vector",
                    context={'user_context': user_context, 'query': query}
                )
                raise APIRequestError("Failed to generate user vector")
            
            SystemLogger.debug("User vector generated successfully", {
                'vector_dimension': len(vector) if vector else 0
            })
            
        except (DatabaseConnectionError, DatabaseQueryError, APIRequestError) as e:
            SystemLogger.error(
                "Database/API error generating user vector for course lookup",
                exception=e,
                context={'user_context': user_context, 'query': query}
            )
            raise
        except Exception as e:
            SystemLogger.error(
                "Unexpected error generating user vector for course lookup",
                exception=e,
                context={'user_context': user_context, 'query': query}
            )
            raise AgentExecutionError(f"Failed to generate user vector: {e}")
        
        try:
            # Check if course data is available
            if not self.impel_data or self.impel_data == "No course data available in database.":
                SystemLogger.error(
                    "Course data not available for lookup - Database may be empty or connection failed",
                    context={'impel_data_length': len(self.impel_data) if self.impel_data else 0}
                )
                raise DatabaseQueryError("Course data not available")
            
            # Generate database lookup response
            SystemLogger.debug("Generating course lookup response with Cohere")
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
            
            if not response or not response.generations or not response.generations[0]:
                SystemLogger.error(
                    "Cohere API returned empty response for course lookup",
                    context={'query': query, 'model': COHERE_GENERATE_MODEL}
                )
                raise APIRequestError("Cohere API returned empty response")
            
            generated_text = response.generations[0].text.strip()
            if not generated_text:
                SystemLogger.error(
                    "Cohere API returned empty text for course lookup",
                    context={'query': query, 'response_structure': str(response)[:200]}
                )
                raise APIRequestError("Cohere API returned empty text")
            
            SystemLogger.debug("Course lookup response generated successfully", {
                'response_length': len(generated_text),
                'model': COHERE_GENERATE_MODEL
            })
            
        except (APIRequestError) as e:
            SystemLogger.error(
                "API error generating course lookup response",
                exception=e,
                context={'query': query, 'model': COHERE_GENERATE_MODEL}
            )
            raise
        except Exception as e:
            SystemLogger.error(
                "Unexpected error generating course lookup response with Cohere",
                exception=e,
                context={'query': query, 'model': COHERE_GENERATE_MODEL}
            )
            raise APIRequestError(f"Failed to generate course lookup response: {e}")

        try:
            # Get similar users' courses
            SystemLogger.debug("Finding similar users and their enrolled courses")
            similar_users = self.neo4j.get_similar_users(vector)
            similar_user_courses = ""
            
            if similar_users:
                SystemLogger.debug(f"Found similar users", {'count': len(similar_users)})
                user_ids = [user["user_id"] for user in similar_users if user.get("user_id")]
                
                if user_ids:
                    courses = self.neo4j.get_enrolled_courses_from_similar_users(user_ids)
                    if courses:
                        unique_courses = sorted(set(courses))
                        similar_user_courses = "\n".join(f"- {c}" for c in unique_courses)
                        SystemLogger.debug(f"Found enrolled courses from similar users", {
                            'unique_courses': len(unique_courses)
                        })
                    else:
                        similar_user_courses = "No similar users who enrolled for IMPEL courses found."
                        SystemLogger.debug("Similar users found but no enrolled courses")
                else:
                    similar_user_courses = "No similar users who enrolled for IMPEL courses found."
                    SystemLogger.debug("Similar users found but no valid user IDs")
            else:
                similar_user_courses = "No similar users who enrolled for IMPEL courses found."
                SystemLogger.debug("No similar users found")
                
        except (DatabaseConnectionError, DatabaseQueryError) as e:
            SystemLogger.error(
                "Database error finding similar users and courses",
                exception=e,
                context={'vector_available': vector is not None}
            )
            # Don't fail the entire request - use fallback message
            similar_user_courses = "Unable to find similar users due to database error."
            SystemLogger.info("Using fallback message for similar users due to database error")
        except Exception as e:
            SystemLogger.error(
                "Unexpected error finding similar users and courses",
                exception=e,
                context={'vector_available': vector is not None}
            )
            # Don't fail the entire request - use fallback message
            similar_user_courses = "Unable to find similar users due to system error."
            SystemLogger.info("Using fallback message for similar users due to system error")

        # Prepare final response
        result = {
            "response": generated_text,
            "similar_user_courses": similar_user_courses,
            "user_vector": vector
        }
        
        SystemLogger.info("DatabaseAgent course lookup completed successfully", {
            'query_preview': query[:100],
            'response_length': len(generated_text),
            'similar_courses_found': len(similar_user_courses) > 50,  # Rough check if courses were found
            'user_vector_dimension': len(vector) if vector else 0
        })
        
        return result