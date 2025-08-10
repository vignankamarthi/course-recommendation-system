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


class CollaborativeAgent:
    """Agent for collaborative filtering recommendations based on similar users."""
    
    def __init__(self):
        SystemLogger.info("Initializing CollaborativeAgent")
        
        try:
            # Initialize database connections
            SystemLogger.debug("Acquiring database connections for CollaborativeAgent")
            self.neo4j = get_neo4j_connection()
            self.mysql = get_mysql_connection()
            
            # Initialize Cohere client
            SystemLogger.debug("Initializing Cohere client for CollaborativeAgent")
            if not cohere_api_key:
                SystemLogger.error(
                    "Cohere API key not configured for CollaborativeAgent - Check environment variables",
                    context={'api_key_provided': bool(cohere_api_key)}
                )
                raise ConfigurationError("Cohere API key not configured")
            
            self.cohere_client = cohere.Client(cohere_api_key)
            
            # Load course data
            SystemLogger.debug("Loading IMPEL course data from MySQL")
            self.impel_data = self._load_impel_courses_and_modules()
            
            SystemLogger.info("CollaborativeAgent initialized successfully", {
                'mysql_connected': self.mysql is not None,
                'neo4j_connected': self.neo4j is not None,
                'cohere_configured': bool(cohere_api_key),
                'course_data_loaded': len(self.impel_data) > 0
            })
            
        except (DatabaseConnectionError, ConfigurationError) as e:
            SystemLogger.error(
                "Failed to initialize CollaborativeAgent - Database or configuration error",
                exception=e,
                context={'initialization_step': 'database_connections'}
            )
            raise
        except Exception as e:
            SystemLogger.error(
                "Unexpected error initializing CollaborativeAgent",
                exception=e,
                context={'initialization_step': 'unknown'}
            )
            raise AgentExecutionError(f"Failed to initialize CollaborativeAgent: {e}")
    
    def _load_impel_courses_and_modules(self):
        """Load and format course data from MySQL for recommendations."""
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

    def generate_recommendations(self, query: str, user_context: dict) -> dict:
        """
        Generate collaborative filtering recommendations based on similar users.
        """
        SystemLogger.info("CollaborativeAgent generating recommendations", {
            'query_preview': query[:100] if query else 'empty',
            'user_context_keys': list(user_context.keys()) if user_context else []
        })
        
        # Input validation
        if not query or not query.strip():
            SystemLogger.error(
                "Recommendation query is empty - Cannot process empty query",
                context={'query': repr(query)}
            )
            raise AgentExecutionError("Query cannot be empty")
        
        if not user_context or not isinstance(user_context, dict):
            SystemLogger.error(
                "User context is missing or invalid for recommendations",
                context={'user_context_type': type(user_context), 'user_context': user_context}
            )
            raise AgentExecutionError("Valid user context is required")
        
        # Validate required user context fields
        required_fields = ["education", "age_group", "profession"]
        missing_fields = [field for field in required_fields if field not in user_context or not user_context[field]]
        if missing_fields:
            SystemLogger.error(
                f"Missing required user context fields for recommendations: {missing_fields}",
                context={'user_context': user_context, 'missing_fields': missing_fields}
            )
            raise AgentExecutionError(f"Missing required user context fields: {missing_fields}")
        
        try:
            # Get user vector for similarity matching
            SystemLogger.debug("Generating user vector for collaborative filtering")
            user_vector = self.neo4j.get_user_vector(
                user_context["education"], 
                user_context["age_group"], 
                user_context["profession"], 
                query
            )
            
            if not user_vector:
                SystemLogger.error(
                    "Failed to generate user vector - Neo4j returned empty vector",
                    context={'user_context': user_context, 'query': query}
                )
                raise APIRequestError("Failed to generate user vector")
            
            SystemLogger.debug("User vector generated successfully", {
                'vector_dimension': len(user_vector) if user_vector else 0
            })
            
        except (DatabaseConnectionError, DatabaseQueryError, APIRequestError) as e:
            SystemLogger.error(
                "Database/API error generating user vector for recommendations",
                exception=e,
                context={'user_context': user_context, 'query': query}
            )
            raise
        except Exception as e:
            SystemLogger.error(
                "Unexpected error generating user vector for recommendations",
                exception=e,
                context={'user_context': user_context, 'query': query}
            )
            raise AgentExecutionError(f"Failed to generate user vector: {e}")
        
        try:
            # Find similar users
            SystemLogger.debug("Finding similar users for collaborative filtering")
            similar_users = self.neo4j.get_similar_users(user_vector)
            
            SystemLogger.debug("Similar users search completed", {
                'similar_users_count': len(similar_users) if similar_users else 0
            })
            
        except (DatabaseConnectionError, DatabaseQueryError) as e:
            SystemLogger.error(
                "Database error finding similar users",
                exception=e,
                context={'vector_available': user_vector is not None}
            )
            # Continue with empty similar_users for fallback behavior
            similar_users = []
            SystemLogger.info("Using empty similar users due to database error")
        except Exception as e:
            SystemLogger.error(
                "Unexpected error finding similar users",
                exception=e,
                context={'vector_available': user_vector is not None}
            )
            # Continue with empty similar_users for fallback behavior
            similar_users = []
            SystemLogger.info("Using empty similar users due to system error")

        # Build recommendation prompt based on similar users
        try:
            SystemLogger.debug("Building recommendation prompt")
            
            if not similar_users:
                SystemLogger.info("No similar users found - using general recommendation approach")
                response_prefix = "No similar users found. Here are some suggested IMPEL courses and modules:\\n\\n"
                prompt = f\"\"\"
You are a course recommendation assistant. Below are courses and their modules from the IMPEL database:
{self.impel_data}
User query: '{query}'
Suggest relevant courses and their modules.
Format:
**Course: <Course Name>**
- <Module 1>
- <Module 2>
\"\"\"
            else:
                SystemLogger.info("Similar users found - using collaborative filtering approach", {
                    'similar_users_count': len(similar_users)
                })
                response_prefix = "Recommended based on similar users' interests:\\n\\n"
                
                try:
                    most_similar_query = similar_users[0].get("query", "")
                    if not most_similar_query:
                        SystemLogger.debug("Most similar user has empty query - using general approach")
                        similar_user_recs = []
                    else:
                        similar_user_recs = self.neo4j.get_recommendations_for_user(most_similar_query)
                        SystemLogger.debug("Retrieved recommendations for most similar user", {
                            'recommendations_count': len(similar_user_recs) if similar_user_recs else 0
                        })
                        
                except Exception as rec_error:
                    SystemLogger.error(
                        "Error retrieving recommendations for similar user - using fallback",
                        exception=rec_error,
                        context={'most_similar_user_id': similar_users[0].get('user_id', 'unknown') if similar_users else 'none'}
                    )
                    similar_user_recs = []
                
                prompt = f\"\"\"
You are a course recommendation assistant. Below are courses and their modules from the IMPEL database:
{self.impel_data}
A similar user was interested in: {similar_user_recs}
Current user query: '{query}'
Suggest relevant courses and modules for this user.
Format:
**Course: <Course Name>**
- <Module 1>
- <Module 2>
\"\"\"

        except Exception as e:
            SystemLogger.error(
                "Error building recommendation prompt",
                exception=e,
                context={'similar_users_available': similar_users is not None}
            )
            raise AgentExecutionError(f"Failed to build recommendation prompt: {e}")

        try:
            # Check if course data is available
            if not self.impel_data or self.impel_data == "No course data available in database.":
                SystemLogger.error(
                    "Course data not available for recommendations - Database may be empty or connection failed",
                    context={'impel_data_length': len(self.impel_data) if self.impel_data else 0}
                )
                raise DatabaseQueryError("Course data not available")
            
            # Generate recommendation using LLM
            SystemLogger.debug("Generating recommendations with Cohere")
            llm_response = self.cohere_client.generate(
                model=COHERE_GENERATE_MODEL,
                prompt=prompt,
                max_tokens=400
            )
            
            if not llm_response or not llm_response.generations or not llm_response.generations[0]:
                SystemLogger.error(
                    "Cohere API returned empty response for recommendations",
                    context={'query': query, 'model': COHERE_GENERATE_MODEL}
                )
                raise APIRequestError("Cohere API returned empty response")
            
            generated_text = llm_response.generations[0].text.strip()
            if not generated_text:
                SystemLogger.error(
                    "Cohere API returned empty text for recommendations",
                    context={'query': query, 'response_structure': str(llm_response)[:200]}
                )
                raise APIRequestError("Cohere API returned empty text")
            
            response = response_prefix + generated_text
            
            SystemLogger.debug("Recommendations generated successfully", {
                'response_length': len(response),
                'model': COHERE_GENERATE_MODEL
            })
            
        except (APIRequestError, DatabaseQueryError) as e:
            SystemLogger.error(
                "API/Database error generating recommendations",
                exception=e,
                context={'query': query, 'model': COHERE_GENERATE_MODEL}
            )
            raise
        except Exception as e:
            SystemLogger.error(
                "Unexpected error generating recommendations with Cohere",
                exception=e,
                context={'query': query, 'model': COHERE_GENERATE_MODEL}
            )
            raise APIRequestError(f"Failed to generate recommendations: {e}")

        try:
            # Get courses similar users enrolled in
            SystemLogger.debug("Finding enrolled courses from similar users")
            similar_user_courses = ""
            
            if similar_users:
                similar_user_ids = [user["user_id"] for user in similar_users if user.get("user_id")]
                
                if similar_user_ids:
                    enrolled_courses = self.neo4j.get_enrolled_courses_from_similar_users(similar_user_ids)
                    if enrolled_courses:
                        unique_courses = sorted(set(enrolled_courses))
                        similar_user_courses = "\\n".join(f"- {name}" for name in unique_courses)
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
                SystemLogger.debug("No similar users available for enrolled courses lookup")
                
        except (DatabaseConnectionError, DatabaseQueryError) as e:
            SystemLogger.error(
                "Database error finding enrolled courses from similar users",
                exception=e,
                context={'similar_users_available': similar_users is not None}
            )
            # Don't fail the entire request - use fallback message
            similar_user_courses = "Unable to find similar user courses due to database error."
            SystemLogger.info("Using fallback message for similar user courses due to database error")
        except Exception as e:
            SystemLogger.error(
                "Unexpected error finding enrolled courses from similar users",
                exception=e,
                context={'similar_users_available': similar_users is not None}
            )
            # Don't fail the entire request - use fallback message
            similar_user_courses = "Unable to find similar user courses due to system error."
            SystemLogger.info("Using fallback message for similar user courses due to system error")

        # Prepare final response
        result = {
            "response": response,
            "similar_user_courses": similar_user_courses,
            "user_vector": user_vector
        }
        
        SystemLogger.info("CollaborativeAgent recommendations completed successfully", {
            'query_preview': query[:100],
            'response_length': len(response),
            'similar_users_found': len(similar_users) if similar_users else 0,
            'similar_courses_found': len(similar_user_courses) > 50,  # Rough check if courses were found
            'user_vector_dimension': len(user_vector) if user_vector else 0
        })
        
        return result