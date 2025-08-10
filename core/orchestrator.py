# LangGraph Workflow
import cohere
from langgraph.graph import StateGraph
from core.config import cohere_api_key, COHERE_GENERATE_MODEL, get_neo4j_connection
from agents.database_agent import DatabaseAgent
from agents.collaborative_agent import CollaborativeAgent
from agents.content_agent import ContentAgent
from utils.logger import SystemLogger
from utils.exceptions import (
    DatabaseConnectionError, APIRequestError, AgentExecutionError, 
    ConfigurationError, WorkflowError
)


class RecommendationSystem:
    """
    Orchestrates different AI agents based on user query intent classification.
    
    This class serves as the main workflow orchestrator that routes user queries
    to appropriate specialized agents (database, collaborative, or content) based
    on intent classification. It manages the complete request lifecycle from
    query processing to result storage.
    
    Attributes
    ----------
    workflow : StateGraph
        LangGraph workflow for state management
    neo4j : Neo4jConnector
        Neo4j database connection for user interactions
    cohere_client : cohere.Client
        Cohere API client for LLM operations
    database_agent : DatabaseAgent
        Agent for direct database course lookups
    collaborative_agent : CollaborativeAgent
        Agent for collaborative filtering recommendations
    content_agent : ContentAgent
        Agent for content-based analysis and web search
        
    Raises
    ------
    ConfigurationError
        If required API keys are not configured
    DatabaseConnectionError
        If database connections cannot be established
    AgentExecutionError
        If agent initialization fails
    """

    def __init__(self):
        SystemLogger.info("Initializing RecommendationSystem orchestrator")

        try:
            # Validate API key before proceeding
            if not cohere_api_key or not cohere_api_key.strip():
                SystemLogger.error(
                    "Cohere API key not available for RecommendationSystem",
                    context={'api_key_provided': bool(cohere_api_key)}
                )
                raise ConfigurationError("Cohere API key not configured")

            # Initialize workflow graph
            SystemLogger.debug("Initializing LangGraph StateGraph")
            self.workflow = StateGraph(dict)

            # Initialize database connections
            SystemLogger.debug("Acquiring Neo4j connection for orchestrator")
            self.neo4j = get_neo4j_connection()

            if not self.neo4j:
                SystemLogger.error(
                    "Neo4j connection not available for RecommendationSystem",
                    context={'neo4j_available': self.neo4j is not None}
                )
                raise DatabaseConnectionError("Neo4j connection not available")

            # Initialize Cohere client
            SystemLogger.debug("Initializing Cohere client for orchestrator")
            self.cohere_client = cohere.Client(cohere_api_key)

            # Initialize agents
            SystemLogger.debug("Initializing agent components")
            self.database_agent = DatabaseAgent()
            self.collaborative_agent = CollaborativeAgent()
            self.content_agent = ContentAgent(cohere_key=cohere_api_key)

            SystemLogger.info("RecommendationSystem orchestrator initialized successfully", {
                'neo4j_connected': self.neo4j is not None,
                'cohere_configured': bool(cohere_api_key),
                'agents_initialized': True
            })

        except (ConfigurationError, DatabaseConnectionError, AgentExecutionError) as e:
            SystemLogger.error(
                "Failed to initialize RecommendationSystem - Configuration or agent error",
                exception=e,
                context={'initialization_step': 'agent_initialization'}
            )
            raise
        except Exception as e:
            SystemLogger.error(
                "Unexpected error initializing RecommendationSystem",
                exception=e,
                context={'initialization_step': 'unknown'}
            )
            raise AgentExecutionError(f"Failed to initialize RecommendationSystem: {e}")

    # TODO: Explore improving this initilization prompt
    def classify_intent(self, query):
        """
        Classify user query intent using Cohere LLM for workflow routing.
        
        Analyzes the user query to determine which specialized agent should
        handle the request. Classifications include database lookup, 
        collaborative recommendations, content analysis, or irrelevant queries.
        
        Parameters
        ----------
        query : str
            User input query to classify for intent determination
            
        Returns
        -------
        str
            Intent classification, one of:
            - 'database_lookup': For specific course/module exploration
            - 'recommendation': For collaborative filtering recommendations  
            - 'content_analysis': For trending skills and market insights
            - 'irrelevant': For queries unrelated to education/data science
            
        Raises
        ------
        AgentExecutionError
            If query is empty or intent classification fails
        APIRequestError
            If Cohere API returns invalid or empty response
        """
        SystemLogger.debug("Classifying user query intent", {
            'query_preview': query[:100] if query else 'empty'
        })

        # Input validation
        if not query or not query.strip():
            SystemLogger.error(
                "Cannot classify empty query - Query is required for intent classification",
                context={'query': repr(query)}
            )
            raise AgentExecutionError("Query cannot be empty for intent classification")

        try:
            prompt = f"""
You are an intent classification assistant. Categorize the user's query as one of the following:
- "database_lookup": if they want to list or explore specific IMPEL courses/modules or descriptions.
- "recommendation": if they are asking what course suits their goal, background, or if they are exploring learning paths, skills or roles in the broad spectrum of Data Science or AI (e.g., how to become a data scientist, what an ML Engineer does, data scientist average salary, etc.).
- "content_analysis": if they are asking about trending skills, job market insights, or want content-based recommendations with research papers.
- "irrelevant": if the query is clearly unrelated to Data Science, AI, Information Technology or education, such as questions about movies, cooking, weather, jokes, casual greetings or personal life.


User query: "{query}"
Only reply with one of the following words: "database_lookup", "recommendation", "content_analysis", or "irrelevant".
"""

            SystemLogger.debug("Invoking Cohere for intent classification")
            response = self.cohere_client.generate(
                model=COHERE_GENERATE_MODEL,
                prompt=prompt,
                max_tokens=5,
                temperature=0
            )

            if not response or not response.generations or not response.generations[0]:
                SystemLogger.error(
                    "Cohere returned empty response for intent classification",
                    context={'query': query, 'model': COHERE_GENERATE_MODEL}
                )
                raise APIRequestError("Cohere returned empty response")

            intent = response.generations[0].text.strip().lower()

            # Validate intent is one of expected values
            valid_intents = ['database_lookup', 'recommendation', 'content_analysis', 'irrelevant']
            if intent not in valid_intents:
                SystemLogger.error(
                    f"Invalid intent classification returned: {intent}",
                    context={'query': query, 'returned_intent': intent, 'valid_intents': valid_intents}
                )
                # Default to recommendation as safest fallback
                intent = 'recommendation'
                SystemLogger.info("Using fallback intent: recommendation")

            SystemLogger.debug("Intent classification completed", {
                'query_preview': query[:50],
                'classified_intent': intent
            })

            return intent

        except APIRequestError as e:
            SystemLogger.error(
                "API error during intent classification",
                exception=e,
                context={'query': query, 'model': COHERE_GENERATE_MODEL}
            )
            raise
        except Exception as e:
            SystemLogger.error(
                "Unexpected error during intent classification",
                exception=e,
                context={'query': query}
            )
            raise APIRequestError(f"Intent classification failed: {e}")

    def collect_user_data(self, state):
        """Collect and vectorize user data from state."""
        SystemLogger.debug("Collecting user data for workflow", {
            'user_id': state.get('user_id', 'unknown'),
            'has_education': bool(state.get('education')),
            'has_profession': bool(state.get('profession'))
        })

        try:
            # Validate required state fields
            required_fields = ['education', 'age_group', 'profession', 'query']
            missing_fields = [field for field in required_fields if field not in state or not state[field]]
            if missing_fields:
                SystemLogger.error(
                    f"Missing required state fields for user data collection: {missing_fields}",
                    context={'state_keys': list(state.keys()), 'missing_fields': missing_fields}
                )
                raise WorkflowError(f"Missing required state fields: {missing_fields}")

            # Generate user vector
            SystemLogger.debug("Generating user vector from demographic data")
            vector = self.neo4j.get_user_vector(
                state["education"], state["age_group"], state["profession"], state["query"]
            )

            if not vector:
                SystemLogger.error(
                    "Failed to generate user vector - Neo4j returned empty vector",
                    context={
                        'education': state["education"],
                        'age_group': state["age_group"],
                        'profession': state["profession"],
                        'query': state["query"][:50] if state.get("query") else ''
                    }
                )
                raise APIRequestError("Failed to generate user vector")

            state["user_vector"] = vector

            # TODO: Demographics is misleading THROUGHOUT THE CODE BASE, from general understanding, demographics means ethinic, socioeconomic, etc., but here, demogrpahics is just education, age_group, and profession. We need to be clear on that.
            state["demographics"] = {
                "education": state["education"],
                "age_group": state["age_group"],
                "profession": state["profession"]
            }

            SystemLogger.debug("User data collection completed successfully", {
                'user_vector_dimension': len(vector) if vector else 0,
                'demographics_fields': len(state["demographics"])
            })

            return state

        except (DatabaseConnectionError, APIRequestError, WorkflowError) as e:
            SystemLogger.error(
                "Database/API/Workflow error collecting user data",
                exception=e,
                context={'user_id': state.get('user_id', 'unknown')}
            )
            raise
        except Exception as e:
            SystemLogger.error(
                "Unexpected error collecting user data",
                exception=e,
                context={'user_id': state.get('user_id', 'unknown')}
            )
            raise WorkflowError(f"User data collection failed: {e}")

    def generate_recommendations(self, state):
        """Delegate to CollaborativeAgent for recommendations."""
        SystemLogger.debug("Generating collaborative recommendations", {
            'user_id': state.get('user_id', 'unknown'),
            'query_preview': state.get('query', '')[:50]
        })

        try:
            # Validate state has required fields
            required_fields = ['education', 'age_group', 'profession', 'query']
            missing_fields = [field for field in required_fields if field not in state or not state[field]]
            if missing_fields:
                SystemLogger.error(
                    f"Missing required state fields for recommendation generation: {missing_fields}",
                    context={'state_keys': list(state.keys()), 'missing_fields': missing_fields}
                )
                raise WorkflowError(f"Missing required state fields: {missing_fields}")

            user_context = {
                "education": state["education"],
                "age_group": state["age_group"],
                "profession": state["profession"]
            }

            SystemLogger.debug("Invoking CollaborativeAgent for recommendations")
            result = self.collaborative_agent.generate_recommendations(
                query=state["query"], 
                user_context=user_context
            )

            if not result or not isinstance(result, dict):
                SystemLogger.error(
                    "CollaborativeAgent returned invalid result",
                    context={'result_type': type(result), 'result': result}
                )
                raise AgentExecutionError("CollaborativeAgent returned invalid result")

            # Validate required result fields
            required_result_fields = ['response', 'similar_user_courses', 'user_vector']
            missing_result_fields = [field for field in required_result_fields if field not in result]
            if missing_result_fields:
                SystemLogger.error(
                    f"CollaborativeAgent result missing required fields: {missing_result_fields}",
                    context={'result_keys': list(result.keys()), 'missing_fields': missing_result_fields}
                )
                raise AgentExecutionError(f"CollaborativeAgent result missing fields: {missing_result_fields}")

            state["response"] = result["response"]
            state["similar_user_courses"] = result["similar_user_courses"]
            state["user_vector"] = result["user_vector"]

            SystemLogger.debug("Collaborative recommendations generated successfully", {
                'response_length': len(result["response"]) if result["response"] else 0,
                'similar_courses_found': len(result["similar_user_courses"]) > 50,
                'user_vector_dimension': len(result["user_vector"]) if result["user_vector"] else 0
            })

            return state

        except (AgentExecutionError, WorkflowError) as e:
            SystemLogger.error(
                "Agent/Workflow error generating recommendations",
                exception=e,
                context={'user_id': state.get('user_id', 'unknown')}
            )
            raise
        except Exception as e:
            SystemLogger.error(
                "Unexpected error generating recommendations",
                exception=e,
                context={'user_id': state.get('user_id', 'unknown')}
            )
            raise WorkflowError(f"Recommendation generation failed: {e}")

    def run_database_agent(self, state):
        """Delegate to DatabaseAgent for course lookups."""
        SystemLogger.debug("Running DatabaseAgent for course lookup", {
            'user_id': state.get('user_id', 'unknown'),
            'query_preview': state.get('query', '')[:50]
        })

        try:
            # Validate state has required fields
            required_fields = ['education', 'age_group', 'profession', 'query']
            missing_fields = [field for field in required_fields if field not in state or not state[field]]
            if missing_fields:
                SystemLogger.error(
                    f"Missing required state fields for database lookup: {missing_fields}",
                    context={'state_keys': list(state.keys()), 'missing_fields': missing_fields}
                )
                raise WorkflowError(f"Missing required state fields: {missing_fields}")

            user_context = {
                "education": state["education"],
                "age_group": state["age_group"],
                "profession": state["profession"]
            }

            SystemLogger.debug("Invoking DatabaseAgent for course lookup")
            result = self.database_agent.lookup_courses(
                query=state["query"],
                user_context=user_context
            )

            if not result or not isinstance(result, dict):
                SystemLogger.error(
                    "DatabaseAgent returned invalid result",
                    context={'result_type': type(result), 'result': result}
                )
                raise AgentExecutionError("DatabaseAgent returned invalid result")

            # Validate required result fields
            required_result_fields = ['response', 'similar_user_courses', 'user_vector']
            missing_result_fields = [field for field in required_result_fields if field not in result]
            if missing_result_fields:
                SystemLogger.error(
                    f"DatabaseAgent result missing required fields: {missing_result_fields}",
                    context={'result_keys': list(result.keys()), 'missing_fields': missing_result_fields}
                )
                raise AgentExecutionError(f"DatabaseAgent result missing fields: {missing_result_fields}")

            state["response"] = result["response"]
            state["similar_user_courses"] = result["similar_user_courses"]
            state["user_vector"] = result["user_vector"]
            state["demographics"] = user_context

            SystemLogger.debug("Database course lookup completed successfully", {
                'response_length': len(result["response"]) if result["response"] else 0,
                'similar_courses_found': len(result["similar_user_courses"]) > 50,
                'user_vector_dimension': len(result["user_vector"]) if result["user_vector"] else 0
            })

            return state

        except (AgentExecutionError, WorkflowError) as e:
            SystemLogger.error(
                "Agent/Workflow error running database lookup",
                exception=e,
                context={'user_id': state.get('user_id', 'unknown')}
            )
            raise
        except Exception as e:
            SystemLogger.error(
                "Unexpected error running database agent",
                exception=e,
                context={'user_id': state.get('user_id', 'unknown')}
            )
            raise WorkflowError(f"Database agent execution failed: {e}")

    def run_content_agent(self, state):
        """Delegate to ContentAgent for content-based recommendations and market analysis."""
        SystemLogger.debug("Running ContentAgent for content analysis", {
            'user_id': state.get('user_id', 'unknown'),
            'query_preview': state.get('query', '')[:50],
            'uploaded_files_count': len(state.get('uploaded_files', []))
        })

        try:
            # Validate state has required query field
            if 'query' not in state or not state['query']:
                SystemLogger.error(
                    "Missing required query field for content agent",
                    context={'state_keys': list(state.keys())}
                )
                raise WorkflowError("Query field is required for content agent")

            uploaded_files = state.get("uploaded_files", [])

            # Validate uploaded files list
            if uploaded_files and not isinstance(uploaded_files, list):
                SystemLogger.error(
                    "Invalid uploaded_files format - must be a list",
                    context={'uploaded_files_type': type(uploaded_files)}
                )
                # Continue with empty list instead of failing
                uploaded_files = []
                SystemLogger.info("Using empty uploaded_files list due to invalid format")

            # ContentAgent handles its own logic and returns formatted response
            SystemLogger.debug("Invoking ContentAgent for content analysis")
            result = self.content_agent.run(
                query=state["query"],
                uploaded_files=uploaded_files
            )

            if not result or not isinstance(result, str):
                SystemLogger.error(
                    "ContentAgent returned invalid result - expected string response",
                    context={'result_type': type(result), 'result_preview': str(result)[:100] if result else 'None'}
                )
                raise AgentExecutionError("ContentAgent returned invalid result")

            state["response"] = result
            state["similar_user_courses"] = ""  # ContentAgent doesn't use collaborative data

            SystemLogger.debug("Content analysis completed successfully", {
                'response_length': len(result),
                'uploaded_files_processed': len(uploaded_files)
            })

            return state

        except (AgentExecutionError, WorkflowError) as e:
            SystemLogger.error(
                "Agent/Workflow error running content analysis",
                exception=e,
                context={'user_id': state.get('user_id', 'unknown')}
            )
            raise
        except Exception as e:
            SystemLogger.error(
                "Unexpected error running content agent",
                exception=e,
                context={'user_id': state.get('user_id', 'unknown')}
            )
            raise WorkflowError(f"Content agent execution failed: {e}")

    def store_result(self, state):
        """Store interaction result in Neo4j database."""
        SystemLogger.debug("Storing workflow result in Neo4j", {
            'user_id': state.get('user_id', 'unknown'),
            'has_response': bool(state.get('response')),
            'has_user_vector': bool(state.get('user_vector'))
        })

        try:
            # Validate required state fields for storage
            required_fields = ['user_id', 'query', 'response']
            missing_fields = [field for field in required_fields if field not in state or state[field] is None]
            if missing_fields:
                SystemLogger.error(
                    f"Missing required state fields for result storage: {missing_fields}",
                    context={'state_keys': list(state.keys()), 'missing_fields': missing_fields}
                )
                raise WorkflowError(f"Missing required state fields for storage: {missing_fields}")

            demographics = state.get("demographics", {})

            # Extract demographics with fallback values
            education = demographics.get("education", state.get("education", ""))
            age_group = demographics.get("age_group", state.get("age_group", ""))
            profession = demographics.get("profession", state.get("profession", ""))

            SystemLogger.debug("Storing interaction in Neo4j database")
            self.neo4j.store_interaction(
                user_id=state["user_id"],
                education=education,
                age_group=age_group,
                profession=profession,
                user_query=state["query"],
                response=state["response"],
                user_vector=state.get("user_vector", [])
            )

            SystemLogger.debug("Workflow result stored successfully", {
                'user_id': state["user_id"],
                'query_length': len(state["query"]),
                'response_length': len(state["response"]),
                'user_vector_dimension': len(state.get("user_vector", []))
            })

            return state

        except (DatabaseConnectionError, WorkflowError) as e:
            SystemLogger.error(
                "Database/Workflow error storing workflow result",
                exception=e,
                context={'user_id': state.get('user_id', 'unknown')}
            )
            raise
        except Exception as e:
            SystemLogger.error(
                "Unexpected error storing workflow result",
                exception=e,
                context={'user_id': state.get('user_id', 'unknown')}
            )
            raise WorkflowError(f"Result storage failed: {e}")

    def build_workflow(self):
        """Build collaborative recommendation workflow."""
        SystemLogger.debug("Building collaborative recommendation workflow")

        try:
            graph = StateGraph(dict)
            graph.add_node("collect_data", self.collect_user_data)
            graph.add_node("generate_recs", self.generate_recommendations)
            graph.add_node("store_result", self.store_result)

            graph.set_entry_point("collect_data")
            graph.add_edge("collect_data", "generate_recs")
            graph.add_edge("generate_recs", "store_result")

            compiled_workflow = graph.compile()
            SystemLogger.debug("Collaborative recommendation workflow built successfully")
            return compiled_workflow

        except Exception as e:
            SystemLogger.error(
                "Error building collaborative recommendation workflow",
                exception=e,
                context={'workflow_type': 'collaborative_recommendation'}
            )
            raise WorkflowError(f"Failed to build collaborative workflow: {e}")

    def build_database_lookup_workflow(self):
        """Build database lookup workflow."""
        SystemLogger.debug("Building database lookup workflow")

        try:
            graph = StateGraph(dict)
            graph.add_node("collect_data", self.collect_user_data)
            graph.add_node("run_database_agent", self.run_database_agent)
            graph.add_node("store_result", self.store_result)

            graph.set_entry_point("collect_data")
            graph.add_edge("collect_data", "run_database_agent")
            graph.add_edge("run_database_agent", "store_result")

            compiled_workflow = graph.compile()
            SystemLogger.debug("Database lookup workflow built successfully")
            return compiled_workflow

        except Exception as e:
            SystemLogger.error(
                "Error building database lookup workflow",
                exception=e,
                context={'workflow_type': 'database_lookup'}
            )
            raise WorkflowError(f"Failed to build database lookup workflow: {e}")

    def build_content_workflow(self):
        """Build content analysis workflow."""
        SystemLogger.debug("Building content analysis workflow")

        try:
            graph = StateGraph(dict)
            graph.add_node("collect_data", self.collect_user_data)
            graph.add_node("run_content_agent", self.run_content_agent)
            graph.add_node("store_result", self.store_result)

            graph.set_entry_point("collect_data")
            graph.add_edge("collect_data", "run_content_agent")
            graph.add_edge("run_content_agent", "store_result")

            compiled_workflow = graph.compile()
            SystemLogger.debug("Content analysis workflow built successfully")
            return compiled_workflow

        except Exception as e:
            SystemLogger.error(
                "Error building content analysis workflow",
                exception=e,
                context={'workflow_type': 'content_analysis'}
            )
            raise WorkflowError(f"Failed to build content workflow: {e}")

    def handle_user_query(self, user_id, education, age_group, profession, query, uploaded_files=None):
        """
        Main entry point for processing user queries through appropriate AI workflows.
        
        Orchestrates the complete user request lifecycle: intent classification,
        workflow routing, agent execution, and result formatting. Handles all
        error conditions gracefully and provides comprehensive logging.
        
        Parameters
        ----------
        user_id : str
            Unique identifier for the user session
        education : str
            User's education level ('High School', 'Undergraduate', 'Graduate')
        age_group : str
            User's age range ('Under 18', '18-25', '26-40', '40+')
        profession : str
            User's professional status ('Student', 'Professional')
        query : str
            User's natural language query for course recommendations
        uploaded_files : list of str, optional
            File paths for uploaded resumes/documents (default: None)
            
        Returns
        -------
        tuple of (str, str or None)
            - response: Formatted recommendation response text
            - similar_courses: Similar user course enrollments or None
            
        Raises
        ------
        WorkflowError
            If required fields are missing or workflow execution fails
        APIRequestError
            If external API calls (Cohere, Tavily) fail
        AgentExecutionError
            If agent initialization or execution fails
            
        Examples
        --------
        >>> system = RecommendationSystem()
        >>> response, courses = system.handle_user_query(
        ...     user_id="user123",
        ...     education="Graduate", 
        ...     age_group="26-40",
        ...     profession="Professional",
        ...     query="I want to become a data scientist"
        ... )
        """
        SystemLogger.info("Handling user query through orchestrator", {
            'user_id': user_id,
            'query_preview': query[:100] if query else 'empty',
            'education': education,
            'profession': profession,
            'uploaded_files_count': len(uploaded_files) if uploaded_files else 0
        })

        try:
            # Input validation
            if not user_id or not user_id.strip():
                SystemLogger.error(
                    "User ID is required for query handling",
                    context={'user_id': repr(user_id)}
                )
                raise WorkflowError("User ID cannot be empty")

            required_fields = {
                'education': education,
                'age_group': age_group,
                'profession': profession,
                'query': query
            }

            missing_fields = [field for field, value in required_fields.items() if not value or not value.strip()]
            if missing_fields:
                SystemLogger.error(
                    f"Missing required user fields: {missing_fields}",
                    context={'user_id': user_id, 'missing_fields': missing_fields}
                )
                raise WorkflowError(f"Missing required fields: {missing_fields}")

            # Classify intent
            SystemLogger.debug("Classifying query intent")
            intent = self.classify_intent(query)

            # Build state
            state = {
                "user_id": user_id,
                "education": education,
                "age_group": age_group,
                "profession": profession,
                "query": query,
                "uploaded_files": uploaded_files or []
            }

            SystemLogger.info(f"Processing query with intent: {intent}", {
                'user_id': user_id,
                'intent': intent,
                'workflow_type': intent
            })

            # Route to appropriate workflow
            if intent == "recommendation":
                try:
                    SystemLogger.debug("Building and invoking collaborative recommendation workflow")
                    app = self.build_workflow()
                    final_state = app.invoke(state)

                    response = final_state.get("response")
                    similar_courses = final_state.get("similar_user_courses")

                    if not response:
                        SystemLogger.error("Collaborative workflow returned empty response")
                        raise WorkflowError("Empty response from collaborative workflow")

                    SystemLogger.info("Collaborative recommendation workflow completed successfully", {
                        'user_id': user_id,
                        'response_length': len(response)
                    })

                    return response, similar_courses

                except Exception as workflow_error:
                    SystemLogger.error(
                        "Error in collaborative recommendation workflow",
                        exception=workflow_error,
                        context={'user_id': user_id, 'intent': intent}
                    )
                    raise WorkflowError(f"Collaborative workflow failed: {workflow_error}")

            elif intent == "database_lookup":
                try:
                    SystemLogger.debug("Building and invoking database lookup workflow")
                    app = self.build_database_lookup_workflow()
                    final_state = app.invoke(state)

                    response = final_state.get("response")
                    similar_courses = final_state.get("similar_user_courses")

                    if not response:
                        SystemLogger.error("Database lookup workflow returned empty response")
                        raise WorkflowError("Empty response from database lookup workflow")

                    SystemLogger.info("Database lookup workflow completed successfully", {
                        'user_id': user_id,
                        'response_length': len(response)
                    })

                    return response, similar_courses

                except Exception as workflow_error:
                    SystemLogger.error(
                        "Error in database lookup workflow",
                        exception=workflow_error,
                        context={'user_id': user_id, 'intent': intent}
                    )
                    raise WorkflowError(f"Database lookup workflow failed: {workflow_error}")

            elif intent == "content_analysis":
                try:
                    SystemLogger.debug("Building and invoking content analysis workflow")
                    app = self.build_content_workflow()
                    final_state = app.invoke(state)

                    response = final_state.get("response")
                    similar_courses = final_state.get("similar_user_courses", "")

                    if not response:
                        SystemLogger.error("Content analysis workflow returned empty response")
                        raise WorkflowError("Empty response from content analysis workflow")

                    SystemLogger.info("Content analysis workflow completed successfully", {
                        'user_id': user_id,
                        'response_length': len(response),
                        'uploaded_files_processed': len(uploaded_files) if uploaded_files else 0
                    })

                    return response, similar_courses

                except Exception as workflow_error:
                    SystemLogger.error(
                        "Error in content analysis workflow",
                        exception=workflow_error,
                        context={'user_id': user_id, 'intent': intent}
                    )
                    raise WorkflowError(f"Content analysis workflow failed: {workflow_error}")

            elif intent == "irrelevant":
                SystemLogger.info("Query classified as irrelevant - returning standard message", {
                    'user_id': user_id,
                    'query_preview': query[:50]
                })
                return (
                    "Sorry, your query seems unrelated to our course offerings. "
                    "Please ask about IMPEL courses or Data Science learning goals.",
                    None
                )

            else:
                SystemLogger.error(
                    f"Unknown intent classification: {intent}",
                    context={'user_id': user_id, 'intent': intent, 'query': query}
                )
                return (
                    "Sorry, I couldn't understand your request. Please try again.",
                    None
                )

        except (WorkflowError, APIRequestError, AgentExecutionError) as e:
            SystemLogger.error(
                "Workflow/API/Agent error handling user query",
                exception=e,
                context={'user_id': user_id, 'query_preview': query[:50] if query else ''}
            )
            return (
                "Encountered an unkown error processing your request.",
                None
            )
        except Exception as e:
            SystemLogger.error(
                "Unexpected error handling user query",
                exception=e,
                context={'user_id': user_id, 'query_preview': query[:50] if query else ''}
            )
            return ("Encountered an unkown error processing your request.", None)
