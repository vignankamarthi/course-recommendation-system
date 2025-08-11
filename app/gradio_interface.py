# Gradio interface 

import gradio as gr
from core.orchestrator import RecommendationSystem
from utils.logger import SystemLogger
from utils.exceptions import (
    WorkflowError, AgentExecutionError, ConfigurationError,
    UserInputValidationError
)

def process_recommendations(user_id, education, age_group, profession, user_query, uploaded_file):
    """
    Process user query through orchestrator and format UI response for Gradio.
    
    Handles the complete user interaction pipeline from input validation through
    orchestrator execution to UI response formatting. Provides comprehensive
    error handling and user feedback for the web interface.
    
    Parameters
    ----------
    user_id : str
        Unique identifier for user session tracking
    education : str
        User's education level selection from UI radio buttons
    age_group : str  
        User's age range selection from UI radio buttons
    profession : str
        User's professional status from UI radio buttons
    user_query : str
        Natural language query input from user
    uploaded_file : str or None
        File path of uploaded resume/document (if any)
        
    Returns
    -------
    tuple of gradio.Update
        Four-element tuple of Gradio update objects for UI components:
        - execution_tag: Status message display
        - cont_output: Content recommendations (hidden)
        - collab_output: Main recommendations display
        - similar_output: Similar user courses display
        
    Examples
    --------
    Used internally by Gradio interface for handling form submissions.
    Returns properly formatted UI updates for success/error states.
    """
    SystemLogger.info("Processing user recommendation request through Gradio interface", {
        'user_id': user_id,
        'education': education,
        'profession': profession,
        'query_preview': user_query[:100] if user_query else 'empty',
        'has_uploaded_file': bool(uploaded_file)
    })
    
    try:
        # Input validation
        if not user_id or not user_id.strip():
            SystemLogger.error(
                "Empty user ID provided to Gradio interface",
                context={'user_id': repr(user_id)}
            )
            return (
                gr.update(value="User ID is required", visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False)
            )
        
        # Validate required user profile fields
        required_fields = {
            'Education': education,
            'Age Group': age_group,
            'Professional Status': profession,
            'Query': user_query
        }
        
        missing_fields = [field for field, value in required_fields.items() if not value or not str(value).strip()]
        if missing_fields:
            SystemLogger.error(
                f"Missing required fields in Gradio interface: {missing_fields}",
                context={'user_id': user_id, 'missing_fields': missing_fields}
            )
            return (
                gr.update(value=f"Please fill in all required fields: {', '.join(missing_fields)}", visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False)
            )
        
        # Initialize orchestrator
        SystemLogger.debug("Initializing RecommendationSystem orchestrator")
        try:
            recommender = RecommendationSystem()
        except (ConfigurationError, AgentExecutionError) as init_error:
            SystemLogger.error(
                "Failed to initialize RecommendationSystem in Gradio interface",
                exception=init_error,
                context={'user_id': user_id}
            )
            return (
                gr.update(value="System initialization error. Please try again or contact support.", visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False)
            )
        
        # Process uploaded file
        SystemLogger.debug("Processing uploaded file for recommendations")
        try:
            # Convert uploaded file to list format expected by orchestrator
            files = []
            if uploaded_file:
                if isinstance(uploaded_file, list):
                    # Handle multiple files (future feature)
                    files = [f for f in uploaded_file if f and f.strip()]
                elif isinstance(uploaded_file, str) and uploaded_file.strip():
                    # Single file path
                    files = [uploaded_file]
                else:
                    SystemLogger.debug("Invalid uploaded file format - using empty list", {
                        'uploaded_file_type': type(uploaded_file),
                        'uploaded_file': uploaded_file
                    })
                    files = []
            
            SystemLogger.debug("File processing completed for Gradio interface", {
                'files_count': len(files)
            })
            
        except Exception as file_error:
            SystemLogger.error(
                "Error processing uploaded file in Gradio interface",
                exception=file_error,
                context={'user_id': user_id, 'uploaded_file': uploaded_file}
            )
            # Continue without files instead of failing completely
            files = []
            SystemLogger.info("Continuing without uploaded files due to processing error")
        
        # Get response from orchestrator (handles all agent routing)
        SystemLogger.debug("Invoking orchestrator for user query processing")
        try:
            response, similar_courses = recommender.handle_user_query(
                user_id=user_id,
                education=education,
                age_group=age_group,
                profession=profession,
                query=user_query,
                uploaded_files=files
            )
        except (WorkflowError, AgentExecutionError) as workflow_error:
            SystemLogger.error(
                "Workflow/Agent error in orchestrator through Gradio interface",
                exception=workflow_error,
                context={'user_id': user_id, 'query_preview': user_query[:50]}
            )
            return (
                gr.update(value="Processing error occurred. Please try again or contact support.", visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False)
            )
        
        # Validate orchestrator response
        if not response:
            SystemLogger.error(
                "Orchestrator returned empty response through Gradio interface",
                context={'user_id': user_id, 'similar_courses': similar_courses}
            )
            return (
                gr.update(value="No response generated. Please try rephrasing your query.", visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False)
            )
        
        # Handle error response format (legacy support)
        if isinstance(response, dict) and "error" in response:
            SystemLogger.info("Orchestrator returned error response format", {
                'user_id': user_id,
                'error': response['error']
            })
            return (
                gr.update(value=f"{response['error']}", visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False)
            )
        
        # Format UI response based on orchestrator result
        SystemLogger.debug("Formatting UI response from orchestrator result")
        if response and not str(response).startswith("Sorry"):
            SystemLogger.info("Successful recommendation response generated", {
                'user_id': user_id,
                'response_length': len(str(response)),
                'similar_courses_available': bool(similar_courses)
            })
            
            return (
                gr.update(value="**Recommendation Ready!**", visible=True),
                gr.update(visible=False),  # No separate content display needed
                gr.update(value=f"{response}", visible=True, label="Recommendations"),
                gr.update(value=f"{similar_courses}" if similar_courses else "", visible=bool(similar_courses))
            )
        else:
            SystemLogger.info("Non-successful response from orchestrator", {
                'user_id': user_id,
                'response_preview': str(response)[:100] if response else 'None'
            })
            
            return (
                gr.update(value=f"{response}", visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False)
            )
    
    except UserInputValidationError as validation_error:
        SystemLogger.error(
            "User input validation error in Gradio interface",
            exception=validation_error,
            context={'user_id': user_id}
        )
        return (
            gr.update(value=f"Input validation error: {validation_error}", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )
    except Exception as unexpected_error:
        SystemLogger.error(
            "Unexpected error in Gradio recommendation processing",
            exception=unexpected_error,
            context={'user_id': user_id, 'query_preview': user_query[:50] if user_query else ''}
        )
        return (
            gr.update(value="An unexpected error occurred. Please try again or contact support.", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )



def create_gradio_interface():
    """
    Create and configure Gradio web interface for course recommendation system.
    
    Builds a comprehensive web interface with input validation, error handling,
    and user feedback mechanisms. Provides interactive forms for user profile data,
    query input, and file upload capabilities with real-time processing feedback.
    
    Returns
    -------
    None
        Launches Gradio interface in debug mode on default port
        
    Raises
    ------
    UserInputValidationError
        If interface creation or component setup fails
        
    Notes
    -----
    Interface includes:
    - User profile inputs (education, age, profession)
    - Natural language query input with placeholder guidance
    - Optional resume/document upload (PDF/DOCX support)
    - Real-time processing status updates
    - Formatted recommendation output display
    - Error handling with user-friendly messages
    
    Examples
    --------
    >>> create_gradio_interface()
    # Launches web interface accessible at http://localhost:7860
    """
    SystemLogger.info("Creating Gradio interface for course recommendation system")
    
    try:
        SystemLogger.debug("Building Gradio Blocks interface")
        
        with gr.Blocks() as demo:
            gr.Markdown("## Personalized Course Recommendation System")
            
            # Input validation helper
            def validate_inputs(user_id, education, age_group, profession, user_query):
                """Validate user inputs before processing."""
                errors = []
                
                if not user_id or not user_id.strip():
                    errors.append("User ID is required")
                if not education:
                    errors.append("Education level is required")
                if not age_group:
                    errors.append("Age group is required")
                if not profession:
                    errors.append("Professional status is required")
                if not user_query or not user_query.strip():
                    errors.append("Query is required")
                
                return errors
            
            try:
                SystemLogger.debug("Creating input components")
                
                with gr.Row():
                    user_id = gr.Textbox(
                        label="User ID", 
                        placeholder="e.g., user123",
                        info="Enter a unique identifier for your session"
                    )
                    education = gr.Radio(
                        ["High School", "Undergraduate", "Graduate"], 
                        label="Education",
                        info="Select your highest education level"
                    )
                    age_group = gr.Radio(
                        ["Under 18", "18-25", "26-40", "40+"], 
                        label="Age Group",
                        info="Select your age range"
                    )
                    profession = gr.Radio(
                        ["Student", "Professional"], 
                        label="Professional Status",
                        info="Are you currently a student or working professional?"
                    )
                    user_query = gr.Textbox(
                        label="Enter Your Query",
                        placeholder="e.g., I want to become a data scientist",
                        info="Describe what you want to learn or achieve"
                    )
                    file_upload = gr.File(
                        label="Upload Resume (PDF or DOCX)", 
                        file_types=[".pdf", ".docx"], 
                        type="filepath",
                        info="Optional: Upload your resume for personalized recommendations"
                    )
                
                submit_button = gr.Button("Get Recommendations", variant="primary")
                
                SystemLogger.debug("Creating output components")
                
                execution_tag = gr.Markdown("")  # Top status message
                
                db_output = gr.Textbox(label="IMPEL Courses and Modules Info", visible=False)
                cont_output = gr.Textbox(label="Content-based Recommendations", visible=False)
                collab_output = gr.Textbox(label="Collaborative Filtering Recommendations", visible=False)
                similar_output = gr.Textbox(label="Similar Users Enrolled In", visible=False)
                
            except Exception as component_error:
                SystemLogger.error(
                    "Error creating Gradio interface components",
                    exception=component_error,
                    context={'component_creation_step': 'input_output_components'}
                )
                raise
            
            # Loading message function with error handling
            def show_processing_message(*args):
                """Display processing message with input validation."""
                try:
                    SystemLogger.debug("Displaying processing message for user request")
                    
                    user_id, education, age_group, profession, user_query, file_upload = args
                    
                    # Quick validation before showing processing message
                    validation_errors = validate_inputs(user_id, education, age_group, profession, user_query)
                    if validation_errors:
                        SystemLogger.info("Input validation failed - showing error message", {
                            'validation_errors': validation_errors
                        })
                        return (
                            gr.update(value=f"Please fix the following: {', '.join(validation_errors)}", visible=True),
                            *[gr.update(visible=False)] * 3
                        )
                    
                    return (
                        gr.update(value="Processing your query...", visible=True),
                        *[gr.update(visible=False)] * 3
                    )
                    
                except Exception as processing_error:
                    SystemLogger.error(
                        "Error in show_processing_message function",
                        exception=processing_error,
                        context={'args_count': len(args) if args else 0}
                    )
                    return (
                        gr.update(value="Error preparing request. Please try again.", visible=True),
                        *[gr.update(visible=False)] * 3
                    )
            
            try:
                SystemLogger.debug("Setting up Gradio event handlers")
                
                # Chain the processing: validation -> processing message -> actual processing
                submit_button.click(
                    fn=show_processing_message,
                    inputs=[user_id, education, age_group, profession, user_query, file_upload],
                    outputs=[execution_tag, cont_output, collab_output, similar_output]
                ).then(
                    fn=process_recommendations,
                    inputs=[user_id, education, age_group, profession, user_query, file_upload],
                    outputs=[execution_tag, cont_output, collab_output, similar_output]
                )
                
                SystemLogger.debug("Gradio event handlers configured successfully")
                
            except Exception as event_error:
                SystemLogger.error(
                    "Error setting up Gradio event handlers",
                    exception=event_error,
                    context={'interface_setup_step': 'event_handlers'}
                )
                raise
        
        SystemLogger.info("Gradio interface created successfully")
        
        try:
            SystemLogger.info("Launching Gradio interface", {
                'debug_mode': True,
                'interface_components': ['user_id', 'education', 'age_group', 'profession', 'query', 'file_upload']
            })
            
            demo.launch(debug=True)
            
        except Exception as launch_error:
            SystemLogger.error(
                "Error launching Gradio interface",
                exception=launch_error,
                context={'launch_config': {'debug': True}}
            )
            raise
    
    except Exception as interface_error:
        SystemLogger.error(
            "Failed to create Gradio interface",
            exception=interface_error,
            context={'interface_creation_step': 'overall'}
        )
        raise UserInputValidationError(f"Gradio interface creation failed: {interface_error}")
