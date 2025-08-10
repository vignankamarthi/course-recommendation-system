# Gradio interface 

import gradio as gr
from core.orchestrator import RecommendationSystem
#TODO: Comments needed for all funtions here. FOLLOW NUMPY

def process_recommendations(user_id, education, age_group, profession, user_query, uploaded_file):
    """
    Process user query through orchestrator and format UI response.
    
    This function is purely for UI formatting - all business logic 
    is handled by the RecommendationSystem orchestrator.
    """
    recommender = RecommendationSystem()
    
    # Convert uploaded file to list format expected by orchestrator
    files = uploaded_file if isinstance(uploaded_file, list) else [uploaded_file] if uploaded_file else []
    
    # Get response from orchestrator (handles all agent routing)
    response, similar_courses = recommender.handle_user_query(
        user_id=user_id,
        education=education,
        age_group=age_group,
        profession=profession,
        query=user_query,
        uploaded_files=files
    )

    if isinstance(response, dict) and "error" in response:
        return (
            gr.update(value=f"⚠️ {response['error']}", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )

    # Format UI response based on orchestrator result
    if response and not response.startswith("Sorry"):
        return (
            gr.update(value="✅ **Recommendation Ready!**", visible=True),
            gr.update(visible=False),  # No separate content display needed
            gr.update(value=f"{response}", visible=True, label="Recommendations"),
            gr.update(value=f"{similar_courses}" if similar_courses else "", visible=bool(similar_courses))
        )
    else:
        return (
            gr.update(value=f"⚠️ {response}", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )



def create_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("## Personalized Course Recommendation System")

        with gr.Row():
            user_id = gr.Textbox(label="User ID", placeholder="e.g., user123")
            education = gr.Radio(["High School", "Undergraduate", "Graduate"], label="Education")
            age_group = gr.Radio(["Under 18", "18-25", "26-40", "40+"], label="Age Group")
            profession = gr.Radio(["Student", "Professional"], label="Professional Status")
            user_query = gr.Textbox(label="Enter Your Query")
            file_upload = gr.File(label="Upload Resume (PDF or DOCX)", file_types=[".pdf", ".docx"], type="filepath")

        submit_button = gr.Button("Get Recommendations")

        execution_tag = gr.Markdown("")  # Top status message

        db_output = gr.Textbox(label="IMPEL Courses and Modules Info", visible=False)
        cont_output = gr.Textbox(label="Content-based Recommendations", visible=False)
        collab_output = gr.Textbox(label="Collaborative Filtering Recommendations", visible=False)
        similar_output = gr.Textbox(label="Similar Users Enrolled In", visible=False)

        # Loading message
        def show_processing_message(*args):
            return (
                gr.update(value="⏳ Processing your query...", visible=True),
                *[gr.update(visible=False)] * 3
            )

        submit_button.click(
            fn=show_processing_message,
            inputs=[user_id, education, age_group, profession, user_query, file_upload],
            outputs=[execution_tag, cont_output, collab_output, similar_output]
        ).then(
            fn=process_recommendations,
            inputs=[user_id, education, age_group, profession, user_query, file_upload],
            outputs=[execution_tag, cont_output, collab_output, similar_output]
        )

    demo.launch(debug=True)
