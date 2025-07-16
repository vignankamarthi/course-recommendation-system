# Gradio interface 

import gradio as gr
from rag_inference import run_rag_inference
from recommender_langgraph import RecommendationSystem
from content_agent import AgenticRAG
from config import cohere_api_key

content_agent = AgenticRAG(cohere_key = cohere_api_key)

def process_recommendations(user_id, education, age_group, profession, user_query, uploaded_file):
    recommender = RecommendationSystem()
    intent = recommender.classify_intent(user_query)

    if intent == "database_lookup":
        response, similar_courses = recommender.handle_user_query(
        user_id=user_id,
        education=education,
        age_group=age_group,
        profession=profession,
        query=user_query
    )
        if response.strip().lower().startswith("sorry, i couldn't find"):
            return (
                gr.update(value=f"⚠️ {response.strip()}", visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False)
            )
        return (
            gr.update(value="ℹ️ **Database Agent activated: Showing course/module info**", visible=True),
            gr.update(value=f"{response}", visible=True, label="Courses and Modules"),
            gr.update(value="", visible=False),
            gr.update(value=f"{similar_courses}", visible=True)
        )

    # Content-based + RAG collaborative recommendation path
    files = uploaded_file if isinstance(uploaded_file, list) else [uploaded_file] if uploaded_file else []
    content_rec = content_agent.run(query=user_query, uploaded_files=files)
    response, similar_courses = run_rag_inference(user_id, education, age_group, profession, user_query, files)

    if isinstance(response, dict) and "error" in response:
        return (
            gr.update(value=f"⚠️ {response['error']}", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )

    return (
        gr.update(value="✅ **Recommendation Agent activated: Personalized Recommendations Ready!**", visible=True),
        gr.update(value=f"{content_rec}", visible=True, label="Content-based Recommendations"),
        gr.update(value=f"{response}", visible=True),
        gr.update(value=f"{similar_courses}", visible=True)
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
