"""
AI Podcast Generator - A Gradio interface for generating podcasts from PDF documents.
"""
import os
import logging
from typing import Dict, List, Tuple, Any, Optional

import gradio as gr
from podcastfy.interface import (
    get_available_tts_models,
    get_available_voices,
    get_existing_projects,
    load_data,
    generate_podcast,
    update_config_preview
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CONFIG_PATH = "./podcastfy/configs/config.yaml"
PROJECTS_DIR = "./projects"
CSS_PATH = "./assets/style.css"

def get_css():
    """Load custom CSS or return default styling."""
    if os.path.exists(CSS_PATH):
        with open(CSS_PATH, "r") as f:
            return f.read()
    else:
        # Default CSS if file doesn't exist
        return """
        .gradio-container {
            font-family: 'Roboto', sans-serif;
        }
        .main-header {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 2rem;
        }
        .tab-content {
            padding: 1rem;
        }
        .action-btn {
            background-color: #3498db;
            color: white;
        }
        .primary-btn {
            background-color: #2ecc71;
            color: white;
        }
        footer {
            text-align: center;
            margin-top: 2rem;
            color: #7f8c8d;
            font-size: 0.8rem;
        }
        """

def update_voices(tts_model):
    """Update available voices based on selected TTS model."""
    try:
        voices = get_available_voices(tts_model)
        choices = [v["display"] for v in voices]
        return gr.update(choices=choices, value=choices[0]), gr.update(choices=choices, value=choices[1] if len(choices) > 1 else choices[0])
    except Exception as e:
        logger.error(f"Error updating voices: {e}")
        return gr.update(choices=["Error loading voices"], value="Error loading voices"), gr.update(choices=["Error loading voices"], value="Error loading voices")

def get_projects_html():
    """Generate HTML list of existing projects."""
    try:
        projects = get_existing_projects()
        if not projects:
            return "<p>No projects found. Create a new project by uploading a PDF.</p>"

        html = "<ul class='projects-list'>"
        for project in projects:
            project_path = os.path.abspath(project)
            html += f'<li><a href="file://{project_path}" target="_blank">{project}</a></li>'
        html += "</ul>"
        return html
    except Exception as e:
        logger.error(f"Error getting projects: {e}")
        return f"<p>Error loading projects: {str(e)}</p>"


def create_interface():
    """Create the Gradio interface for the AI Podcast Generator."""
    with gr.Blocks(title="AI Podcast Generator", css=get_css(), fill_width=True) as demo:
        gr.Markdown("# AI Podcast Generator", elem_classes=["main-header"])

        with gr.Tabs() as tabs:
            with gr.Tab("Upload & Configure", elem_classes=["tab-content"]):
                with gr.Row():
                    with gr.Column(scale=1):
                        pdf_file = gr.File(label="Upload PDF Document", file_types=[".pdf"])

                        with gr.Group():
                            gr.Markdown("## Basic Settings")
                            output_language = gr.Dropdown(
                                choices=["English", "Spanish", "French", "German"],
                                label="Output Language",
                                value="English"
                            )
                            podcast_name = gr.Textbox(label="Podcast Name", value="AI Talks Science")
                            podcast_tagline = gr.Textbox(label="Tag line", value="Where AI talks Science")
                            doctitle = gr.Textbox(label="Document Title", value="")

                            with gr.Row():
                                person1_name = gr.Textbox(label="Host Name", value="Host")
                                person2_name = gr.Textbox(label="Guest Name", value="Guest")

                            with gr.Row():
                                host_role = gr.Textbox(label="Host Role", value="Science Communicator")
                                guest_role = gr.Textbox(label="Guest Role", value="Paper Author")

                        with gr.Group():
                            gr.Markdown("## Content Settings")
                            conversation_style = gr.CheckboxGroup(
                                choices=["engaging", "fast-paced", "enthusiastic", "formal", "educational", "debate"],
                                label="Conversation Style",
                                value=["engaging", "educational"]
                            )

                            document_type = gr.Radio(
                                choices=["paper", "review"],
                                label="Document Type",
                                value="research paper"
                            )

                            dialogue_structure = gr.CheckboxGroup(
                                choices=["Introduction with Main Content Summary",
                                        "Interview going over the list of questions",
                                        "Conclusion"],
                                label="Dialogue Structure",
                                value=["Introduction with Main Content Summary", "Conclusion"]
                            )

                            engagement_techniques = gr.CheckboxGroup(
                                choices=["rhetorical questions", "anecdotes", "analogies", "humor"],
                                label="Engagement Techniques",
                                value=["rhetorical questions", "humor"]
                            )

                            with gr.Row():
                                creativity = gr.Slider(label="Creativity", value=0.5, minimum=0, maximum=1, step=0.1)
                                podcast_duration = gr.Dropdown(
                                choices=["5 min", "10 min", "15 min", "20 min", "25 min", "30 min", "45 min"],
                                label="Podcast Duration",
                                value="15 min"
                            )

                        with gr.Group():
                            gr.Markdown("## Model Settings")
                            base_config_path = gr.Textbox(
                                label="Base Config Path",
                                value=DEFAULT_CONFIG_PATH
                            )

                            model_name = gr.Dropdown(
                                choices=["gemini-1.5-pro", "gemini-2.0-flash", "claude-3-opus"],
                                label="LLM Model",
                                value="gemini-2.0-flash"
                            )

                        with gr.Group():
                            gr.Markdown("## Text-to-Speech Settings")
                            tts_model = gr.Dropdown(
                                choices=get_available_tts_models(),
                                label="TTS Model",
                                value="googleneural2"
                            )

                            with gr.Row():
                                host_voice = gr.Dropdown(
                                    label="Host Voice",
                                    choices=[f"{v['name']} ({v['gender']})" for v in get_available_voices()],
                                    value="en-US-Neural2-A (Male)"
                                )
                                guest_voice = gr.Dropdown(
                                    label="Guest Voice",
                                    choices=[f"{v['name']} ({v['gender']})" for v in get_available_voices()],
                                    value="en-US-Neural2-C (Female)"
                                )

                            audio_format = gr.Dropdown(
                                label="Audio Format",
                                choices=["mp3", "wav", "ogg"],
                                value="mp3"
                            )

                            ending_message = gr.Textbox(label="Ending Message", value="Bye Bye!")

                    with gr.Column(scale=1):
                        update_button = gr.Button("Update Configuration", elem_classes=["action-btn"])
                        config_data = gr.State({})
                        config_preview = gr.Code(label="Configuration Preview", language="yaml")

                        load_button = gr.Button("Load Data", variant="primary", elem_classes=["primary-btn"])
                        load_output = gr.Textbox(label="Load Status")
                        current_project_path = gr.State(None)
                        current_pdf_path = gr.State(None)
                        current_config_path = gr.State(None)

                        generate_button = gr.Button("Generate Podcast", variant="primary", elem_classes=["primary-btn"])
                        generate_output = gr.Textbox(label="Generation Status")
                        audio_output = gr.Audio(label="Generated Podcast")

            with gr.Tab("Projects", elem_classes=["tab-content"]):
                refresh_button = gr.Button("Refresh Projects", elem_classes=["action-btn"])
                projects_list = gr.HTML(label="Existing Projects")

        # Footer
        gr.Markdown(
            "### AI Podcast Generator v1.0\n"
            "Created with ❤️ using Gradio and Podcastfy",
            elem_classes=["footer"]
        )

        # Register event handlers
        tts_model.change(
            fn=update_voices,
            inputs=[tts_model],
            outputs=[host_voice, guest_voice]
        )

        update_button.click(
            fn=update_config_preview,
            inputs=[output_language, podcast_name, podcast_tagline, doctitle, person1_name, person2_name,
                   host_role, guest_role, conversation_style, document_type, # Added document_type here
                   dialogue_structure, engagement_techniques, creativity,
                   podcast_duration, tts_model, host_voice, guest_voice, audio_format, ending_message],
            outputs=[config_data, config_preview]
        )

        load_button.click(
            fn=load_data,
            inputs=[pdf_file, config_data],
            outputs=[load_output, current_project_path, current_pdf_path, current_config_path]
        )

        generate_button.click(
            fn=generate_podcast,
            inputs=[current_project_path, current_pdf_path, current_config_path, base_config_path, model_name, tts_model],
            outputs=[generate_output, audio_output]
        )

        refresh_button.click(fn=get_projects_html, outputs=projects_list)

        # Initialize projects list
        demo.load(fn=get_projects_html, outputs=projects_list)

    return demo

def main():
    """Main entry point for the application."""
    try:
        # Make sure projects and assets directory exists
        os.makedirs(PROJECTS_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(CSS_PATH), exist_ok=True)

        # Launch the interface
        app = create_interface()
        app.launch(
            quiet=True,
            show_api=False,
            inbrowser=True,
            server_name="127.0.0.1",
            server_port=7860
        )
    except Exception as e:
        logger.error(f"Error launching application: {e}")
        raise

if __name__ == "__main__":
    # Set httpx logging to WARNING to reduce output
    logging.getLogger("httpx").setLevel(logging.INFO)
    main()
