import os
import glob
import shutil
import yaml
from podcastfy.utils.config import Config
from podcastfy.utils.config_conversation import ConversationConfig
from podcastfy.client import process_content
from podcastfy.tts.providers.kokoro import KOKORO_VOICES
from podcastfy.tts.providers.google_neural2 import GOOGLE_VOICES

# Get available TTS models and voices (this would be replaced with actual function calls)
def get_available_tts_models():
    return ["googleneural2", "googleneural", "elevenlabs", "openai", "kokoro"]


def get_available_voices(tts_model="googleneural2"):
    """Get available voices based on the selected TTS model."""
    if tts_model.lower() == "kokoro":
        # Return Kokoro voices
        voices = []
        for voice_id, voice_data in KOKORO_VOICES.items():
            voices.append({
                "name": voice_id,
                "gender": voice_data["gender"],
                "country": voice_data["country"],
                "display": voice_data["display"]
            })
        return voices
    elif tts_model.lower() == "googleneural2":
        # Return Google Neural2 voices
        voices = []
        for voice_id, voice_data in GOOGLE_VOICES.items():
            voices.append({
                "name": voice_id,
                "gender": voice_data["gender"],
                "country": voice_data["country"],
                "display": voice_data["display"]
            })
        return voices
    else:
        # Return a minimal default list for other providers
        return [
            {"name": "en-US-Neural2-A", "gender": "Male", "display": "en-US-Neural2-A (Male)"},
            {"name": "en-US-Neural2-C", "gender": "Female", "display": "en-US-Neural2-C (Female)"},
            {"name": "en-US-Neural2-D", "gender": "Male", "display": "en-US-Neural2-D (Male)"},
            {"name": "en-US-Neural2-F", "gender": "Female", "display": "en-US-Neural2-F (Female)"},
            {"name": "en-US-Neural2-J", "gender": "Male", "display": "en-US-Neural2-J (Male)"}
        ]

def get_existing_projects():
    projects = glob.glob("./projects/project_*")
    projects.sort(key=lambda x: int(x.split('_')[-1]))
    return projects

def get_next_project_number():
    projects = get_existing_projects()
    if not projects:
        return 1
    last_project = projects[-1]
    return int(last_project.split('_')[-1]) + 1

def create_project_folder(project_number):
    project_path = f"./projects/project_{project_number}"
    os.makedirs(project_path, exist_ok=True)
    os.makedirs(os.path.join(project_path, "transcripts"), exist_ok=True)
    os.makedirs(os.path.join(project_path, "audio"), exist_ok=True)
    return project_path

def load_data(pdf_file, config_data):
    project_number = get_next_project_number()
    project_path = create_project_folder(project_number)
    
    # Save uploaded PDF
    pdf_path = os.path.join(project_path, os.path.basename(pdf_file.name))
    shutil.copy(pdf_file.name, pdf_path)
    
    # Update paths in config to match the project structure
    config_data['text_to_speech']['output_directories']['transcripts'] = f"./projects/project_{project_number}/transcripts"
    config_data['text_to_speech']['output_directories']['audio'] = f"./projects/project_{project_number}/audio"
    
    # Save conversation config
    config_path = os.path.join(project_path, "conversation_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)
    
    return f"Data loaded to {project_path}", project_path, pdf_path, config_path

def generate_podcast(project_path, pdf_path, config_path, base_config_path, model_name, tts_model):
    try:
        # Load configs
        config = Config(base_config_path)
        conversation_config = ConversationConfig(path=config_path)
        
        # Generate podcast
        audio_file = process_content(
            urls=[pdf_path],
            config=config,
            conversation_config=conversation_config.to_dict(),
            model_name=model_name,
            tts_model=tts_model
        )
        
        return f"Podcast generated successfully: {audio_file}", audio_file
    except Exception as e:
        return f"Error generating podcast: {str(e)}", None

def get_voice(selected_voice, tts_model):
    if tts_model == "google_neural2":
        for voice_key, voice_data in GOOGLE_VOICES.items():
            if voice_data["display"] == selected_voice:
                return voice_key
    else:
        for voice_key, voice_data in KOKORO_VOICES.items():
            if voice_data["display"] == selected_voice:
                return voice_key
    return selected_voice    

def update_config_preview(output_language, podcast_name, podcast_tagline, doctitle, person1_name, person2_name, 
                         host_role, guest_role, conversation_style, dialogue_structure, 
                         engagement_techniques, creativity, max_num_chunks, min_chunk_size,
                         podcast_duration, tts_model, host_voice, guest_voice, audio_format, ending_message):
    
    # Extract duration in minutes as an integer
    duration_minutes = podcast_duration
    
    config_data = {
        "output_language": output_language,
        "podcast_name": podcast_name,
        "podcast_tagline": podcast_tagline,
        "roles_person1": host_role,
        "roles_person2": guest_role,
        "person1_name": person1_name,
        "person2_name": person2_name,
        "doctitle": doctitle,
        "conversation_style": conversation_style,
        "dialogue_structure": dialogue_structure,
        "engagement_techniques": engagement_techniques,
        "creativity": float(creativity),
        "max_num_chunks": int(max_num_chunks),
        "min_chunk_size": int(min_chunk_size),
        "duration": duration_minutes,
        "text_to_speech": {
            "default_tts_model": tts_model,
            "output_directories": {
                "transcripts": "./transcripts",  # Will be updated during project creation
                "audio": "./audio"  # Will be updated during project creation
            },
            tts_model: {
                "default_voices": {
                    "question": get_voice(host_voice, tts_model),
                    "answer": get_voice(guest_voice, tts_model),
                }
            },
            "audio_format": audio_format,
            "temp_audio_dir": "data/audio/tmp/",
            "ending_message": ending_message
        }
    }
    
    return config_data, yaml.dump(config_data, sort_keys=False)
