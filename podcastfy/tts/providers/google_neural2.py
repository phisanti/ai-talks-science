"""Google Neural2 TTS provider."""

import re
import logging
from typing import List, Optional, Any, Dict

from google.cloud import texttospeech
from ..base import TTSProvider

logger = logging.getLogger(__name__)

# Google Neural2 and Chirp HD voice mapping
GOOGLE_VOICES = {
    # British English Voices
    "en-GB-Chirp-HD-D": {"name": "en-GB-Chirp-HD-D", "gender": "Male", "country": "British", "display": "en-GB-Chirp-HD-D (Male)"},
    "en-GB-Chirp-HD-F": {"name": "en-GB-Chirp-HD-F", "gender": "Female", "country": "British", "display": "en-GB-Chirp-HD-F (Female)"},
    "en-GB-Chirp-HD-O": {"name": "en-GB-Chirp-HD-O", "gender": "Female", "country": "British", "display": "en-GB-Chirp-HD-O (Female)"},
    "en-GB-Chirp3-HD-Aoede": {"name": "en-GB-Chirp3-HD-Aoede", "gender": "Female", "country": "British", "display": "en-GB-Chirp3-HD-Aoede (Female)"},
    "en-GB-Chirp3-HD-Charon": {"name": "en-GB-Chirp3-HD-Charon", "gender": "Male", "country": "British", "display": "en-GB-Chirp3-HD-Charon (Male)"},
    "en-GB-Chirp3-HD-Fenrir": {"name": "en-GB-Chirp3-HD-Fenrir", "gender": "Male", "country": "British", "display": "en-GB-Chirp3-HD-Fenrir (Male)"},
    "en-GB-Chirp3-HD-Kore": {"name": "en-GB-Chirp3-HD-Kore", "gender": "Female", "country": "British", "display": "en-GB-Chirp3-HD-Kore (Female)"},
    "en-GB-Chirp3-HD-Leda": {"name": "en-GB-Chirp3-HD-Leda", "gender": "Female", "country": "British", "display": "en-GB-Chirp3-HD-Leda (Female)"},
    "en-GB-Chirp3-HD-Orus": {"name": "en-GB-Chirp3-HD-Orus", "gender": "Male", "country": "British", "display": "en-GB-Chirp3-HD-Orus (Male)"},
    "en-GB-Chirp3-HD-Puck": {"name": "en-GB-Chirp3-HD-Puck", "gender": "Male", "country": "British", "display": "en-GB-Chirp3-HD-Puck (Male)"},
    "en-GB-Chirp3-HD-Zephyr": {"name": "en-GB-Chirp3-HD-Zephyr", "gender": "Female", "country": "British", "display": "en-GB-Chirp3-HD-Zephyr (Female)"},
    "en-GB-Neural2-A": {"name": "en-GB-Neural2-A", "gender": "Female", "country": "British", "display": "en-GB-Neural2-A (Female)"},
    "en-GB-Neural2-B": {"name": "en-GB-Neural2-B", "gender": "Male", "country": "British", "display": "en-GB-Neural2-B (Male)"},
    "en-GB-Neural2-C": {"name": "en-GB-Neural2-C", "gender": "Female", "country": "British", "display": "en-GB-Neural2-C (Female)"},
    "en-GB-Neural2-D": {"name": "en-GB-Neural2-D", "gender": "Male", "country": "British", "display": "en-GB-Neural2-D (Male)"},
    "en-GB-Neural2-F": {"name": "en-GB-Neural2-F", "gender": "Female", "country": "British", "display": "en-GB-Neural2-F (Female)"},
    "en-GB-Neural2-N": {"name": "en-GB-Neural2-N", "gender": "Female", "country": "British", "display": "en-GB-Neural2-N (Female)"},
    "en-GB-Neural2-O": {"name": "en-GB-Neural2-O", "gender": "Male", "country": "British", "display": "en-GB-Neural2-O (Male)"},
    
    # American English Voices
    "en-US-Chirp-HD-D": {"name": "en-US-Chirp-HD-D", "gender": "Male", "country": "American", "display": "en-US-Chirp-HD-D (Male)"},
    "en-US-Chirp-HD-F": {"name": "en-US-Chirp-HD-F", "gender": "Female", "country": "American", "display": "en-US-Chirp-HD-F (Female)"},
    "en-US-Chirp-HD-O": {"name": "en-US-Chirp-HD-O", "gender": "Female", "country": "American", "display": "en-US-Chirp-HD-O (Female)"},
    "en-US-Chirp3-HD-Aoede": {"name": "en-US-Chirp3-HD-Aoede", "gender": "Female", "country": "American", "display": "en-US-Chirp3-HD-Aoede (Female)"},
    "en-US-Chirp3-HD-Charon": {"name": "en-US-Chirp3-HD-Charon", "gender": "Male", "country": "American", "display": "en-US-Chirp3-HD-Charon (Male)"},
    "en-US-Chirp3-HD-Fenrir": {"name": "en-US-Chirp3-HD-Fenrir", "gender": "Male", "country": "American", "display": "en-US-Chirp3-HD-Fenrir (Male)"},
    "en-US-Chirp3-HD-Kore": {"name": "en-US-Chirp3-HD-Kore", "gender": "Female", "country": "American", "display": "en-US-Chirp3-HD-Kore (Female)"},
    "en-US-Chirp3-HD-Leda": {"name": "en-US-Chirp3-HD-Leda", "gender": "Female", "country": "American", "display": "en-US-Chirp3-HD-Leda (Female)"},
    "en-US-Chirp3-HD-Orus": {"name": "en-US-Chirp3-HD-Orus", "gender": "Male", "country": "American", "display": "en-US-Chirp3-HD-Orus (Male)"},
    "en-US-Chirp3-HD-Puck": {"name": "en-US-Chirp3-HD-Puck", "gender": "Male", "country": "American", "display": "en-US-Chirp3-HD-Puck (Male)"},
    "en-US-Chirp3-HD-Zephyr": {"name": "en-US-Chirp3-HD-Zephyr", "gender": "Female", "country": "American", "display": "en-US-Chirp3-HD-Zephyr (Female)"},
    "en-US-Neural2-A": {"name": "en-US-Neural2-A", "gender": "Male", "country": "American", "display": "en-US-Neural2-A (Male)"},
    "en-US-Neural2-C": {"name": "en-US-Neural2-C", "gender": "Female", "country": "American", "display": "en-US-Neural2-C (Female)"},
    "en-US-Neural2-D": {"name": "en-US-Neural2-D", "gender": "Male", "country": "American", "display": "en-US-Neural2-D (Male)"},
    "en-US-Neural2-E": {"name": "en-US-Neural2-E", "gender": "Female", "country": "American", "display": "en-US-Neural2-E (Female)"},
    "en-US-Neural2-F": {"name": "en-US-Neural2-F", "gender": "Female", "country": "American", "display": "en-US-Neural2-F (Female)"},
    "en-US-Neural2-G": {"name": "en-US-Neural2-G", "gender": "Female", "country": "American", "display": "en-US-Neural2-G (Female)"},
    "en-US-Neural2-H": {"name": "en-US-Neural2-H", "gender": "Female", "country": "American", "display": "en-US-Neural2-H (Female)"},
    "en-US-Neural2-I": {"name": "en-US-Neural2-I", "gender": "Male", "country": "American", "display": "en-US-Neural2-I (Male)"},
    "en-US-Neural2-J": {"name": "en-US-Neural2-J", "gender": "Male", "country": "American", "display": "en-US-Neural2-J (Male)"}
}

class GoogleNeural2TTS(TTSProvider):
    """Google Neural2 TTS provider class."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the Google Neural2 TTS provider.
        
        Args:
            api_key (Optional[str]): Google Cloud API key
            model (Optional[str]): Default voice model to use
        """
        self.model = model or "google_neural2"
        try:
            self.client = texttospeech.TextToSpeechClient(
                client_options={'api_key': api_key} if api_key else None
            )
            logger.info("Successfully initialized GoogleNeural2TTS client")
        except Exception as e:
            logger.error(f"Failed to initialize GoogleNeural2TTS client: {str(e)}")
            raise
        
    def generate_audio(self, text: str, voice: Optional[str] = None, 
                      model: Optional[str] = None, **kwargs) -> bytes:
        """
        Generate audio from text using Google Neural2 TTS.
        
        Args:
            text: The text to convert to speech
            voice: The voice to use for the speech
            model: The model to use for speech generation
            **kwargs: Additional arguments
            
        Returns:
            Bytes of the audio content
        """
        try:
            # Default voice if none provided
            voice = voice or "en-US-Neural2-F"
            
            # Clean text for processing
            text = self.clean_text(text)
            
            # Set the text input to be synthesized
            synthesis_input = texttospeech.SynthesisInput(text=text)

            # Parse voice name (expects format like "en-US-Neural2-F")
            voice_parts = voice.split('-')
            if len(voice_parts) < 4:
                logger.warning(f"Invalid voice format: {voice}. Using default.")
                language_code = "en-US"
                name = "en-US-Neural2-F"
            else:
                language_code = f"{voice_parts[0]}-{voice_parts[1]}"
                name = voice
            # Build the voice request
            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=name,
            )

            # Select the type of audio file to return
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=1.0,
                pitch=0.0,
            )

            # Perform the text-to-speech request
            response = self.client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )
            return response.audio_content
            
        except Exception as e:
            logger.error(f"Error generating audio with Google Neural2: {str(e)}")
            raise

    def clean_text(self, text: str) -> str:
        """
        Clean and prepare text for TTS processing.
        
        Args:
            text: The text to clean
        
        Returns:
            Cleaned text ready for TTS
        """
        # Basic cleaning
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Other cleaning as needed
        return text