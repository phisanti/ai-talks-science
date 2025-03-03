"""Google Neural2 TTS provider."""

import re
import logging
from typing import List, Optional, Any, Dict

from google.cloud import texttospeech
from ..base import TTSProvider

logger = logging.getLogger(__name__)

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
            if len(voice_parts) != 4:
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