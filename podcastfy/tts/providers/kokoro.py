"""Kokoro TTS provider."""

import io
import re
import logging
import torch
import numpy as np
import soundfile as sf
from typing import List, Optional, Any, Dict, Tuple, Generator

from kokoro import KPipeline
from ..base import TTSProvider

logger = logging.getLogger(__name__)

"""Kokoro voice mapping."""

# Voice file names to structured information mapping
KOKORO_VOICES = {
    # American Female voices
    "af_alloy": {"name": "af_alloy", "gender": "Female", "country": "American", "display": "Alloy (American Female)"},
    "af_aoede": {"name": "af_aoede", "gender": "Female", "country": "American", "display": "Aoede (American Female)"},
    "af_bella": {"name": "af_bella", "gender": "Female", "country": "American", "display": "Bella (American Female)"},
    "af_heart": {"name": "af_heart", "gender": "Female", "country": "American", "display": "Heart (American Female)"},
    "af_jessica": {"name": "af_jessica", "gender": "Female", "country": "American", "display": "Jessica (American Female)"},
    "af_kore": {"name": "af_kore", "gender": "Female", "country": "American", "display": "Kore (American Female)"},
    "af_nicole": {"name": "af_nicole", "gender": "Female", "country": "American", "display": "Nicole (American Female)"},
    "af_nova": {"name": "af_nova", "gender": "Female", "country": "American", "display": "Nova (American Female)"},
    "af_river": {"name": "af_river", "gender": "Female", "country": "American", "display": "River (American Female)"},
    "af_sarah": {"name": "af_sarah", "gender": "Female", "country": "American", "display": "Sarah (American Female)"},
    "af_sky": {"name": "af_sky", "gender": "Female", "country": "American", "display": "Sky (American Female)"},
    
    # American Male voices
    "am_adam": {"name": "am_adam", "gender": "Male", "country": "American", "display": "Adam (American Male)"},
    "am_echo": {"name": "am_echo", "gender": "Male", "country": "American", "display": "Echo (American Male)"},
    "am_eric": {"name": "am_eric", "gender": "Male", "country": "American", "display": "Eric (American Male)"},
    "am_fenrir": {"name": "am_fenrir", "gender": "Male", "country": "American", "display": "Fenrir (American Male)"},
    "am_liam": {"name": "am_liam", "gender": "Male", "country": "American", "display": "Liam (American Male)"},
    "am_michael": {"name": "am_michael", "gender": "Male", "country": "American", "display": "Michael (American Male)"},
    "am_onyx": {"name": "am_onyx", "gender": "Male", "country": "American", "display": "Onyx (American Male)"},
    "am_puck": {"name": "am_puck", "gender": "Male", "country": "American", "display": "Puck (American Male)"},
    "am_santa": {"name": "am_santa", "gender": "Male", "country": "American", "display": "Santa (American Male)"},
    
    # British Female voices
    "bf_alice": {"name": "bf_alice", "gender": "Female", "country": "British", "display": "Alice (British Female)"},
    "bf_emma": {"name": "bf_emma", "gender": "Female", "country": "British", "display": "Emma (British Female)"},
    "bf_isabella": {"name": "bf_isabella", "gender": "Female", "country": "British", "display": "Isabella (British Female)"},
    "bf_lily": {"name": "bf_lily", "gender": "Female", "country": "British", "display": "Lily (British Female)"},
    
    # British Male voices
    "bm_daniel": {"name": "bm_daniel", "gender": "Male", "country": "British", "display": "Daniel (British Male)"},
    "bm_fable": {"name": "bm_fable", "gender": "Male", "country": "British", "display": "Fable (British Male)"},
    "bm_george": {"name": "bm_george", "gender": "Male", "country": "British", "display": "George (British Male)"},
    "bm_lewis": {"name": "bm_lewis", "gender": "Male", "country": "British", "display": "Lewis (British Male)"},    
}


def get_device():
    """Get the best available device in order of preference: CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    #elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #    return torch.device("mps")
    # Currently off due to the following error:
        # NotImplementedError: The operator 'aten::angle' is not currently implemented for the MPS device. If you want 
        # this op to be considered for addition please comment on https://github.com/pytorch/pytorch/issues/141287 and 
        # mention use-case, that resulted in missing op as well as commit hash 2236df1770800ffea5697b11b0bb0d910b2e59e1. 
        # As a temporary fix, you can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` 
        # to use the CPU as a fallback for this op. WARNING: this will be slower than running natively on MPS.
    else:
        return torch.device("cpu")


class KokoroTTS(TTSProvider):
    """Kokoro TTS provider class for text-to-speech conversion."""

    def __init__(self, model: Optional[str] = None):
        """
        Initialize the Kokoro TTS provider.
        
        Args:
            api_key (Optional[str]): Not used for Kokoro
            model (Optional[str]): Default language code to use (e.g., 'a' for American English)
        """
        self.model = model or "kokoro"
        
        # Map model name to language code if needed
        lang_map = {
            "kokoro": "a",       # American English
            "kokoro_en_us": "a", # American English
            "kokoro_en_gb": "b", # British English
            "kokoro_es": "e",    # Spanish
            "kokoro_fr": "f",    # French
            "kokoro_hi": "h",    # Hindi
            "kokoro_it": "i",    # Italian
            "kokoro_pt": "p",    # Portuguese
            "kokoro_jp": "j",    # Japanese
            "kokoro_zh": "z"     # Chinese
        }
        
        # Get language code from model name or use 'a' as default
        self.lang_code = lang_map.get(self.model.lower(), 'a')
        
        try:
            # Initialize the Kokoro pipeline
            self.pipeline = KPipeline(lang_code=self.lang_code, device=get_device())
            logger.info(f"Successfully initialized KokoroTTS pipeline with language {self.lang_code}")
        except Exception as e:
            logger.error(f"Failed to initialize KokoroTTS pipeline: {str(e)}")
            raise
    
    def get_supported_tags(self) -> List[str]:
        """
        Get the list of tags supported by this provider.
        
        Returns:
            List of supported tag names
        """
        return ["Person1", "Person2"]
        
    def generate_audio(self, text: str, voice: Optional[str] = None, 
                      model: Optional[str] = None, **kwargs) -> bytes:
        """
        Generate audio from text using Kokoro TTS.
        
        Args:
            text: The text to convert to speech
            voice: The voice to use for the speech (e.g., 'af_heart')
            model: The model language code to use (e.g., 'a' for American English)
            **kwargs: Additional arguments including:
                - ending_message: Optional text to append at the end
                - voice2: Optional second voice for multi-speaker functionality
            
        Returns:
            Bytes of the audio content
        """
        try:
            # Default voice if none provided
            voice = voice or "af_heart"
            ending_message = kwargs.get("ending_message", "")
            
            # Clean text for processing
            text = self.clean_text(text)
            
            # Use model parameter to override default language if provided
            if model and model in "abefhijpz":  # Valid language codes
                if model != self.lang_code:
                    self.pipeline = KPipeline(lang_code=model)
                    self.lang_code = model
                    logger.info(f"Switched to language code: {self.lang_code}")
            
            # Check if we're handling multi-speaker content
            voice2 = kwargs.get("voice2")
            if voice2:
                logger.info(f"Using multi-speaker mode with voices {voice} and {voice2}")
                return self._generate_multi_speaker_audio(text, voice, voice2, ending_message)
            
            # Generate audio using Kokoro for single speaker
            all_audio = []
            generator = self.pipeline(text, voice=voice, speed=1.0, split_pattern=r'\n+')
            
            for gs, ps, audio in generator:
                if audio is not None:
                    logger.debug(f"Generated segment: {gs[:30]}...")
                    all_audio.append(torch.tensor(audio))
            
            # Add ending message if provided
            if ending_message:
                generator = self.pipeline(ending_message, voice=voice, speed=1.0)
                for _, _, audio in generator:
                    if audio is not None:
                        all_audio.append(torch.tensor(audio))
            
            # Combine all audio segments
            if not all_audio:
                raise ValueError("No audio was generated")
                
            combined_audio = torch.cat(all_audio, dim=0).numpy()
            
            # Convert the audio to bytes
            buffer = io.BytesIO()
            sf.write(buffer, combined_audio, 24000, format='mp3')
            buffer.seek(0)
            return buffer.read()
            
        except Exception as e:
            logger.error(f"Error generating audio with Kokoro: {str(e)}")
            raise
    
    def _generate_multi_speaker_audio(self, text: str, voice1: str, voice2: str, 
                                      ending_message: str) -> bytes:
        """
        Generate audio with alternating speakers.
        
        Args:
            text: The text to convert
            voice1: Voice for first speaker
            voice2: Voice for second speaker
            ending_message: Optional ending message
            
        Returns:
            Audio data as bytes
        """
        all_audio = []
        
        # Split text into chunks for different speakers
        qa_pairs = self.split_qa(text, ending_message, self.get_supported_tags())
        
        for question, answer in qa_pairs:
            # Process question with voice1
            if question:
                generator = self.pipeline(question, voice=voice1, speed=1.0)
                for _, _, audio in generator:
                    if audio is not None:
                        all_audio.append(torch.tensor(audio))
            
            # Process answer with voice2
            if answer:
                generator = self.pipeline(answer, voice=voice2, speed=1.0)
                for _, _, audio in generator:
                    if audio is not None:
                        all_audio.append(torch.tensor(audio))
        
        # Add ending message if provided
        if ending_message:
            generator = self.pipeline(ending_message, voice=voice1, speed=1.0)
            for _, _, audio in generator:
                if audio is not None:
                    all_audio.append(torch.tensor(audio))
        
        # Combine all audio segments
        if not all_audio:
            raise ValueError("No audio was generated in multi-speaker mode")
            
        combined_audio = torch.cat(all_audio, dim=0).numpy()
        
        # Convert the audio to bytes
        buffer = io.BytesIO()
        sf.write(buffer, combined_audio, 24000, format='mp3')
        buffer.seek(0)
        return buffer.read()
    
    def split_qa(self, text: str, ending_message: str, supported_tags: List[str]) -> List[Tuple[str, str]]:
        """
        Split text into question and answer pairs based on tags.
        
        Args:
            text: The text to split
            ending_message: Ending message to append
            supported_tags: List of supported tags
            
        Returns:
            List of (question, answer) tuples
        """
        qa_pairs = []
        
        # Extract Person1 and Person2 sections
        person1_pattern = r"<Person1>(.*?)</Person1>"
        person2_pattern = r"<Person2>(.*?)</Person2>"
        
        person1_sections = re.findall(person1_pattern, text, re.DOTALL)
        person2_sections = re.findall(person2_pattern, text, re.DOTALL)
        
        # Match questions with answers
        for i in range(max(len(person1_sections), len(person2_sections))):
            question = person1_sections[i].strip() if i < len(person1_sections) else ""
            answer = person2_sections[i].strip() if i < len(person2_sections) else ""
            qa_pairs.append((question, answer))
        
        # If text doesn't have tags, treat the whole text as one answer
        if not qa_pairs:
            qa_pairs = [("", text.strip())]
            
        return qa_pairs

    def clean_text(self, text: str) -> str:
        """
        Clean and prepare text for TTS processing.
        
        Args:
            text: The text to clean
        
        Returns:
            Cleaned text ready for TTS
        """
        if not text:
            return ""
        
        # Remove HTML-like tags if present and not Person tags
        text = re.sub(r'<(?!/?Person[12]>).*?>', '', text)
        
        # Basic text cleaning
        text = text.strip()
        
        return text


if __name__ == "__main__":
    import os
    import logging
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Sample conversation with 6 short sentences in the specified format
    test_text = """
    <Person1>Hello, how are you?</Person1>
    <Person2>Fine, thank you. How about you?</Person2>
    <Person1>I'm doing well. What are you working on today?</Person1>
    <Person2>I'm analyzing some data for my research project.</Person2>
    <Person1>That sounds interesting! What kind of research?</Person1>
    <Person2>It's about natural language processing and text-to-speech systems.</Person2>
    """
    print(test_text)
    try:
        # Initialize the Kokoro TTS provider
        tts = KokoroTTS(model="kokoro")
        
        # Set output path
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "kokoro_test_conversation.mp3")
        
        # Generate and save audio
        logger.info(f"Generating audio for test conversation...")
        audio_data = tts.generate_audio(
            text=test_text,
            voice="af_heart",  # Voice for Person1
            voice2="af_bella",  # Voice for Person2
        )
        
        # Save audio to file
        with open(output_file, "wb") as f:
            f.write(audio_data)
        
        logger.info(f"Audio saved to {output_file}")
        
        # Additional info about the process
        logger.info(f"Used language code: {tts.lang_code}")
        logger.info("Test completed successfully")
        
    except Exception as e:
        logger.error(f"Error in KokoroTTS test: {str(e)}")
        raise