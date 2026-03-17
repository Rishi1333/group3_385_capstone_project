"""
Text-to-Speech Service using Kokoro-82M

This module provides TTS functionality using the Kokoro-82M model
for natural-sounding speech synthesis.

Repository: https://github.com/hexgrad/kokoro
"""

import io
import logging
from typing import Optional, Tuple
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# Available Kokoro voices
KOKORO_VOICES = {
    # American English
    'af_bella': {'name': 'Bella', 'gender': 'female', 'locale': 'en-US'},
    'af_sarah': {'name': 'Sarah', 'gender': 'female', 'locale': 'en-US'},
    'am_adam': {'name': 'Adam', 'gender': 'male', 'locale': 'en-US'},
    'am_michael': {'name': 'Michael', 'gender': 'male', 'locale': 'en-US'},
    # British English
    'bf_emma': {'name': 'Emma', 'gender': 'female', 'locale': 'en-GB'},
    'bm_george': {'name': 'George', 'gender': 'male', 'locale': 'en-GB'},
}

DEFAULT_VOICE = 'af_bella'
DEFAULT_SPEED = 0.9
SAMPLE_RATE = 24000


class TTSService:
    """
    Text-to-Speech service using Kokoro-82M model.
    
    Provides natural-sounding speech synthesis with multiple voice options.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern to ensure only one model instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the TTS service with Kokoro model."""
        if TTSService._initialized:
            return
            
        self.model = None
        self.pipeline = None
        self._available = False
        
        try:
            self._load_model()
            TTSService._initialized = True
        except Exception as e:
            logger.warning(f"TTS service initialization failed: {e}")
            logger.info("TTS will use fallback mode (Web Speech Synthesis)")
    
    def _load_model(self):
        """Load the Kokoro-82M model."""
        try:
            from kokoro import KPipeline
            
            logger.info("Loading Kokoro-82M model...")
            # KPipeline with model=True will automatically load the model
            # lang_code='a' for American English
            self.pipeline = KPipeline(lang_code='a', model=True, device='cpu')
            self._available = True
            logger.info("Kokoro-82M model loaded successfully")
        except ImportError:
            logger.warning("Kokoro not installed. Install with: pip install kokoro")
            raise
        except Exception as e:
            logger.error(f"Failed to load Kokoro model: {e}")
            raise
    
    @property
    def available(self) -> bool:
        """Check if TTS service is available."""
        return self._available
    
    def get_available_voices(self) -> dict:
        """
        Get dictionary of available voices.
        
        Returns:
            dict: Voice ID to voice info mapping
        """
        return KOKORO_VOICES.copy()
    
    def synthesize(
        self,
        text: str,
        voice: str = DEFAULT_VOICE,
        speed: float = DEFAULT_SPEED
    ) -> Tuple[Optional[np.ndarray], int]:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            voice: Voice ID (default: af_bella)
            speed: Speech speed multiplier (default: 0.9)
            
        Returns:
            Tuple of (audio_array, sample_rate) or (None, 0) if failed
        """
        if not self._available:
            logger.warning("TTS service not available")
            return None, 0
        
        if voice not in KOKORO_VOICES:
            logger.warning(f"Unknown voice '{voice}', using default")
            voice = DEFAULT_VOICE
        
        try:
            # Truncate very long text to prevent timeout
            max_chars = 2000
            if len(text) > max_chars:
                logger.info(f"Truncating text from {len(text)} to {max_chars} chars")
                text = text[:max_chars] + "..."
            
            # Generate speech using the pipeline callable
            # The pipeline returns a generator of Result objects
            audio_segments = []
            for result in self.pipeline(text, voice=voice, speed=speed):
                if result.audio is not None:
                    audio_segments.append(result.audio)
            
            if not audio_segments:
                logger.warning("No audio segments generated")
                return None, 0
            
            # Concatenate all audio segments
            audio = np.concatenate(audio_segments)
            return audio, SAMPLE_RATE
            
        except Exception as e:
            logger.error(f"Speech synthesis failed: {e}")
            return None, 0
    
    def synthesize_to_wav(
        self, 
        text: str, 
        voice: str = DEFAULT_VOICE,
        speed: float = DEFAULT_SPEED
    ) -> Optional[io.BytesIO]:
        """
        Synthesize speech and return as WAV file bytes.
        
        Args:
            text: Text to synthesize
            voice: Voice ID (default: af_bella)
            speed: Speech speed multiplier (default: 0.9)
            
        Returns:
            BytesIO buffer containing WAV audio, or None if failed
        """
        audio, sample_rate = self.synthesize(text, voice, speed)
        
        if audio is None:
            return None
        
        try:
            import soundfile as sf
            
            # Create in-memory WAV file
            buffer = io.BytesIO()
            sf.write(buffer, audio, sample_rate, format='WAV')
            buffer.seek(0)
            
            return buffer
            
        except ImportError:
            logger.error("soundfile not installed. Install with: pip install soundfile")
            return None
        except Exception as e:
            logger.error(f"Failed to create WAV buffer: {e}")
            return None


# Global TTS service instance
_tts_service: Optional[TTSService] = None


def get_tts_service() -> TTSService:
    """
    Get or create the global TTS service instance.
    
    Returns:
        TTSService instance
    """
    global _tts_service
    
    if _tts_service is None:
        _tts_service = TTSService()
    
    return _tts_service


def is_tts_available() -> bool:
    """
    Check if TTS service is available.
    
    Returns:
        True if TTS is ready to use
    """
    service = get_tts_service()
    return service.available
