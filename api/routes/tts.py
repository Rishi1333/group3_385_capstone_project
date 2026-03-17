"""
Text-to-Speech API Routes

Provides endpoints for speech synthesis using Kokoro-82M.
"""

import logging
from flask import Blueprint, request, jsonify, send_file

from services.tts_service import (
    get_tts_service, 
    is_tts_available, 
    KOKORO_VOICES,
    DEFAULT_VOICE,
    DEFAULT_SPEED
)

logger = logging.getLogger(__name__)

# Create blueprint
tts_bp = Blueprint('tts', __name__, url_prefix='/tts')


@tts_bp.route('/status', methods=['GET'])
def get_status():
    """
    Get TTS service status and available voices.
    
    Returns:
        JSON with service status and voice options
    """
    service = get_tts_service()
    
    return jsonify({
        'available': service.available,
        'voices': [
            {
                'id': voice_id,
                'name': info['name'],
                'gender': info['gender'],
                'locale': info['locale']
            }
            for voice_id, info in KOKORO_VOICES.items()
        ],
        'default_voice': DEFAULT_VOICE,
        'default_speed': DEFAULT_SPEED
    })


@tts_bp.route('/speak', methods=['POST'])
def speak():
    """
    Synthesize speech from text.
    
    Request Body:
        {
            "text": "Text to speak",
            "voice": "af_bella" (optional),
            "speed": 0.9 (optional)
        }
    
    Returns:
        WAV audio file or error response
    """
    # Check if TTS is available
    if not is_tts_available():
        return jsonify({
            'error': 'TTS service unavailable',
            'message': 'Kokoro model not loaded. Please check server configuration.',
            'fallback': True
        }), 503
    
    # Parse request
    data = request.get_json()
    
    if not data:
        return jsonify({
            'error': 'Invalid request',
            'message': 'Request body must be JSON'
        }), 400
    
    text = data.get('text', '').strip()
    
    if not text:
        return jsonify({
            'error': 'Invalid input',
            'message': 'Text field is required and cannot be empty'
        }), 400
    
    voice = data.get('voice', DEFAULT_VOICE)
    speed = data.get('speed', DEFAULT_SPEED)
    
    # Validate voice
    if voice not in KOKORO_VOICES:
        logger.warning(f"Invalid voice '{voice}', using default")
        voice = DEFAULT_VOICE
    
    # Validate speed
    try:
        speed = float(speed)
        if not 0.1 <= speed <= 2.0:
            logger.warning(f"Speed {speed} out of range, using default")
            speed = DEFAULT_SPEED
    except (TypeError, ValueError):
        speed = DEFAULT_SPEED
    
    logger.info(f"TTS request: text_len={len(text)}, voice={voice}, speed={speed}")
    
    # Get TTS service and synthesize
    service = get_tts_service()
    wav_buffer = service.synthesize_to_wav(text, voice, speed)
    
    if wav_buffer is None:
        return jsonify({
            'error': 'Synthesis failed',
            'message': 'Failed to generate speech audio'
        }), 500
    
    # Return audio file
    return send_file(
        wav_buffer,
        mimetype='audio/wav',
        as_attachment=False,
        download_name='speech.wav'
    )


@tts_bp.route('/voices', methods=['GET'])
def get_voices():
    """
    Get list of available voices.
    
    Returns:
        JSON array of available voices
    """
    return jsonify([
        {
            'id': voice_id,
            'name': info['name'],
            'gender': info['gender'],
            'locale': info['locale']
        }
        for voice_id, info in KOKORO_VOICES.items()
    ])


def create_tts_blueprint():
    """
    Factory function to create TTS blueprint.
    
    Returns:
        Flask Blueprint for TTS routes
    """
    return tts_bp
