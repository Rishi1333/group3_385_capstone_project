/**
 * Speech Service Module
 * 
 * Provides Speech-to-Text (STT) and Text-to-Speech (TTS) functionality
 * for the AI Virtual Clinic chatbot.
 * 
 * STT: Uses Web Speech API for browser-native speech recognition
 * TTS: Uses backend Kokoro-82M service with Web Speech Synthesis fallback
 */

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:3000';

// Default configuration
const DEFAULT_CONFIG = {
  stt: {
    language: 'en-US',
    continuous: false,
    interimResults: true,
    maxAlternatives: 1
  },
  tts: {
    voice: 'af_bella',
    speed: 0.9,
    useBackend: true // Use Kokoro backend by default
  }
};

// Available Kokoro voices (fetched from backend)
let availableVoices = [];
let ttsAvailable = false;

/**
 * Check if Web Speech API is supported
 */
export function isSpeechRecognitionSupported() {
  return 'SpeechRecognition' in window || 'webkitSpeechRecognition' in window;
}

/**
 * Check if Web Speech Synthesis is supported
 */
export function isSpeechSynthesisSupported() {
  return 'speechSynthesis' in window;
}

/**
 * Initialize TTS service and fetch available voices
 */
export async function initializeTTS() {
  try {
    const response = await fetch(`${API_BASE}/tts/status`);
    if (response.ok) {
      const data = await response.json();
      ttsAvailable = data.available;
      availableVoices = data.voices || [];
      console.log('TTS service initialized:', { available: ttsAvailable, voices: availableVoices.length });
      return data;
    }
  } catch (error) {
    console.warn('TTS backend not available, using fallback:', error.message);
    ttsAvailable = false;
  }
  return { available: false, voices: [] };
}

/**
 * Get available TTS voices
 */
export function getAvailableVoices() {
  return availableVoices;
}

/**
 * Check if backend TTS is available
 */
export function isBackendTTSAvailable() {
  return ttsAvailable;
}

/**
 * Speech-to-Text Class
 * Uses Web Speech API for browser-native speech recognition
 */
export class SpeechRecognition {
  constructor(options = {}) {
    if (!isSpeechRecognitionSupported()) {
      throw new Error('Speech recognition not supported in this browser');
    }

    const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
    this.recognition = new SpeechRecognitionAPI();
    
    // Configure recognition
    this.recognition.lang = options.language || DEFAULT_CONFIG.stt.language;
    this.recognition.continuous = options.continuous ?? DEFAULT_CONFIG.stt.continuous;
    this.recognition.interimResults = options.interimResults ?? DEFAULT_CONFIG.stt.interimResults;
    this.recognition.maxAlternatives = options.maxAlternatives || DEFAULT_CONFIG.stt.maxAlternatives;

    // State
    this.isListening = false;
    this.onResultCallback = null;
    this.onInterimCallback = null;
    this.onErrorCallback = null;
    this.onEndCallback = null;

    // Set up event handlers
    this.recognition.onresult = (event) => {
      let interimTranscript = '';
      let finalTranscript = '';

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalTranscript += transcript;
        } else {
          interimTranscript += transcript;
        }
      }

      if (interimTranscript && this.onInterimCallback) {
        this.onInterimCallback(interimTranscript);
      }

      if (finalTranscript && this.onResultCallback) {
        this.onResultCallback(finalTranscript.trim());
      }
    };

    this.recognition.onerror = (event) => {
      console.error('Speech recognition error:', event.error);
      this.isListening = false;
      if (this.onErrorCallback) {
        this.onErrorCallback(event.error);
      }
    };

    this.recognition.onend = () => {
      this.isListening = false;
      if (this.onEndCallback) {
        this.onEndCallback();
      }
    };
  }

  /**
   * Start listening for speech
   */
  start() {
    if (this.isListening) return;
    
    try {
      this.recognition.start();
      this.isListening = true;
    } catch (error) {
      console.error('Failed to start speech recognition:', error);
    }
  }

  /**
   * Stop listening
   */
  stop() {
    if (!this.isListening) return;
    
    try {
      this.recognition.stop();
      this.isListening = false;
    } catch (error) {
      console.error('Failed to stop speech recognition:', error);
    }
  }

  /**
   * Set callback for final result
   */
  onResult(callback) {
    this.onResultCallback = callback;
  }

  /**
   * Set callback for interim results
   */
  onInterim(callback) {
    this.onInterimCallback = callback;
  }

  /**
   * Set callback for errors
   */
  onError(callback) {
    this.onErrorCallback = callback;
  }

  /**
   * Set callback for end of recognition
   */
  onEnd(callback) {
    this.onEndCallback = callback;
  }
}

/**
 * Text-to-Speech Class
 * Uses backend Kokoro service with Web Speech Synthesis fallback
 */
export class TextToSpeech {
  constructor(options = {}) {
    this.voice = options.voice || DEFAULT_CONFIG.tts.voice;
    this.speed = options.speed || DEFAULT_CONFIG.tts.speed;
    this.useBackend = options.useBackend ?? DEFAULT_CONFIG.tts.useBackend;
    
    // State
    this.isPlaying = false;
    this.currentAudio = null;
    this.onEndCallback = null;
    this.onStartCallback = null;
    this.onErrorCallback = null;

    // Web Speech Synthesis fallback
    if (isSpeechSynthesisSupported()) {
      this.synthesis = window.speechSynthesis;
      this.synthesisVoices = [];
      
      // Load voices
      this.loadVoices();
      
      // Chrome requires this event
      if (speechSynthesis.onvoiceschanged !== undefined) {
        speechSynthesis.onvoiceschanged = () => this.loadVoices();
      }
    }
  }

  /**
   * Load Web Speech Synthesis voices
   */
  loadVoices() {
    if (this.synthesis) {
      this.synthesisVoices = this.synthesis.getVoices();
    }
  }

  /**
   * Speak text using TTS
   * @param {string} text - Text to speak
   * @param {object} options - Optional voice and speed overrides
   */
  async speak(text, options = {}) {
    const voice = options.voice || this.voice;
    const speed = options.speed || this.speed;

    // Stop any current speech
    this.stop();

    // Try backend TTS first
    if (this.useBackend && ttsAvailable) {
      try {
        await this.speakWithBackend(text, voice, speed);
        return;
      } catch (error) {
        console.warn('Backend TTS failed, using fallback:', error.message);
      }
    }

    // Fallback to Web Speech Synthesis
    if (isSpeechSynthesisSupported()) {
      this.speakWithSynthesis(text, voice, speed);
    } else {
      console.error('No TTS available');
      if (this.onErrorCallback) {
        this.onErrorCallback('TTS not available');
      }
    }
  }

  /**
   * Speak using backend Kokoro service
   */
  async speakWithBackend(text, voice, speed) {
    try {
      const response = await fetch(`${API_BASE}/tts/speak`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          // Include auth header if available
          ...(this.getAuthHeader() || {})
        },
        body: JSON.stringify({ text, voice, speed })
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'TTS request failed');
      }

      // Get audio blob
      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      
      // Create and play audio
      this.currentAudio = new Audio(audioUrl);
      this.isPlaying = true;

      return new Promise((resolve, reject) => {
        this.currentAudio.onended = () => {
          this.isPlaying = false;
          URL.revokeObjectURL(audioUrl);
          if (this.onEndCallback) this.onEndCallback();
          resolve();
        };

        this.currentAudio.onerror = (error) => {
          this.isPlaying = false;
          URL.revokeObjectURL(audioUrl);
          if (this.onErrorCallback) this.onErrorCallback(error);
          reject(error);
        };

        if (this.onStartCallback) this.onStartCallback();
        this.currentAudio.play();
      });
    } catch (error) {
      this.isPlaying = false;
      throw error;
    }
  }

  /**
   * Speak using Web Speech Synthesis (fallback)
   */
  speakWithSynthesis(text, voice, speed) {
    const utterance = new SpeechSynthesisUtterance(text);
    
    // Try to find a matching voice
    const voiceMatch = this.synthesisVoices.find(v => 
      v.lang.startsWith('en') && 
      (voice.includes('female') ? v.name.toLowerCase().includes('female') : true)
    );
    
    if (voiceMatch) {
      utterance.voice = voiceMatch;
    }
    
    utterance.rate = speed;
    utterance.pitch = 1.0;

    utterance.onstart = () => {
      this.isPlaying = true;
      if (this.onStartCallback) this.onStartCallback();
    };

    utterance.onend = () => {
      this.isPlaying = false;
      if (this.onEndCallback) this.onEndCallback();
    };

    utterance.onerror = (error) => {
      this.isPlaying = false;
      if (this.onErrorCallback) this.onErrorCallback(error);
    };

    this.synthesis.speak(utterance);
  }

  /**
   * Stop current speech
   */
  stop() {
    // Stop backend audio
    if (this.currentAudio) {
      this.currentAudio.pause();
      this.currentAudio.currentTime = 0;
      this.currentAudio = null;
    }

    // Stop Web Speech Synthesis
    if (this.synthesis) {
      this.synthesis.cancel();
    }

    this.isPlaying = false;
  }

  /**
   * Get auth header from localStorage
   */
  getAuthHeader() {
    const auth = localStorage.getItem('virtual_clinic_auth');
    if (auth) {
      try {
        const { token } = JSON.parse(auth);
        return { Authorization: `Bearer ${token}` };
      } catch {
        return null;
      }
    }
    return null;
  }

  /**
   * Set callback for speech start
   */
  onStart(callback) {
    this.onStartCallback = callback;
  }

  /**
   * Set callback for speech end
   */
  onEnd(callback) {
    this.onEndCallback = callback;
  }

  /**
   * Set callback for errors
   */
  onError(callback) {
    this.onErrorCallback = callback;
  }

  /**
   * Set voice
   */
  setVoice(voice) {
    this.voice = voice;
  }

  /**
   * Set speed
   */
  setSpeed(speed) {
    this.speed = speed;
  }
}

// Create singleton instances
let speechRecognitionInstance = null;
let textToSpeechInstance = null;

/**
 * Get or create SpeechRecognition instance
 */
export function getSpeechRecognition(options = {}) {
  if (!speechRecognitionInstance && isSpeechRecognitionSupported()) {
    speechRecognitionInstance = new SpeechRecognition(options);
  }
  return speechRecognitionInstance;
}

/**
 * Get or create TextToSpeech instance
 */
export function getTextToSpeech(options = {}) {
  if (!textToSpeechInstance) {
    textToSpeechInstance = new TextToSpeech(options);
  }
  return textToSpeechInstance;
}

export default {
  isSpeechRecognitionSupported,
  isSpeechSynthesisSupported,
  initializeTTS,
  getAvailableVoices,
  isBackendTTSAvailable,
  SpeechRecognition,
  TextToSpeech,
  getSpeechRecognition,
  getTextToSpeech
};
