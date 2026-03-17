/**
 * Voice Storage Service
 * 
 * Persists user voice preferences to localStorage.
 */

const STORAGE_KEY = 'virtual_clinic_voice_settings';

// Default settings
const DEFAULT_SETTINGS = {
  voice: 'af_bella',
  speed: 0.9,
  autoSpeak: false,        // Auto-speak bot responses
  showVoiceUI: true,       // Show voice controls
  speechRecognitionLang: 'en-US'
};

/**
 * Get voice settings from localStorage
 * @returns {object} Voice settings
 */
export function getVoiceSettings() {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      return { ...DEFAULT_SETTINGS, ...JSON.parse(stored) };
    }
  } catch (error) {
    console.warn('Failed to load voice settings:', error);
  }
  return { ...DEFAULT_SETTINGS };
}

/**
 * Save voice settings to localStorage
 * @param {object} settings - Settings to save
 */
export function saveVoiceSettings(settings) {
  try {
    const current = getVoiceSettings();
    const updated = { ...current, ...settings };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
    return updated;
  } catch (error) {
    console.warn('Failed to save voice settings:', error);
    return getVoiceSettings();
  }
}

/**
 * Reset voice settings to defaults
 */
export function resetVoiceSettings() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(DEFAULT_SETTINGS));
    return { ...DEFAULT_SETTINGS };
  } catch (error) {
    console.warn('Failed to reset voice settings:', error);
    return getVoiceSettings();
  }
}

/**
 * Update a single setting
 * @param {string} key - Setting key
 * @param {any} value - Setting value
 */
export function updateVoiceSetting(key, value) {
  return saveVoiceSettings({ [key]: value });
}

/**
 * Get a specific setting
 * @param {string} key - Setting key
 * @returns {any} Setting value
 */
export function getVoiceSetting(key) {
  const settings = getVoiceSettings();
  return settings[key];
}

/**
 * Check if voice features should be shown
 */
export function shouldShowVoiceUI() {
  return getVoiceSetting('showVoiceUI') ?? true;
}

/**
 * Check if auto-speak is enabled
 */
export function isAutoSpeakEnabled() {
  return getVoiceSetting('autoSpeak') ?? false;
}

export default {
  getVoiceSettings,
  saveVoiceSettings,
  resetVoiceSettings,
  updateVoiceSetting,
  getVoiceSetting,
  shouldShowVoiceUI,
  isAutoSpeakEnabled
};
