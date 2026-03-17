"""
Safety Guardrails Service

Implements LLM-based semantic classification for emergency detection.
Replaces keyword matching with intelligent semantic analysis.
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional
import ollama

# Configure logging
logger = logging.getLogger(__name__)


class SafetyGuardrails:
    """
    LLM-based safety guardrails for medical triage.
    
    Uses semantic classification to detect emergency situations,
    avoiding false positives/negatives from keyword matching.
    """
    
    # Severity levels
    SEVERITY_CRITICAL = "CRITICAL"
    SEVERITY_HIGH = "HIGH"
    SEVERITY_MODERATE = "MODERATE"
    SEVERITY_LOW = "LOW"
    
    # Recommended actions
    ACTION_EMERGENCY = "EMERGENCY_SERVICES"
    ACTION_URGENT = "URGENT_CARE"
    ACTION_ROUTINE = "ROUTINE_APPOINTMENT"
    ACTION_SELF_CARE = "SELF_CARE"
    
    def __init__(self, model: str = "gemini-3-flash-preview"):
        """
        Initialize the safety guardrails.
        
        Args:
            model: LLM model to use for classification
        """
        self.model = model
    
    def detect_emergency(self, text: str) -> Dict[str, Any]:
        """
        Use LLM to semantically detect emergency situations.
        
        Args:
            text: User input text
            
        Returns:
            Dict with keys:
                - is_emergency: bool
                - severity: str (CRITICAL, HIGH, MODERATE, LOW)
                - recommended_action: str
                - reasoning: str
        """
        prompt = f"""
You are a medical safety classifier. Analyze the following user input and determine if it describes an emergency situation.

USER INPUT: "{text}"

CLASSIFICATION RULES:
1. CRITICAL - Life-threatening, requires immediate emergency services (911, 999, 112)
   Examples: chest pain with shortness of breath, suspected stroke symptoms, severe bleeding, 
   overdose, suicidal ideation with intent, can't breathe, severe allergic reaction
   
2. HIGH - Urgent medical situation requiring care within hours
   Examples: high fever with confusion, severe pain, moderate bleeding, suspected fracture
   
3. MODERATE - Should see a doctor within 24-48 hours
   Examples: persistent symptoms, worsening condition, new concerning symptoms
   
4. LOW - Can be addressed with routine appointment or self-care
   Examples: mild symptoms, general questions, follow-up inquiries

OUTPUT FORMAT (JSON only, no markdown):
{{
    "is_emergency": true/false,
    "severity": "CRITICAL/HIGH/MODERATE/LOW",
    "recommended_action": "EMERGENCY_SERVICES/URGENT_CARE/ROUTINE_APPOINTMENT/SELF_CARE",
    "reasoning": "Brief explanation of classification"
}}

Return ONLY valid JSON.
"""
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            raw_content = response["message"]["content"]
            # Clean thinking tags if present
            cleaned = re.sub(r'<[^>]+>', '', raw_content)
            cleaned = cleaned.strip()
            
            # Try to parse JSON
            try:
                result = json.loads(cleaned)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                match = re.search(r'\{.*\}', cleaned, re.DOTALL)
                if match:
                    result = json.loads(match.group(0))
                else:
                    # Fallback to safe defaults
                    logger.warning(f"Failed to parse LLM response: {cleaned}")
                    return self._fallback_detection(text)
            
            # Validate and normalize result
            return self._normalize_result(result)
            
        except Exception as e:
            logger.error(f"Error in emergency detection: {e}")
            # Fallback to keyword-based detection on error
            return self._fallback_detection(text)
    
    def _fallback_detection(self, text: str) -> Dict[str, Any]:
        """
        Fallback keyword-based detection if LLM fails.
        """
        text_lower = text.lower()
        
        # Critical emergency keywords
        critical_keywords = [
            "suicide", "suicidal", "kill myself", "end my life",
            "can't breathe", "chest pain", "heart attack", "stroke",
            "bleeding won't stop", "severe burn", "overdose", "poisoning",
            "unconscious", "not breathing"
        ]
        
        for keyword in critical_keywords:
            if keyword in text_lower:
                return {
                    "is_emergency": True,
                    "severity": self.SEVERITY_CRITICAL,
                    "recommended_action": self.ACTION_EMERGENCY,
                    "reasoning": f"Fallback: Critical keyword detected: {keyword}"
                }
        
        # High severity keywords
        high_keywords = [
            "severe pain", "high fever", "can't move", "broken bone",
            "seizure", "confusion", "severe headache"
        ]
        
        for keyword in high_keywords:
            if keyword in text_lower:
                return {
                    "is_emergency": True,
                    "severity": self.SEVERITY_HIGH,
                    "recommended_action": self.ACTION_URGENT,
                    "reasoning": f"Fallback: High severity keyword detected: {keyword}"
                }
        
        return {
            "is_emergency": False,
            "severity": self.SEVERITY_LOW,
            "recommended_action": self.ACTION_SELF_CARE,
            "reasoning": "Fallback: No emergency keywords detected"
        }
    
    def _normalize_result(self, result: Dict) -> Dict[str, Any]:
        """Normalize and validate the classification result."""
        is_emergency = result.get("is_emergency", False)
        
        # Normalize severity
        severity = result.get("severity", self.SEVERITY_LOW).upper()
        if severity not in [self.SEVERITY_CRITICAL, self.SEVERITY_HIGH, 
                          self.SEVERITY_MODERATE, self.SEVERITY_LOW]:
            severity = self.SEVERITY_LOW
        
        # Normalize action
        action = result.get("recommended_action", self.ACTION_SELF_CARE).upper()
        valid_actions = [self.ACTION_EMERGENCY, self.ACTION_URGENT, 
                        self.ACTION_ROUTINE, self.ACTION_SELF_CARE]
        if action not in valid_actions:
            action = self.ACTION_SELF_CARE
        
        # Override if critical
        if severity == self.SEVERITY_CRITICAL:
            is_emergency = True
            action = self.ACTION_EMERGENCY
        
        return {
            "is_emergency": is_emergency,
            "severity": severity,
            "recommended_action": action,
            "reasoning": result.get("reasoning", "No reasoning provided")
        }
    
    def get_emergency_response(self) -> str:
        """
        Get the emergency response message with contact information.
        
        Returns:
            Formatted emergency response string
        """
        return """
🚨 IMMEDIATE ASSISTANCE NEEDED 🚨

If you or someone you know is in immediate danger, please contact:

📞 EMERGENCY SERVICES: 911 (US) / 999 (UK) / 112 (EU)
📞 NATIONAL SUICIDE PREVENTION: 988 (US)

For immediate support:
📱 Crisis Text Line: Text HOME to 741741
📞 SAMHSA National Helpline: 1-800-662-4357

⚠️ DISCLAIMER: This is an AI assistant and cannot provide emergency medical care. 
Please seek immediate human assistance for any life-threatening condition.
"""
    
    def should_bypass_triage(self, detection_result: Dict[str, Any]) -> bool:
        """
        Determine if the triage flow should be bypassed based on detection.
        
        Args:
            detection_result: Result from detect_emergency()
            
        Returns:
            True if triage should be bypassed
        """
        return detection_result.get("is_emergency", False)


# Singleton instance
_safety_guardrails = None

def get_safety_guardrails() -> SafetyGuardrails:
    """Get or create the singleton SafetyGuardrails instance."""
    global _safety_guardrails
    if _safety_guardrails is None:
        _safety_guardrails = SafetyGuardrails()
    return _safety_guardrails
