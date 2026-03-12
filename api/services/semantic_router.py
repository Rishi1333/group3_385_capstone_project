"""
Semantic Router Service

Implements LLM-based intent classification for routing user inputs
to the appropriate handler (emergency, triage, image analysis, or RAG).
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional
import ollama

# Configure logging
logger = logging.getLogger(__name__)


# Intent categories
class Intent:
    EMERGENCY = "EMERGENCY"
    SYMPTOM_TRIAGE = "SYMPTOM_TRIAGE"
    IMAGE_ANALYSIS = "IMAGE_ANALYSIS"
    GENERAL_QUERY = "GENERAL_QUERY"
    
    ALL = [EMERGENCY, SYMPTOM_TRIAGE, IMAGE_ANALYSIS, GENERAL_QUERY]


class SemanticRouter:
    """
    LLM-based semantic router for medical triage.
    
    Uses intent classification to route user inputs to appropriate handlers
    instead of brittle keyword matching.
    """
    
    def __init__(self, model: str = "gemini-3-flash-preview"):
        """
        Initialize the semantic router.
        
        Args:
            model: LLM model to use for classification
        """
        self.model = model
    
    def classify_intent(self, text: str, has_image: bool = False) -> Dict[str, Any]:
        """
        Classify user intent using LLM.
        
        Args:
            text: User input text
            has_image: Whether an image was uploaded
            
        Returns:
            Dict with keys:
                - intent: str (EMERGENCY, SYMPTOM_TRIAGE, IMAGE_ANALYSIS, GENERAL_QUERY)
                - confidence: float (0-1)
                - reasoning: str
                - suggested_action: str
        """
        # If image is present, default to IMAGE_ANALYSIS
        if has_image:
            return {
                "intent": Intent.IMAGE_ANALYSIS,
                "confidence": 1.0,
                "reasoning": "Image upload detected, routing to image analysis",
                "suggested_action": "analyze_image"
            }
        
        prompt = f"""
You are a medical intent classifier. Analyze the following user input and classify their intent.

USER INPUT: "{text}"

INTENT CATEGORIES:

1. EMERGENCY - User describes life-threatening symptoms or needs immediate help
   Examples: "my chest hurts and I can't breathe", "I think I'm having a heart attack", 
   "I want to hurt myself", "severe bleeding"
   
2. SYMPTOM_TRIAGE - User describes symptoms and wants medical assessment
   Examples: "I've had a fever for 3 days", "my head hurts", "feeling tired all the time",
   "stomach pain after eating"
   
3. IMAGE_ANALYSIS - User uploads or describes a medical image
   Examples: "here's my X-ray", "can you look at this rash", "what's this mark on my skin"
   (This should be selected if the user mentions uploading an image)
   
4. GENERAL_QUERY - User asks general health questions or wants information
   Examples: "what is diabetes", "how do I lower blood pressure", "is coffee healthy"

OUTPUT FORMAT (JSON only, no markdown):
{{
    "intent": "EMERGENCY/SYMPTOM_TRIAGE/IMAGE_ANALYSIS/GENERAL_QUERY",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of why this intent was selected",
    "suggested_action": "emergency_services/triage/image_analysis/rag_query"
}}

Return ONLY valid JSON.
"""
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            raw_content = response["message"]["content"]
            # Clean any tags
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
                    # Default to symptom triage on parse failure
                    logger.warning(f"Failed to parse intent classification: {cleaned}")
                    return self._default_intent()
            
            return self._normalize_intent_result(result)
            
        except Exception as e:
            logger.error(f"Error in intent classification: {e}")
            # Default to symptom triage on error
            return self._default_intent()
    
    def _default_intent(self) -> Dict[str, Any]:
        """Default intent when classification fails."""
        return {
            "intent": Intent.SYMPTOM_TRIAGE,
            "confidence": 0.5,
            "reasoning": "Default fallback due to classification error",
            "suggested_action": "triage"
        }
    
    def _normalize_intent_result(self, result: Dict) -> Dict[str, Any]:
        """Normalize and validate the intent classification result."""
        intent = result.get("intent", Intent.SYMPTOM_TRIAGE).upper()
        
        # Validate intent
        if intent not in Intent.ALL:
            intent = Intent.SYMPTOM_TRIAGE
        
        # Normalize confidence
        try:
            confidence = float(result.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            confidence = 0.5
        
        # Map intent to action
        action_map = {
            Intent.EMERGENCY: "emergency_services",
            Intent.SYMPTOM_TRIAGE: "triage",
            Intent.IMAGE_ANALYSIS: "image_analysis",
            Intent.GENERAL_QUERY: "rag_query"
        }
        
        return {
            "intent": intent,
            "confidence": confidence,
            "reasoning": result.get("reasoning", "No reasoning provided"),
            "suggested_action": action_map.get(intent, "triage")
        }
    
    def route(self, text: str, has_image: bool = False) -> str:
        """
        Simple routing method returning just the intent.
        
        Args:
            text: User input text
            has_image: Whether an image was uploaded
            
        Returns:
            Intent string
        """
        result = self.classify_intent(text, has_image)
        return result["intent"]
    
    def should_start_intake(self, intent: str) -> bool:
        """
        Determine if the intent requires starting the intake flow.
        
        Args:
            intent: Classified intent
            
        Returns:
            True if intake should begin
        """
        return intent in [Intent.SYMPTOM_TRIAGE, Intent.IMAGE_ANALYSIS]
    
    def should_use_rag(self, intent: str) -> bool:
        """
        Determine if RAG should be used for this intent.
        
        Args:
            intent: Classified intent
            
        Returns:
            True if RAG should be used
        """
        return intent == Intent.GENERAL_QUERY


# Singleton instance
_semantic_router = None

def get_semantic_router() -> SemanticRouter:
    """Get or create the singleton SemanticRouter instance."""
    global _semantic_router
    if _semantic_router is None:
        _semantic_router = SemanticRouter()
    return _semantic_router
