import re
import spacy
import logging
from typing import Dict, List, Optional, Tuple, Any

# Configure logging
logger = logging.getLogger(__name__)


class ModelRouter:
    """
    Routes user input to the appropriate prediction model based on content analysis.
    Uses keyword matching and NLP to determine the most relevant model.
    
    Supported models:
    - heart: Heart disease prediction
    - diabetes: Diabetes prediction
    - mental_health: Mental health screening
    - symptom: General symptom-based disease prediction
    
    Supports RAG fallback when models are unavailable.
    """
    
    # Keywords that strongly indicate heart-related concerns
    HEART_KEYWORDS = [
        "heart", "cardiac", "chest pain", "angina", "heart attack", "heart disease",
        "palpitation", "heart rate", "blood pressure", "cholesterol", "ecg", "ekg",
        "shortness of breath", "arrhythmia", "cardiovascular", "coronary", "stent",
        "bypass", "heart failure", "atrial", "ventricular", "tachycardia", "bradycardia"
    ]
    
    # Keywords that indicate diabetes-related concerns
    DIABETES_KEYWORDS = [
        "diabetes", "blood sugar", "glucose", "insulin", "diabetic", "prediabetes",
        "a1c", "hemoglobin a1c", "hyperglycemia", "hypoglycemia", "polyuria",
        "polydipsia", "increased thirst", "frequent urination", "gestational diabetes",
        "type 1", "type 2", "sugar level", "fasting glucose", "bmi", "pregnancies"
    ]
    
    # Keywords that indicate mental health concerns
    MENTAL_HEALTH_KEYWORDS = [
        "depression", "anxiety", "mental health", "stress", "panic", "therapy",
        "counseling", "psychologist", "psychiatrist", "antidepressant", "mood",
        "bipolar", "ptsd", "trauma", "suicide", "self-harm", "mental illness",
        "emotional", "anxious", "depressed", "overwhelmed", "burnout", "work stress",
        "family history mental", "treatment", "wellness", "therapy options"
    ]
    
    # Keywords that indicate general symptom concerns
    SYMPTOM_KEYWORDS = [
        "fever", "cough", "headache", "nausea", "vomiting", "fatigue", "tired",
        "sore throat", "chills", "body pain", "stomach", "diarrhea", "flu",
        "cold", "infection", "allergy", "rash", "dizziness", "sweating"
    ]
    
    # Heart-specific feature indicators (from heart model features)
    HEART_FEATURE_INDICATORS = [
        "age", "sex", "chest pain", "blood pressure", "cholesterol", "fasting blood sugar",
        "ecg", "heart rate", "exercise", "angina", "st depression", "slope", 
        "vessels", "thalassemia", "thal"
    ]
    
    # Diabetes-specific feature indicators
    DIABETES_FEATURE_INDICATORS = [
        "pregnancies", "glucose", "blood pressure", "skin thickness", "insulin",
        "bmi", "diabetes pedigree", "age", "outcome"
    ]
    
    # Mental health feature indicators
    MENTAL_HEALTH_FEATURE_INDICATORS = [
        "family history", "work interfere", "employees", "remote work", "tech company",
        "benefits", "care options", "wellness program", "seek help", "anonymity",
        "leave", "mental health consequence", "coworkers", "supervisor"
    ]

    # Per-model confidence thresholds
    # Lower thresholds for critical conditions (heart, mental health)
    # Standard threshold for diabetes
    # Symptom is the default fallback
    MODEL_THRESHOLDS = {
        "heart": 0.50,           # Lower threshold - critical condition
        "mental_health": 0.50,   # Lower threshold - early intervention important
        "diabetes": 0.70,        # Standard threshold
        "symptom": 0.30          # Very low - acts as fallback
    }

    def __init__(self, threshold: float = 0.7, rag_service=None):
        """
        Initialize the model router.
        
        Args:
            threshold: Default minimum confidence threshold for model selection
            rag_service: Optional RAG service for fallback predictions
        """
        self.threshold = threshold
        self.rag_service = rag_service
        try:
            self.nlp = spacy.load("en_core_web_md")
        except OSError:
            # Fallback to smaller model if medium not available
            self.nlp = spacy.load("en_core_web_sm")

    def preprocess(self, text: str) -> str:
        """Clean and normalize input text."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return text

    def _calculate_keyword_score(self, text: str, keywords: List[str]) -> Tuple[int, List[str]]:
        """
        Calculate how many keywords match the text.
        Returns (score, matched_keywords).
        """
        text_lower = text.lower()
        matched = []
        
        for keyword in keywords:
            if keyword.lower() in text_lower:
                matched.append(keyword)
        
        return len(matched), matched

    def _calculate_similarity_score(self, text: str, keywords: List[str]) -> float:
        """
        Calculate semantic similarity score between text and keywords.
        Uses spaCy word vectors for semantic matching.
        """
        doc = self.nlp(text)
        if not doc.vector.any():
            return 0.0
        
        max_sim = 0.0
        for keyword in keywords:
            keyword_doc = self.nlp(keyword)
            if keyword_doc.vector.any():
                sim = doc.similarity(keyword_doc)
                max_sim = max(max_sim, sim)
        
        return max_sim

    def analyze_input(self, text: str) -> Dict:
        """
        Analyze input text and return routing decision with confidence.
        
        Returns:
            {
                "model_type": "heart" | "diabetes" | "mental_health" | "symptom" | "unknown",
                "confidence": float,
                "scores": dict,
                "matched_keywords": dict
            }
        """
        text = self.preprocess(text)
        
        # Calculate keyword-based scores for each model type
        heart_kw_score, heart_matched = self._calculate_keyword_score(
            text, self.HEART_KEYWORDS + self.HEART_FEATURE_INDICATORS
        )
        diabetes_kw_score, diabetes_matched = self._calculate_keyword_score(
            text, self.DIABETES_KEYWORDS + self.DIABETES_FEATURE_INDICATORS
        )
        mental_health_kw_score, mental_health_matched = self._calculate_keyword_score(
            text, self.MENTAL_HEALTH_KEYWORDS + self.MENTAL_HEALTH_FEATURE_INDICATORS
        )
        symptom_kw_score, symptom_matched = self._calculate_keyword_score(
            text, self.SYMPTOM_KEYWORDS
        )
        
        # Calculate semantic similarity scores
        heart_sim_score = self._calculate_similarity_score(text, self.HEART_KEYWORDS)
        diabetes_sim_score = self._calculate_similarity_score(text, self.DIABETES_KEYWORDS)
        mental_health_sim_score = self._calculate_similarity_score(text, self.MENTAL_HEALTH_KEYWORDS)
        symptom_sim_score = self._calculate_similarity_score(text, self.SYMPTOM_KEYWORDS)
        
        # Combine scores (weighted average)
        # Keyword matches are more reliable, so weight them higher
        heart_score = (heart_kw_score * 0.7) + (heart_sim_score * 0.3)
        diabetes_score = (diabetes_kw_score * 0.7) + (diabetes_sim_score * 0.3)
        mental_health_score = (mental_health_kw_score * 0.7) + (mental_health_sim_score * 0.3)
        symptom_score = (symptom_kw_score * 0.7) + (symptom_sim_score * 0.3)
        
        # Store all scores
        scores = {
            "heart": heart_score,
            "diabetes": diabetes_score,
            "mental_health": mental_health_score,
            "symptom": symptom_score
        }
        
        # Normalize scores
        total = heart_score + diabetes_score + mental_health_score + symptom_score
        if total > 0:
            heart_confidence = heart_score / total
            diabetes_confidence = diabetes_score / total
            mental_health_confidence = mental_health_score / total
            symptom_confidence = symptom_score / total
        else:
            # Default to symptom model if no clear indicators
            heart_confidence = 0.1
            diabetes_confidence = 0.1
            mental_health_confidence = 0.1
            symptom_confidence = 0.7
        
        # Find the model with highest confidence
        confidences = {
            "heart": heart_confidence,
            "diabetes": diabetes_confidence,
            "mental_health": mental_health_confidence,
            "symptom": symptom_confidence
        }
        
        best_model = max(confidences, key=confidences.get)
        best_confidence = confidences[best_model]
        
        # Apply per-model thresholds
        # Check if the best model meets its threshold requirement
        model_threshold = self.MODEL_THRESHOLDS.get(best_model, self.threshold)
        
        # If best model doesn't meet its threshold, check other models in priority order
        # Priority: heart > mental_health > diabetes > symptom
        if best_confidence < model_threshold:
            # Check critical models first with their lower thresholds
            for model in ["heart", "mental_health", "diabetes"]:
                if confidences[model] >= self.MODEL_THRESHOLDS.get(model, self.threshold):
                    best_model = model
                    best_confidence = confidences[model]
                    break
            else:
                # Default to symptom as fallback
                best_model = "symptom"
                best_confidence = symptom_confidence
        
        return {
            "model_type": best_model,
            "confidence": best_confidence,
            "scores": scores,
            "matched_keywords": {
                "heart": heart_matched,
                "diabetes": diabetes_matched,
                "mental_health": mental_health_matched,
                "symptom": symptom_matched
            }
        }

    def route(self, text: str) -> str:
        """
        Determine which model should handle the input.
        
        Returns:
            Model type string: "heart", "diabetes", "mental_health", or "symptom"
        """
        analysis = self.analyze_input(text)
        return analysis["model_type"]

    def get_routing_explanation(self, text: str) -> str:
        """
        Get a human-readable explanation of the routing decision.
        """
        analysis = self.analyze_input(text)
        
        explanation = f"Routed to {analysis['model_type']} model (confidence: {analysis['confidence']:.2%}). "
        
        matched = analysis.get("matched_keywords", {})
        
        if matched.get("heart"):
            explanation += f"Heart-related terms: {', '.join(matched['heart'][:3])}. "
        
        if matched.get("diabetes"):
            explanation += f"Diabetes-related terms: {', '.join(matched['diabetes'][:3])}. "
        
        if matched.get("mental_health"):
            explanation += f"Mental health terms: {', '.join(matched['mental_health'][:3])}. "
        
        if matched.get("symptom"):
            explanation += f"Symptom-related terms: {', '.join(matched['symptom'][:3])}."
        
        return explanation

    def route_with_fallback(
        self,
        text: str,
        model_availability: Dict[str, bool]
    ) -> Dict[str, Any]:
        """
        Route input to appropriate handler with RAG fallback support.
        
        Args:
            text: User input text
            model_availability: Dictionary mapping model types to availability status
            
        Returns:
            {
                "handler": "model" | "rag",
                "model_type": str,
                "confidence": float,
                "fallback_reason": str | None,
                "rag_available": bool
            }
        """
        analysis = self.analyze_input(text)
        model_type = analysis["model_type"]
        confidence = analysis["confidence"]
        
        # Check if the preferred model is available
        if model_availability.get(model_type, False):
            return {
                "handler": "model",
                "model_type": model_type,
                "confidence": confidence,
                "fallback_reason": None,
                "rag_available": self.rag_service is not None and self.rag_service.is_available(),
                "analysis": analysis
            }
        
        # Model not available - check if RAG is available
        rag_available = self.rag_service is not None and self.rag_service.is_available()
        
        if rag_available:
            logger.info(
                f"Model '{model_type}' not available. Falling back to RAG for: {text[:50]}..."
            )
            return {
                "handler": "rag",
                "model_type": model_type,
                "confidence": confidence,
                "fallback_reason": f"Model '{model_type}' not available",
                "rag_available": True,
                "analysis": analysis
            }
        
        # No fallback available - try to find an available model
        for alt_model in ["symptom", "heart", "diabetes", "mental_health"]:
            if model_availability.get(alt_model, False):
                logger.info(
                    f"Preferred model '{model_type}' not available. "
                    f"Using alternative model '{alt_model}'."
                )
                return {
                    "handler": "model",
                    "model_type": alt_model,
                    "confidence": analysis["scores"].get(alt_model, 0),
                    "fallback_reason": f"Preferred model '{model_type}' not available, using alternative",
                    "rag_available": False,
                    "analysis": analysis
                }
        
        # No models available and no RAG
        return {
            "handler": "none",
            "model_type": model_type,
            "confidence": 0,
            "fallback_reason": "No prediction models or RAG available",
            "rag_available": False,
            "analysis": analysis
        }

    def set_rag_service(self, rag_service):
        """
        Set the RAG service for fallback predictions.
        
        Args:
            rag_service: RAGService instance
        """
        self.rag_service = rag_service
        logger.info("RAG service configured for model router")

    def get_routing_info(self) -> Dict[str, Any]:
        """
        Get information about the router configuration.
        
        Returns:
            Dictionary with router configuration details
        """
        return {
            "threshold": self.threshold,
            "model_thresholds": self.MODEL_THRESHOLDS,
            "rag_service_available": self.rag_service is not None,
            "rag_ready": self.rag_service.is_available() if self.rag_service else False,
            "supported_models": ["heart", "diabetes", "mental_health", "symptom"]
        }
