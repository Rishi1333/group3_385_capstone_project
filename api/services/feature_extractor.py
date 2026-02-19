import json
import re
import spacy
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extractors."""
    
    @abstractmethod
    def extract(self, text: str) -> dict:
        """Extract features from text."""
        pass
    
    @abstractmethod
    def get_missing_features(self, extracted: dict) -> list:
        """Return list of features that need to be collected."""
        pass


class SymptomFeatureExtractor(BaseFeatureExtractor):
    """
    Extracts symptom features from natural language text.
    Uses spaCy for NLP and keyword matching with similarity scoring.
    """
    
    def __init__(self, keyword_file: str, config: dict, threshold: float = 0.85):
        with open(keyword_file) as f:
            self.keyword_map = json.load(f)

        self.features = config["features"]
        self.nlp = spacy.load("en_core_web_md")
        self.threshold = threshold

    def preprocess(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return text

    def extract(self, text: str) -> dict:
        text = self.preprocess(text)
        doc = self.nlp(text)

        vector = {feature: 0 for feature in self.features}

        for feature in self.features:
            keywords = self.keyword_map.get(feature, [])
            for kw in keywords:
                kw_doc = self.nlp(kw.lower())

                # Check similarity with each sentence
                for sent in doc.sents:
                    if sent.similarity(kw_doc) >= self.threshold:
                        vector[feature] = 1
                        break
                if vector[feature] == 1:
                    break

                # Fallback: check token overlap (lemmas)
                tokens = [t.lemma_ for t in doc]
                kw_tokens = [t.lemma_ for t in kw_doc]
                if all(tok in tokens for tok in kw_tokens):
                    vector[feature] = 1
                    break

        return vector
    
    def get_missing_features(self, extracted: dict) -> list:
        """Return features that weren't detected (value = 0)."""
        return [f for f in self.features if extracted.get(f, 0) == 0]
    
    def to_model_vector(self, feature_dict: dict) -> list:
        """Convert feature dict to a list/vector for model input."""
        return [feature_dict[f] for f in self.features]


class HeartFeatureExtractor(BaseFeatureExtractor):
    """
    Extracts heart disease related features from natural language text.
    Uses spaCy for NLP and keyword matching with similarity scoring.
    """
    
    def __init__(self, keyword_file: str, config: dict, threshold: float = 0.75):
        # Use utf-8-sig to handle BOM (Byte Order Mark) in JSON files
        with open(keyword_file, encoding='utf-8-sig') as f:
            mapping_data = json.load(f)
        
        self.feature_aliases = mapping_data.get("feature_aliases", {})
        self.categorical_values = mapping_data.get("categorical_values", {})
        self.features = config["features"]
        self.numeric_features = config.get("numeric_features", [])
        self.categorical_features = config.get("categorical_features", [])
        self.categorical_encodings = config.get("categorical_encodings", {})
        
        self.nlp = spacy.load("en_core_web_md")
        self.threshold = threshold

    def preprocess(self, text: str) -> str:
        """Clean and normalize input text."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return text

    def _extract_numeric(self, text: str, feature: str) -> Optional[float]:
        """
        Extract numeric values for features like age, blood pressure, cholesterol.
        Looks for patterns like "age 45", "blood pressure 120", etc.
        """
        doc = self.nlp(text)
        aliases = self.feature_aliases.get(feature, [feature])
        
        for alias in aliases:
            alias_lower = alias.lower()
            for i, token in enumerate(doc):
                if alias_lower in token.text.lower() or token.text.lower() in alias_lower:
                    # Look for numeric value nearby
                    for j in range(max(0, i-3), min(len(doc), i+4)):
                        if doc[j].like_num:
                            try:
                                return float(doc[j].text)
                            except ValueError:
                                continue
        
        return None

    def _extract_categorical(self, text: str, feature: str) -> Optional[str]:
        """
        Extract categorical values for features like sex, cp (chest pain type), etc.
        Maps natural language to categorical values.
        """
        text_lower = text.lower()
        feature_values = self.categorical_values.get(feature, {})
        
        # Direct string matching
        for value, aliases in feature_values.items():
            for alias in aliases:
                if alias.lower() in text_lower:
                    return value
        
        # Try similarity matching for categorical values
        doc = self.nlp(text)
        for value, aliases in feature_values.items():
            for alias in aliases:
                alias_doc = self.nlp(alias.lower())
                for sent in doc.sents:
                    if sent.similarity(alias_doc) >= self.threshold:
                        return value
        
        return None

    def extract(self, text: str) -> dict:
        """
        Extract all heart disease features from text.
        Returns a dict with extracted values or defaults.
        """
        text = self.preprocess(text)
        
        result = {}
        
        for feature in self.features:
            if feature in self.numeric_features:
                # Try to extract numeric value
                val = self._extract_numeric(text, feature)
                result[feature] = val if val is not None else None
            elif feature in self.categorical_features:
                # Try to extract categorical value
                val = self._extract_categorical(text, feature)
                if val is not None:
                    # Convert to numeric using encoding
                    mapping = self.categorical_encodings.get(feature, {}).get("mapping", {})
                    result[feature] = mapping.get(val, None)
                else:
                    result[feature] = None
            else:
                result[feature] = None
        
        return result
    
    def get_missing_features(self, extracted: dict) -> list:
        """Return features that weren't successfully extracted."""
        missing = []
        for feature in self.features:
            val = extracted.get(feature)
            if val is None:
                missing.append(feature)
        return missing

    def get_feature_context(self, feature: str) -> dict:
        """
        Get context for generating questions about a feature.
        Returns info about the feature for LLM question generation.
        """
        aliases = self.feature_aliases.get(feature, [feature])
        
        if feature in self.categorical_features:
            values = self.categorical_values.get(feature, {})
            return {
                "feature": feature,
                "type": "categorical",
                "aliases": aliases,
                "possible_values": list(values.keys())
            }
        else:
            return {
                "feature": feature,
                "type": "numeric",
                "aliases": aliases,
                "possible_values": None
            }


class FeatureExtractorFactory:
    """
    Factory class for creating appropriate feature extractors based on model type.
    """
    
    @staticmethod
    def create(model_type: str, keyword_file: str, config: dict, threshold: float = 0.75) -> BaseFeatureExtractor:
        """
        Create a feature extractor for the specified model type.
        
        Args:
            model_type: Type of model ('symptom', 'heart', etc.)
            keyword_file: Path to the keyword/feature mapping JSON file
            config: Model configuration containing features list
            threshold: Similarity threshold for NLP matching
            
        Returns:
            Appropriate feature extractor instance
        """
        extractors = {
            "symptom": SymptomFeatureExtractor,
            "heart": HeartFeatureExtractor,
        }
        
        extractor_class = extractors.get(model_type)
        if not extractor_class:
            raise ValueError(f"Unknown model type: {model_type}. Available types: {list(extractors.keys())}")
        
        return extractor_class(keyword_file, config, threshold)


# Backward compatibility - keep the original FeatureExtractor class
class FeatureExtractor(SymptomFeatureExtractor):
    """
    Legacy wrapper for backward compatibility.
    Delegates to SymptomFeatureExtractor.
    """
    pass
