"""
RAG (Retrieval-Augmented Generation) Service for fallback predictions.

This module provides a RAG-based fallback system that retrieves relevant
medical information from a knowledge base when ML models are unavailable.
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from .vector_store import VectorStore, Document
from .embedding_service import EmbeddingService, EmbeddingServiceFactory

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Represents a RAG-based response."""
    query: str
    possible_conditions: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    retrieved_documents: List[Document] = field(default_factory=list)
    confidence: float = 0.0
    sources: List[str] = field(default_factory=list)
    disclaimer: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            "query": self.query,
            "possible_conditions": self.possible_conditions,
            "recommendations": self.recommendations,
            "confidence": self.confidence,
            "sources": self.sources,
            "disclaimer": self.disclaimer
        }


class RAGService:
    """
    Retrieval-Augmented Generation service for fallback predictions.
    
    Provides medical information retrieval when ML models are unavailable.
    Uses vector similarity search to find relevant medical knowledge and
    generates responses using the LLM.
    """
    
    # Default disclaimer for RAG-based responses
    DEFAULT_DISCLAIMER = (
        "This response is generated using our knowledge base as the prediction "
        "model is unavailable. This is NOT a medical diagnosis. Please consult "
        "a healthcare professional for proper medical advice."
    )
    
    # Model type to category mapping
    MODEL_CATEGORIES = {
        "symptom": ["disease", "symptom", "general"],
        "heart": ["cardiac", "heart", "cardiovascular"],
        "diabetes": ["diabetes", "endocrine", "metabolic"],
        "mental_health": ["mental_health", "psychiatric", "psychological"]
    }
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService = None,
        llm_client = None,
        top_k: int = 5,
        min_relevance_score: float = 0.3
    ):
        """
        Initialize the RAG service.
        
        Args:
            vector_store: Vector store for document retrieval
            embedding_service: Service for generating embeddings
            llm_client: LLM client for response generation (Ollama)
            top_k: Number of documents to retrieve
            min_relevance_score: Minimum relevance score for retrieved documents
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service or EmbeddingServiceFactory.get_instance()
        self.llm_client = llm_client
        self.top_k = top_k
        self.min_relevance_score = min_relevance_score
        
        # Track if initialized with documents
        self._initialized = False
    
    def is_available(self) -> bool:
        """
        Check if the RAG service is available.
        
        Returns:
            True if the service can process requests
        """
        return self.vector_store.count() > 0
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        filters: Dict = None
    ) -> List[Document]:
        """
        Retrieve relevant documents from the vector store.
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve (uses default if None)
            filters: Optional metadata filters
            
        Returns:
            List of relevant Document objects
        """
        top_k = top_k or self.top_k
        
        # Generate query embedding
        query_embedding = self.embedding_service.embed(query)
        
        # Search vector store
        documents = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            k=top_k,
            filters=filters
        )
        
        # Filter by relevance score
        relevant_docs = []
        for doc in documents:
            distance = doc.metadata.get("distance", 1.0)
            relevance = 1.0 - distance  # Convert distance to similarity
            
            if relevance >= self.min_relevance_score:
                doc.metadata["relevance_score"] = relevance
                relevant_docs.append(doc)
        
        logger.info(
            f"Retrieved {len(relevant_docs)} relevant documents for query: {query[:50]}..."
        )
        
        return relevant_docs
    
    def generate_response(
        self,
        query: str,
        context: List[Document],
        model_type: str = "symptom"
    ) -> str:
        """
        Generate a response using retrieved context.
        
        Args:
            query: User query
            context: Retrieved documents as context
            model_type: Type of model for specialized response
            
        Returns:
            Generated response string
        """
        if not context:
            return self._generate_no_context_response(query, model_type)
        
        # Build context string
        context_text = "\n\n".join([
            f"[{doc.metadata.get('source', 'Unknown')}]: {doc.content}"
            for doc in context[:3]  # Use top 3 documents
        ])
        
        # Generate response using LLM if available
        if self.llm_client:
            return self._generate_with_llm(query, context_text, model_type)
        
        # Fallback to template-based response
        return self._generate_template_response(query, context, model_type)
    
    def _generate_with_llm(
        self,
        query: str,
        context: str,
        model_type: str
    ) -> str:
        """Generate response using LLM."""
        import ollama
        
        prompt = f"""
        You are a medical triage assistant. A patient has described their symptoms.
        
        PATIENT QUERY: {query}
        
        RELEVANT MEDICAL INFORMATION:
        {context}
        
        Based on the information provided, give a helpful response that:
        1. Acknowledges the patient's concerns
        2. Provides relevant information from the medical context
        3. Suggests when to seek medical attention
        4. Includes appropriate disclaimers
        
        Keep the response concise and informative. Do not make a diagnosis.
        """
        
        try:
            response = ollama.chat(
                model="gpt-oss:120b-cloud",
                messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._generate_template_response(query, [], model_type)
    
    def _generate_template_response(
        self,
        query: str,
        context: List[Document],
        model_type: str
    ) -> str:
        """Generate response using templates when LLM is unavailable."""
        conditions = []
        for doc in context:
            if doc.metadata.get("category") == "disease":
                conditions.append(doc.metadata.get("disease_name", "Unknown condition"))
        
        if conditions:
            conditions_str = ", ".join(conditions[:3])
            return (
                f"Based on your symptoms, some conditions that may be relevant include: "
                f"{conditions_str}. Please note this is not a diagnosis. "
                f"Consult a healthcare professional for proper evaluation."
            )
        
        return (
            "I found some relevant medical information, but cannot provide specific "
            "recommendations without more details. Please consult a healthcare professional."
        )
    
    def _generate_no_context_response(self, query: str, model_type: str) -> str:
        """Generate response when no relevant context is found."""
        return (
            "I don't have enough information to provide specific guidance. "
            "Please consult a healthcare professional for proper evaluation of your symptoms."
        )
    
    def predict_with_rag(
        self,
        symptoms: List[str],
        features: Dict = None,
        model_type: str = "symptom"
    ) -> RAGResponse:
        """
        Main entry point for RAG-based prediction.
        
        Args:
            symptoms: List of symptom strings
            features: Optional feature dictionary
            model_type: Type of model (symptom, heart, diabetes, mental_health)
            
        Returns:
            RAGResponse with prediction results
        """
        # Build query from symptoms
        query = self._build_query(symptoms, features, model_type)
        
        # Get category filters for model type
        categories = self.MODEL_CATEGORIES.get(model_type, ["general"])
        
        # Retrieve relevant documents
        documents = self.retrieve(query)
        
        # Filter documents by category if available
        category_docs = [
            doc for doc in documents
            if doc.metadata.get("category") in categories or
               doc.metadata.get("model_type") == model_type
        ]
        
        # Use category-filtered docs if available, otherwise use all
        relevant_docs = category_docs if category_docs else documents
        
        # Extract possible conditions
        conditions = self._extract_conditions(relevant_docs, symptoms)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            symptoms, conditions, model_type
        )
        
        # Calculate confidence based on retrieval quality
        confidence = self._calculate_confidence(relevant_docs, symptoms)
        
        # Get unique sources
        sources = list(set(
            doc.metadata.get("source", "unknown")
            for doc in relevant_docs
        ))
        
        return RAGResponse(
            query=query,
            possible_conditions=conditions,
            recommendations=recommendations,
            retrieved_documents=relevant_docs,
            confidence=confidence,
            sources=sources,
            disclaimer=self.DEFAULT_DISCLAIMER
        )
    
    def _build_query(
        self,
        symptoms: List[str],
        features: Dict = None,
        model_type: str = "symptom"
    ) -> str:
        """Build a search query from symptoms and features."""
        parts = []
        
        if symptoms:
            parts.extend(symptoms)
        
        if features:
            # Add relevant features based on model type
            if model_type == "heart":
                heart_features = ["chest pain", "shortness of breath", "palpitations"]
                for key, value in features.items():
                    if value and key.lower() in heart_features:
                        parts.append(f"{key}: {value}")
            elif model_type == "diabetes":
                if features.get("Glucose"):
                    parts.append(f"blood glucose {features['Glucose']}")
                if features.get("BMI"):
                    parts.append(f"BMI {features['BMI']}")
        
        return " ".join(parts) if parts else "general symptoms"
    
    def _extract_conditions(
        self,
        documents: List[Document],
        symptoms: List[str]
    ) -> List[Dict[str, Any]]:
        """Extract possible conditions from retrieved documents."""
        conditions = []
        seen_conditions = set()
        
        for doc in documents:
            disease_name = doc.metadata.get("disease_name")
            
            if disease_name and disease_name not in seen_conditions:
                # Calculate relevance based on symptom overlap
                doc_symptoms = doc.metadata.get("symptoms", [])
                if isinstance(doc_symptoms, str):
                    doc_symptoms = [s.strip() for s in doc_symptoms.split(",")]
                
                matching_symptoms = [
                    s for s in symptoms
                    if any(s.lower() in ds.lower() for ds in doc_symptoms)
                ]
                
                relevance = doc.metadata.get("relevance_score", 0.5)
                if matching_symptoms:
                    relevance = min(1.0, relevance + 0.1 * len(matching_symptoms))
                
                conditions.append({
                    "condition": disease_name,
                    "relevance_score": round(relevance, 2),
                    "matching_symptoms": matching_symptoms,
                    "source": doc.metadata.get("source", "unknown")
                })
                seen_conditions.add(disease_name)
        
        # Sort by relevance
        conditions.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return conditions[:5]  # Return top 5 conditions
    
    def _generate_recommendations(
        self,
        symptoms: List[str],
        conditions: List[Dict],
        model_type: str
    ) -> List[str]:
        """Generate recommendations based on symptoms and conditions."""
        recommendations = []
        
        # General recommendations
        if any(s.lower() in ["fever", "high temperature"] for s in symptoms):
            recommendations.append("Monitor your temperature regularly and stay hydrated")
        
        if any(s.lower() in ["chest pain", "difficulty breathing"] for s in symptoms):
            recommendations.append(
                "Seek immediate medical attention if symptoms worsen or become severe"
            )
        
        # Model-specific recommendations
        if model_type == "heart":
            recommendations.extend([
                "Avoid strenuous activity until evaluated by a doctor",
                "Keep a log of when symptoms occur"
            ])
        elif model_type == "diabetes":
            recommendations.extend([
                "Monitor your blood sugar levels",
                "Follow your prescribed diet and medication plan"
            ])
        elif model_type == "mental_health":
            recommendations.extend([
                "Consider speaking with a mental health professional",
                "Reach out to trusted friends or family for support"
            ])
        
        # Add general recommendation
        if not recommendations:
            recommendations.append(
                "Consult a healthcare professional for proper evaluation"
            )
        
        return recommendations
    
    def _calculate_confidence(
        self,
        documents: List[Document],
        symptoms: List[str]
    ) -> float:
        """Calculate confidence score for the RAG response."""
        if not documents:
            return 0.0
        
        # Base confidence on retrieval scores
        scores = [
            doc.metadata.get("relevance_score", 0.5)
            for doc in documents
        ]
        
        if not scores:
            return 0.3
        
        avg_score = sum(scores) / len(scores)
        
        # Boost confidence if multiple documents agree
        if len(documents) >= 3:
            avg_score = min(1.0, avg_score * 1.1)
        
        # Reduce confidence if few symptoms provided
        if len(symptoms) < 2:
            avg_score *= 0.8
        
        return round(avg_score, 2)
    
    def search(
        self,
        query: str,
        model_type: str = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Direct search endpoint for testing and debugging.
        
        Args:
            query: Search query
            model_type: Optional model type filter
            top_k: Number of results
            
        Returns:
            Dictionary with search results
        """
        filters = None
        if model_type:
            filters = {"model_type": model_type}
        
        documents = self.retrieve(query, top_k=top_k, filters=filters)
        
        return {
            "query": query,
            "results": [
                {
                    "id": doc.id,
                    "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                    "metadata": doc.metadata,
                    "relevance_score": doc.metadata.get("relevance_score", 0)
                }
                for doc in documents
            ],
            "total": len(documents)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG service.
        
        Returns:
            Dictionary with service statistics
        """
        return {
            "available": self.is_available(),
            "document_count": self.vector_store.count(),
            "top_k": self.top_k,
            "min_relevance_score": self.min_relevance_score,
            "vector_store_stats": self.vector_store.get_stats(),
            "embedding_info": self.embedding_service.get_model_info()
        }


class RAGServiceFactory:
    """Factory for creating RAG service instances."""
    
    _instance: Optional[RAGService] = None
    
    @classmethod
    def create(
        cls,
        vector_store: VectorStore = None,
        embedding_service: EmbeddingService = None,
        llm_client = None,
        top_k: int = 5,
        min_relevance_score: float = 0.3,
        persist_directory: str = None
    ) -> RAGService:
        """
        Create a RAG service instance.
        
        Args:
            vector_store: Optional pre-configured vector store
            embedding_service: Optional pre-configured embedding service
            llm_client: Optional LLM client
            top_k: Number of documents to retrieve
            min_relevance_score: Minimum relevance score
            persist_directory: Directory for vector store persistence
            
        Returns:
            RAGService instance
        """
        if vector_store is None:
            from .vector_store import VectorStoreFactory
            vector_store = VectorStoreFactory.create(
                backend="chromadb",
                persist_directory=persist_directory
            )
        
        if embedding_service is None:
            embedding_service = EmbeddingServiceFactory.get_instance()
        
        return RAGService(
            vector_store=vector_store,
            embedding_service=embedding_service,
            llm_client=llm_client,
            top_k=top_k,
            min_relevance_score=min_relevance_score
        )
    
    @classmethod
    def get_instance(cls) -> Optional[RAGService]:
        """Get the singleton instance."""
        return cls._instance
    
    @classmethod
    def set_instance(cls, instance: RAGService):
        """Set the singleton instance."""
        cls._instance = instance
    
    @classmethod
    def reset(cls):
        """Reset the singleton instance."""
        cls._instance = None
