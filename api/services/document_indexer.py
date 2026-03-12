"""
Document Indexer for building the RAG knowledge base.

This module processes medical datasets and configurations to create
searchable documents for the RAG system.
"""

import os
import json
import csv
import logging
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass

from .vector_store import Document, VectorStore
from .embedding_service import EmbeddingService, EmbeddingServiceFactory

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class IndexingStats:
    """Statistics for document indexing."""
    total_documents: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
    
    def to_dict(self) -> Dict:
        return {
            "total_documents": self.total_documents,
            "successful": self.successful,
            "failed": self.failed,
            "skipped": self.skipped,
            "errors": self.errors[:10]  # Limit error messages
        }


class DocumentIndexer:
    """
    Indexes medical documents into the vector store.
    
    Processes datasets and creates embeddings for retrieval.
    Supports multiple document types:
    - Symptom-disease mappings
    - Feature configurations
    - Medical knowledge articles
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService = None,
        batch_size: int = 100,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Initialize the document indexer.
        
        Args:
            vector_store: Vector store to index documents into
            embedding_service: Service for generating embeddings
            batch_size: Number of documents to process per batch
            chunk_size: Maximum characters per document chunk
            chunk_overlap: Overlap between chunks
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service or EmbeddingServiceFactory.get_instance()
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Track indexed document IDs
        self._indexed_ids: set = set()
    
    def index_all(
        self,
        project_root: str,
        rebuild: bool = False
    ) -> IndexingStats:
        """
        Index all available medical documents.
        
        Args:
            project_root: Root directory of the project
            rebuild: Whether to rebuild the index from scratch
            
        Returns:
            IndexingStats with results
        """
        stats = IndexingStats()
        
        # Clear existing index if rebuilding
        if rebuild:
            logger.info("Clearing existing vector store...")
            self.vector_store.clear()
            self._indexed_ids.clear()
        
        # Check if already indexed
        if self.vector_store.count() > 0 and not rebuild:
            logger.info(
                f"Vector store already contains {self.vector_store.count()} documents. "
                "Use rebuild=True to reindex."
            )
            stats.skipped = self.vector_store.count()
            return stats
        
        # Index symptom-disease mapping
        try:
            mapping_path = os.path.join(
                project_root,
                "datasets/Symptom-Disease Extended/mapping.json"
            )
            if os.path.exists(mapping_path):
                symptom_stats = self.index_symptom_disease_mapping(mapping_path)
                stats.successful += symptom_stats.successful
                stats.failed += symptom_stats.failed
        except Exception as e:
            logger.error(f"Failed to index symptom-disease mapping: {e}")
            stats.errors.append(f"Symptom mapping error: {str(e)}")
        
        # Index disease prediction dataset
        try:
            dataset_path = os.path.join(project_root, "datasets/disease_prediction.csv")
            if os.path.exists(dataset_path):
                dataset_stats = self.index_disease_dataset(dataset_path)
                stats.successful += dataset_stats.successful
                stats.failed += dataset_stats.failed
        except Exception as e:
            logger.error(f"Failed to index disease dataset: {e}")
            stats.errors.append(f"Disease dataset error: {str(e)}")
        
        # Index feature mappings
        try:
            config_dir = os.path.join(project_root, "api/config")
            if os.path.exists(config_dir):
                config_stats = self.index_feature_mappings(config_dir)
                stats.successful += config_stats.successful
                stats.failed += config_stats.failed
        except Exception as e:
            logger.error(f"Failed to index feature mappings: {e}")
            stats.errors.append(f"Feature mapping error: {str(e)}")
        
        # Index medical knowledge documents
        try:
            knowledge_stats = self.index_medical_knowledge()
            stats.successful += knowledge_stats.successful
            stats.failed += knowledge_stats.failed
        except Exception as e:
            logger.error(f"Failed to index medical knowledge: {e}")
            stats.errors.append(f"Medical knowledge error: {str(e)}")
        
        stats.total_documents = stats.successful + stats.failed
        logger.info(
            f"Indexing complete: {stats.successful} successful, "
            f"{stats.failed} failed, {stats.skipped} skipped"
        )
        
        return stats
    
    def index_symptom_disease_mapping(
        self,
        mapping_path: str
    ) -> IndexingStats:
        """
        Index symptom-disease relationships from mapping file.
        
        Args:
            mapping_path: Path to the mapping JSON file
            
        Returns:
            IndexingStats with results
        """
        stats = IndexingStats()
        
        logger.info(f"Indexing symptom-disease mapping from {mapping_path}")
        
        try:
            with open(mapping_path, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
            
            documents = []
            
            # Create documents for each disease
            for disease_name, disease_id in mapping.items():
                doc_id = f"disease_{disease_id}"
                
                if doc_id in self._indexed_ids:
                    continue
                
                # Create content describing the disease
                content = f"Disease: {disease_name}. "
                content += f"This is a medical condition that may be associated with various symptoms."
                
                doc = Document(
                    id=doc_id,
                    content=content,
                    metadata={
                        "source": "symptom_disease_mapping",
                        "category": "disease",
                        "model_type": "symptom",
                        "disease_name": disease_name,
                        "disease_id": disease_id
                    }
                )
                documents.append(doc)
                self._indexed_ids.add(doc_id)
            
            # Index documents in batches
            stats = self._index_documents_batch(documents)
            
        except Exception as e:
            logger.error(f"Error indexing symptom-disease mapping: {e}")
            stats.errors.append(str(e))
        
        return stats
    
    def index_disease_dataset(
        self,
        dataset_path: str,
        max_rows: int = 5000
    ) -> IndexingStats:
        """
        Index disease prediction dataset.
        
        Args:
            dataset_path: Path to the CSV dataset
            max_rows: Maximum number of rows to process
            
        Returns:
            IndexingStats with results
        """
        stats = IndexingStats()
        
        logger.info(f"Indexing disease dataset from {dataset_path}")
        
        try:
            documents = []
            
            with open(dataset_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for i, row in enumerate(reader):
                    if i >= max_rows:
                        break
                    
                    # Extract disease and symptoms
                    disease = row.get('Disease', row.get('disease', 'Unknown'))
                    symptoms = [
                        row.get(key, '').strip()
                        for key in row.keys()
                        if 'symptom' in key.lower() and row.get(key, '').strip()
                    ]
                    
                    doc_id = f"dataset_row_{i}"
                    if doc_id in self._indexed_ids:
                        continue
                    
                    # Create content
                    content = f"Disease: {disease}. "
                    if symptoms:
                        content += f"Associated symptoms: {', '.join(symptoms)}."
                    
                    doc = Document(
                        id=doc_id,
                        content=content,
                        metadata={
                            "source": "disease_prediction_dataset",
                            "category": "disease",
                            "model_type": "symptom",
                            "disease_name": disease,
                            "symptoms": symptoms
                        }
                    )
                    documents.append(doc)
                    self._indexed_ids.add(doc_id)
            
            # Index documents in batches
            stats = self._index_documents_batch(documents)
            
        except Exception as e:
            logger.error(f"Error indexing disease dataset: {e}")
            stats.errors.append(str(e))
        
        return stats
    
    def index_feature_mappings(
        self,
        config_dir: str
    ) -> IndexingStats:
        """
        Index feature mapping configurations.
        
        Args:
            config_dir: Path to the config directory
            
        Returns:
            IndexingStats with results
        """
        stats = IndexingStats()
        
        logger.info(f"Indexing feature mappings from {config_dir}")
        
        mapping_files = {
            "diabetes_feature_mapping.json": "diabetes",
            "heart_feature_mapping.json": "heart",
            "mental_health_feature_mapping.json": "mental_health",
            "symptom_mapping.json": "symptom"
        }
        
        documents = []
        
        for filename, model_type in mapping_files.items():
            filepath = os.path.join(config_dir, filename)
            
            if not os.path.exists(filepath):
                continue
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    mapping_data = json.load(f)
                
                # Create documents from feature mappings
                if isinstance(mapping_data, dict):
                    for key, value in mapping_data.items():
                        doc_id = f"feature_{model_type}_{key}"
                        
                        if doc_id in self._indexed_ids:
                            continue
                        
                        # Create content describing the feature
                        content = self._create_feature_content(key, value, model_type)
                        
                        doc = Document(
                            id=doc_id,
                            content=content,
                            metadata={
                                "source": "feature_mapping",
                                "category": "feature",
                                "model_type": model_type,
                                "feature_name": key,
                                "feature_value": str(value) if not isinstance(value, dict) else json.dumps(value)
                            }
                        )
                        documents.append(doc)
                        self._indexed_ids.add(doc_id)
                        
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                stats.errors.append(f"{filename}: {str(e)}")
        
        # Index documents in batches
        batch_stats = self._index_documents_batch(documents)
        stats.successful += batch_stats.successful
        stats.failed += batch_stats.failed
        
        return stats
    
    def index_medical_knowledge(self) -> IndexingStats:
        """
        Index built-in medical knowledge documents.
        
        Returns:
            IndexingStats with results
        """
        stats = IndexingStats()
        
        logger.info("Indexing built-in medical knowledge")
        
        # Built-in medical knowledge documents
        knowledge_docs = self._get_builtin_knowledge()
        
        documents = []
        for i, knowledge in enumerate(knowledge_docs):
            doc_id = f"knowledge_{knowledge.get('id', i)}"
            
            if doc_id in self._indexed_ids:
                continue
            
            doc = Document(
                id=doc_id,
                content=knowledge.get("content", ""),
                metadata={
                    "source": "medical_knowledge",
                    "category": knowledge.get("category", "general"),
                    "model_type": knowledge.get("model_type", "general"),
                    "title": knowledge.get("title", ""),
                    **knowledge.get("extra_metadata", {})
                }
            )
            documents.append(doc)
            self._indexed_ids.add(doc_id)
        
        # Index documents in batches
        stats = self._index_documents_batch(documents)
        
        return stats
    
    def _get_builtin_knowledge(self) -> List[Dict]:
        """Get built-in medical knowledge documents."""
        return [
            # Heart-related knowledge
            {
                "id": "heart_symptoms",
                "content": (
                    "Heart disease symptoms may include chest pain or discomfort, "
                    "shortness of breath, palpitations, dizziness, and fatigue. "
                    "Risk factors include high blood pressure, high cholesterol, "
                    "smoking, diabetes, and family history. Early detection and "
                    "lifestyle changes can significantly reduce risk."
                ),
                "category": "cardiac",
                "model_type": "heart",
                "title": "Heart Disease Symptoms and Risk Factors"
            },
            {
                "id": "heart_healthy_lifestyle",
                "content": (
                    "To maintain heart health: exercise regularly (150 minutes per week), "
                    "eat a balanced diet low in saturated fats, maintain healthy weight, "
                    "avoid smoking, limit alcohol, manage stress, and monitor blood pressure. "
                    "Regular check-ups are important for early detection."
                ),
                "category": "cardiac",
                "model_type": "heart",
                "title": "Heart Healthy Lifestyle Recommendations"
            },
            # Diabetes-related knowledge
            {
                "id": "diabetes_symptoms",
                "content": (
                    "Diabetes symptoms include increased thirst, frequent urination, "
                    "hunger, fatigue, blurred vision, and slow-healing sores. "
                    "Type 1 diabetes often develops quickly, while Type 2 may develop "
                    "slowly over years. Blood tests can diagnose diabetes."
                ),
                "category": "diabetes",
                "model_type": "diabetes",
                "title": "Diabetes Symptoms Overview"
            },
            {
                "id": "diabetes_management",
                "content": (
                    "Diabetes management involves monitoring blood sugar levels, "
                    "taking prescribed medications, following a healthy diet, "
                    "regular physical activity, and maintaining a healthy weight. "
                    "Regular check-ups with healthcare providers are essential."
                ),
                "category": "diabetes",
                "model_type": "diabetes",
                "title": "Diabetes Management Guidelines"
            },
            # Mental health knowledge
            {
                "id": "mental_health_signs",
                "content": (
                    "Signs of mental health concerns include persistent sadness, "
                    "excessive worry, mood changes, withdrawal from activities, "
                    "changes in sleep or appetite, difficulty concentrating, "
                    "and physical symptoms without clear cause. Professional "
                    "help should be sought if symptoms persist."
                ),
                "category": "mental_health",
                "model_type": "mental_health",
                "title": "Mental Health Warning Signs"
            },
            {
                "id": "mental_health_support",
                "content": (
                    "Mental health support options include therapy (counseling, CBT), "
                    "medication, support groups, lifestyle changes, and self-care. "
                    "Employers may offer Employee Assistance Programs (EAP). "
                    "Crisis hotlines are available 24/7 for immediate support."
                ),
                "category": "mental_health",
                "model_type": "mental_health",
                "title": "Mental Health Support Options"
            },
            # General symptom knowledge
            {
                "id": "fever_management",
                "content": (
                    "Fever is a common symptom indicating the body is fighting infection. "
                    "For adults, seek medical attention if fever exceeds 103°F (39.4°C), "
                    "lasts more than 3 days, or is accompanied by severe symptoms. "
                    "Stay hydrated, rest, and use fever reducers as directed."
                ),
                "category": "symptom",
                "model_type": "symptom",
                "title": "Fever Management Guidelines"
            },
            {
                "id": "common_cold",
                "content": (
                    "Common cold symptoms include runny nose, sore throat, cough, "
                    "congestion, mild fever, and fatigue. Most colds resolve within "
                    "7-10 days. Rest, fluids, and over-the-counter medications can "
                    "help manage symptoms. See a doctor if symptoms worsen or persist."
                ),
                "category": "symptom",
                "model_type": "symptom",
                "title": "Common Cold Information"
            },
            {
                "id": "when_to_seek_care",
                "content": (
                    "Seek immediate medical attention for: difficulty breathing, "
                    "chest pain, severe bleeding, loss of consciousness, severe "
                    "allergic reactions, signs of stroke (FAST: face drooping, "
                    "arm weakness, speech difficulty), or any life-threatening condition."
                ),
                "category": "general",
                "model_type": "general",
                "title": "When to Seek Medical Care"
            },
            {
                "id": "preventive_care",
                "content": (
                    "Preventive care includes regular check-ups, age-appropriate "
                    "screenings, vaccinations, and healthy lifestyle choices. "
                    "Early detection of conditions like cancer, diabetes, and heart "
                    "disease significantly improves outcomes. Follow recommended "
                    "screening schedules for your age group."
                ),
                "category": "general",
                "model_type": "general",
                "title": "Preventive Care Guidelines"
            }
        ]
    
    def _create_feature_content(
        self,
        key: str,
        value: Any,
        model_type: str
    ) -> str:
        """Create content describing a feature."""
        # Format the key for readability
        formatted_key = key.replace('_', ' ').title()
        
        if isinstance(value, dict):
            # Handle dictionary values (e.g., mappings)
            value_desc = ", ".join([f"{k}: {v}" for k, v in list(value.items())[:5]])
            if len(value) > 5:
                value_desc += f" (and {len(value) - 5} more)"
            return f"{model_type.title()} feature '{formatted_key}': {value_desc}"
        
        elif isinstance(value, list):
            items = ", ".join(str(v) for v in value[:5])
            if len(value) > 5:
                items += f" (and {len(value) - 5} more)"
            return f"{model_type.title()} feature '{formatted_key}': {items}"
        
        else:
            return f"{model_type.title()} feature '{formatted_key}': {value}"
    
    def _index_documents_batch(
        self,
        documents: List[Document]
    ) -> IndexingStats:
        """
        Index documents in batches with embeddings.
        
        Args:
            documents: List of documents to index
            
        Returns:
            IndexingStats with results
        """
        stats = IndexingStats(total_documents=len(documents))
        
        if not documents:
            return stats
        
        # Filter out documents with empty content
        valid_documents = []
        for doc in documents:
            if doc.content and doc.content.strip():
                valid_documents.append(doc)
            else:
                stats.failed += 1
                logger.warning(f"Skipping document with empty content: {doc.id}")
        
        if not valid_documents:
            return stats
        
        # Process in batches
        for i in range(0, len(valid_documents), self.batch_size):
            batch = valid_documents[i:i + self.batch_size]
            
            try:
                # Generate embeddings for batch
                contents = [doc.content for doc in batch]
                embeddings = self.embedding_service.embed_batch(contents)
                
                # Add to vector store
                added = self.vector_store.add_documents(batch, embeddings)
                stats.successful += added
                stats.failed += len(batch) - added
                
                if added < len(batch):
                    logger.warning(
                        f"Batch {i//self.batch_size}: {added}/{len(batch)} documents indexed "
                        f"({len(batch) - added} may be duplicates)"
                    )
                
            except Exception as e:
                logger.error(f"Error indexing batch {i//self.batch_size}: {e}")
                stats.failed += len(batch)
                stats.errors.append(f"Batch {i//self.batch_size}: {str(e)}")
        
        return stats
    
    def create_document_chunks(
        self,
        text: str,
        doc_id_prefix: str = "chunk"
    ) -> List[Document]:
        """
        Split a long text into retrievable chunks.
        
        Args:
            text: Text to split
            doc_id_prefix: Prefix for document IDs
            
        Returns:
            List of Document chunks
        """
        chunks = []
        
        # Simple chunking by character count with overlap
        start = 0
        chunk_num = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end
                last_period = text.rfind('.', start, end)
                if last_period > start + self.chunk_size // 2:
                    end = last_period + 1
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                doc = Document(
                    id=f"{doc_id_prefix}_{chunk_num}",
                    content=chunk_text,
                    metadata={
                        "source": "chunked_document",
                        "chunk_number": chunk_num,
                        "start_char": start,
                        "end_char": end
                    }
                )
                chunks.append(doc)
                chunk_num += 1
            
            start = end - self.chunk_overlap if end < len(text) else len(text)
        
        return chunks
    
    def get_indexing_status(self) -> Dict[str, Any]:
        """
        Get current indexing status.
        
        Returns:
            Dictionary with indexing status
        """
        return {
            "indexed_document_count": len(self._indexed_ids),
            "vector_store_count": self.vector_store.count(),
            "batch_size": self.batch_size,
            "chunk_size": self.chunk_size
        }


class DocumentIndexerFactory:
    """Factory for creating document indexer instances."""
    
    @staticmethod
    def create(
        vector_store: VectorStore = None,
        embedding_service: EmbeddingService = None,
        batch_size: int = 100,
        chunk_size: int = 500,
        persist_directory: str = None
    ) -> DocumentIndexer:
        """
        Create a document indexer instance.
        
        Args:
            vector_store: Optional pre-configured vector store
            embedding_service: Optional pre-configured embedding service
            batch_size: Batch size for indexing
            chunk_size: Chunk size for document splitting
            persist_directory: Directory for vector store persistence
            
        Returns:
            DocumentIndexer instance
        """
        if vector_store is None:
            from .vector_store import VectorStoreFactory
            vector_store = VectorStoreFactory.create(
                backend="chromadb",
                persist_directory=persist_directory
            )
        
        if embedding_service is None:
            embedding_service = EmbeddingServiceFactory.get_instance()
        
        return DocumentIndexer(
            vector_store=vector_store,
            embedding_service=embedding_service,
            batch_size=batch_size,
            chunk_size=chunk_size
        )
