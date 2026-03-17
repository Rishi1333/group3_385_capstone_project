"""
Medical Knowledge Ingestor

Processes and indexes medical knowledge from downloaded sources into the RAG vector store.
Handles multiple content formats and applies medical-specific chunking strategies.
"""

import os
import json
import logging
import re
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass, field
from datetime import datetime

from .vector_store import Document, VectorStore, VectorStoreFactory
from .embedding_service import EmbeddingService, EmbeddingServiceFactory

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class IngestionStats:
    """Statistics for knowledge ingestion."""
    total_documents: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    by_source: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "total_documents": self.total_documents,
            "successful": self.successful,
            "failed": self.failed,
            "skipped": self.skipped,
            "by_source": self.by_source,
            "errors": self.errors[:10]
        }


@dataclass
class MedicalDocument:
    """Represents a processed medical document."""
    id: str
    title: str
    content: str
    source: str
    category: str
    symptoms: List[str] = field(default_factory=list)
    urgency: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_document(self) -> Document:
        """Convert to VectorStore Document."""
        return Document(
            id=self.id,
            content=self.content,
            metadata={
                "title": self.title,
                "source": self.source,
                "category": self.category,
                "symptoms": self.symptoms,
                "urgency": self.urgency,
                **self.metadata
            }
        )


class MedicalKnowledgeIngestor:
    """
    Ingests medical knowledge from various sources into the vector store.
    
    Supports:
    - Downloaded JSON content from CDC, NHS, MedlinePlus, Mayo Clinic
    - Red flags emergency definitions
    - Custom medical knowledge files
    """
    
    # Categories for medical content
    CATEGORIES = {
        "cardiovascular": ["heart", "cardiac", "chest pain", "blood pressure", "cholesterol"],
        "neurological": ["brain", "stroke", "headache", "seizure", "neural"],
        "respiratory": ["lung", "breathing", "asthma", "copd", "pneumonia", "cough"],
        "gastrointestinal": ["stomach", "digestive", "abdominal", "bowel", "liver"],
        "psychiatric": ["mental", "depression", "anxiety", "psychiatric", "suicide"],
        "trauma": ["injury", "fracture", "wound", "bleeding", "trauma"],
        "pediatric": ["child", "infant", "baby", "pediatric", "newborn"],
        "obstetric": ["pregnancy", "pregnant", "prenatal", "labor", "maternal"],
        "endocrine": ["diabetes", "thyroid", "hormone", "insulin", "glucose"],
        "infectious": ["infection", "fever", "bacterial", "viral", "antibiotic"],
        "dermatological": ["skin", "rash", "dermatitis", "eczema", "lesion"],
        "general": []
    }
    
    def __init__(
        self,
        vector_store: VectorStore = None,
        embedding_service: EmbeddingService = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Initialize the medical knowledge ingestor.
        
        Args:
            vector_store: Vector store for document storage
            embedding_service: Service for generating embeddings
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks
        """
        self.vector_store = vector_store or VectorStoreFactory.get_instance()
        self.embedding_service = embedding_service or EmbeddingServiceFactory.get_instance()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.stats = IngestionStats()
    
    def _generate_id(self, source: str, title: str, index: int = 0) -> str:
        """Generate a unique document ID."""
        base = f"{source}_{title}_{index}".lower()
        base = re.sub(r'[^a-z0-9_]', '_', base)
        return base[:100]
    
    def _categorize_content(self, title: str, content: str) -> str:
        """Determine the medical category for content."""
        text = (title + " " + content).lower()
        
        for category, keywords in self.CATEGORIES.items():
            for keyword in keywords:
                if keyword in text:
                    return category
        
        return "general"
    
    def _extract_symptoms(self, content: str, symptoms_list: List[str] = None) -> List[str]:
        """Extract symptom keywords from content."""
        symptoms = set(symptoms_list or [])
        
        # Common symptom patterns
        symptom_patterns = [
            r'symptoms?[:\s]+([^.]*)',
            r'signs?[:\s]+([^.]*)',
            r'you may (?:also )?experience[:\s]+([^.]*)',
            r'common (?:symptoms? )?include[:\s]+([^.]*)'
        ]
        
        for pattern in symptom_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                # Split by common delimiters
                items = re.split(r'[,;]', match)
                for item in items:
                    item = item.strip().lower()
                    if 3 < len(item) < 50:  # Reasonable symptom length
                        symptoms.add(item)
        
        return list(symptoms)[:20]  # Limit to 20 symptoms
    
    def _chunk_content(self, content: str) -> Generator[str, None, None]:
        """
        Chunk content into smaller pieces for embedding.
        
        Uses medical-aware chunking to preserve context.
        """
        if len(content) <= self.chunk_size:
            yield content
            return
        
        # Split by paragraphs first
        paragraphs = content.split('\n\n')
        
        current_chunk = ""
        
        for para in paragraphs:
            # If paragraph alone exceeds chunk size, split by sentences
            if len(para) > self.chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) > self.chunk_size:
                        if current_chunk:
                            yield current_chunk.strip()
                        current_chunk = sentence
                    else:
                        current_chunk += " " + sentence
            else:
                if len(current_chunk) + len(para) > self.chunk_size:
                    if current_chunk:
                        yield current_chunk.strip()
                    current_chunk = para
                else:
                    current_chunk += "\n\n" + para
        
        if current_chunk:
            yield current_chunk.strip()
    
    def ingest_json_file(
        self,
        filepath: str,
        source_name: str,
        category: str = None
    ) -> int:
        """
        Ingest a JSON file of medical content.
        
        Args:
            filepath: Path to JSON file
            source_name: Name of the source (e.g., 'cdc', 'nhs')
            category: Optional category override
            
        Returns:
            Number of documents ingested
        """
        logger.info(f"Ingesting JSON file: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON file: {e}")
            self.stats.errors.append(f"Failed to load {filepath}: {str(e)}")
            return 0
        
        documents_ingested = 0
        
        # Handle different JSON structures
        if isinstance(data, dict):
            # Could be structured content or single document
            if 'content' in data:
                # Single document
                count = self._process_single_entry(data, source_name, category)
                documents_ingested += count
            else:
                # Multiple documents keyed by ID
                for key, entry in data.items():
                    if isinstance(entry, dict):
                        entry['_key'] = key
                        count = self._process_single_entry(entry, source_name, category)
                        documents_ingested += count
        
        elif isinstance(data, list):
            # List of documents
            for entry in data:
                if isinstance(entry, dict):
                    count = self._process_single_entry(entry, source_name, category)
                    documents_ingested += count
        
        # Update stats
        if source_name not in self.stats.by_source:
            self.stats.by_source[source_name] = 0
        self.stats.by_source[source_name] += documents_ingested
        
        logger.info(f"Ingested {documents_ingested} documents from {filepath}")
        return documents_ingested
    
    def _process_single_entry(
        self,
        entry: Dict[str, Any],
        source_name: str,
        category: str = None
    ) -> int:
        """Process a single entry from JSON data."""
        # Extract fields
        title = entry.get('title') or entry.get('name') or entry.get('_key', 'Untitled')
        content = entry.get('content') or entry.get('description') or entry.get('text', '')
        url = entry.get('url', '')
        symptoms = entry.get('symptoms', [])
        urgency = entry.get('urgency') or entry.get('severity')
        
        if not content:
            return 0
        
        # Determine category
        if not category:
            category = self._categorize_content(title, content)
        
        # Extract additional symptoms
        all_symptoms = self._extract_symptoms(content, symptoms)
        
        # Chunk content
        chunks = list(self._chunk_content(content))
        
        documents_added = 0
        
        for i, chunk in enumerate(chunks):
            doc_id = self._generate_id(source_name, title, i)
            
            # Create medical document
            med_doc = MedicalDocument(
                id=doc_id,
                title=title,
                content=chunk,
                source=source_name,
                category=category,
                symptoms=all_symptoms,
                urgency=urgency,
                metadata={
                    "url": url,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "ingested_at": datetime.utcnow().isoformat()
                }
            )
            
            # Generate embedding and add to vector store
            try:
                embedding = self.embedding_service.embed(chunk)
                doc = med_doc.to_document()
                self.vector_store.add(doc, embedding)
                documents_added += 1
                self.stats.successful += 1
            except Exception as e:
                logger.error(f"Failed to add document {doc_id}: {e}")
                self.stats.failed += 1
        
        self.stats.total_documents += len(chunks)
        return documents_added
    
    def ingest_red_flags(self, filepath: str = None) -> int:
        """
        Ingest red flags emergency definitions.
        
        Args:
            filepath: Path to red_flags.json (uses default if not provided)
            
        Returns:
            Number of documents ingested
        """
        if filepath is None:
            # Use default path
            api_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            filepath = os.path.join(api_dir, 'config', 'red_flags.json')
        
        logger.info(f"Ingesting red flags from: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load red flags: {e}")
            self.stats.errors.append(f"Failed to load red flags: {str(e)}")
            return 0
        
        documents_ingested = 0
        
        # Process red flags by category
        red_flags = data.get('red_flags', {})
        
        for category, urgency_levels in red_flags.items():
            for urgency_level, flags in urgency_levels.items():
                for flag in flags:
                    # Create a searchable document for each red flag
                    symptom = flag.get('symptom', '')
                    display_name = flag.get('display_name', symptom)
                    context_indicators = flag.get('context_indicators', [])
                    questions = flag.get('questions', [])
                    potential_conditions = flag.get('potential_conditions', [])
                    immediate_action = flag.get('immediate_action', '')
                    
                    # Build content for embedding
                    content_parts = [
                        f"Emergency Symptom: {display_name}",
                        f"Category: {category}",
                        f"Urgency: {urgency_level}",
                        f"Context indicators: {', '.join(context_indicators)}",
                        f"Questions to ask: {' '.join(questions)}",
                    ]
                    
                    if potential_conditions:
                        content_parts.append(f"Potential conditions: {', '.join(potential_conditions)}")
                    
                    if immediate_action:
                        content_parts.append(f"Immediate action: {immediate_action}")
                    
                    content = '\n'.join(content_parts)
                    
                    doc_id = self._generate_id('red_flags', f"{category}_{symptom}", 0)
                    
                    med_doc = MedicalDocument(
                        id=doc_id,
                        title=display_name,
                        content=content,
                        source='red_flags',
                        category=category,
                        symptoms=[symptom, display_name] + context_indicators[:5],
                        urgency=urgency_level,
                        metadata={
                            "flag_id": flag.get('id', ''),
                            "context_indicators": context_indicators,
                            "questions": questions,
                            "potential_conditions": potential_conditions,
                            "immediate_action": immediate_action,
                            "is_red_flag": True,
                            "ingested_at": datetime.utcnow().isoformat()
                        }
                    )
                    
                    try:
                        embedding = self.embedding_service.embed(content)
                        doc = med_doc.to_document()
                        self.vector_store.add(doc, embedding)
                        documents_ingested += 1
                        self.stats.successful += 1
                    except Exception as e:
                        logger.error(f"Failed to add red flag document: {e}")
                        self.stats.failed += 1
        
        # Update stats
        if 'red_flags' not in self.stats.by_source:
            self.stats.by_source['red_flags'] = 0
        self.stats.by_source['red_flags'] += documents_ingested
        self.stats.total_documents += documents_ingested
        
        logger.info(f"Ingested {documents_ingested} red flag documents")
        return documents_ingested
    
    def ingest_directory(
        self,
        directory: str,
        source_name: str = None,
        recursive: bool = True
    ) -> int:
        """
        Ingest all JSON files from a directory.
        
        Args:
            directory: Directory path
            source_name: Source name (uses directory name if not provided)
            recursive: Whether to search subdirectories
            
        Returns:
            Total documents ingested
        """
        if source_name is None:
            source_name = os.path.basename(directory)
        
        logger.info(f"Ingesting directory: {directory}")
        
        total_ingested = 0
        
        if recursive:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith('.json'):
                        filepath = os.path.join(root, file)
                        # Use subdirectory as category if present
                        rel_path = os.path.relpath(root, directory)
                        category = rel_path if rel_path != '.' else None
                        
                        count = self.ingest_json_file(filepath, source_name, category)
                        total_ingested += count
        else:
            for file in os.listdir(directory):
                if file.endswith('.json'):
                    filepath = os.path.join(directory, file)
                    count = self.ingest_json_file(filepath, source_name)
                    total_ingested += count
        
        return total_ingested
    
    def ingest_all_medical_knowledge(
        self,
        project_root: str,
        include_downloaded: bool = True,
        include_red_flags: bool = True
    ) -> IngestionStats:
        """
        Ingest all available medical knowledge.
        
        Args:
            project_root: Root directory of the project
            include_downloaded: Whether to include downloaded content
            include_red_flags: Whether to include red flags
            
        Returns:
            Ingestion statistics
        """
        logger.info("Starting full medical knowledge ingestion...")
        
        # Reset stats
        self.stats = IngestionStats()
        
        # Ingest red flags
        if include_red_flags:
            self.ingest_red_flags()
        
        # Ingest downloaded content
        if include_downloaded:
            knowledge_dir = os.path.join(project_root, 'datasets', 'medical_knowledge')
            
            if os.path.exists(knowledge_dir):
                for source_dir in os.listdir(knowledge_dir):
                    source_path = os.path.join(knowledge_dir, source_dir)
                    if os.path.isdir(source_path):
                        self.ingest_directory(source_path, source_name=source_dir)
        
        logger.info(f"Ingestion complete: {self.stats.to_dict()}")
        return self.stats
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current ingestion statistics."""
        return self.stats.to_dict()


# Singleton instance
_ingestor = None


def get_medical_knowledge_ingestor() -> MedicalKnowledgeIngestor:
    """Get or create the singleton MedicalKnowledgeIngestor instance."""
    global _ingestor
    if _ingestor is None:
        _ingestor = MedicalKnowledgeIngestor()
    return _ingestor
