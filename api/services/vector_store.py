"""
Vector Store Service for RAG-based medical knowledge retrieval.

This module provides a vector database interface for storing and retrieving
medical documents using semantic similarity search.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a document in the vector store."""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict:
        """Convert document to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "embedding": self.embedding
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Document":
        """Create document from dictionary."""
        return cls(
            id=data.get("id", ""),
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding")
        )


class VectorStore:
    """
    Vector store for medical knowledge retrieval.
    
    Supports multiple backends: ChromaDB (default), FAISS, or in-memory.
    Provides semantic similarity search for RAG-based fallback predictions.
    """
    
    def __init__(
        self,
        backend: str = "chromadb",
        persist_directory: str = None,
        collection_name: str = "medical_knowledge",
        embedding_dimension: int = 384
    ):
        """
        Initialize the vector store.
        
        Args:
            backend: Vector database backend ("chromadb", "faiss", or "memory")
            persist_directory: Directory to persist the vector store
            collection_name: Name of the collection to use
            embedding_dimension: Dimension of the embedding vectors
        """
        self.backend = backend
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self.persist_directory = persist_directory or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "chroma_db"
        )
        
        # Initialize the appropriate backend
        self._client = None
        self._collection = None
        self._documents: Dict[str, Document] = {}  # For in-memory fallback
        
        if backend == "chromadb":
            self._init_chromadb()
        elif backend == "faiss":
            self._init_faiss()
        else:
            logger.info("Using in-memory vector store")
    
    def _init_chromadb(self):
        """Initialize ChromaDB backend."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Ensure persist directory exists
            os.makedirs(self.persist_directory, exist_ok=True)
            
            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"ChromaDB initialized with {self._collection.count()} documents")
            
        except ImportError:
            logger.warning("ChromaDB not installed. Falling back to in-memory store.")
            self.backend = "memory"
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}. Falling back to in-memory store.")
            self.backend = "memory"
    
    def _init_faiss(self):
        """Initialize FAISS backend."""
        try:
            import faiss
            import numpy as np
            
            self._faiss_index = faiss.IndexFlatIP(self.embedding_dimension)
            self._faiss_id_map = {}
            logger.info("FAISS index initialized")
            
        except ImportError:
            logger.warning("FAISS not installed. Falling back to in-memory store.")
            self.backend = "memory"
    
    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None
    ) -> int:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
            embeddings: Optional pre-computed embeddings for each document
            
        Returns:
            Number of documents added
        """
        if not documents:
            return 0
        
        if self.backend == "chromadb":
            return self._add_to_chromadb(documents, embeddings)
        elif self.backend == "faiss":
            return self._add_to_faiss(documents, embeddings)
        else:
            return self._add_to_memory(documents, embeddings)
    
    def _add_to_chromadb(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None
    ) -> int:
        """Add documents to ChromaDB using upsert to handle duplicates."""
        if not self._collection:
            logger.error("ChromaDB collection not initialized")
            return 0
        
        # Filter out documents with empty content
        valid_docs = [(doc, emb) for doc, emb in zip(documents, embeddings or [None]*len(documents))
                      if doc.content and doc.content.strip()]
        
        if not valid_docs:
            return 0
        
        ids = [doc.id for doc, _ in valid_docs]
        contents = [doc.content for doc, _ in valid_docs]
        metadatas = [doc.metadata for doc, _ in valid_docs]
        valid_embeddings = [emb for _, emb in valid_docs if emb is not None]
        
        try:
            if valid_embeddings:
                # Use upsert to handle duplicates (update if exists, add if new)
                self._collection.upsert(
                    ids=ids,
                    documents=contents,
                    metadatas=metadatas,
                    embeddings=valid_embeddings
                )
            else:
                # Let ChromaDB handle embedding (uses default embedding function)
                self._collection.upsert(
                    ids=ids,
                    documents=contents,
                    metadatas=metadatas
                )
            
            logger.info(f"Added/updated {len(valid_docs)} documents in ChromaDB")
            return len(valid_docs)
            
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
            # Try adding one by one to identify problematic documents
            added = 0
            for i, (doc, emb) in enumerate(valid_docs):
                try:
                    if emb:
                        self._collection.upsert(
                            ids=[doc.id],
                            documents=[doc.content],
                            metadatas=[doc.metadata],
                            embeddings=[emb]
                        )
                    else:
                        self._collection.upsert(
                            ids=[doc.id],
                            documents=[doc.content],
                            metadatas=[doc.metadata]
                        )
                    added += 1
                except Exception as doc_error:
                    logger.warning(f"Failed to index document {doc.id}: {doc_error}")
            return added
    
    def _add_to_faiss(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None
    ) -> int:
        """Add documents to FAISS index."""
        if embeddings is None:
            logger.error("FAISS requires pre-computed embeddings")
            return 0
        
        import numpy as np
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings_array)
        
        # Add to index
        start_id = len(self._faiss_id_map)
        self._faiss_index.add(embeddings_array)
        
        # Update ID mapping
        for i, doc in enumerate(documents):
            self._faiss_id_map[start_id + i] = doc
            self._documents[doc.id] = doc
        
        return len(documents)
    
    def _add_to_memory(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None
    ) -> int:
        """Add documents to in-memory store."""
        for i, doc in enumerate(documents):
            if embeddings and i < len(embeddings):
                doc.embedding = embeddings[i]
            self._documents[doc.id] = doc
        
        logger.info(f"Added {len(documents)} documents to in-memory store")
        return len(documents)
    
    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Document]:
        """
        Find similar documents using vector similarity.
        
        Args:
            query_embedding: Embedding vector of the query
            k: Number of results to return
            filters: Optional metadata filters to apply
            
        Returns:
            List of similar Document objects
        """
        if self.backend == "chromadb":
            return self._search_chromadb(query_embedding, k, filters)
        elif self.backend == "faiss":
            return self._search_faiss(query_embedding, k)
        else:
            return self._search_memory(query_embedding, k, filters)
    
    def _search_chromadb(
        self,
        query_embedding: List[float],
        k: int,
        filters: Optional[Dict] = None
    ) -> List[Document]:
        """Search ChromaDB for similar documents."""
        if not self._collection:
            return []
        
        try:
            # Build query parameters
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": k
            }
            
            if filters:
                query_params["where"] = filters
            
            results = self._collection.query(**query_params)
            
            documents = []
            if results and results.get("ids"):
                for i, doc_id in enumerate(results["ids"][0]):
                    doc = Document(
                        id=doc_id,
                        content=results["documents"][0][i] if results.get("documents") else "",
                        metadata=results["metadatas"][0][i] if results.get("metadatas") else {},
                        embedding=results["embeddings"][0][i] if results.get("embeddings") else None
                    )
                    # Add distance score to metadata
                    if results.get("distances"):
                        doc.metadata["distance"] = results["distances"][0][i]
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            return []
    
    def _search_faiss(
        self,
        query_embedding: List[float],
        k: int
    ) -> List[Document]:
        """Search FAISS index for similar documents."""
        import numpy as np
        
        query_array = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_array)
        
        distances, indices = self._faiss_index.search(query_array, k)
        
        documents = []
        for i, idx in enumerate(indices[0]):
            if idx in self._faiss_id_map:
                doc = self._faiss_id_map[idx]
                doc.metadata["distance"] = float(distances[0][i])
                documents.append(doc)
        
        return documents
    
    def _search_memory(
        self,
        query_embedding: List[float],
        k: int,
        filters: Optional[Dict] = None
    ) -> List[Document]:
        """Search in-memory documents using cosine similarity."""
        import numpy as np
        
        query_array = np.array(query_embedding)
        query_norm = np.linalg.norm(query_array)
        
        if query_norm == 0:
            return []
        
        similarities = []
        
        for doc_id, doc in self._documents.items():
            # Apply filters if provided
            if filters:
                match = all(
                    doc.metadata.get(key) == value
                    for key, value in filters.items()
                )
                if not match:
                    continue
            
            if doc.embedding:
                doc_array = np.array(doc.embedding)
                doc_norm = np.linalg.norm(doc_array)
                
                if doc_norm > 0:
                    similarity = np.dot(query_array, doc_array) / (query_norm * doc_norm)
                    similarities.append((doc, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k documents
        result = []
        for doc, sim in similarities[:k]:
            doc.metadata["distance"] = 1.0 - sim  # Convert similarity to distance
            result.append(doc)
        
        return result
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve a specific document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document if found, None otherwise
        """
        if self.backend == "chromadb" and self._collection:
            try:
                results = self._collection.get(ids=[doc_id])
                if results and results.get("ids"):
                    return Document(
                        id=results["ids"][0],
                        content=results["documents"][0] if results.get("documents") else "",
                        metadata=results["metadatas"][0] if results.get("metadatas") else {}
                    )
            except Exception as e:
                logger.error(f"Failed to get document {doc_id}: {e}")
        
        return self._documents.get(doc_id)
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the store.
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            True if deleted, False otherwise
        """
        if self.backend == "chromadb" and self._collection:
            try:
                self._collection.delete(ids=[doc_id])
                logger.info(f"Deleted document {doc_id} from ChromaDB")
                return True
            except Exception as e:
                logger.error(f"Failed to delete document {doc_id}: {e}")
                return False
        
        if doc_id in self._documents:
            del self._documents[doc_id]
            return True
        
        return False
    
    def clear(self) -> bool:
        """
        Clear all documents from the store.
        
        Returns:
            True if successful, False otherwise
        """
        if self.backend == "chromadb" and self._client:
            try:
                self._client.delete_collection(self.collection_name)
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info("Cleared ChromaDB collection")
                return True
            except Exception as e:
                logger.error(f"Failed to clear ChromaDB: {e}")
                return False
        
        self._documents.clear()
        return True
    
    def count(self) -> int:
        """
        Get the number of documents in the store.
        
        Returns:
            Number of documents
        """
        if self.backend == "chromadb" and self._collection:
            return self._collection.count()
        
        return len(self._documents)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with store statistics
        """
        stats = {
            "backend": self.backend,
            "collection_name": self.collection_name,
            "document_count": self.count(),
            "embedding_dimension": self.embedding_dimension
        }
        
        if self.backend == "chromadb":
            stats["persist_directory"] = self.persist_directory
        
        return stats


class VectorStoreFactory:
    """Factory for creating vector store instances."""
    
    @staticmethod
    def create(
        backend: str = "chromadb",
        persist_directory: str = None,
        collection_name: str = "medical_knowledge",
        embedding_dimension: int = 384
    ) -> VectorStore:
        """
        Create a vector store instance.
        
        Args:
            backend: Backend type ("chromadb", "faiss", or "memory")
            persist_directory: Directory for persistence
            collection_name: Name of the collection
            embedding_dimension: Dimension of embeddings
            
        Returns:
            VectorStore instance
        """
        return VectorStore(
            backend=backend,
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_dimension=embedding_dimension
        )
