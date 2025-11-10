"""Vector store management with Qdrant."""
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from config import config


class VectorStoreManager:
    """Manage Qdrant vector store."""
    
    def __init__(self):
        """Initialize vector store manager."""
        self.client = QdrantClient(
            host=config.qdrant.host,
            port=config.qdrant.port
        )
        self.embeddings = OllamaEmbeddings(
            base_url=config.ollama.base_url,
            model=config.ollama.embedding_model
        )
        self.collection_name = config.qdrant.collection_name
        self.vector_store: Optional[Qdrant] = None
    
    def create_collection(self, vector_size: int = 768):
        """Create Qdrant collection if not exists.
        
        Args:
            vector_size: Dimension of embedding vectors (nomic-embed-text uses 768)
        """
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"Created collection: {self.collection_name}")
        else:
            print(f"Collection {self.collection_name} already exists")
    
    def add_documents(self, documents: List[Document]):
        """Add documents to vector store.
        
        Args:
            documents: List of documents to add
        """
        print(f"Adding {len(documents)} documents to vector store...")
        
        # Create collection if not exists
        self.create_collection()
        
        # Add documents to Qdrant
        self.vector_store = Qdrant.from_documents(
            documents,
            self.embeddings,
            url=f"http://{config.qdrant.host}:{config.qdrant.port}",
            collection_name=self.collection_name,
            force_recreate=False
        )
        
        print("Documents added successfully")
    
    def get_vector_store(self) -> Qdrant:
        """Get or create vector store.
        
        Returns:
            Qdrant vector store instance
        """
        if self.vector_store is None:
            self.vector_store = Qdrant(
                client=self.client,
                collection_name=self.collection_name,
                embeddings=self.embeddings
            )
        return self.vector_store
    
    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        if k is None:
            k = config.rag.top_k_results
        
        vector_store = self.get_vector_store()
        return vector_store.similarity_search(query, k=k)
    
    def delete_collection(self):
        """Delete the collection."""
        self.client.delete_collection(collection_name=self.collection_name)
        print(f"Deleted collection: {self.collection_name}")


if __name__ == "__main__":
    # Test vector store
    manager = VectorStoreManager()
    manager.create_collection()
