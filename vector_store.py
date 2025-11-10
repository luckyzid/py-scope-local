"""Vector store management with Qdrant."""
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from config import config
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np


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

        # Initialize CLIP for image embeddings
        self.clip_model = CLIPModel.from_pretrained(config.image.embedding_model)
        self.clip_processor = CLIPProcessor.from_pretrained(config.image.embedding_model)
        self.image_collection_name = config.image.collection_name
        self.image_vector_store: Optional[Qdrant] = None
    
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

    def create_image_collection(self, vector_size: int = 512):
        """Create Qdrant collection for images if not exists.

        Args:
            vector_size: Dimension of CLIP embedding vectors (512 for ViT-B/32)
        """
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if self.image_collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.image_collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"Created image collection: {self.image_collection_name}")
        else:
            print(f"Image collection {self.image_collection_name} already exists")
    
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

    def add_images(self, images: List[Tuple[Image.Image, dict]]):
        """Add images to vector store.

        Args:
            images: List of tuples containing PIL Image and metadata
        """
        print(f"Adding {len(images)} images to vector store...")

        # Create image collection if not exists
        self.create_image_collection()

        # Generate embeddings for images
        image_embeddings = []
        metadatas = []

        for image, metadata in images:
            # Generate CLIP embedding
            inputs = self.clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                # Normalize the features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                embedding = image_features.squeeze().numpy()

            image_embeddings.append(embedding)
            metadatas.append(metadata)

        # Create documents for Qdrant (using dummy text since we're storing image embeddings)
        image_documents = []
        for i, (embedding, metadata) in enumerate(zip(image_embeddings, metadatas)):
            doc = Document(
                page_content=f"Image {i}",  # Dummy content
                metadata=metadata
            )
            image_documents.append(doc)

        # Add to Qdrant with pre-computed embeddings
        self.image_vector_store = Qdrant.from_documents(
            image_documents,
            embedding=self.embeddings,  # This will be overridden
            url=f"http://{config.qdrant.host}:{config.qdrant.port}",
            collection_name=self.image_collection_name,
            force_recreate=False
        )

        # Manually add vectors since Qdrant.from_documents doesn't support pre-computed embeddings well
        from qdrant_client.models import PointStruct
        points = []
        for i, (embedding, metadata) in enumerate(zip(image_embeddings, metadatas)):
            point = PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload=metadata
            )
            points.append(point)

        self.client.upsert(
            collection_name=self.image_collection_name,
            points=points
        )

        print("Images added successfully")
    
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

    def get_image_vector_store(self) -> Qdrant:
        """Get or create image vector store.

        Returns:
            Qdrant vector store instance for images
        """
        if self.image_vector_store is None:
            self.image_vector_store = Qdrant(
                client=self.client,
                collection_name=self.image_collection_name,
                embeddings=self.embeddings  # Dummy, we'll use CLIP directly
            )
        return self.image_vector_store
    
    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
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

    def similarity_search_image(self, query_image: Image.Image, k: Optional[int] = None) -> List[dict]:
        """Search for similar images.

        Args:
            query_image: PIL Image to search for
            k: Number of results to return

        Returns:
            List of similar image metadata
        """
        if k is None:
            k = config.rag.top_k_results

        # Generate embedding for query image
        inputs = self.clip_processor(images=query_image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            # Normalize the features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            query_embedding = image_features.squeeze().numpy()

        # Search in Qdrant
        search_result = self.client.search(
            collection_name=self.image_collection_name,
            query_vector=query_embedding.tolist(),
            limit=k
        )

        # Convert to dict format
        results = []
        for hit in search_result:
            result = {
                "metadata": hit.payload,
                "score": hit.score,
                "id": hit.id
            }
            results.append(result)

        return results
    
    def delete_collection(self):
        """Delete the collection."""
        self.client.delete_collection(collection_name=self.collection_name)
        print(f"Deleted collection: {self.collection_name}")

    def delete_image_collection(self):
        """Delete the image collection."""
        self.client.delete_collection(collection_name=self.image_collection_name)
        print(f"Deleted image collection: {self.image_collection_name}")


if __name__ == "__main__":
    # Test vector store
    manager = VectorStoreManager()
    manager.create_collection()
