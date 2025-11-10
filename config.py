"""Configuration management for RAG pipeline."""
import os
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class OllamaConfig(BaseModel):
    """Ollama configuration."""
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model: str = os.getenv("OLLAMA_MODEL", "exaone3.5:2.4b")
    embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")


class QdrantConfig(BaseModel):
    """Qdrant configuration."""
    host: str = os.getenv("QDRANT_HOST", "localhost")
    port: int = int(os.getenv("QDRANT_PORT", "6333"))
    collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "documents")


class RedisConfig(BaseModel):
    """Redis configuration."""
    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", "6379"))
    ttl: int = int(os.getenv("REDIS_TTL", "3600"))


class DocumentConfig(BaseModel):
    """Document processing configuration."""
    pdf_path: str = os.getenv("PDF_PATH", "./pdf")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))


class RAGConfig(BaseModel):
    """RAG pipeline configuration."""
    top_k_results: int = int(os.getenv("TOP_K_RESULTS", "5"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))


class Config(BaseModel):
    """Main configuration."""
    ollama: OllamaConfig = OllamaConfig()
    qdrant: QdrantConfig = QdrantConfig()
    redis: RedisConfig = RedisConfig()
    document: DocumentConfig = DocumentConfig()
    rag: RAGConfig = RAGConfig()


# Global config instance
config = Config()
