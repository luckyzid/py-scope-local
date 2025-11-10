"""Advanced logging system with structured logging and metrics."""
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from loguru import logger
import json


class RAGLogger:
    """Centralized logging system for RAG pipeline."""
    
    def __init__(self, log_dir: str = "./logs"):
        """Initialize logger.
        
        Args:
            log_dir: Directory for log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Remove default handler
        logger.remove()
        
        # Console handler with color
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
            level="INFO",
            colorize=True
        )
        
        # File handler - all logs
        logger.add(
            self.log_dir / "rag_{time:YYYY-MM-DD}.log",
            rotation="00:00",
            retention="30 days",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        )
        
        # File handler - errors only
        logger.add(
            self.log_dir / "errors_{time:YYYY-MM-DD}.log",
            rotation="00:00",
            retention="90 days",
            level="ERROR",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        )
        
        # Query logs (structured JSON)
        self.query_log_file = self.log_dir / "queries.jsonl"
        
        self.logger = logger
    
    def log_query(
        self,
        question: str,
        answer: str,
        sources: list,
        response_time: float,
        cache_hit: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log a query with structured data.
        
        Args:
            question: User question
            answer: System answer
            sources: List of source documents
            response_time: Response time in seconds
            cache_hit: Whether cache was used
            metadata: Additional metadata
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer[:200] + "..." if len(answer) > 200 else answer,
            "num_sources": len(sources),
            "response_time": response_time,
            "cache_hit": cache_hit,
            "metadata": metadata or {}
        }
        
        # Append to JSONL file
        with open(self.query_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        
        # Log to console
        cache_status = "CACHE" if cache_hit else "COMPUTE"
        self.logger.info(
            f"Query [{cache_status}] | Time: {response_time:.2f}s | "
            f"Sources: {len(sources)} | Q: {question[:50]}..."
        )
    
    def log_ingestion(
        self,
        num_documents: int,
        num_chunks: int,
        duration: float,
        success: bool = True
    ):
        """Log document ingestion.
        
        Args:
            num_documents: Number of documents processed
            num_chunks: Number of chunks created
            duration: Processing duration in seconds
            success: Whether ingestion succeeded
        """
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(
            f"Ingestion [{status}] | Docs: {num_documents} | "
            f"Chunks: {num_chunks} | Time: {duration:.2f}s"
        )
    
    def log_error(self, error: Exception, context: str = ""):
        """Log an error with context.
        
        Args:
            error: Exception object
            context: Additional context
        """
        self.logger.error(f"{context} | Error: {str(error)}", exc_info=True)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)


# Global logger instance
rag_logger = RAGLogger()


if __name__ == "__main__":
    # Test logging
    rag_logger.info("Testing logger")
    rag_logger.debug("Debug message")
    rag_logger.warning("Warning message")
    rag_logger.error("Error message")
    
    # Test query logging
    rag_logger.log_query(
        question="Test question?",
        answer="Test answer",
        sources=[{"source": "test.pdf"}],
        response_time=1.23,
        cache_hit=False
    )
    
    print(f"\nLogs saved to: {rag_logger.log_dir}")
