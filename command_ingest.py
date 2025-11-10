"""Ingest documents into vector store."""
import argparse
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager


def ingest_documents(pdf_path: str = None, reset: bool = False):
    """Ingest documents from PDF directory.
    
    Args:
        pdf_path: Path to PDF directory
        reset: Whether to reset the collection
    """
    print("=== Starting Document Ingestion ===")
    
    # Initialize components
    processor = DocumentProcessor()
    vector_store = VectorStoreManager()
    
    # Reset collection if requested
    if reset:
        try:
            vector_store.delete_collection()
            print("Collection reset successfully")
        except Exception as e:
            print(f"No existing collection to delete: {e}")
    
    # Process documents
    print("\n1. Processing PDF documents...")
    documents = processor.process_directory(pdf_path)
    
    print("\n2. Splitting documents into chunks...")
    chunks = processor.split_documents(documents)
    
    print("\n3. Creating embeddings and storing in Qdrant...")
    vector_store.add_documents(chunks)
    
    print("\n=== Ingestion Complete ===")
    print(f"Total documents processed: {len(documents)}")
    print(f"Total chunks created: {len(chunks)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into vector store")
    parser.add_argument(
        "--pdf-path",
        type=str,
        default=None,
        help="Path to PDF directory (default: from config)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the collection before ingestion"
    )
    
    args = parser.parse_args()
    ingest_documents(args.pdf_path, args.reset)
