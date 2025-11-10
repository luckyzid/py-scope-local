"""Ingest documents into vector store."""
import argparse
from typing import Optional
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager


def ingest_documents(pdf_path: Optional[str] = None, reset: bool = False, include_images: bool = True):
    """Ingest documents and images from PDF directory.

    Args:
        pdf_path: Path to PDF directory
        reset: Whether to reset the collection
        include_images: Whether to extract and store images
    """
    print("=== Starting Document Ingestion ===")

    # Initialize components
    processor = DocumentProcessor()
    vector_store = VectorStoreManager()

    # Reset collections if requested
    if reset:
        try:
            vector_store.delete_collection()
            print("Text collection reset successfully")
        except Exception as e:
            print(f"No existing text collection to delete: {e}")

        if include_images:
            try:
                vector_store.delete_image_collection()
                print("Image collection reset successfully")
            except Exception as e:
                print(f"No existing image collection to delete: {e}")

    # Process text documents
    print("\n1. Processing PDF documents...")
    documents = processor.process_directory(pdf_path)

    print("\n2. Splitting documents into chunks...")
    chunks = processor.split_documents(documents)

    print("\n3. Creating text embeddings and storing in Qdrant...")
    vector_store.add_documents(chunks)

    # Process images if requested
    images = []
    if include_images:
        print("\n4. Extracting images from PDFs...")
        images = processor.process_images_directory(pdf_path)

        if images:
            print("\n5. Creating image embeddings and storing in Qdrant...")
            vector_store.add_images(images)
        else:
            print("\n5. No images found in PDFs")

    print("\n=== Ingestion Complete ===")
    print(f"Total documents processed: {len(documents)}")
    print(f"Total chunks created: {len(chunks)}")
    if include_images:
        print(f"Total images extracted: {len(images)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into vector store")
    parser.add_argument(
        "--pdf-path",
        type=str,
        default="",
        help="Path to PDF directory (default: from config)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the collection before ingestion"
    )
    
    args = parser.parse_args()
    ingest_documents(args.pdf_path if args.pdf_path else None, args.reset)
