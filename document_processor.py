"""Document processing using Docling."""
import os
from pathlib import Path
from typing import List, Tuple
from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import config
from tqdm import tqdm
from PIL import Image
import io


class DocumentProcessor:
    """Process documents using Docling."""
    
    def __init__(self):
        """Initialize document processor."""
        self.converter = DocumentConverter()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.document.chunk_size,
            chunk_overlap=config.document.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_pdf(self, pdf_path: str) -> str:
        """Load PDF using Docling.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        try:
            result = self.converter.convert(pdf_path)
            return result.document.export_to_markdown()
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            return ""
    
    def process_directory(self, directory_path: str = None) -> List[Document]:
        """Process all PDFs in directory.
        
        Args:
            directory_path: Directory containing PDF files
            
        Returns:
            List of processed documents
        """
        if directory_path is None:
            directory_path = config.document.pdf_path
        
        pdf_dir = Path(directory_path)
        if not pdf_dir.exists():
            raise ValueError(f"Directory {directory_path} does not exist")
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in {directory_path}")
        
        print(f"Found {len(pdf_files)} PDF files")
        
        documents = []
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            text = self.load_pdf(str(pdf_file))
            if text:
                # Create document with metadata
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": str(pdf_file),
                        "filename": pdf_file.name
                    }
                )
                documents.append(doc)
        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks.

        Args:
            documents: List of documents

        Returns:
            List of chunked documents
        """
        chunks = self.text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks

    def extract_images_from_pdf(self, pdf_path: str) -> List[Tuple[Image.Image, dict]]:
        """Extract images from PDF using Docling.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of tuples containing PIL Image and metadata
        """
        try:
            result = self.converter.convert(pdf_path)
            images = []

            # Extract images from document
            for item in result.document.body.items:
                if hasattr(item, 'image') and item.image:
                    # Convert image bytes to PIL Image
                    image_bytes = item.image.get_image()
                    if image_bytes:
                        image = Image.open(io.BytesIO(image_bytes))
                        # Resize if too large
                        if max(image.size) > config.image.max_image_size:
                            image.thumbnail((config.image.max_image_size, config.image.max_image_size))
                        metadata = {
                            "source": pdf_path,
                            "filename": Path(pdf_path).name,
                            "page": getattr(item, 'prov', [None])[0].page_no if hasattr(item, 'prov') and item.prov else None
                        }
                        images.append((image, metadata))

            return images
        except Exception as e:
            print(f"Error extracting images from {pdf_path}: {e}")
            return []

    def process_images_directory(self, directory_path: str = None) -> List[Tuple[Image.Image, dict]]:
        """Process all images from PDFs in directory.

        Args:
            directory_path: Directory containing PDF files

        Returns:
            List of image tuples with metadata
        """
        if directory_path is None:
            directory_path = config.document.pdf_path

        pdf_dir = Path(directory_path)
        if not pdf_dir.exists():
            raise ValueError(f"Directory {directory_path} does not exist")

        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in {directory_path}")

        print(f"Extracting images from {len(pdf_files)} PDF files...")

        all_images = []
        for pdf_file in tqdm(pdf_files, desc="Extracting images"):
            images = self.extract_images_from_pdf(str(pdf_file))
            all_images.extend(images)

        print(f"Extracted {len(all_images)} images total")
        return all_images


if __name__ == "__main__":
    # Test document processing
    processor = DocumentProcessor()
    docs = processor.process_directory()
    chunks = processor.split_documents(docs)
    print(f"Total chunks: {len(chunks)}")
    if chunks:
        print(f"Sample chunk: {chunks[0].page_content[:200]}...")
