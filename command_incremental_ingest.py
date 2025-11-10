"""Incremental document ingestion with change detection."""
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Set
from datetime import datetime
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from logger import rag_logger


class IncrementalIngestion:
    """Manage incremental document updates."""
    
    def __init__(self, index_file: str = ".document_index.json"):
        """Initialize incremental ingestion.
        
        Args:
            index_file: File to store document metadata
        """
        self.index_file = Path(index_file)
        self.document_index: Dict[str, Dict] = {}
        self.load_index()
        
        rag_logger.info(f"Incremental ingestion initialized ({len(self.document_index)} docs tracked)")
    
    def load_index(self):
        """Load document index from file."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    self.document_index = json.load(f)
                rag_logger.debug(f"Loaded index: {len(self.document_index)} documents")
            except Exception as e:
                rag_logger.warning(f"Failed to load index: {e}")
                self.document_index = {}
        else:
            self.document_index = {}
    
    def save_index(self):
        """Save document index to file."""
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.document_index, f, indent=2, ensure_ascii=False)
        rag_logger.debug(f"Saved index: {len(self.document_index)} documents")
    
    def compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hash string
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def get_file_metadata(self, file_path: Path) -> Dict:
        """Get file metadata.
        
        Args:
            file_path: Path to file
            
        Returns:
            Metadata dict
        """
        stat = file_path.stat()
        return {
            'path': str(file_path),
            'size': stat.st_size,
            'modified': stat.st_mtime,
            'hash': self.compute_file_hash(str(file_path)),
            'last_indexed': datetime.now().isoformat()
        }
    
    def detect_changes(self, pdf_directory: str) -> Dict[str, List[str]]:
        """Detect document changes.
        
        Args:
            pdf_directory: Directory containing PDFs
            
        Returns:
            Dict with 'new', 'modified', 'deleted' lists
        """
        pdf_dir = Path(pdf_directory)
        current_files = set(str(f) for f in pdf_dir.glob("*.pdf"))
        indexed_files = set(self.document_index.keys())
        
        changes = {
            'new': [],
            'modified': [],
            'deleted': [],
            'unchanged': []
        }
        
        # Check for new and modified files
        for file_path_str in current_files:
            file_path = Path(file_path_str)
            
            if file_path_str not in indexed_files:
                # New file
                changes['new'].append(file_path_str)
            else:
                # Check if modified
                current_hash = self.compute_file_hash(file_path_str)
                indexed_hash = self.document_index[file_path_str].get('hash', '')
                
                if current_hash != indexed_hash:
                    changes['modified'].append(file_path_str)
                else:
                    changes['unchanged'].append(file_path_str)
        
        # Check for deleted files
        changes['deleted'] = list(indexed_files - current_files)
        
        rag_logger.info(
            f"Change detection: {len(changes['new'])} new, "
            f"{len(changes['modified'])} modified, "
            f"{len(changes['deleted'])} deleted"
        )
        
        return changes
    
    def process_incremental_updates(
        self,
        pdf_directory: str,
        force_reindex: bool = False
    ) -> Dict:
        """Process incremental updates.
        
        Args:
            pdf_directory: Directory containing PDFs
            force_reindex: Force reindexing of all documents
            
        Returns:
            Summary of processing
        """
        start_time = datetime.now()
        
        if force_reindex:
            rag_logger.info("Force reindex requested, processing all documents")
            self.document_index = {}
        
        # Detect changes
        changes = self.detect_changes(pdf_directory)
        
        # Files to process
        files_to_process = changes['new'] + changes['modified']
        
        if not files_to_process and not changes['deleted']:
            rag_logger.info("No changes detected, skipping ingestion")
            return {
                'status': 'skipped',
                'reason': 'no changes',
                'changes': changes
            }
        
        # Initialize components
        processor = DocumentProcessor()
        vector_store = VectorStoreManager()
        
        # Process new and modified files
        if files_to_process:
            rag_logger.info(f"Processing {len(files_to_process)} files...")
            
            documents = []
            for file_path in files_to_process:
                try:
                    text = processor.load_pdf(file_path)
                    if text:
                        from langchain_core.documents import Document
                        doc = Document(
                            page_content=text,
                            metadata={
                                'source': file_path,
                                'filename': Path(file_path).name
                            }
                        )
                        documents.append(doc)
                        
                        # Update index
                        self.document_index[file_path] = self.get_file_metadata(Path(file_path))
                    
                except Exception as e:
                    rag_logger.error(f"Failed to process {file_path}: {e}")
            
            if documents:
                # Split and add to vector store
                chunks = processor.split_documents(documents)
                vector_store.add_documents(chunks)
                rag_logger.info(f"Added {len(chunks)} chunks to vector store")
        
        # Handle deleted files
        if changes['deleted']:
            rag_logger.info(f"Removing {len(changes['deleted'])} deleted files from index")
            for file_path in changes['deleted']:
                del self.document_index[file_path]
            # Note: Qdrant doesn't easily support deletion by metadata
            # In production, you'd need to implement proper deletion
        
        # Save index
        self.save_index()
        
        duration = (datetime.now() - start_time).total_seconds()
        
        summary = {
            'status': 'completed',
            'duration': duration,
            'changes': changes,
            'processed': len(files_to_process),
            'total_indexed': len(self.document_index)
        }
        
        rag_logger.info(f"Incremental ingestion completed in {duration:.2f}s")
        return summary
    
    def get_index_status(self) -> Dict:
        """Get current index status.
        
        Returns:
            Index status dict
        """
        return {
            'total_documents': len(self.document_index),
            'documents': list(self.document_index.keys()),
            'index_file': str(self.index_file),
            'last_update': max(
                (doc['last_indexed'] for doc in self.document_index.values()),
                default='Never'
            )
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Incremental document ingestion")
    parser.add_argument(
        "--pdf-path",
        type=str,
        default="./pdf",
        help="PDF directory"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reindex all documents"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show index status"
    )
    
    args = parser.parse_args()
    
    ingestion = IncrementalIngestion()
    
    if args.status:
        status = ingestion.get_index_status()
        print(json.dumps(status, indent=2, ensure_ascii=False))
    else:
        summary = ingestion.process_incremental_updates(
            args.pdf_path,
            force_reindex=args.force
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
