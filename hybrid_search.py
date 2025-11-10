"""Hybrid search combining vector similarity and BM25 keyword search."""
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
import numpy as np
from logger import rag_logger


class HybridSearcher:
    """Hybrid search combining semantic and keyword search."""
    
    def __init__(self, documents: List[Document], alpha: float = 0.5):
        """Initialize hybrid searcher.
        
        Args:
            documents: List of documents to search
            alpha: Weight for vector search (1-alpha for BM25)
                  0.0 = only BM25, 1.0 = only vector search
        """
        self.documents = documents
        self.alpha = alpha
        
        # Build BM25 index
        self._build_bm25_index()
        
        rag_logger.info(f"Initialized hybrid search with {len(documents)} documents (alpha={alpha})")
    
    def _build_bm25_index(self):
        """Build BM25 index from documents."""
        # Tokenize documents
        tokenized_corpus = [
            doc.page_content.lower().split() 
            for doc in self.documents
        ]
        
        # Create BM25 index
        self.bm25 = BM25Okapi(tokenized_corpus)
        rag_logger.debug(f"Built BM25 index with {len(tokenized_corpus)} documents")
    
    def bm25_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """BM25 keyword search.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of (document, score) tuples
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_k_idx = np.argsort(scores)[-k:][::-1]
        
        results = []
        for idx in top_k_idx:
            # Include all results, even with zero scores (important for small document sets)
            results.append((self.documents[idx], float(scores[idx])))
        
        return results
    
    def hybrid_search(
        self,
        query: str,
        vector_results: List[Document],
        k: int = 5
    ) -> List[Document]:
        """Combine vector and BM25 search results.
        
        Args:
            query: Search query
            vector_results: Results from vector search (List of Documents)
            k: Number of final results
            
        Returns:
            Reranked list of documents
        """
        # Get BM25 results
        bm25_results = self.bm25_search(query, k=k*2)
        
        # Create default scores for vector results (equal weight)
        vector_scores = [1.0] * len(vector_results)
        bm25_scores_raw = [score for _, score in bm25_results]
        
        # Normalize scores
        vector_scores_norm = self._normalize_scores(vector_scores)
        bm25_scores_norm = self._normalize_scores(bm25_scores_raw)
        
        # Create score dict
        doc_scores = {}
        
        # Add vector scores
        for doc, norm_score in zip(vector_results, vector_scores_norm):
            doc_id = id(doc)
            doc_scores[doc_id] = {
                'doc': doc,
                'score': self.alpha * norm_score
            }
        
        # Add BM25 scores
        for (doc, _), norm_score in zip(bm25_results, bm25_scores_norm):
            doc_id = id(doc)
            if doc_id in doc_scores:
                doc_scores[doc_id]['score'] += (1 - self.alpha) * norm_score
            else:
                doc_scores[doc_id] = {
                    'doc': doc,
                    'score': (1 - self.alpha) * norm_score
                }
        
        # Sort by combined score
        ranked_docs = sorted(
            doc_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        # Return top k documents
        final_docs = [item['doc'] for item in ranked_docs[:k]]
        
        rag_logger.debug(
            f"Hybrid search: {len(vector_results)} vector + "
            f"{len(bm25_results)} BM25 → {len(final_docs)} final"
        )
        
        return final_docs
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range.
        
        Args:
            scores: List of scores
            
        Returns:
            Normalized scores
        """
        if not scores or max(scores) == min(scores):
            return [1.0] * len(scores)
        
        min_score = min(scores)
        max_score = max(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    def update_documents(self, documents: List[Document]):
        """Update document index.
        
        Args:
            documents: New list of documents
        """
        self.documents = documents
        self._build_bm25_index()
        rag_logger.info(f"Updated hybrid search index with {len(documents)} documents")


if __name__ == "__main__":
    # Test hybrid search
    test_docs = [
        Document(page_content="Galaxy S22 has a 3700mAh battery"),
        Document(page_content="iPhone 15 Pro features advanced camera"),
        Document(page_content="LG ThinQ smart features for home"),
    ]
    
    searcher = HybridSearcher(test_docs, alpha=0.5)
    
    # Test BM25 search
    results = searcher.bm25_search("Galaxy battery", k=2)
    print("BM25 Results:")
    for doc, score in results:
        print(f"  Score: {score:.2f} | {doc.page_content[:50]}")
