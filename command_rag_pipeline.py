"""RAG pipeline with Redis semantic caching."""
from typing import List, Optional
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
CACHE_AVAILABLE = False
from vector_store import VectorStoreManager
from config import config
import redis
from PIL import Image


class RAGPipeline:
    """RAG pipeline with semantic caching."""
    
    def __init__(self, use_cache: bool = True):
        """Initialize RAG pipeline.
        
        Args:
            use_cache: Whether to use Redis semantic cache
        """
        self.vector_store_manager = VectorStoreManager()
        self.use_cache = use_cache
        
        # Initialize LLM
        self.llm = Ollama(
            base_url=config.ollama.base_url,
            model=config.ollama.model,
            temperature=config.rag.temperature
        )
        
        # Setup semantic cache
        if use_cache:
            self._setup_cache()
        
        # Setup retrieval chain
        self._setup_chain()
    
    def _setup_cache(self):
        """Setup Redis cache."""
        try:
            self.redis_client = redis.Redis(
                host=config.redis.host,
                port=config.redis.port,
                decode_responses=False  # Store binary data
            )
            
            # Test connection
            self.redis_client.ping()
            print("✅ Redis cache enabled")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not setup Redis cache: {e}")
            print("Continuing without cache...")
            self.use_cache = False
            self.redis_client = None
    
    def _setup_chain(self):
        """Setup retrieval components."""
        # Get vector store
        self.vector_store = self.vector_store_manager.get_vector_store()
        
        # Create prompt template
        self.prompt_template = """당신은 문서 기반 질문 답변 AI 어시스턴트입니다. 
주어진 컨텍스트를 바탕으로 질문에 정확하고 상세하게 답변해주세요.
컨텍스트에 없는 내용은 답변하지 마세요.

컨텍스트:
{context}

질문: {question}

답변:"""
    
    def query(self, question: str, return_sources: bool = True) -> dict:
        """Query the RAG pipeline.
        
        Args:
            question: User question
            return_sources: Whether to return source documents
            
        Returns:
            Dictionary with answer and optional source documents
        """
        import hashlib
        import json
        
        # Check cache
        cache_hit = False
        if self.use_cache and self.redis_client:
            cache_key = f"rag:query:{hashlib.md5(question.encode()).hexdigest()}"
            try:
                cached = self.redis_client.get(cache_key)
                if cached:
                    cache_hit = True
                    print("⚡ Cache hit!")
                    return json.loads(cached)
            except Exception as e:
                print(f"Cache read error: {e}")
        
        # 1. Retrieve relevant documents
        docs = self.vector_store.similarity_search(
            question, 
            k=config.rag.top_k_results
        )
        
        # 2. Build context from documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 3. Generate prompt
        prompt = self.prompt_template.format(
            context=context,
            question=question
        )
        
        # 4. Generate answer
        answer = self.llm.invoke(prompt)
        
        # 5. Build response
        response = {
            "answer": answer,
            "question": question,
            "cache_hit": cache_hit
        }
        
        if return_sources:
            sources = []
            for doc in docs:
                sources.append({
                    "content": doc.page_content[:200] + "...",
                    "source": doc.metadata.get("source", "Unknown"),
                    "filename": doc.metadata.get("filename", "Unknown")
                })
            response["sources"] = sources
        
        # Store in cache
        if self.use_cache and self.redis_client and not cache_hit:
            try:
                self.redis_client.setex(
                    cache_key,
                    config.redis.ttl,  # TTL in seconds
                    json.dumps(response, ensure_ascii=False)
                )
            except Exception as e:
                print(f"Cache write error: {e}")
        
        return response
    
    def search_similar(self, query: str, k: int = None) -> List[Document]:
        """Search for similar documents.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of similar documents
        """
        return self.vector_store_manager.similarity_search(query, k)

    def search_similar_images(self, query_image: Image.Image, k: int = None) -> List[dict]:
        """Search for similar images.

        Args:
            query_image: PIL Image to search for
            k: Number of results

        Returns:
            List of similar image metadata with scores
        """
        if k is None:
            k = config.rag.top_k_results
        return self.vector_store_manager.similarity_search_image(query_image, k)


def interactive_mode():
    """Run interactive Q&A mode."""
    print("\n=== RAG Q&A System ===")
    print("Type 'quit' or 'exit' to stop")
    print("Type 'image <path>' to search for similar images\n")

    # Initialize pipeline
    try:
        pipeline = RAGPipeline(use_cache=True)
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        return

    while True:
        user_input = input("\n질문을 입력하세요 (또는 'image <path>'로 이미지 검색): ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("시스템을 종료합니다.")
            break

        if not user_input:
            continue

        # Check if it's an image search command
        if user_input.startswith('image '):
            image_path = user_input[6:].strip()
            try:
                query_image = Image.open(image_path)
                print(f"\n이미지 '{image_path}'로 유사 이미지 검색 중...")
                results = pipeline.search_similar_images(query_image)

                print("\n" + "="*60)
                print(f"쿼리 이미지: {image_path}")
                print("="*60)

                if results:
                    print("\n유사 이미지:")
                    for i, result in enumerate(results, 1):
                        metadata = result['metadata']
                        print(f"\n{i}. {metadata.get('filename', 'Unknown')}")
                        print(f"   출처: {metadata.get('source', 'Unknown')}")
                        print(f"   유사도: {result['score']:.3f}")
                        if metadata.get('page'):
                            print(f"   페이지: {metadata['page']}")
                else:
                    print("\n유사 이미지를 찾을 수 없습니다.")

                print("="*60)

            except Exception as e:
                print(f"이미지 검색 오류: {e}")
            continue

        # Text query
        try:
            print("\n답변 생성 중...")
            result = pipeline.query(user_input)

            print("\n" + "="*60)
            print(f"질문: {result['question']}")
            print("="*60)
            print(f"\n답변:\n{result['answer']}")

            if result.get('sources'):
                print("\n" + "-"*60)
                print("참고 문서:")
                for i, source in enumerate(result['sources'], 1):
                    print(f"\n{i}. {source['filename']}")
                    print(f"   내용: {source['content']}")
            print("="*60)

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    interactive_mode()
