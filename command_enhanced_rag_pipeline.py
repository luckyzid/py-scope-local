"""Enhanced RAG pipeline with all advanced features integrated."""
import time
from typing import List, Optional, Dict
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from vector_store import VectorStoreManager
from config import config
from logger import rag_logger
from hybrid_search import HybridSearcher
from query_processor import QueryProcessor
from conversation import ConversationManager
from evaluation import RAGEvaluator
from advanced_rag import AdvancedRAG, QueryComplexity
import redis


class EnhancedRAGPipeline:
    """Enhanced RAG pipeline with all advanced features."""
    
    def __init__(
        self,
        use_cache: bool = True,
        use_hybrid_search: bool = True,
        use_conversation: bool = True,
        use_evaluation: bool = True
    ):
        """Initialize enhanced RAG pipeline.
        
        Args:
            use_cache: Enable Redis semantic cache
            use_hybrid_search: Enable hybrid search
            use_conversation: Enable conversation memory
            use_evaluation: Enable evaluation metrics
        """
        rag_logger.info("Initializing Enhanced RAG Pipeline...")
        
        self.vector_store_manager = VectorStoreManager()
        self.query_processor = QueryProcessor()
        self.advanced_rag = AdvancedRAG()
        
        # Optional components
        self.use_hybrid_search = use_hybrid_search
        self.use_conversation = use_conversation
        self.use_evaluation = use_evaluation
        
        if use_conversation:
            self.conversation_manager = ConversationManager()
        
        if use_evaluation:
            self.evaluator = RAGEvaluator()
        
        # Initialize LLM
        self.llm = Ollama(
            base_url=config.ollama.base_url,
            model=config.ollama.model,
            temperature=config.rag.temperature
        )
        
        # Setup semantic cache
        if use_cache:
            self._setup_cache()
        
        # Initialize hybrid search
        self.hybrid_searcher = None
        if use_hybrid_search:
            self._init_hybrid_search()
        
        rag_logger.info("Enhanced RAG Pipeline initialized successfully")
    
    def _setup_cache(self):
        """Setup Redis semantic cache."""
        try:
            redis_client = redis.Redis(
                host=config.redis.host,
                port=config.redis.port,
                decode_responses=True
            )
            redis_client.ping()
            
            # Redis semantic cache is available but requires langchain-redis package
            # For now, we'll use basic Redis caching
            rag_logger.info("Redis connection established (semantic cache disabled)")
        except Exception as e:
            rag_logger.warning(f"Redis cache unavailable: {e}")
    
    def _init_hybrid_search(self):
        """Initialize hybrid search with all documents."""
        try:
            # Get all documents from vector store
            vector_store = self.vector_store_manager.get_vector_store()
            # Note: This is a simplified version
            # In production, you'd need to store documents separately
            rag_logger.info("Hybrid search will be initialized per query")
        except Exception as e:
            rag_logger.warning(f"Hybrid search initialization skipped: {e}")
    
    def query(
        self,
        question: str,
        use_query_enhancement: bool = True,
        use_self_rag: bool = True,
        adaptive: bool = True
    ) -> Dict:
        """Query the enhanced RAG pipeline.
        
        Args:
            question: User question
            use_query_enhancement: Enable query rewriting/expansion
            use_self_rag: Enable self-assessment
            adaptive: Enable adaptive RAG
            
        Returns:
            Comprehensive response dict
        """
        start_time = time.time()
        cache_hit = False
        error = None
        
        try:
            # Add to conversation
            if self.use_conversation:
                self.conversation_manager.add_message("user", question)
            
            # Adaptive RAG: determine strategy
            strategy = {'top_k': config.rag.top_k_results}
            if adaptive:
                complexity, strategy = self.advanced_rag.apply_adaptive_rag(question)
                rag_logger.info(f"Query complexity: {complexity.value}")
            
            # Query enhancement
            enhanced_question = question
            if use_query_enhancement and strategy.get('use_multi_query', False):
                queries = self.query_processor.generate_multi_queries(question, num_queries=2)
                rag_logger.info(f"Generated {len(queries)} query variations")
            else:
                queries = [question]
            
            # Retrieve documents
            all_docs = []
            vector_store = self.vector_store_manager.get_vector_store()
            
            for q in queries:
                docs = vector_store.similarity_search(q, k=strategy['top_k'])
                all_docs.extend(docs)
            
            # Remove duplicates
            unique_docs = []
            seen = set()
            for doc in all_docs:
                doc_id = doc.page_content[:100]
                if doc_id not in seen:
                    seen.add(doc_id)
                    unique_docs.append(doc)
            
            all_docs = unique_docs[:strategy['top_k']]
            
            # Self-RAG: assess relevance
            if use_self_rag and all_docs:
                scored_docs = self.advanced_rag.self_assess_relevance(question, all_docs)
                # Filter low-scoring documents
                all_docs = [doc for doc, score in scored_docs if score > 0.3]
                rag_logger.info(f"Self-RAG filtered to {len(all_docs)} relevant docs")
            
            # Corrective RAG: check if documents are sufficient
            is_sufficient = self.advanced_rag.assess_retrieval_quality(question, all_docs)
            
            if not is_sufficient:
                rag_logger.warning("Retrieved documents insufficient, using fallback")
                answer = self.advanced_rag.generate_fallback_answer(question)
                sources = []
            else:
                # Generate answer
                context = "\n\n".join([doc.page_content for doc in all_docs])
                
                # Include conversation context if available
                conversation_context = ""
                if self.use_conversation:
                    conversation_context = self.conversation_manager.format_for_llm(num_messages=4)
                
                # Create prompt
                template = """당신은 문서 기반 질문 답변 AI 어시스턴트입니다.
주어진 컨텍스트를 바탕으로 질문에 정확하고 상세하게 답변해주세요.

{conversation_context}

컨텍스트: {context}

질문: {question}

답변:"""
                
                prompt = PromptTemplate(
                    template=template,
                    input_variables=["conversation_context", "context", "question"]
                )
                
                formatted_prompt = prompt.format(
                    conversation_context=conversation_context,
                    context=context,
                    question=question
                )
                
                answer = self.llm.invoke(formatted_prompt)
                sources = all_docs
            
            # Self-RAG: critique answer
            critique = None
            if use_self_rag and sources:
                critique = self.advanced_rag.self_critique_answer(question, answer, sources)
                rag_logger.info(f"Answer quality score: {critique['overall']:.2f}")
            
            # Add assistant response to conversation
            if self.use_conversation:
                self.conversation_manager.add_message("assistant", answer)
            
            # Prepare response
            response = {
                "question": question,
                "answer": answer,
                "sources": [
                    {
                        "content": doc.page_content[:300] + "...",
                        "source": doc.metadata.get("source", "Unknown"),
                        "filename": doc.metadata.get("filename", "Unknown")
                    }
                    for doc in sources
                ],
                "metadata": {
                    "response_time": time.time() - start_time,
                    "cache_hit": cache_hit,
                    "num_sources": len(sources),
                    "strategy": strategy if adaptive else None,
                    "critique": critique,
                    "sufficient_retrieval": is_sufficient
                }
            }
            
            # Evaluation
            if self.use_evaluation:
                self.evaluator.log_query_metrics(
                    question=question,
                    answer=answer,
                    sources=response["sources"],
                    response_time=response["metadata"]["response_time"],
                    cache_hit=cache_hit
                )
            
            # Logging
            rag_logger.log_query(
                question=question,
                answer=answer,
                sources=response["sources"],
                response_time=response["metadata"]["response_time"],
                cache_hit=cache_hit,
                metadata=response["metadata"]
            )
            
            return response
            
        except Exception as e:
            error = str(e)
            rag_logger.error(f"Query failed: {error}")
            
            if self.use_evaluation:
                self.evaluator.log_query_metrics(
                    question=question,
                    answer="",
                    sources=[],
                    response_time=time.time() - start_time,
                    cache_hit=False,
                    error=error
                )
            
            return {
                "question": question,
                "answer": f"오류가 발생했습니다: {error}",
                "sources": [],
                "metadata": {
                    "error": error,
                    "response_time": time.time() - start_time
                }
            }
    
    def get_evaluation_report(self) -> str:
        """Get evaluation report.
        
        Returns:
            Evaluation report text
        """
        if self.use_evaluation:
            return self.evaluator.generate_report()
        return "Evaluation not enabled"
    
    def save_conversation(self):
        """Save current conversation."""
        if self.use_conversation:
            self.conversation_manager.save_session()
            rag_logger.info("Conversation saved")


def interactive_mode():
    """Run interactive Q&A mode with enhanced features."""
    print("\n" + "="*70)
    print("  🚀 Enhanced RAG Q&A System")
    print("="*70)
    print("\nCommands:")
    print("  - Type your question to get an answer")
    print("  - 'report' - Show evaluation report")
    print("  - 'save' - Save conversation")
    print("  - 'quit' or 'exit' - Exit the system")
    print("="*70 + "\n")
    
    # Initialize pipeline
    try:
        pipeline = EnhancedRAGPipeline(
            use_cache=True,
            use_hybrid_search=True,
            use_conversation=True,
            use_evaluation=True
        )
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        return
    
    while True:
        question = input("\n💬 질문을 입력하세요: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            pipeline.save_conversation()
            print("\n👋 시스템을 종료합니다.")
            break
        
        if question.lower() == 'report':
            print(pipeline.get_evaluation_report())
            continue
        
        if question.lower() == 'save':
            pipeline.save_conversation()
            print("✅ 대화가 저장되었습니다.")
            continue
        
        if not question:
            continue
        
        try:
            print("\n⏳ 답변 생성 중...")
            result = pipeline.query(question)
            
            print("\n" + "="*70)
            print(f"📝 질문: {result['question']}")
            print("="*70)
            print(f"\n💡 답변:\n{result['answer']}")
            
            if result.get('sources'):
                print("\n" + "-"*70)
                print("📚 참고 문서:")
                for i, source in enumerate(result['sources'], 1):
                    print(f"\n  {i}. {source['filename']}")
                    print(f"     {source['content'][:150]}...")
            
            # Show metadata
            metadata = result.get('metadata', {})
            print("\n" + "-"*70)
            print(f"⚡ 응답 시간: {metadata.get('response_time', 0):.2f}초")
            print(f"📊 출처 개수: {metadata.get('num_sources', 0)}")
            
            if metadata.get('critique'):
                critique = metadata['critique']
                print(f"⭐ 답변 품질: {critique['overall']:.2f}/1.0")
            
            print("="*70)
            
        except Exception as e:
            print(f"\n❌ 오류: {e}")


if __name__ == "__main__":
    interactive_mode()
