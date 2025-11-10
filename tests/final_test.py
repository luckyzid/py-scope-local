"""Final comprehensive test of all features."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_result(feature, success, message=""):
    emoji = "✅" if success else "❌"
    print(f"{emoji} {feature}")
    if message:
        print(f"   {message}")

def main():
    print_header("🚀 RAG 파이프라인 종합 테스트")
    
    results = {}
    
    # Test 1: Services
    print_header("1️⃣  서비스 연결 테스트")
    try:
        from test_connection import test_redis, test_qdrant, test_ollama, test_pdf_files
        
        redis_ok = test_redis()
        qdrant_ok = test_qdrant()
        ollama_ok = test_ollama()
        pdf_ok = test_pdf_files()
        
        results['redis'] = redis_ok
        results['qdrant'] = qdrant_ok
        results['ollama'] = ollama_ok
        results['pdf'] = pdf_ok
        
        print_result("Redis", redis_ok)
        print_result("Qdrant", qdrant_ok)
        print_result("Ollama", ollama_ok)
        print_result("PDF Files", pdf_ok)
    except Exception as e:
        print_result("서비스 연결", False, str(e))
        results['services'] = False
    
    # Test 2: Basic RAG
    print_header("2️⃣  기본 RAG 파이프라인")
    try:
        from vector_store import VectorStoreManager
        from langchain_community.llms import Ollama
        from config import config
        
        vector_store = VectorStoreManager()
        llm = Ollama(
            base_url=config.ollama.base_url,
            model=config.ollama.model,
            temperature=0.7
        )
        
        # Search test
        docs = vector_store.similarity_search("Galaxy S22", k=3)
        search_ok = len(docs) > 0
        print_result("벡터 검색", search_ok, f"{len(docs)}개 문서 검색")
        
        # Answer generation test
        if search_ok:
            context = docs[0].page_content[:200]
            prompt = f"Context: {context}\n\nQuestion: What is this about?\n\nAnswer:"
            answer = llm.invoke(prompt)
            answer_ok = len(answer) > 10
            print_result("답변 생성", answer_ok)
            results['basic_rag'] = True
        else:
            results['basic_rag'] = False
            
    except Exception as e:
        print_result("기본 RAG", False, str(e))
        results['basic_rag'] = False
    
    # Test 3: Logging
    print_header("3️⃣  로깅 시스템")
    try:
        from logger import rag_logger
        rag_logger.info("Test log message")
        print_result("구조화된 로깅", True)
        results['logging'] = True
    except Exception as e:
        print_result("로깅", False, str(e))
        results['logging'] = False
    
    # Test 4: Query Processing
    print_header("4️⃣  쿼리 처리")
    try:
        from query_processor import QueryProcessor
        processor = QueryProcessor()
        
        # Query rewriting
        rewritten = processor.rewrite_query("배터리")
        print_result("쿼리 재작성", len(rewritten) > 0, f"'{rewritten}'")
        
        # Multi-query
        queries = processor.generate_multi_queries("배터리", num_queries=3)
        print_result("다중 쿼리 생성", len(queries) >= 3, f"{len(queries)}개 생성")
        
        results['query_processing'] = True
    except Exception as e:
        print_result("쿼리 처리", False, str(e))
        results['query_processing'] = False
    
    # Test 5: Hybrid Search
    print_header("5️⃣  하이브리드 검색")
    try:
        from hybrid_search import HybridSearcher
        from langchain_core.documents import Document
        
        # Use 3+ documents for better BM25 scoring
        docs = [
            Document(page_content="Galaxy S22 has amazing camera"),
            Document(page_content="iPhone 15 Pro features"),
            Document(page_content="Galaxy camera is very good"),
        ]
        searcher = HybridSearcher(docs, alpha=0.5)
        bm25_results = searcher.bm25_search("Galaxy camera", k=2)
        bm25_ok = len(bm25_results) >= 1
        print_result("BM25 검색", bm25_ok, f"{len(bm25_results)}개 결과")
        results['hybrid_search'] = bm25_ok
    except Exception as e:
        print_result("하이브리드 검색", False, str(e))
        results['hybrid_search'] = False
    
    # Test 6: Conversation
    print_header("6️⃣  대화 관리")
    try:
        from conversation import ConversationManager
        manager = ConversationManager()
        session_id = manager.start_session()
        manager.add_message("user", "안녕하세요")
        manager.add_message("assistant", "안녕하세요!")
        context = manager.get_context_string()
        print_result("대화 컨텍스트", len(context) > 0)
        results['conversation'] = True
    except Exception as e:
        print_result("대화 관리", False, str(e))
        results['conversation'] = False
    
    # Test 7: Evaluation
    print_header("7️⃣  평가 시스템")
    try:
        from evaluation import RAGEvaluator
        evaluator = RAGEvaluator()
        evaluator.log_query_metrics("test", "answer", 0.5, False, 1)
        print_result("메트릭 로깅", True)
        results['evaluation'] = True
    except Exception as e:
        print_result("평가 시스템", False, str(e))
        results['evaluation'] = False
    
    # Final Summary
    print_header("📊 테스트 결과 요약")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\n통과: {passed}/{total} ({passed/total*100:.1f}%)")
    print("\n상세 결과:")
    for feature, success in results.items():
        emoji = "✅" if success else "❌"
        print(f"  {emoji} {feature}")
    
    print("\n" + "=" * 70)
    if passed == total:
        print("🎉 모든 테스트 통과!")
    elif passed >= total * 0.7:
        print("✅ 대부분의 기능 정상 작동")
    else:
        print("⚠️  일부 기능 수정 필요")
    print("=" * 70)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
