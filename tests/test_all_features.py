"""Comprehensive test script for all RAG features."""
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from logger import rag_logger

def print_section(title):
    """Print section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def test_logging():
    """Test logging system."""
    print_section("1️⃣  Testing Logging System")
    
    try:
        from logger import RAGLogger
        logger = RAGLogger()
        
        logger.info("Test info message")
        logger.debug("Test debug message")
        logger.warning("Test warning message")
        
        logger.log_query(
            question="Test question",
            answer="Test answer",
            sources=[{"source": "test.pdf"}],
            response_time=1.5,
            cache_hit=False
        )
        
        print(f"✅ Logging system works!")
        print(f"   Log directory: {logger.log_dir}")
        return True
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        return False


def test_hybrid_search():
    """Test hybrid search."""
    print_section("2️⃣  Testing Hybrid Search")
    
    try:
        from hybrid_search import HybridSearcher
        from langchain_core.documents import Document
        
        docs = [
            Document(page_content="Galaxy S22 has 3700mAh battery"),
            Document(page_content="iPhone 15 Pro camera features"),
            Document(page_content="LG ThinQ smart home features"),
        ]
        
        searcher = HybridSearcher(docs, alpha=0.5)
        results = searcher.bm25_search("Galaxy battery", k=2)
        
        print(f"✅ Hybrid search works!")
        print(f"   Found {len(results)} results")
        return True
    except Exception as e:
        print(f"❌ Hybrid search test failed: {e}")
        return False


def test_query_processor():
    """Test query processing."""
    print_section("3️⃣  Testing Query Processor")
    
    try:
        from query_processor import QueryProcessor
        processor = QueryProcessor()
        
        # Test query rewriting
        query = "갤럭시 배터리"
        rewritten = processor.rewrite_query(query)
        
        print(f"✅ Query processor works!")
        print(f"   Original: {query}")
        print(f"   Rewritten: {rewritten}")
        return True
    except Exception as e:
        print(f"❌ Query processor test failed: {e}")
        return False


def test_conversation():
    """Test conversation management."""
    print_section("4️⃣  Testing Conversation Manager")
    
    try:
        from conversation import ConversationManager
        manager = ConversationManager(max_history=5)
        
        session_id = manager.start_session()
        manager.add_message("user", "Test question?")
        manager.add_message("assistant", "Test answer.")
        
        context = manager.get_context_string()
        
        print(f"✅ Conversation manager works!")
        print(f"   Session: {session_id}")
        print(f"   Messages: {len(manager.current_session.messages)}")
        return True
    except Exception as e:
        print(f"❌ Conversation test failed: {e}")
        return False


def test_evaluation():
    """Test evaluation system."""
    print_section("5️⃣  Testing Evaluation System")
    
    try:
        from evaluation import RAGEvaluator
        evaluator = RAGEvaluator()
        
        metrics = evaluator.evaluate_relevance(
            question="Test question?",
            answer="Test answer with relevant information.",
            sources=[{"content": "relevant information"}]
        )
        
        print(f"✅ Evaluation system works!")
        print(f"   Quality score: {metrics['quality_score']:.2f}")
        return True
    except Exception as e:
        print(f"❌ Evaluation test failed: {e}")
        return False


def test_incremental_ingest():
    """Test incremental ingestion."""
    print_section("6️⃣  Testing Incremental Ingestion")
    
    try:
        from command_incremental_ingest import IncrementalIngestion
        ingestion = IncrementalIngestion()
        
        status = ingestion.get_index_status()
        
        print(f"✅ Incremental ingestion works!")
        print(f"   Tracked documents: {status['total_documents']}")
        return True
    except Exception as e:
        print(f"❌ Incremental ingestion test failed: {e}")
        return False


def test_advanced_rag():
    """Test advanced RAG techniques."""
    print_section("7️⃣  Testing Advanced RAG")
    
    try:
        from advanced_rag import AdvancedRAG
        advanced = AdvancedRAG()
        
        # Test query complexity
        complexity, strategy = advanced.apply_adaptive_rag(
            "Galaxy S22의 배터리 용량은?"
        )
        
        print(f"✅ Advanced RAG works!")
        print(f"   Complexity: {complexity.value}")
        print(f"   Strategy: {strategy}")
        return True
    except Exception as e:
        print(f"❌ Advanced RAG test failed: {e}")
        return False


def test_enhanced_pipeline():
    """Test enhanced RAG pipeline."""
    print_section("8️⃣  Testing Enhanced RAG Pipeline")
    
    try:
        from command_enhanced_rag_pipeline import EnhancedRAGPipeline
        
        print("   Initializing pipeline...")
        pipeline = EnhancedRAGPipeline(
            use_cache=False,  # Skip cache for test
            use_hybrid_search=False,
            use_conversation=True,
            use_evaluation=True
        )
        
        print("✅ Enhanced pipeline works!")
        print("   All components initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Enhanced pipeline test failed: {e}")
        print(f"   Note: This may fail if documents are not ingested yet")
        return False


def test_connections():
    """Test service connections."""
    print_section("🔌 Testing Service Connections")
    
    try:
        from test_connection import (
            test_redis,
            test_qdrant,
            test_ollama,
            test_pdf_files
        )
        
        results = {
            "Redis": test_redis(),
            "Qdrant": test_qdrant(),
            "Ollama": test_ollama(),
            "PDF Files": test_pdf_files()
        }
        
        all_ok = all(results.values())
        
        if all_ok:
            print("\n✅ All services are connected!")
        else:
            print("\n⚠️  Some services are not available:")
            for service, status in results.items():
                if not status:
                    print(f"   - {service}")
        
        return all_ok
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "🚀"*35)
    print("  COMPREHENSIVE RAG SYSTEM TEST SUITE")
    print("🚀"*35)
    
    tests = [
        ("Service Connections", test_connections),
        ("Logging System", test_logging),
        ("Hybrid Search", test_hybrid_search),
        ("Query Processor", test_query_processor),
        ("Conversation Manager", test_conversation),
        ("Evaluation System", test_evaluation),
        ("Incremental Ingestion", test_incremental_ingest),
        ("Advanced RAG", test_advanced_rag),
        ("Enhanced Pipeline", test_enhanced_pipeline),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ Unexpected error in {test_name}: {e}")
            results[test_name] = False
        
        time.sleep(0.5)
    
    # Summary
    print_section("📊 TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n{'='*70}")
    print(f"  Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"{'='*70}\n")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("  1. Run: python enhanced_rag_pipeline.py")
        print("  2. Or run web UI: streamlit run app_streamlit.py")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\nCommon fixes:")
        print("  - Start services: docker-compose up -d")
        print("  - Check Ollama: ollama list")
        print("  - Ingest documents: python ingest.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())
