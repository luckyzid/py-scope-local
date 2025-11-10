"""Manual test for rag_pipeline.py with Redis cache"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from command_rag_pipeline import RAGPipeline

print("=" * 60)
print("Testing RAG Pipeline with RedisSemanticCache")
print("=" * 60)

try:
    print("\n1. Initializing pipeline with cache...")
    pipeline = RAGPipeline(use_cache=True)
    print("✅ Pipeline initialized successfully!")
    
    print("\n2. Testing query...")
    question = "Galaxy S22의 배터리 용량은?"
    print(f"   Question: {question}")
    
    result = pipeline.query(question)
    
    print("\n" + "=" * 60)
    print(f"📝 질문: {result['question']}")
    print("=" * 60)
    print(f"\n💡 답변:\n{result['answer']}")
    
    if result.get('sources'):
        print("\n" + "-" * 60)
        print("📚 참고 문서:")
        for i, source in enumerate(result['sources'][:2], 1):
            print(f"\n{i}. {source['filename']}")
            print(f"   {source['content']}")
    print("=" * 60)
    
    print("\n3. Testing cache (same question)...")
    result2 = pipeline.query(question)
    print("✅ Second query successful (should use cache)")
    
    print("\n✅ All tests passed!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
