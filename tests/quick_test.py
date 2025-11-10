"""Quick test of advanced features."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vector_store import VectorStoreManager
from langchain_community.llms import Ollama
from config import config
from hybrid_search import HybridSearcher
from query_processor import QueryProcessor
from logger import rag_logger

def main():
    print("=" * 70)
    print("🚀 고급 RAG 기능 테스트")
    print("=" * 70)
    
    # Initialize
    print("\n1️⃣  초기화 중...")
    vector_store = VectorStoreManager()
    llm = Ollama(
        base_url=config.ollama.base_url,
        model=config.ollama.model,
        temperature=config.rag.temperature
    )
    query_processor = QueryProcessor()
    
    # Test query
    question = "Galaxy S22의 주요 특징은?"
    print(f"\n2️⃣  원본 질문: {question}")
    
    # Query rewriting
    print("\n3️⃣  쿼리 재작성...")
    try:
        rewritten = query_processor.rewrite_query(question)
        print(f"   재작성된 질문: {rewritten}")
    except Exception as e:
        print(f"   ⚠️  재작성 실패 (계속 진행): {e}")
        rewritten = question
    
    # Multi-query generation
    print("\n4️⃣  다중 쿼리 생성...")
    try:
        multi_queries = query_processor.generate_multi_queries(question, num_queries=3)
        for i, q in enumerate(multi_queries, 1):
            print(f"   {i}. {q}")
    except Exception as e:
        print(f"   ⚠️  다중 쿼리 생성 실패 (계속 진행): {e}")
    
    # Vector search
    print("\n5️⃣  벡터 검색...")
    docs = vector_store.similarity_search(rewritten, k=5)
    print(f"   찾은 문서 수: {len(docs)}")
    
    # Hybrid search
    print("\n6️⃣  하이브리드 검색...")
    try:
        all_docs = vector_store.similarity_search("", k=50)  # Get more docs for BM25
        searcher = HybridSearcher(all_docs, alpha=0.5)
        hybrid_results = searcher.hybrid_search(rewritten, docs, k=3)
        print(f"   하이브리드 결과 수: {len(hybrid_results)}")
    except Exception as e:
        print(f"   ⚠️  하이브리드 검색 실패 (벡터 검색 사용): {e}")
        hybrid_results = docs[:3]
    
    # Generate answer
    print("\n7️⃣  답변 생성 중...")
    context = "\n\n".join([doc.page_content for doc in hybrid_results[:3]])
    
    prompt = f"""다음 문맥을 기반으로 질문에 답변하세요.

문맥:
{context}

질문: {question}

답변:"""
    
    answer = llm.invoke(prompt)
    
    # Display results
    print("\n" + "=" * 70)
    print("📝 질문:", question)
    print("=" * 70)
    print("\n💡 답변:")
    print(answer)
    print("\n" + "=" * 70)
    print("📚 출처:")
    for i, doc in enumerate(hybrid_results[:3], 1):
        print(f"\n{i}. {doc.metadata.get('filename', 'Unknown')}")
        print(f"   {doc.page_content[:150]}...")
    print("=" * 70)
    
    # Log
    rag_logger.log_query(
        question=question,
        answer=answer,
        sources=[doc.metadata.get('filename', 'Unknown') for doc in hybrid_results[:3]],
        response_time=0.0
    )
    
    print("\n✅ 테스트 완료!")

if __name__ == "__main__":
    main()
