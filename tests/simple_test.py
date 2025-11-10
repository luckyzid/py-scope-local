"""Simple RAG test without complex dependencies."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.globals import set_verbose
set_verbose(False)

from vector_store import VectorStoreManager
from langchain_community.llms import Ollama
from config import config

def test_rag():
    print("=" * 60)
    print("Simple RAG Test")
    print("=" * 60)
    
    # 1. Initialize components
    print("\n1. Initializing components...")
    vector_store = VectorStoreManager()
    llm = Ollama(
        base_url=config.ollama.base_url,
        model=config.ollama.model,
        temperature=config.rag.temperature
    )
    
    # 2. Test search
    question = "Galaxy S22의 배터리 용량은?"
    print(f"\n2. Question: {question}")
    
    print("\n3. Searching documents...")
    docs = vector_store.similarity_search(question, k=3)
    print(f"Found {len(docs)} relevant documents")
    
    # 4. Display sources
    print("\n4. Relevant sources:")
    for i, doc in enumerate(docs, 1):
        print(f"\n--- Source {i} ---")
        print(f"Content: {doc.page_content[:200]}...")
        if doc.metadata:
            print(f"Metadata: {doc.metadata}")
    
    # 5. Generate answer
    print("\n5. Generating answer...")
    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""다음 문맥을 기반으로 질문에 답변하세요.

문맥:
{context}

질문: {question}

답변:"""
    
    answer = llm.invoke(prompt)
    
    print("\n" + "=" * 60)
    print("Answer:")
    print("=" * 60)
    print(answer)
    print("=" * 60)

if __name__ == "__main__":
    test_rag()
