"""Advanced RAG techniques: Self-RAG, Corrective RAG, Adaptive RAG."""
from typing import List, Dict, Tuple, Optional
from enum import Enum
from langchain_core.documents import Document
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from config import config
from logger import rag_logger


class QueryComplexity(Enum):
    """Query complexity levels."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class AdvancedRAG:
    """Advanced RAG techniques implementation."""
    
    def __init__(self):
        """Initialize advanced RAG."""
        self.llm = Ollama(
            base_url=config.ollama.base_url,
            model=config.ollama.model,
            temperature=0.3
        )
        rag_logger.info("Advanced RAG initialized")
    
    # ========== Self-RAG ==========
    
    def self_assess_relevance(self, question: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """Assess document relevance to question (Self-RAG).
        
        Args:
            question: User question
            documents: Retrieved documents
            
        Returns:
            List of (document, relevance_score) tuples
        """
        template = """문서가 질문에 얼마나 관련이 있는지 평가해주세요.
0.0 (전혀 관련없음) ~ 1.0 (매우 관련있음) 사이의 점수만 출력하세요.

질문: {question}

문서: {document}

관련성 점수 (0.0~1.0):"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["question", "document"]
        )
        
        scored_docs = []
        
        for doc in documents:
            try:
                response = self.llm.invoke(
                    prompt.format(
                        question=question,
                        document=doc.page_content[:500]
                    )
                )
                
                # Extract score
                try:
                    score = float(response.strip().split()[0])
                    score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
                except:
                    score = 0.5  # Default score
                
                scored_docs.append((doc, score))
                
            except Exception as e:
                rag_logger.warning(f"Failed to assess relevance: {e}")
                scored_docs.append((doc, 0.5))
        
        # Sort by score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        rag_logger.debug(f"Self-assessed {len(documents)} documents")
        return scored_docs
    
    def self_critique_answer(self, question: str, answer: str, sources: List[Document]) -> Dict:
        """Critique generated answer (Self-RAG).
        
        Args:
            question: User question
            answer: Generated answer
            sources: Source documents
            
        Returns:
            Critique dict with scores and feedback
        """
        template = """다음 답변을 평가해주세요:

질문: {question}
답변: {answer}

다음 항목을 각각 평가하고 점수(0.0~1.0)를 매겨주세요:
1. 정확성 (Accuracy):
2. 완전성 (Completeness):
3. 출처 기반 (Grounded):

각 줄에 "항목: 점수" 형식으로 답변하세요."""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["question", "answer"]
        )
        
        try:
            response = self.llm.invoke(prompt.format(question=question, answer=answer))
            
            # Parse scores
            critique = {
                'accuracy': 0.7,
                'completeness': 0.7,
                'grounded': 0.7,
                'overall': 0.7
            }
            
            lines = response.strip().split('\n')
            for line in lines:
                if ':' in line:
                    parts = line.split(':')
                    if len(parts) >= 2:
                        try:
                            score = float(parts[1].strip().split()[0])
                            score = max(0.0, min(1.0, score))
                            
                            if 'accuracy' in parts[0].lower() or '정확성' in parts[0]:
                                critique['accuracy'] = score
                            elif 'complete' in parts[0].lower() or '완전성' in parts[0]:
                                critique['completeness'] = score
                            elif 'ground' in parts[0].lower() or '출처' in parts[0]:
                                critique['grounded'] = score
                        except:
                            pass
            
            # Calculate overall
            critique['overall'] = (
                critique['accuracy'] + 
                critique['completeness'] + 
                critique['grounded']
            ) / 3
            
            rag_logger.debug(f"Self-critique: {critique['overall']:.2f}")
            return critique
            
        except Exception as e:
            rag_logger.warning(f"Failed to critique answer: {e}")
            return {
                'accuracy': 0.5,
                'completeness': 0.5,
                'grounded': 0.5,
                'overall': 0.5
            }
    
    # ========== Corrective RAG ==========
    
    def assess_retrieval_quality(self, question: str, documents: List[Document]) -> bool:
        """Assess if retrieved documents are sufficient (Corrective RAG).
        
        Args:
            question: User question
            documents: Retrieved documents
            
        Returns:
            True if documents are sufficient
        """
        if not documents:
            return False
        
        template = """다음 문서들이 질문에 답하기에 충분한 정보를 포함하고 있나요?
'예' 또는 '아니오'로만 답하세요.

질문: {question}

문서 수: {num_docs}
문서 내용 샘플: {sample}

충분한가요?"""
        
        sample = "\n".join([doc.page_content[:200] for doc in documents[:3]])
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["question", "num_docs", "sample"]
        )
        
        try:
            response = self.llm.invoke(
                prompt.format(
                    question=question,
                    num_docs=len(documents),
                    sample=sample
                )
            )
            
            response = response.strip().lower()
            is_sufficient = '예' in response or 'yes' in response
            
            rag_logger.debug(f"Retrieval quality: {'sufficient' if is_sufficient else 'insufficient'}")
            return is_sufficient
            
        except Exception as e:
            rag_logger.warning(f"Failed to assess retrieval quality: {e}")
            return True  # Assume sufficient on error
    
    def generate_fallback_answer(self, question: str) -> str:
        """Generate fallback answer when retrieval fails (Corrective RAG).
        
        Args:
            question: User question
            
        Returns:
            Fallback answer
        """
        template = """질문에 대한 일반적인 답변을 제공해주세요.
특정 문서 정보는 없지만, 일반적인 지식으로 답변해주세요.

질문: {question}

답변:"""
        
        prompt = PromptTemplate(template=template, input_variables=["question"])
        
        try:
            answer = self.llm.invoke(prompt.format(question=question))
            rag_logger.info("Generated fallback answer")
            return answer.strip()
        except Exception as e:
            rag_logger.error(f"Failed to generate fallback: {e}")
            return "죄송합니다. 해당 질문에 대한 정보를 찾을 수 없습니다."
    
    # ========== Adaptive RAG ==========
    
    def classify_query_complexity(self, question: str) -> QueryComplexity:
        """Classify query complexity (Adaptive RAG).
        
        Args:
            question: User question
            
        Returns:
            QueryComplexity enum
        """
        template = """질문의 복잡도를 분류해주세요:
- simple: 단순한 사실 확인 질문
- medium: 약간의 추론이 필요한 질문
- complex: 여러 정보를 종합해야 하는 복잡한 질문

'simple', 'medium', 'complex' 중 하나만 답하세요.

질문: {question}

복잡도:"""
        
        prompt = PromptTemplate(template=template, input_variables=["question"])
        
        try:
            response = self.llm.invoke(prompt.format(question=question))
            response = response.strip().lower()
            
            if 'simple' in response:
                complexity = QueryComplexity.SIMPLE
            elif 'complex' in response:
                complexity = QueryComplexity.COMPLEX
            else:
                complexity = QueryComplexity.MEDIUM
            
            rag_logger.debug(f"Query complexity: {complexity.value}")
            return complexity
            
        except Exception as e:
            rag_logger.warning(f"Failed to classify complexity: {e}")
            return QueryComplexity.MEDIUM
    
    def get_adaptive_strategy(self, complexity: QueryComplexity) -> Dict:
        """Get RAG strategy based on query complexity (Adaptive RAG).
        
        Args:
            complexity: Query complexity
            
        Returns:
            Strategy configuration dict
        """
        strategies = {
            QueryComplexity.SIMPLE: {
                'top_k': 3,
                'use_reranking': False,
                'use_multi_query': False,
                'temperature': 0.3
            },
            QueryComplexity.MEDIUM: {
                'top_k': 5,
                'use_reranking': True,
                'use_multi_query': False,
                'temperature': 0.5
            },
            QueryComplexity.COMPLEX: {
                'top_k': 10,
                'use_reranking': True,
                'use_multi_query': True,
                'temperature': 0.7
            }
        }
        
        strategy = strategies.get(complexity, strategies[QueryComplexity.MEDIUM])
        rag_logger.debug(f"Adaptive strategy: {strategy}")
        return strategy
    
    def apply_adaptive_rag(self, question: str) -> Tuple[QueryComplexity, Dict]:
        """Apply adaptive RAG based on query complexity.
        
        Args:
            question: User question
            
        Returns:
            Tuple of (complexity, strategy)
        """
        complexity = self.classify_query_complexity(question)
        strategy = self.get_adaptive_strategy(complexity)
        
        rag_logger.info(f"Adaptive RAG: {complexity.value} → {strategy}")
        return complexity, strategy


if __name__ == "__main__":
    # Test advanced RAG
    advanced_rag = AdvancedRAG()
    
    # Test query complexity
    questions = [
        "Galaxy S22의 배터리 용량은?",
        "Galaxy S22와 iPhone 15 Pro의 카메라를 비교해줘",
        "스마트폰 선택 시 고려해야 할 요소들을 제품 사양을 바탕으로 분석해줘"
    ]
    
    print("Query Complexity Classification:")
    for q in questions:
        complexity, strategy = advanced_rag.apply_adaptive_rag(q)
        print(f"\nQ: {q}")
        print(f"Complexity: {complexity.value}")
        print(f"Strategy: {strategy}")
