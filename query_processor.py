"""Query processing including rewriting, expansion, and HyDE."""
from typing import List, Optional
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from config import config
from logger import rag_logger


class QueryProcessor:
    """Process and enhance user queries."""
    
    def __init__(self):
        """Initialize query processor."""
        self.llm = Ollama(
            base_url=config.ollama.base_url,
            model=config.ollama.model,
            temperature=0.3  # Lower temperature for more focused queries
        )
        rag_logger.info("Query processor initialized")
    
    def rewrite_query(self, query: str) -> str:
        """Rewrite query for better search results.
        
        Args:
            query: Original user query
            
        Returns:
            Rewritten query
        """
        template = """사용자의 질문을 더 명확하고 검색하기 좋은 형태로 다시 작성해주세요.
핵심 키워드를 유지하면서 더 구체적이고 명확하게 만들어주세요.

원본 질문: {query}

개선된 질문:"""
        
        prompt = PromptTemplate(template=template, input_variables=["query"])
        
        try:
            rewritten = self.llm.invoke(prompt.format(query=query))
            rewritten = rewritten.strip()
            rag_logger.debug(f"Query rewritten: '{query}' → '{rewritten}'")
            return rewritten
        except Exception as e:
            rag_logger.warning(f"Query rewrite failed: {e}, using original")
            return query
    
    def generate_multi_queries(self, query: str, num_queries: int = 3) -> List[str]:
        """Generate multiple variations of the query.
        
        Args:
            query: Original query
            num_queries: Number of variations to generate
            
        Returns:
            List of query variations
        """
        template = """사용자의 질문을 {num} 가지 다른 방식으로 표현해주세요.
각 질문은 원본과 같은 의미를 가지지만 다른 단어나 관점을 사용해야 합니다.

원본 질문: {query}

변형 질문들 (한 줄에 하나씩):
1."""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["query", "num"]
        )
        
        try:
            response = self.llm.invoke(prompt.format(query=query, num=num_queries))
            
            # Parse response
            queries = [query]  # Include original
            lines = response.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                # Remove numbering
                if line and len(line) > 3:
                    cleaned = line
                    for prefix in ['1.', '2.', '3.', '4.', '5.', '-', '*']:
                        if cleaned.startswith(prefix):
                            cleaned = cleaned[len(prefix):].strip()
                    if cleaned and cleaned not in queries:
                        queries.append(cleaned)
            
            rag_logger.debug(f"Generated {len(queries)} query variations")
            return queries[:num_queries + 1]
            
        except Exception as e:
            rag_logger.warning(f"Multi-query generation failed: {e}")
            return [query]
    
    def generate_hyde_document(self, query: str) -> str:
        """Generate a hypothetical document that would answer the query (HyDE).
        
        Args:
            query: User query
            
        Returns:
            Hypothetical document
        """
        template = """다음 질문에 대한 이상적인 답변이 포함된 문서를 작성해주세요.
실제 사실이 아니어도 되며, 질문에 대한 완벽한 답변 형태의 문서를 만들어주세요.

질문: {query}

이상적인 문서 내용:"""
        
        prompt = PromptTemplate(template=template, input_variables=["query"])
        
        try:
            hyde_doc = self.llm.invoke(prompt.format(query=query))
            hyde_doc = hyde_doc.strip()
            rag_logger.debug(f"Generated HyDE document ({len(hyde_doc)} chars)")
            return hyde_doc
        except Exception as e:
            rag_logger.warning(f"HyDE generation failed: {e}")
            return query
    
    def extract_keywords(self, query: str) -> List[str]:
        """Extract key terms from query.
        
        Args:
            query: User query
            
        Returns:
            List of keywords
        """
        template = """다음 질문에서 핵심 키워드만 추출해주세요.
제품명, 기능명, 중요한 명사 등을 쉼표로 구분해서 나열해주세요.

질문: {query}

키워드:"""
        
        prompt = PromptTemplate(template=template, input_variables=["query"])
        
        try:
            response = self.llm.invoke(prompt.format(query=query))
            keywords = [k.strip() for k in response.strip().split(',')]
            keywords = [k for k in keywords if k]
            rag_logger.debug(f"Extracted keywords: {keywords}")
            return keywords
        except Exception as e:
            rag_logger.warning(f"Keyword extraction failed: {e}")
            return query.split()
    
    def decompose_query(self, query: str) -> List[str]:
        """Decompose complex query into simpler sub-queries.
        
        Args:
            query: Complex query
            
        Returns:
            List of sub-queries
        """
        template = """다음 복잡한 질문을 더 간단한 하위 질문들로 분해해주세요.
각 하위 질문은 독립적으로 답변 가능해야 합니다.

복잡한 질문: {query}

하위 질문들 (한 줄에 하나씩):
1."""
        
        prompt = PromptTemplate(template=template, input_variables=["query"])
        
        try:
            response = self.llm.invoke(prompt.format(query=query))
            
            sub_queries = []
            lines = response.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if line and len(line) > 5:
                    cleaned = line
                    for prefix in ['1.', '2.', '3.', '4.', '5.', '-', '*']:
                        if cleaned.startswith(prefix):
                            cleaned = cleaned[len(prefix):].strip()
                    if cleaned:
                        sub_queries.append(cleaned)
            
            if not sub_queries:
                sub_queries = [query]
            
            rag_logger.debug(f"Decomposed into {len(sub_queries)} sub-queries")
            return sub_queries
            
        except Exception as e:
            rag_logger.warning(f"Query decomposition failed: {e}")
            return [query]


if __name__ == "__main__":
    processor = QueryProcessor()
    
    # Test query rewriting
    query = "갤럭시 배터리"
    rewritten = processor.rewrite_query(query)
    print(f"Original: {query}")
    print(f"Rewritten: {rewritten}\n")
    
    # Test multi-query
    queries = processor.generate_multi_queries(query, num_queries=3)
    print(f"Multi-queries:")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")
