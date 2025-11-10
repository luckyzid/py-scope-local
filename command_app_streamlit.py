"""Streamlit web UI for Enhanced RAG system."""
import streamlit as st
import time
from datetime import datetime
from command_enhanced_rag_pipeline import EnhancedRAGPipeline
from logger import rag_logger
import pandas as pd
import json


# Page configuration
st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stTextInput > div > div > input {
        font-size: 1.1rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .metric-box {
        background-color: #e8f4f8;
        padding: 0.5rem;
        border-radius: 0.3rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_pipeline():
    """Initialize RAG pipeline (cached)."""
    try:
        pipeline = EnhancedRAGPipeline(
            use_cache=True,
            use_hybrid_search=True,
            use_conversation=True,
            use_evaluation=True
        )
        return pipeline, None
    except Exception as e:
        return None, str(e)


def render_sidebar():
    """Render sidebar with settings and statistics."""
    st.sidebar.title("⚙️ 설정")
    
    # Advanced features toggles
    st.sidebar.subheader("고급 기능")
    use_query_enhancement = st.sidebar.checkbox(
        "쿼리 향상 (Multi-Query)",
        value=True,
        help="질문을 여러 방식으로 변형하여 검색"
    )
    use_self_rag = st.sidebar.checkbox(
        "Self-RAG (자가 평가)",
        value=True,
        help="문서 관련성 및 답변 품질 자동 평가"
    )
    adaptive = st.sidebar.checkbox(
        "Adaptive RAG (적응형)",
        value=True,
        help="질문 복잡도에 따라 전략 자동 조정"
    )
    
    st.sidebar.divider()
    
    # Statistics
    st.sidebar.subheader("📊 통계")
    if 'query_count' in st.session_state:
        col1, col2 = st.sidebar.columns(2)
        col1.metric("총 질문", st.session_state.query_count)
        col2.metric("평균 시간", f"{st.session_state.avg_time:.2f}s")
    
    st.sidebar.divider()
    
    # Actions
    st.sidebar.subheader("🔧 작업")
    if st.sidebar.button("📝 대화 저장"):
        if st.session_state.pipeline:
            st.session_state.pipeline.save_conversation()
            st.sidebar.success("✅ 대화가 저장되었습니다!")
    
    if st.sidebar.button("🗑️ 대화 초기화"):
        st.session_state.messages = []
        st.session_state.query_count = 0
        st.session_state.total_time = 0.0
        st.sidebar.success("✅ 대화가 초기화되었습니다!")
    
    if st.sidebar.button("📊 평가 리포트"):
        st.session_state.show_report = True
    
    return use_query_enhancement, use_self_rag, adaptive


def render_message(role, content, metadata=None):
    """Render a chat message."""
    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.write(content)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.write(content)
            
            # Show metadata if available
            if metadata:
                with st.expander("📊 상세 정보"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "응답 시간",
                            f"{metadata.get('response_time', 0):.2f}s"
                        )
                    
                    with col2:
                        st.metric(
                            "출처 개수",
                            metadata.get('num_sources', 0)
                        )
                    
                    with col3:
                        if metadata.get('critique'):
                            quality = metadata['critique'].get('overall', 0)
                            st.metric(
                                "답변 품질",
                                f"{quality:.2f}/1.0"
                            )
                
                # Show sources
                sources = metadata.get('sources', [])
                if sources:
                    st.subheader("📚 참고 문서")
                    for i, source in enumerate(sources, 1):
                        with st.container():
                            st.markdown(f"**{i}. {source.get('filename', 'Unknown')}**")
                            st.text(source.get('content', '')[:200] + "...")
                            st.divider()


def main():
    """Main Streamlit app."""
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'query_count' not in st.session_state:
        st.session_state.query_count = 0
    if 'total_time' not in st.session_state:
        st.session_state.total_time = 0.0
    if 'show_report' not in st.session_state:
        st.session_state.show_report = False
    
    # Header
    st.markdown('<div class="main-header">🤖 RAG Q&A System</div>', unsafe_allow_html=True)
    st.markdown("##### 로컬 LLM 기반 문서 질의응답 시스템")
    
    # Initialize pipeline
    if st.session_state.pipeline is None:
        with st.spinner("🚀 시스템 초기화 중..."):
            pipeline, error = initialize_pipeline()
            if error:
                st.error(f"❌ 초기화 실패: {error}")
                st.info("""
                **문제 해결:**
                1. Docker 서비스 확인: `docker-compose ps`
                2. Ollama 실행 확인: `ollama list`
                3. 문서 수집 확인: `python ingest.py`
                """)
                st.stop()
            else:
                st.session_state.pipeline = pipeline
                st.success("✅ 시스템이 준비되었습니다!")
                time.sleep(1)
                st.rerun()
    
    # Sidebar
    use_query_enhancement, use_self_rag, adaptive = render_sidebar()
    
    # Show evaluation report if requested
    if st.session_state.show_report:
        st.subheader("📊 평가 리포트")
        report = st.session_state.pipeline.get_evaluation_report()
        st.text(report)
        st.session_state.show_report = False
    
    # Display chat history
    for message in st.session_state.messages:
        render_message(
            message['role'],
            message['content'],
            message.get('metadata')
        )
    
    # Chat input
    if prompt := st.chat_input("질문을 입력하세요..."):
        # Add user message
        st.session_state.messages.append({
            'role': 'user',
            'content': prompt
        })
        render_message('user', prompt)
        
        # Get response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("답변 생성 중..."):
                try:
                    result = st.session_state.pipeline.query(
                        question=prompt,
                        use_query_enhancement=use_query_enhancement,
                        use_self_rag=use_self_rag,
                        adaptive=adaptive
                    )
                    
                    answer = result['answer']
                    metadata = result.get('metadata', {})
                    metadata['sources'] = result.get('sources', [])
                    
                    # Update statistics
                    st.session_state.query_count += 1
                    st.session_state.total_time += metadata.get('response_time', 0)
                    st.session_state.avg_time = st.session_state.total_time / st.session_state.query_count
                    
                    # Display answer
                    st.write(answer)
                    
                    # Show metadata
                    with st.expander("📊 상세 정보", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "응답 시간",
                                f"{metadata.get('response_time', 0):.2f}s"
                            )
                        
                        with col2:
                            st.metric(
                                "출처 개수",
                                metadata.get('num_sources', 0)
                            )
                        
                        with col3:
                            if metadata.get('critique'):
                                quality = metadata['critique'].get('overall', 0)
                                st.metric(
                                    "답변 품질",
                                    f"{quality:.2f}/1.0"
                                )
                        
                        # Show strategy if adaptive
                        if adaptive and metadata.get('strategy'):
                            st.json(metadata['strategy'])
                    
                    # Show sources
                    sources = metadata.get('sources', [])
                    if sources:
                        with st.expander(f"📚 참고 문서 ({len(sources)}개)", expanded=False):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**{i}. {source.get('filename', 'Unknown')}**")
                                st.text(source.get('content', '')[:300] + "...")
                                if i < len(sources):
                                    st.divider()
                    
                    # Add to messages
                    st.session_state.messages.append({
                        'role': 'assistant',
                        'content': answer,
                        'metadata': metadata
                    })
                    
                except Exception as e:
                    st.error(f"❌ 오류 발생: {str(e)}")
                    rag_logger.error(f"Streamlit query error: {e}")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.9rem;'>
        Powered by Ollama, LangChain, Qdrant, Redis | 100% Open Source
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
