#!/bin/bash

# 로컬 RAG 파이프라인 자동 설정 스크립트

set -e

echo "========================================="
echo "  로컬 RAG 파이프라인 설정 시작"
echo "========================================="
echo ""

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 1. Docker 확인
echo -e "${YELLOW}[1/8] Docker 확인 중...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker가 설치되어 있지 않습니다.${NC}"
    echo "Docker를 먼저 설치해주세요: https://docs.docker.com/get-docker/"
    exit 1
fi

# Docker Compose 확인 (docker-compose 또는 docker compose)
DOCKER_COMPOSE_CMD=""
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker compose"
    echo -e "${GREEN}✓ Docker Compose (v2) 확인 완료${NC}"
elif command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker-compose"
    echo -e "${GREEN}✓ Docker Compose (v1) 확인 완료${NC}"
else
    echo -e "${RED}Error: Docker Compose가 설치되어 있지 않습니다.${NC}"
    echo "Docker Compose를 먼저 설치해주세요: https://docs.docker.com/compose/install/"
    exit 1
fi
echo ""

# 2. Python 확인
echo -e "${YELLOW}[3/8] Python 확인 중...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python3가 설치되어 있지 않습니다.${NC}"
    exit 1
fi
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo -e "${GREEN}✓ Python $PYTHON_VERSION 확인 완료${NC}"
echo ""

# 3. 시스템 의존성 설치
echo -e "${YELLOW}[3/8] 시스템 의존성 설치 중...${NC}"
if command -v brew &> /dev/null; then
    echo "문서 처리에 필요한 패키지 설치 중... (macOS)"
    brew install poppler tesseract libmagic imagemagick
    echo -e "${GREEN}✓ 시스템 의존성 설치 완료${NC}"
elif command -v apt &> /dev/null; then
    echo "문서 처리에 필요한 패키지 설치 중... (Ubuntu/Debian)"
    sudo apt update && sudo apt install -y poppler-utils tesseract-ocr libmagic-dev imagemagick
    echo -e "${GREEN}✓ 시스템 의존성 설치 완료${NC}"
elif command -v yum &> /dev/null; then
    echo "문서 처리에 필요한 패키지 설치 중... (CentOS/RHEL)"
    sudo yum install -y poppler-utils tesseract libmagic imagemagick
    echo -e "${GREEN}✓ 시스템 의존성 설치 완료${NC}"
elif command -v dnf &> /dev/null; then
    echo "문서 처리에 필요한 패키지 설치 중... (Fedora)"
    sudo dnf install -y poppler-utils tesseract libmagic imagemagick
    echo -e "${GREEN}✓ 시스템 의존성 설치 완료${NC}"
else
    echo -e "${YELLOW}Warning: 지원되는 패키지 매니저를 찾을 수 없습니다.${NC}"
    echo "문서 처리에 필요한 시스템 패키지를 수동으로 설치해주세요:"
    echo "  Ubuntu/Debian: sudo apt install poppler-utils tesseract-ocr libmagic-dev imagemagick"
    echo "  CentOS/RHEL: sudo yum install poppler-utils tesseract libmagic imagemagick"
    echo "  Fedora: sudo dnf install poppler-utils tesseract libmagic imagemagick"
    echo "  macOS: brew install poppler tesseract libmagic imagemagick"
    echo ""
    read -p "계속 진행하시겠습니까? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
echo ""

# 4. Ollama 설치
echo -e "${YELLOW}[4/8] Ollama 확인 및 설치 중...${NC}"
if ! command -v ollama &> /dev/null; then
    echo "Ollama가 설치되어 있지 않습니다. 자동으로 설치합니다..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl -fsSL https://ollama.com/install.sh | sh
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew &> /dev/null; then
            brew install ollama
        else
            echo -e "${RED}Error: macOS에서는 Homebrew가 필요합니다. brew install ollama${NC}"
            exit 1
        fi
    else
        echo -e "${RED}Error: 지원되지 않는 OS입니다.${NC}"
        exit 1
    fi
    if command -v ollama &> /dev/null; then
        echo -e "${GREEN}✓ Ollama 설치 완료${NC}"
    else
        echo -e "${RED}Error: Ollama 설치 실패${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ Ollama 확인 완료${NC}"
fi
echo ""

# 5. 환경 변수 파일 생성
echo -e "${YELLOW}[5/8] 환경 변수 파일 생성 중...${NC}"
if [ ! -f .env ]; then
    cp .env.example .env
    echo -e "${GREEN}✓ .env 파일 생성 완료${NC}"
else
    echo -e "${GREEN}✓ .env 파일이 이미 존재합니다${NC}"
fi
echo ""

# 6. Docker 서비스 시작
echo -e "${YELLOW}[6/8] Docker 서비스 시작 중...${NC}"
# 확인된 Docker Compose 명령어 사용
$DOCKER_COMPOSE_CMD up -d 2>&1 | grep -v "attribute .version. is obsolete" || true
echo "Qdrant와 Redis가 시작되기를 기다리는 중..."
sleep 5
echo -e "${GREEN}✓ Docker 서비스 시작 완료${NC}"
echo ""

# 7. Python 가상환경 및 의존성 설치
echo -e "${YELLOW}[7/8] Python 의존성 설치 중...${NC}"
if [ ! -d "venv" ]; then
    echo "가상환경 생성 중..."
    python3 -m venv venv
fi

echo "의존성 설치 중... (시간이 걸릴 수 있습니다)"
source venv/bin/activate
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt
echo -e "${GREEN}✓ Python 의존성 설치 완료${NC}"
echo ""

# 7. Ollama 모델 다운로드
echo -e "${YELLOW}[8/8] Ollama 모델 다운로드${NC}"
if command -v ollama &> /dev/null; then
    echo "Ollama 서버 시작 중..."
    ollama serve &
    sleep 5
    echo "nomic-embed-text 모델 다운로드 중..."
    ollama pull nomic-embed-text
    
    echo ""
    echo "LLM 모델을 선택하세요:"
    echo "1) exaone3.5:2.4b (추천 - 가벼움, 2.4B 파라미터)"
    echo "2) llama3.2:3b (대안 - 3B 파라미터)"
    echo "3) 건너뛰기 (나중에 수동으로 설치)"
    read -p "선택 (1-3): " choice
    
    case $choice in
        1)
            echo "exaone3.5:2.4b 다운로드 중..."
            ollama pull exaone3.5:2.4b
            ;;
        2)
            echo "llama3.2:3b 다운로드 중..."
            ollama pull llama3.2:3b
            sed -i 's/OLLAMA_MODEL=exaone3.5:2.4b/OLLAMA_MODEL=llama3.2:3b/' .env
            ;;
        3)
            echo "모델 다운로드를 건너뜁니다."
            ;;
        *)
            echo "잘못된 선택입니다. 건너뜁니다."
            ;;
    esac
    echo -e "${GREEN}✓ Ollama 모델 다운로드 완료${NC}"
else
    echo -e "${YELLOW}! Ollama가 설치되지 않아 모델 다운로드를 건너뜁니다${NC}"
fi
echo ""

echo "========================================="
echo -e "${GREEN}  설정 완료!${NC}"
echo "========================================="
echo ""
echo "다음 단계:"
echo ""
echo "1. 가상환경 활성화:"
echo "   source venv/bin/activate"
echo ""
echo "2. 서비스 연결 테스트:"
echo "   python tests/test_connection.py"
echo ""
echo "3. PDF 문서 수집:"
echo "   python command_ingest.py --reset"
echo ""
echo "4. RAG 시스템 테스트:"
echo "   python tests/simple_test.py      (기본 테스트)"
echo "   python tests/quick_test.py       (고급 기능 테스트)"
echo "   python tests/final_test.py       (종합 테스트)"
echo ""
echo "5. RAG 시스템 실행:"
echo "   streamlit run command_app_streamlit.py  (웹 UI - 추천!)"
echo "   python command_enhanced_rag_pipeline.py (고급 CLI)"
echo "   python command_rag_pipeline.py          (기본 CLI)"
echo "   python api.py                   (REST API 서버)"
echo ""
echo "Docker 서비스 관리:"
if [ "$DOCKER_COMPOSE_CMD" = "docker compose" ]; then
    echo "  - 중지: docker compose stop"
    echo "  - 시작: docker compose start"
    echo "  - 상태: docker compose ps"
    echo "  - 로그: docker compose logs"
else
    echo "  - 중지: docker-compose stop"
    echo "  - 시작: docker-compose start"
    echo "  - 상태: docker-compose ps"
    echo "  - 로그: docker-compose logs"
fi
echo ""
echo "문서:"
echo "  - 기본 가이드: README.md"
echo ""
