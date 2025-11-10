"""Test connections to all services."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config


def test_redis():
    """Test Redis connection."""
    try:
        import redis
        client = redis.Redis(
            host=config.redis.host,
            port=config.redis.port,
            decode_responses=True
        )
        client.ping()
        print("✓ Redis: Connected")
        return True
    except Exception as e:
        print(f"✗ Redis: Failed - {e}")
        return False


def test_qdrant():
    """Test Qdrant connection."""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(
            host=config.qdrant.host,
            port=config.qdrant.port
        )
        collections = client.get_collections()
        print(f"✓ Qdrant: Connected ({len(collections.collections)} collections)")
        return True
    except Exception as e:
        print(f"✗ Qdrant: Failed - {e}")
        return False


def test_ollama():
    """Test Ollama connection."""
    try:
        import requests
        response = requests.get(f"{config.ollama.base_url}/api/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            print(f"✓ Ollama: Connected ({len(models)} models available)")
            
            # Check required models
            required_models = [
                config.ollama.embedding_model,
                config.ollama.model
            ]
            
            missing = []
            for model in required_models:
                found = any(model in m for m in model_names)
                if found:
                    print(f"  ✓ {model}")
                else:
                    print(f"  ✗ {model} (not found)")
                    missing.append(model)
            
            if missing:
                print(f"\n  Missing models: {', '.join(missing)}")
                print("  Download with: ollama pull <model_name>")
                return False
            
            return True
        else:
            print(f"✗ Ollama: Failed - Status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Ollama: Failed - {e}")
        return False


def test_pdf_files():
    """Test PDF files availability."""
    try:
        from pathlib import Path
        pdf_path = Path(config.document.pdf_path)
        if not pdf_path.exists():
            print(f"✗ PDF Directory: Not found - {pdf_path}")
            return False
        
        pdf_files = list(pdf_path.glob("*.pdf"))
        if not pdf_files:
            print(f"✗ PDF Files: No PDF files found in {pdf_path}")
            return False
        
        print(f"✓ PDF Files: {len(pdf_files)} files found")
        for pdf in pdf_files:
            print(f"  - {pdf.name}")
        return True
    except Exception as e:
        print(f"✗ PDF Files: Failed - {e}")
        return False


def main():
    """Run all connection tests."""
    print("="*60)
    print("  RAG Pipeline Connection Test")
    print("="*60)
    print()
    
    results = {
        "Redis": test_redis(),
        "Qdrant": test_qdrant(),
        "Ollama": test_ollama(),
        "PDF Files": test_pdf_files()
    }
    
    print()
    print("="*60)
    print("  Test Summary")
    print("="*60)
    
    all_passed = all(results.values())
    
    for service, status in results.items():
        status_icon = "✓" if status else "✗"
        print(f"{status_icon} {service}: {'OK' if status else 'FAILED'}")
    
    print()
    
    if all_passed:
        print("✓ All tests passed! You're ready to go.")
        print()
        print("Next steps:")
        print("1. Run: python ingest.py")
        print("2. Run: python rag_pipeline.py")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        print()
        print("Common fixes:")
        print("- Start Docker services: docker-compose up -d")
        print("- Start Ollama: systemctl start ollama")
        print("- Download models: ollama pull <model_name>")
        return 1


if __name__ == "__main__":
    sys.exit(main())
