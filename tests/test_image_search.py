"""Test image search functionality."""
import unittest
import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image, ImageDraw
from vector_store import VectorStoreManager
from command_rag_pipeline import RAGPipeline
from command_enhanced_rag_pipeline import EnhancedRAGPipeline


class TestImageSearch(unittest.TestCase):
    """Test cases for image search functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = VectorStoreManager()
        self.rag_pipeline = RAGPipeline()
        self.enhanced_pipeline = EnhancedRAGPipeline()

        # Create temporary directory for test images
        self.temp_dir = tempfile.mkdtemp()

        # Create test images
        self.test_images = []
        colors = ['red', 'blue', 'green', 'yellow']

        for color in colors:
            img_path = os.path.join(self.temp_dir, f'test_{color}.png')
            img = Image.new('RGB', (50, 50), color=color)
            draw = ImageDraw.Draw(img)
            draw.text((5, 20), color.upper(), fill='white')
            img.save(img_path)

            metadata = {
                'source': img_path,
                'filename': f'test_{color}.png',
                'type': 'test_image'
            }
            self.test_images.append((img, metadata))

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        for img, metadata in self.test_images:
            img_path = metadata['source']
            if os.path.exists(img_path):
                os.remove(img_path)
        os.rmdir(self.temp_dir)

        # Clean up vector store collections
        try:
            self.manager.delete_image_collection()
        except:
            pass

    def test_image_storage(self):
        """Test storing images in vector store."""
        # Add images to vector store
        self.manager.add_images(self.test_images)

        # Verify images were stored (by checking search returns results)
        query_image = self.test_images[0][0]  # First image
        results = self.manager.similarity_search_image(query_image, k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['metadata']['filename'], 'test_red.png')
        self.assertAlmostEqual(results[0]['score'], 1.0, places=2)

    def test_image_similarity_search(self):
        """Test image similarity search."""
        # Add images to vector store
        self.manager.add_images(self.test_images)

        # Search with red image
        red_image = None
        for img, metadata in self.test_images:
            if metadata['filename'] == 'test_red.png':
                red_image = img
                break

        self.assertIsNotNone(red_image)

        results = self.manager.similarity_search_image(red_image, k=4)

        # Should find all 4 images
        self.assertEqual(len(results), 4)

        # First result should be the query image itself
        self.assertEqual(results[0]['metadata']['filename'], 'test_red.png')
        self.assertAlmostEqual(results[0]['score'], 1.0, places=2)

        # Other results should have lower similarity scores
        for i in range(1, len(results)):
            self.assertLess(results[i]['score'], results[0]['score'])

    def test_rag_pipeline_image_search(self):
        """Test image search in RAG pipeline."""
        # Add images to vector store
        self.manager.add_images(self.test_images)

        # Test image search through RAG pipeline
        query_image = self.test_images[0][0]  # Red image
        results = self.rag_pipeline.search_similar_images(query_image, k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['metadata']['filename'], 'test_red.png')

    def test_enhanced_rag_pipeline_image_search(self):
        """Test image search in Enhanced RAG pipeline."""
        # Add images to vector store
        self.manager.add_images(self.test_images)

        # Test image search through Enhanced RAG pipeline
        query_image = self.test_images[1][0]  # Blue image
        results = self.enhanced_pipeline.search_similar_images(query_image, k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['metadata']['filename'], 'test_blue.png')

    def test_image_metadata(self):
        """Test image metadata is preserved."""
        # Add images to vector store
        self.manager.add_images(self.test_images)

        # Search and verify metadata
        query_image = self.test_images[2][0]  # Green image
        results = self.manager.similarity_search_image(query_image, k=1)

        metadata = results[0]['metadata']
        self.assertEqual(metadata['filename'], 'test_green.png')
        self.assertEqual(metadata['type'], 'test_image')
        self.assertTrue(metadata['source'].endswith('test_green.png'))


if __name__ == '__main__':
    unittest.main()