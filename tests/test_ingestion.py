import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pdf_processor import PDFProcessor

class TestPDFProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = PDFProcessor()
    
    @patch('src.pdf_processor.os.path.exists')
    def test_initialization_without_credentials(self, mock_exists):
        """Test PDFProcessor initialization without Google Drive credentials"""
        mock_exists.return_value = False
        processor = PDFProcessor()
        self.assertIsNone(processor.drive_client)
    
    def test_chunk_text_basic(self):
        """Test basic text chunking functionality"""
        text = "This is a test sentence. " * 50  # Create longer text
        chunks = self.processor.chunk_text(text, chunk_size=20, overlap=5)
        
        self.assertGreater(len(chunks), 1)
        self.assertIsInstance(chunks, list)
        self.assertTrue(all(isinstance(chunk, str) for chunk in chunks))
    
    def test_chunk_text_small_text(self):
        """Test chunking with text smaller than chunk size"""
        text = "Short text"
        chunks = self.processor.chunk_text(text, chunk_size=100, overlap=10)
        
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)
    
    def test_get_text_stats(self):
        """Test text statistics calculation"""
        text = "Hello world! This is a test."
        stats = self.processor.get_text_stats(text)
        
        self.assertIn('characters', stats)
        self.assertIn('words', stats)
        self.assertIn('tokens', stats)
        self.assertGreater(stats['characters'], 0)
        self.assertGreater(stats['words'], 0)
        self.assertGreater(stats['tokens'], 0)
    
    @patch('src.pdf_processor.os.path.exists')
    def test_process_local_pdf_not_found(self, mock_exists):
        """Test processing non-existent local PDF"""
        mock_exists.return_value = False
        
        result = self.processor.process_local_pdf("nonexistent.pdf")
        self.assertEqual(result, [])
    
    def test_get_pdfs_from_drive_no_client(self):
        """Test getting PDFs without Drive client"""
        self.processor.drive_client = None
        result = self.processor.get_pdfs_from_drive("test_folder_id")
        self.assertEqual(result, [])

if __name__ == '__main__':
    unittest.main()