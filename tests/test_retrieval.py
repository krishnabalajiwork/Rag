import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from elastic_client import ElasticClient
from embeddings import EmbeddingModel

class TestElasticClient(unittest.TestCase):
    @patch('src.elastic_client.Elasticsearch')
    def setUp(self, mock_elasticsearch):
        """Set up test client with mocked Elasticsearch"""
        self.mock_es = MagicMock()
        mock_elasticsearch.return_value = self.mock_es
        self.mock_es.ping.return_value = True
        
        self.client = ElasticClient()
    
    def test_format_search_results(self):
        """Test formatting of search results"""
        mock_response = {
            'hits': {
                'hits': [
                    {
                        '_id': 'doc1',
                        '_score': 0.95,
                        '_source': {
                            'text': 'Test document content',
                            'metadata': {'filename': 'test.pdf'}
                        }
                    },
                    {
                        '_id': 'doc2', 
                        '_score': 0.85,
                        '_source': {
                            'text': 'Another document',
                            'metadata': {'filename': 'test2.pdf'}
                        }
                    }
                ]
            }
        }
        
        results = self.client._format_search_results(mock_response)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['id'], 'doc1')
        self.assertEqual(results[0]['score'], 0.95)
        self.assertEqual(results[0]['text'], 'Test document content')
        self.assertEqual(results[1]['id'], 'doc2')
    
    def test_reciprocal_rank_fusion(self):
        """Test Reciprocal Rank Fusion algorithm"""
        result_list_1 = [
            {'id': 'doc1', 'score': 0.95},
            {'id': 'doc2', 'score': 0.85},
            {'id': 'doc3', 'score': 0.75}
        ]
        
        result_list_2 = [
            {'id': 'doc2', 'score': 0.90},
            {'id': 'doc1', 'score': 0.80},
            {'id': 'doc4', 'score': 0.70}
        ]
        
        result_lists = [result_list_1, result_list_2]
        fused_results = self.client._reciprocal_rank_fusion(result_lists, top_k=3)
        
        self.assertEqual(len(fused_results), 3)
        # doc1 and doc2 should rank higher due to appearing in both lists
        result_ids = [doc['id'] for doc in fused_results]
        self.assertIn('doc1', result_ids)
        self.assertIn('doc2', result_ids)
    
    def test_reciprocal_rank_fusion_empty_lists(self):
        """Test RRF with empty result lists"""
        result_lists = [[], []]
        fused_results = self.client._reciprocal_rank_fusion(result_lists, top_k=5)
        
        self.assertEqual(len(fused_results), 0)
    
    def test_get_document_count_error_handling(self):
        """Test document count with Elasticsearch error"""
        self.mock_es.count.side_effect = Exception("Connection error")
        
        count = self.client.get_document_count()
        self.assertEqual(count, 0)

class TestEmbeddingModel(unittest.TestCase):
    @patch('src.embeddings.SentenceTransformer')
    def setUp(self, mock_sentence_transformer):
        """Set up test embedding model"""
        self.mock_model = MagicMock()
        mock_sentence_transformer.return_value = self.mock_model
        
        self.embedding_model = EmbeddingModel()
    
    def test_encode_dense_basic(self):
        """Test basic dense encoding"""
        # Mock the model's encode method
        self.mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        
        texts = ["Test text 1", "Test text 2"]
        embeddings = self.embedding_model.encode_dense(texts)
        
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(embeddings.shape, (2, 3))
        self.mock_model.encode.assert_called_once_with(texts, convert_to_tensor=False)
    
    def test_encode_sparse_local(self):
        """Test sparse encoding (placeholder implementation)"""
        texts = ["Test text 1", "Test text 2"]
        sparse_embeddings = self.embedding_model.encode_sparse_local(texts)
        
        self.assertEqual(len(sparse_embeddings), 2)
        self.assertIsInstance(sparse_embeddings, list)
        # Current implementation returns empty dicts as placeholders
        self.assertTrue(all(isinstance(emb, dict) for emb in sparse_embeddings))
    
    def test_get_embedding_dimension(self):
        """Test getting embedding dimensions"""
        self.mock_model.get_sentence_embedding_dimension.return_value = 384
        
        dim = self.embedding_model.get_embedding_dimension()
        self.assertEqual(dim, 384)
    
    def test_encode_dense_no_model(self):
        """Test dense encoding when model is not loaded"""
        self.embedding_model.dense_model = None
        
        with self.assertRaises(ValueError):
            self.embedding_model.encode_dense(["test"])

if __name__ == '__main__':
    unittest.main()