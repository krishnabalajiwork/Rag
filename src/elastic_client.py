import logging
from typing import List, Dict, Any, Optional
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError
from langchain_elasticsearch import ElasticsearchStore
from langchain_elasticsearch.vectorstores import BM25Strategy, DenseVectorStrategy, SparseVectorStrategy
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ElasticClient:
    def __init__(self):
        self.client = None
        self.vector_store = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Elasticsearch client"""
        try:
            self.client = Elasticsearch(
                settings.elasticsearch_url,
                basic_auth=(settings.elasticsearch_username, settings.elasticsearch_password),
                verify_certs=False,
                request_timeout=30
            )
            
            # Test connection
            if self.client.ping():
                logger.info("Connected to Elasticsearch")
            else:
                logger.error("Failed to connect to Elasticsearch")
                
        except Exception as e:
            logger.error(f"Error initializing Elasticsearch client: {e}")
            raise
    
    def create_index(self, index_name: str = None):
        """Create index with proper mappings for hybrid search"""
        if not index_name:
            index_name = settings.index_name
            
        mapping = {
            "properties": {
                "text": {"type": "text"},
                "text_expansion": {
                    "type": "sparse_vector"
                },
                "dense_vector": {
                    "type": "dense_vector",
                    "dims": 384,  # all-MiniLM-L6-v2 dimensions
                    "index": True,
                    "similarity": "cosine"
                },
                "metadata": {
                    "properties": {
                        "filename": {"type": "keyword"},
                        "chunk_id": {"type": "keyword"},
                        "drive_url": {"type": "keyword"},
                        "page_number": {"type": "integer"}
                    }
                }
            }
        }
        
        try:
            if not self.client.indices.exists(index=index_name):
                self.client.indices.create(index=index_name, body={"mappings": mapping})
                logger.info(f"Created index: {index_name}")
            else:
                logger.info(f"Index {index_name} already exists")
                
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise
    
    def index_documents(self, documents: List[Dict[str, Any]], index_name: str = None):
        """Index documents with hybrid embeddings"""
        if not index_name:
            index_name = settings.index_name
            
        try:
            for doc in documents:
                self.client.index(
                    index=index_name,
                    body=doc
                )
            
            self.client.indices.refresh(index=index_name)
            logger.info(f"Indexed {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            raise
    
    def search_bm25(self, query: str, index_name: str = None, top_k: int = None) -> List[Dict]:
        """BM25 keyword search"""
        if not index_name:
            index_name = settings.index_name
        if not top_k:
            top_k = settings.top_k
            
        search_body = {
            "query": {
                "match": {
                    "text": query
                }
            },
            "size": top_k
        }
        
        try:
            response = self.client.search(index=index_name, body=search_body)
            return self._format_search_results(response)
        except Exception as e:
            logger.error(f"Error in BM25 search: {e}")
            return []
    
    def search_dense(self, query_vector: List[float], index_name: str = None, top_k: int = None) -> List[Dict]:
        """Dense vector search"""
        if not index_name:
            index_name = settings.index_name
        if not top_k:
            top_k = settings.top_k
            
        search_body = {
            "query": {
                "knn": {
                    "field": "dense_vector",
                    "query_vector": query_vector,
                    "k": top_k,
                    "num_candidates": top_k * 2
                }
            },
            "size": top_k
        }
        
        try:
            response = self.client.search(index=index_name, body=search_body)
            return self._format_search_results(response)
        except Exception as e:
            logger.error(f"Error in dense search: {e}")
            return []
    
    def search_sparse(self, query_expansion: Dict, index_name: str = None, top_k: int = None) -> List[Dict]:
        """ELSER sparse vector search"""
        if not index_name:
            index_name = settings.index_name
        if not top_k:
            top_k = settings.top_k
            
        search_body = {
            "query": {
                "text_expansion": {
                    "text_expansion": {
                        "model_id": ".elser_model_2",
                        "model_text": query_expansion
                    }
                }
            },
            "size": top_k
        }
        
        try:
            response = self.client.search(index=index_name, body=search_body)
            return self._format_search_results(response)
        except Exception as e:
            logger.error(f"Error in sparse search: {e}")
            return []
    
    def hybrid_search(self, query: str, query_vector: List[float], query_expansion: Dict, 
                     index_name: str = None, top_k: int = None) -> List[Dict]:
        """Hybrid search combining BM25, dense, and ELSER with RRF"""
        if not top_k:
            top_k = settings.top_k
            
        # Get results from each method
        bm25_results = self.search_bm25(query, index_name, top_k)
        dense_results = self.search_dense(query_vector, index_name, top_k)
        sparse_results = self.search_sparse(query_expansion, index_name, top_k)
        
        # Apply Reciprocal Rank Fusion
        return self._reciprocal_rank_fusion([bm25_results, dense_results, sparse_results], top_k)
    
    def _reciprocal_rank_fusion(self, result_lists: List[List[Dict]], top_k: int, k: int = 60) -> List[Dict]:
        """Apply Reciprocal Rank Fusion to combine multiple result lists"""
        doc_scores = {}
        
        for results in result_lists:
            for rank, doc in enumerate(results, 1):
                doc_id = doc.get('id', doc.get('_id', ''))
                if doc_id:
                    if doc_id not in doc_scores:
                        doc_scores[doc_id] = {'doc': doc, 'score': 0}
                    doc_scores[doc_id]['score'] += 1.0 / (k + rank)
        
        # Sort by combined score and return top_k
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x['score'], reverse=True)
        return [item['doc'] for item in sorted_docs[:top_k]]
    
    def _format_search_results(self, response: Dict) -> List[Dict]:
        """Format Elasticsearch response to standard format"""
        results = []
        for hit in response['hits']['hits']:
            result = {
                'id': hit['_id'],
                'score': hit['_score'],
                'text': hit['_source'].get('text', ''),
                'metadata': hit['_source'].get('metadata', {})
            }
            results.append(result)
        return results
    
    def delete_index(self, index_name: str = None):
        """Delete index"""
        if not index_name:
            index_name = settings.index_name
            
        try:
            if self.client.indices.exists(index=index_name):
                self.client.indices.delete(index=index_name)
                logger.info(f"Deleted index: {index_name}")
        except Exception as e:
            logger.error(f"Error deleting index: {e}")
            raise
    
    def get_document_count(self, index_name: str = None) -> int:
        """Get document count in index"""
        if not index_name:
            index_name = settings.index_name
            
        try:
            response = self.client.count(index=index_name)
            return response['count']
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0