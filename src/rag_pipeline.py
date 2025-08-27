import logging
from typing import List, Dict, Any, Optional
from src.elastic_client import ElasticClient
from src.pdf_processor import PDFProcessor
from src.embeddings import EmbeddingModel
from src.llm_client import LLMClient
from src.guardrails import GuardrailsManager
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self):
        self.elastic_client = ElasticClient()
        self.pdf_processor = PDFProcessor()
        self.embedding_model = EmbeddingModel()
        self.llm_client = LLMClient()
        self.guardrails = GuardrailsManager()
        
        # Initialize index
        self.elastic_client.create_index()
    
    def ingest_documents(self, folder_id: str = None) -> Dict[str, Any]:
        """Ingest PDFs from Google Drive and index them"""
        try:
            logger.info("Starting document ingestion...")
            
            # Process PDFs to chunks
            chunks = self.pdf_processor.process_all_pdfs(folder_id)
            
            if not chunks:
                return {
                    'success': False,
                    'message': 'No documents found or processed',
                    'document_count': 0,
                    'chunk_count': 0
                }
            
            logger.info(f"Processing {len(chunks)} chunks for indexing...")
            
            # Generate embeddings for chunks
            texts = [chunk['text'] for chunk in chunks]
            
            # Generate dense embeddings
            dense_embeddings = self.embedding_model.encode_dense(texts)
            
            # Generate sparse embeddings (placeholder for ELSER)
            sparse_embeddings = self.embedding_model.encode_sparse_local(texts)
            
            # Prepare documents for indexing
            indexed_docs = []
            for i, chunk in enumerate(chunks):
                doc = {
                    'text': chunk['text'],
                    'dense_vector': dense_embeddings[i].tolist(),
                    'text_expansion': sparse_embeddings[i],  # ELSER will be handled by Elasticsearch
                    'metadata': chunk['metadata']
                }
                indexed_docs.append(doc)
            
            # Index documents
            self.elastic_client.index_documents(indexed_docs)
            
            # Get stats
            unique_files = set(chunk['metadata']['filename'] for chunk in chunks)
            
            logger.info(f"Successfully indexed {len(chunks)} chunks from {len(unique_files)} documents")
            
            return {
                'success': True,
                'message': f'Successfully indexed {len(chunks)} chunks from {len(unique_files)} documents',
                'document_count': len(unique_files),
                'chunk_count': len(chunks)
            }
            
        except Exception as e:
            logger.error(f"Error in document ingestion: {e}")
            return {
                'success': False,
                'message': f'Error during ingestion: {str(e)}',
                'document_count': 0,
                'chunk_count': 0
            }
    
    def query(self, question: str, retrieval_mode: str = "hybrid", top_k: int = None) -> Dict[str, Any]:
        """Query the RAG system"""
        if not top_k:
            top_k = settings.top_k
            
        try:
            logger.info(f"Processing query: {question[:100]}...")
            
            # Apply guardrails to query
            query_check = self.guardrails.check_query_safety(question)
            
            if query_check['should_refuse']:
                return {
                    'success': False,
                    'answer': "I cannot provide information on that topic. Please ask questions related to the available documents.",
                    'citations': [],
                    'metadata': {
                        'retrieval_mode': retrieval_mode,
                        'documents_found': 0,
                        'is_safe': False,
                        'is_grounded': False,
                        'reason': query_check['reason']
                    }
                }
            
            # Retrieve relevant documents
            if retrieval_mode == "elser_only":
                results = self._retrieve_elser_only(question, top_k)
            elif retrieval_mode == "hybrid":
                results = self._retrieve_hybrid(question, top_k)
            else:
                results = self._retrieve_bm25_only(question, top_k)
            
            if not results:
                return {
                    'success': True,
                    'answer': "I don't have any relevant documents to answer this question.",
                    'citations': [],
                    'metadata': {
                        'retrieval_mode': retrieval_mode,
                        'documents_found': 0,
                        'is_safe': True,
                        'is_grounded': False,
                        'reason': 'No relevant documents found'
                    }
                }
            
            # Generate answer
            raw_answer = self.llm_client.generate_answer(question, results)
            
            # Apply guardrails to answer
            guardrail_result = self.guardrails.apply_guardrails(question, raw_answer, results)
            
            # Format with citations
            formatted_result = self.guardrails.format_answer_with_citations(
                guardrail_result['final_answer'], 
                results
            )
            
            # Log interaction
            self.guardrails.log_interaction(question, raw_answer, guardrail_result)
            
            return {
                'success': True,
                'answer': formatted_result['answer'],
                'citations': formatted_result['citations'],
                'sources': formatted_result['sources'],
                'metadata': {
                    'retrieval_mode': retrieval_mode,
                    'documents_found': len(results),
                    'is_safe': guardrail_result['is_safe'],
                    'is_grounded': guardrail_result['is_grounded'],
                    'confidence': guardrail_result.get('confidence', 0.0),
                    'reason': guardrail_result.get('reason', '')
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'success': False,
                'answer': "I apologize, but I encountered an error while processing your question.",
                'citations': [],
                'metadata': {
                    'retrieval_mode': retrieval_mode,
                    'documents_found': 0,
                    'is_safe': True,
                    'is_grounded': False,
                    'reason': f'System error: {str(e)}'
                }
            }
    
    def _retrieve_bm25_only(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        """Retrieve using BM25 only"""
        return self.elastic_client.search_bm25(question, top_k=top_k)
    
    def _retrieve_elser_only(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        """Retrieve using ELSER only"""
        # For ELSER, we would need the model to generate query expansion
        # For now, using sparse embeddings as placeholder
        query_expansion = self.embedding_model.encode_sparse_local([question])[0]
        return self.elastic_client.search_sparse(query_expansion, top_k=top_k)
    
    def _retrieve_hybrid(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        """Retrieve using hybrid approach (BM25 + dense + ELSER)"""
        # Generate dense embedding for query
        query_vector = self.embedding_model.encode_dense([question])[0].tolist()
        
        # Generate sparse embedding for query
        query_expansion = self.embedding_model.encode_sparse_local([question])[0]
        
        # Perform hybrid search
        return self.elastic_client.hybrid_search(
            question, 
            query_vector, 
            query_expansion, 
            top_k=top_k
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and health"""
        try:
            # Check Elasticsearch connection
            es_connected = self.elastic_client.client.ping()
            
            # Check document count
            doc_count = self.elastic_client.get_document_count()
            
            # Check LLM model
            llm_loaded = self.llm_client.is_model_loaded()
            
            # Check embedding model
            embedding_loaded = self.embedding_model.dense_model is not None
            
            return {
                'elasticsearch_connected': es_connected,
                'document_count': doc_count,
                'llm_model_loaded': llm_loaded,
                'embedding_model_loaded': embedding_loaded,
                'index_name': settings.index_name,
                'system_healthy': all([es_connected, llm_loaded, embedding_loaded])
            }
            
        except Exception as e:
            logger.error(f"Error checking system status: {e}")
            return {
                'elasticsearch_connected': False,
                'document_count': 0,
                'llm_model_loaded': False,
                'embedding_model_loaded': False,
                'index_name': settings.index_name,
                'system_healthy': False,
                'error': str(e)
            }
    
    def delete_all_documents(self) -> Dict[str, Any]:
        """Delete all documents from the index"""
        try:
            self.elastic_client.delete_index()
            self.elastic_client.create_index()
            
            return {
                'success': True,
                'message': 'All documents deleted successfully'
            }
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return {
                'success': False,
                'message': f'Error deleting documents: {str(e)}'
            }