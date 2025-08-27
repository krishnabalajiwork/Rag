import logging
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingModel:
    def __init__(self):
        self.dense_model = None
        self.sparse_model = None
        self._load_models()
    
    def _load_models(self):
        """Load embedding models"""
        try:
            # Load dense embedding model
            self.dense_model = SentenceTransformer(settings.embedding_model_name)
            logger.info(f"Loaded dense embedding model: {settings.embedding_model_name}")
            
        except Exception as e:
            logger.error(f"Error loading embedding models: {e}")
            raise
    
    def encode_dense(self, texts: List[str]) -> np.ndarray:
        """Generate dense embeddings for texts"""
        if not self.dense_model:
            raise ValueError("Dense model not loaded")
            
        try:
            embeddings = self.dense_model.encode(texts, convert_to_tensor=False)
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"Error generating dense embeddings: {e}")
            raise
    
    def encode_sparse_local(self, texts: List[str]) -> List[dict]:
        """Generate sparse embeddings using local SPLADE model (fallback)"""
        # This is a simplified implementation
        # In practice, you would use a proper SPLADE model
        try:
            # For now, return empty sparse vectors as placeholder
            # In real implementation, use models like naver/splade-v3-distilbert
            sparse_vectors = []
            for text in texts:
                # Placeholder sparse vector
                sparse_vectors.append({})
            
            logger.info(f"Generated {len(sparse_vectors)} sparse embeddings (placeholder)")
            return sparse_vectors
            
        except Exception as e:
            logger.error(f"Error generating sparse embeddings: {e}")
            return []
    
    def get_embedding_dimension(self) -> int:
        """Get dimension of dense embeddings"""
        if self.dense_model:
            return self.dense_model.get_sentence_embedding_dimension()
        return 384  # Default for all-MiniLM-L6-v2