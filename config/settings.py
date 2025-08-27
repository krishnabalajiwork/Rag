import os
from typing import List, Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Elasticsearch Configuration
    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_username: str = "elastic"
    elasticsearch_password: str = "changeme"
    
    # Google Drive Configuration
    google_drive_folder_id: str = ""
    google_credentials_path: str = "credentials.json"
    
    # Hugging Face Configuration
    hf_token: Optional[str] = None
    
    # Application Settings
    index_name: str = "rag_documents"
    chunk_size: int = 300
    chunk_overlap: int = 50
    top_k: int = 5
    
    # LLM Configuration
    llm_model_name: str = "google/flan-t5-small"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Streamlit Configuration
    streamlit_port: int = 8501
    
    class Config:
        env_file = ".env"

settings = Settings()
