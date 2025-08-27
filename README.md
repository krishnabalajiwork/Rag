# RAG System with Elastic + Open LLM

This is a Retrieval-Augmented Generation (RAG) system built with Elasticsearch for hybrid retrieval (ELSER + dense embeddings + BM25) and open-source LLMs for answer generation.

## 🚀 Features

- **Hybrid Retrieval**: ELSER sparse embeddings + dense vectors + BM25 keyword search
- **PDF Ingestion**: Load and process PDFs from Google Drive
- **FastAPI Backend**: RESTful API endpoints for querying and ingestion
- **Streamlit UI**: Interactive web interface with citations
- **Guardrails**: Safe, grounded, and reliable responses
- **Open Source**: Uses free models only (no paid APIs)

## 📁 Project Structure

```
rag-system/
├── src/
│   ├── rag_pipeline.py          # Core RAG pipeline logic
│   ├── elastic_client.py        # Elasticsearch operations
│   ├── pdf_processor.py         # PDF ingestion and chunking
│   ├── embeddings.py            # Embedding models
│   ├── llm_client.py            # LLM integration
│   └── guardrails.py            # Safety and grounding checks
├── config/
│   └── settings.py              # Configuration settings
├── tests/
│   ├── test_ingestion.py        # Unit tests for ingestion
│   └── test_retrieval.py        # Unit tests for retrieval
├── api.py                       # FastAPI backend
├── streamlit_app.py             # Streamlit frontend
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd rag-system
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Setup

Create a `.env` file in the project root:

```env
# Elasticsearch Configuration
ELASTICSEARCH_URL=http://localhost:9200
ELASTICSEARCH_USERNAME=elastic
ELASTICSEARCH_PASSWORD=your_password

# Google Drive Configuration  
GOOGLE_DRIVE_FOLDER_ID=your_folder_id
GOOGLE_CREDENTIALS_PATH=path/to/credentials.json

# Hugging Face Configuration
HF_TOKEN=your_huggingface_token

# Application Settings
INDEX_NAME=rag_documents
CHUNK_SIZE=300
CHUNK_OVERLAP=50
TOP_K=5
```

### 4. Start Elasticsearch

#### Using Docker:
```bash
docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  docker.elastic.co/elasticsearch/elasticsearch:8.11.1
```

#### Or install locally following [Elasticsearch installation guide](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html)

### 5. Google Drive Setup

1. Create a Google Cloud Project
2. Enable Google Drive API
3. Create service account credentials
4. Download credentials JSON file
5. Share your Google Drive folder with the service account email
6. Get the folder ID from the Drive URL

## 🚀 Running the Application

### Method 1: Run Both Services

1. **Start FastAPI Backend:**
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Start Streamlit Frontend:**
   ```bash
   streamlit run streamlit_app.py --server.port 8501
   ```

3. **Access the Application:**
   - Streamlit UI: http://localhost:8501
   - FastAPI Docs: http://localhost:8000/docs

### Method 2: Streamlit Only (Recommended for Streamlit Cloud)

```bash
streamlit run streamlit_app.py
```

## 📚 Usage

### 1. Document Ingestion

- Use the Streamlit interface to trigger PDF ingestion from Google Drive
- Or call the API endpoint: `POST /ingest`

### 2. Querying

- Enter questions in the Streamlit chat interface
- Toggle between ELSER-only and Hybrid retrieval modes
- View answers with citations and source links

### 3. API Endpoints

- `POST /query` - Submit questions and get answers
- `POST /ingest` - Load/re-index documents  
- `GET /healthz` - Health check

## 🔧 Configuration

### Retrieval Modes

- **ELSER-only**: Uses Elastic Learned Sparse Encoder for semantic search
- **Hybrid**: Combines ELSER + dense embeddings + BM25 with Reciprocal Rank Fusion

### Embedding Models

- Dense: `sentence-transformers/all-MiniLM-L6-v2`
- Sparse: ELSER v2 model in Elasticsearch

### LLM Models

Default models (configurable):
- `microsoft/DialoGPT-medium`
- `google/flan-t5-base`  
- Or any Hugging Face model

## 🧪 Testing

Run unit tests:

```bash
pytest tests/ -v
```

## 🚢 Deployment

### Streamlit Cloud

1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Add secrets in Streamlit Cloud dashboard
4. Deploy automatically

### Local Docker

```bash
# Build and run with docker-compose
docker-compose up -d
```

## 🔒 Security & Guardrails

- Input sanitization and validation
- Refusal of unsafe/harmful queries  
- Evidence grounding checks
- Rate limiting and timeout protection

## 📈 Performance

- Target latency: ≤ 3 seconds for queries
- Supports datasets up to 10,000 documents
- Configurable chunk size and retrieval parameters

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **Elasticsearch Connection Error**
   - Check if Elasticsearch is running
   - Verify URL and credentials in `.env`

2. **Google Drive Access Error**  
   - Ensure credentials file path is correct
   - Check service account permissions
   - Verify folder sharing settings

3. **Memory Issues**
   - Reduce `CHUNK_SIZE` or `TOP_K`
   - Use smaller embedding models
   - Increase system memory allocation

4. **Streamlit Deployment Issues**
   - Check secrets configuration
   - Verify all dependencies in requirements.txt
   - Check Streamlit Cloud logs

## 📞 Support

For issues and questions:
- Create an issue in the GitHub repository
- Check the troubleshooting section above
- Review Elasticsearch and Streamlit documentation

---

Built with ❤️ using Elasticsearch, FastAPI, Streamlit, and Hugging Face 🤗