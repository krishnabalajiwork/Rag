import streamlit as st
import requests
import json
import time
from typing import Dict, Any, List
import logging


es_url = st.secrets["ELASTICSEARCH_URL"]
es_user = st.secrets["ELASTICSEARCH_USERNAME"]
es_pass = st.secrets["ELASTICSEARCH_PASSWORD"]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="RAG System - Document Q&A",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .status-healthy {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .status-unhealthy {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .citation-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .sidebar-info {
        background-color: #e9ecef;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = "http://localhost:8000"  # Change this for deployment

class RAGInterface:
    def __init__(self):
        self.api_base_url = API_BASE_URL
        
    def call_api(self, endpoint: str, method: str = "GET", data: Dict = None) -> Dict[str, Any]:
        """Make API calls with error handling"""
        url = f"{self.api_base_url}{endpoint}"
        
        try:
            if method == "GET":
                response = requests.get(url, timeout=30)
            elif method == "POST":
                response = requests.post(url, json=data, timeout=30)
            elif method == "DELETE":
                response = requests.delete(url, timeout=30)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.ConnectionError:
            return {"error": "Cannot connect to the API server. Please make sure the FastAPI server is running."}
        except requests.exceptions.Timeout:
            return {"error": "Request timed out. Please try again."}
        except requests.exceptions.HTTPError as e:
            return {"error": f"HTTP error: {e.response.status_code} - {e.response.text}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status"""
        return self.call_api("/healthz")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get detailed system status"""
        return self.call_api("/status")
    
    def query_documents(self, question: str, retrieval_mode: str = "hybrid", top_k: int = 5) -> Dict[str, Any]:
        """Query the RAG system"""
        data = {
            "question": question,
            "retrieval_mode": retrieval_mode,
            "top_k": top_k
        }
        return self.call_api("/query", method="POST", data=data)
    
    def ingest_documents(self, folder_id: str = None) -> Dict[str, Any]:
        """Ingest documents from Google Drive"""
        data = {"folder_id": folder_id} if folder_id else {}
        return self.call_api("/ingest", method="POST", data=data)
    
    def delete_documents(self) -> Dict[str, Any]:
        """Delete all documents"""
        return self.call_api("/documents", method="DELETE")

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "system_status" not in st.session_state:
        st.session_state.system_status = None
    if "rag_interface" not in st.session_state:
        st.session_state.rag_interface = RAGInterface()

def display_system_status():
    """Display system status in sidebar"""
    st.sidebar.header("🔍 System Status")
    
    # Get status
    status = st.session_state.rag_interface.get_health_status()
    
    if "error" not in status:
        if status.get("status") == "healthy":
            st.sidebar.markdown("""
            <div class="status-box status-healthy">
                <strong>✅ System Healthy</strong><br>
                All components operational
            </div>
            """, unsafe_allow_html=True)
        else:
            st.sidebar.markdown("""
            <div class="status-box status-unhealthy">
                <strong>⚠️ System Issues</strong><br>
                Some components may be offline
            </div>
            """, unsafe_allow_html=True)
        
        # Display component status
        st.sidebar.subheader("Components")
        components = {
            "Elasticsearch": status.get("elasticsearch", False),
            "LLM Model": status.get("llm_model", False),
            "Embedding Model": status.get("embedding_model", False)
        }
        
        for component, is_healthy in components.items():
            emoji = "✅" if is_healthy else "❌"
            st.sidebar.write(f"{emoji} {component}")
        
        # Document count
        doc_count = status.get("document_count", 0)
        st.sidebar.metric("Documents Indexed", doc_count)
        
    else:
        st.sidebar.error(f"Status check failed: {status['error']}")

def display_document_management():
    """Display document management section"""
    st.sidebar.header("📚 Document Management")
    
    # Document ingestion
    st.sidebar.subheader("Ingest Documents")
    
    folder_id = st.sidebar.text_input(
        "Google Drive Folder ID",
        help="Leave empty to use default folder from settings"
    )
    
    if st.sidebar.button("🔄 Ingest Documents", type="secondary"):
        with st.spinner("Ingesting documents..."):
            result = st.session_state.rag_interface.ingest_documents(folder_id if folder_id else None)
            
            if "error" not in result:
                if result.get("success"):
                    st.sidebar.success(f"✅ {result['message']}")
                else:
                    st.sidebar.error(f"❌ {result['message']}")
            else:
                st.sidebar.error(f"❌ {result['error']}")
    
    # Document deletion
    st.sidebar.subheader("Reset System")
    if st.sidebar.button("🗑️ Delete All Documents", type="secondary"):
        if st.sidebar.checkbox("I understand this will delete all documents"):
            with st.spinner("Deleting documents..."):
                result = st.session_state.rag_interface.delete_documents()
                
                if "error" not in result:
                    if result.get("success"):
                        st.sidebar.success("✅ All documents deleted")
                    else:
                        st.sidebar.error(f"❌ {result['message']}")
                else:
                    st.sidebar.error(f"❌ {result['error']}")

def display_query_settings():
    """Display query settings"""
    st.sidebar.header("⚙️ Query Settings")
    
    retrieval_mode = st.sidebar.selectbox(
        "Retrieval Mode",
        ["hybrid", "elser_only", "bm25_only"],
        index=0,
        help="Choose the retrieval strategy"
    )
    
    top_k = st.sidebar.slider(
        "Number of Documents",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of documents to retrieve"
    )
    
    return retrieval_mode, top_k

def display_chat_interface():
    """Display the main chat interface"""
    st.markdown('<div class="main-header">🤖 RAG System - Document Q&A</div>', unsafe_allow_html=True)
    
    # Description
    st.markdown("""
    Welcome to the RAG (Retrieval-Augmented Generation) system! Ask questions about your uploaded documents 
    and get accurate, cited answers.
    
    **Features:**
    - 🔍 Hybrid search combining BM25, dense vectors, and ELSER
    - 📚 Google Drive integration for document ingestion
    - 🛡️ Built-in guardrails for safe and grounded responses
    - 📎 Automatic citations with source links
    """)
    
    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display citations if available
            if message["role"] == "assistant" and "citations" in message:
                display_citations(message["citations"])

def display_citations(citations: List[Dict]):
    """Display citations in a formatted way"""
    if not citations:
        return
    
    st.markdown("**Sources:**")
    for citation in citations:
        with st.expander(f"📄 {citation['filename']} (Page {citation['page_number']})"):
            st.markdown(f"**Relevance Score:** {citation.get('relevance_score', 0):.3f}")
            st.markdown(f"**Snippet:** {citation['snippet']}")
            if citation.get('url') and citation['url'] != '#':
                st.markdown(f"[🔗 View Document]({citation['url']})")

def main():
    """Main application"""
    initialize_session_state()
    
    # Sidebar
    display_system_status()
    display_document_management()
    retrieval_mode, top_k = display_query_settings()
    
    # Main interface
    display_chat_interface()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            with st.spinner("Thinking..."):
                # Query the RAG system
                result = st.session_state.rag_interface.query_documents(
                    question=prompt,
                    retrieval_mode=retrieval_mode,
                    top_k=top_k
                )
                
                if "error" not in result:
                    if result.get("success"):
                        answer = result["answer"]
                        citations = result.get("citations", [])
                        metadata = result.get("metadata", {})
                        
                        # Display answer
                        message_placeholder.markdown(answer)
                        
                        # Display metadata
                        with st.expander("🔍 Query Details"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Documents Found", metadata.get("documents_found", 0))
                            with col2:
                                st.metric("Confidence", f"{metadata.get('confidence', 0):.2f}")
                            with col3:
                                safety_status = "✅ Safe" if metadata.get("is_safe", False) else "⚠️ Filtered"
                                st.metric("Safety", safety_status)
                        
                        # Display citations
                        if citations:
                            display_citations(citations)
                        
                        # Add assistant message to chat history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "citations": citations,
                            "metadata": metadata
                        })
                        
                    else:
                        error_msg = f"Query failed: {result.get('answer', 'Unknown error')}"
                        message_placeholder.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                else:
                    error_msg = f"API Error: {result['error']}"
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        RAG System v1.0 | Built with Elasticsearch, FastAPI, and Streamlit
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
