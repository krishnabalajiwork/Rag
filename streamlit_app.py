import streamlit as st
import requests
import json
import time
from typing import Dict, Any, List, Optional
import logging

# Use Streamlit secrets or fallback to environment variables
API_BASE_URL = st.secrets.get("API_BASE_URL") or "http://localhost:8000"
ES_URL = st.secrets.get("ELASTICSEARCH_URL", "")
ES_USER = st.secrets.get("ELASTICSEARCH_USERNAME", "")
ES_PASS = st.secrets.get("ELASTICSEARCH_PASSWORD", "")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="RAG System - Upload Documents",
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

class RAGInterface:
    def __init__(self):
        self.api_base_url = API_BASE_URL
        
    def call_api(self, endpoint: str, method: str = "GET", data: Optional[dict] = None, files: Optional[list] = None) -> Dict[str, Any]:
        url = f"{self.api_base_url}{endpoint}"
        try:
            if method == "GET":
                response = requests.get(url, timeout=30)
            elif method == "POST":
                if files:
                    response = requests.post(url, files=files, data=data, timeout=60)
                else:
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
    def ingest_documents(self, files) -> Dict[str, Any]:
        file_data = []
        for uploaded_file in files:
            file_data.append(("files", (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)))
        return self.call_api("/ingest", method="POST", files=file_data)
    
    def get_health_status(self) -> Dict[str, Any]:
        return self.call_api("/healthz")

def initialize_session_state():
    if "system_status" not in st.session_state:
        st.session_state.system_status = None
    if "rag_interface" not in st.session_state:
        st.session_state.rag_interface = RAGInterface()
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

def display_system_status():
    st.sidebar.header("🔍 System Status")
    status = st.session_state.rag_interface.get_health_status()
    if "error" not in status:
        if status.get("status") == "healthy":
            st.sidebar.markdown("""
            <div class="status-box status-healthy">
                <strong>✅ System Healthy</strong><br>
                All components operational
            </div>""", unsafe_allow_html=True)
        else:
            st.sidebar.markdown("""
            <div class="status-box status-unhealthy">
                <strong>⚠️ System Issues</strong><br>
                Some components may be offline
            </div>""", unsafe_allow_html=True)
    else:
        st.sidebar.error(f"Status check failed: {status['error']}")

def main():
    st.markdown('<div class="main-header">📄 Upload Documents (Max 20 PDF files)</div>', unsafe_allow_html=True)
    initialize_session_state()
    display_system_status()

    uploaded_files = st.file_uploader(
        label="Upload PDF Documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload up to 20 PDF files",
        key="file_uploader"
    )
    if uploaded_files:
        if len(uploaded_files) > 20:
            st.error("You can upload a maximum of 20 files at a time. Please reduce the number of files.")
            return
        st.session_state.uploaded_files = uploaded_files
    if st.button("📤 Upload & Process", key="upload_process_btn") and st.session_state.uploaded_files:
        with st.spinner("Uploading and processing documents..."):
            result = st.session_state.rag_interface.ingest_documents(st.session_state.uploaded_files)
            if "error" not in result:
                if result.get("success"):
                    st.success(f"✅ {result['message']}")
                else:
                    st.error(f"❌ {result['message']}")
            else:
                st.error(f"❌ Error: {result['error']}")
    elif st.button("Upload Warning Placeholder", key="upload_warn_btn"):
        # This branch only triggers if user presses button without files;
        # to avoid duplicate StreamlitDuplicateElementId error
        st.warning("Please upload PDF files before processing.")

if __name__ == "__main__":
    main()
