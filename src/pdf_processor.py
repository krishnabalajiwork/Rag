import os
import io
import logging
from typing import List, Dict, Any, Optional
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from google.oauth2.service_account import Credentials
import PyPDF2
import pdfplumber
import tiktoken
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self):
        self.drive_client = None
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self._initialize_drive_client()
    
    def _initialize_drive_client(self):
        """Initialize Google Drive client"""
        try:
            if os.path.exists(settings.google_credentials_path):
                # Service account authentication
                creds = Credentials.from_service_account_file(
                    settings.google_credentials_path,
                    scopes=['https://www.googleapis.com/auth/drive.readonly']
                )
                
                gauth = GoogleAuth()
                gauth.credentials = creds
                self.drive_client = GoogleDrive(gauth)
                logger.info("Google Drive client initialized")
            else:
                logger.warning("Google Drive credentials not found. Drive functionality disabled.")
                
        except Exception as e:
            logger.error(f"Error initializing Google Drive client: {e}")
            self.drive_client = None
    
    def get_pdfs_from_drive(self, folder_id: str = None) -> List[Dict[str, Any]]:
        """Get list of PDF files from Google Drive folder"""
        if not self.drive_client:
            logger.error("Google Drive client not initialized")
            return []
            
        if not folder_id:
            folder_id = settings.google_drive_folder_id
            
        if not folder_id:
            logger.error("No Google Drive folder ID provided")
            return []
        
        try:
            # Query for PDF files in the specified folder
            query = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
            file_list = self.drive_client.ListFile({'q': query}).GetList()
            
            pdf_files = []
            for file in file_list:
                pdf_info = {
                    'id': file['id'],
                    'title': file['title'],
                    'drive_url': f"https://drive.google.com/file/d/{file['id']}/view",
                    'size': int(file.get('fileSize', 0)),
                    'modified_date': file.get('modifiedDate', '')
                }
                pdf_files.append(pdf_info)
            
            logger.info(f"Found {len(pdf_files)} PDF files in Google Drive folder")
            return pdf_files
            
        except Exception as e:
            logger.error(f"Error getting PDFs from Google Drive: {e}")
            return []
    
    def download_pdf_content(self, file_id: str) -> Optional[bytes]:
        """Download PDF content from Google Drive"""
        if not self.drive_client:
            logger.error("Google Drive client not initialized")
            return None
            
        try:
            file = self.drive_client.CreateFile({'id': file_id})
            content = file.GetContentIOBuffer()
            return content.getvalue()
            
        except Exception as e:
            logger.error(f"Error downloading PDF content: {e}")
            return None
    
    def extract_text_from_pdf(self, pdf_content: bytes, filename: str) -> List[Dict[str, Any]]:
        """Extract text from PDF content and return pages with metadata"""
        pages = []
        
        try:
            # Try pdfplumber first (better text extraction)
            with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text and text.strip():
                        pages.append({
                            'text': text.strip(),
                            'page_number': page_num,
                            'filename': filename
                        })
                        
        except Exception as e1:
            logger.warning(f"pdfplumber failed for {filename}: {e1}. Trying PyPDF2...")
            
            try:
                # Fallback to PyPDF2
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    if text and text.strip():
                        pages.append({
                            'text': text.strip(),
                            'page_number': page_num,
                            'filename': filename
                        })
                        
            except Exception as e2:
                logger.error(f"Both PDF extraction methods failed for {filename}: {e2}")
                return []
        
        logger.info(f"Extracted {len(pages)} pages from {filename}")
        return pages
    
    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split text into chunks with overlap"""
        if not chunk_size:
            chunk_size = settings.chunk_size
        if not overlap:
            overlap = settings.chunk_overlap
            
        # Tokenize the text
        tokens = self.tokenizer.encode(text)
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            # Get chunk of tokens
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            
            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # Move start position with overlap
            start = end - overlap
            
            # Break if we're at the end
            if end >= len(tokens):
                break
        
        return chunks
    
    def process_pdf_to_chunks(self, file_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a single PDF file to chunks with metadata"""
        file_id = file_info['id']
        filename = file_info['title']
        drive_url = file_info['drive_url']
        
        # Download PDF content
        pdf_content = self.download_pdf_content(file_id)
        if not pdf_content:
            logger.error(f"Failed to download {filename}")
            return []
        
        # Extract text by pages
        pages = self.extract_text_from_pdf(pdf_content, filename)
        if not pages:
            logger.error(f"No text extracted from {filename}")
            return []
        
        # Process each page into chunks
        all_chunks = []
        for page in pages:
            page_text = page['text']
            page_chunks = self.chunk_text(page_text)
            
            for i, chunk_text in enumerate(page_chunks):
                chunk_id = f"{filename}_page{page['page_number']}_chunk{i+1}"
                
                chunk_doc = {
                    'text': chunk_text,
                    'metadata': {
                        'filename': filename,
                        'page_number': page['page_number'],
                        'chunk_id': chunk_id,
                        'drive_url': drive_url,
                        'file_id': file_id
                    }
                }
                all_chunks.append(chunk_doc)
        
        logger.info(f"Created {len(all_chunks)} chunks from {filename}")
        return all_chunks
    
    def process_all_pdfs(self, folder_id: str = None) -> List[Dict[str, Any]]:
        """Process all PDFs in Google Drive folder to chunks"""
        # Get list of PDF files
        pdf_files = self.get_pdfs_from_drive(folder_id)
        
        if not pdf_files:
            logger.warning("No PDF files found")
            return []
        
        all_chunks = []
        
        for file_info in pdf_files:
            try:
                chunks = self.process_pdf_to_chunks(file_info)
                all_chunks.extend(chunks)
                
            except Exception as e:
                logger.error(f"Error processing {file_info['title']}: {e}")
                continue
        
        logger.info(f"Processed {len(pdf_files)} files into {len(all_chunks)} total chunks")
        return all_chunks
    
    def process_local_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a local PDF file to chunks"""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []
        
        filename = os.path.basename(file_path)
        
        try:
            with open(file_path, 'rb') as file:
                pdf_content = file.read()
            
            # Create file info structure
            file_info = {
                'id': 'local_file',
                'title': filename,
                'drive_url': f'file://{file_path}'
            }
            
            return self.process_pdf_to_chunks(file_info)
            
        except Exception as e:
            logger.error(f"Error processing local PDF {file_path}: {e}")
            return []
    
    def get_text_stats(self, text: str) -> Dict[str, int]:
        """Get statistics about text"""
        tokens = self.tokenizer.encode(text)
        return {
            'characters': len(text),
            'words': len(text.split()),
            'tokens': len(tokens)
        }