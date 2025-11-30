"""
Document Intelligence System - Streamlit Application
Python 3.15.5
Fully Fixed & Optimized with Enhanced UI
"""

import os
import tempfile
import shutil
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import streamlit as st
import pandas as pd
import requests
from dotenv import load_dotenv

# Fix for torch classes error - disable Streamlit file watcher for torch
os.environ['STREAMLIT_SERVER_ENABLE_STATIC_FILE_WATCHING'] = 'false'
# Force disable the problematic watcher completely
import streamlit.web.bootstrap as bootstrap
bootstrap._on_server_start = lambda _: None

# LangChain imports
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_classic.agents import AgentExecutor, AgentType, initialize_agent
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_classic import hub

# Docling imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat, DocumentStream
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import DoclingDocument

# Streamlit extras
from streamlit_extras.bottom_container import bottom

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===========================
# Environment & Config
# ===========================
os.environ['HF_HOME'] = os.path.join(tempfile.gettempdir(), 'hf_cache')
os.environ['TRANSFORMERS_CACHE'] = os.environ['HF_HOME']
Path(os.environ['HF_HOME']).mkdir(exist_ok=True, parents=True)
load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "granite-embedding:30m")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3:1.7b")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "8"))
SUPPORTED_FORMATS = ["pdf", "docx", "pptx", "html"]

SYSTEM_PROMPT = """You are a helpful document intelligence assistant. You have access to documents that have been uploaded and processed.

GUIDELINES:
- Use the search_documents tool to find relevant information
- Be efficient: one well-crafted search is usually sufficient
- Only search again if the first results are clearly incomplete
- Provide clear, accurate answers based on the document contents
- Always cite your sources with filenames
- If information isn't found, say so clearly
- Be concise but thorough

When answering:
1. Search the documents with a focused query
2. Synthesize a clear answer from the results
3. Include source citations (filenames)
"""

# ===========================
# Helper: File Hashing (Skip Duplicates)
# ===========================
def compute_file_hash(uploaded_file) -> str:
    """Compute SHA256 hash of uploaded file to detect duplicates."""
    uploaded_file.seek(0)
    hasher = hashlib.sha256()
    hasher.update(uploaded_file.read())
    uploaded_file.seek(0)  # Reset pointer
    return hasher.hexdigest()

# ===========================
# Optimized Document Processor with Direct Docling
# ===========================
class DocumentProcessor:
    def __init__(self):
        # Configure pipeline options for optimal performance
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.do_ocr = True
        self.pipeline_options.do_table_structure = True
        self.pipeline_options.generate_picture_images = True
        self.pipeline_options.images_scale = 2.0

        # Initialize converter with optimized settings
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options),
                InputFormat.DOCX: None,  # Use default options
                InputFormat.PPTX: None,
                InputFormat.HTML: None,
            }
        )

    def _extract_document_metadata(self, docling_doc: DoclingDocument) -> Dict[str, Any]:
        """Extract comprehensive metadata from DoclingDocument."""
        metadata = {
            "filename": getattr(docling_doc, 'name', 'unknown'),
            "total_pages": len(getattr(docling_doc, 'pages', {})),
            "total_tables": len(getattr(docling_doc, 'tables', [])),
            "total_images": len(getattr(docling_doc, 'pictures', [])),
            "total_text_items": len(getattr(docling_doc, 'texts', [])),
        }
        
        # Extract text statistics
        texts = getattr(docling_doc, 'texts', [])
        if texts:
            total_chars = sum(len(getattr(text, 'text', '')) for text in texts)
            metadata["total_characters"] = total_chars
            metadata["avg_text_length"] = total_chars / len(texts)
            
        return metadata

    def _extract_structured_content(self, docling_doc: DoclingDocument) -> Dict[str, Any]:
        """Extract structured content from DoclingDocument."""
        structure = {
            "headings": [],
            "paragraphs": [],
            "tables": [],
            "images": []
        }
        
        # Extract headings and paragraphs
        for text_item in getattr(docling_doc, 'texts', []):
            text_content = getattr(text_item, 'text', '').strip()
            if not text_content:
                continue
                
            label = getattr(text_item, 'label', '').lower()
            prov = getattr(text_item, 'prov', [])
            page_no = prov[0].page_no if prov else None
            
            item_data = {
                "text": text_content,
                "page": page_no,
                "label": label
            }
            
            if "heading" in label or "title" in label:
                structure["headings"].append(item_data)
            else:
                structure["paragraphs"].append(item_data)
        
        # Extract tables
        for i, table in enumerate(getattr(docling_doc, 'tables', []), 1):
            try:
                df = table.export_to_dataframe(doc=docling_doc)
                prov = getattr(table, 'prov', [])
                page_no = prov[0].page_no if prov else None
                
                # Fixed caption extraction - call the method instead of accessing it as a property
                caption = table.caption_text(doc=docling_doc) if hasattr(table, 'caption_text') and callable(table.caption_text) else None
                
                structure["tables"].append({
                    "table_number": i,
                    "page": page_no,
                    "dataframe": df,
                    "shape": df.shape,
                    "caption": caption,
                    "is_empty": df.empty  # Add is_empty field
                })
            except Exception as e:
                logger.warning(f"Warning: Could not process table {i}: {e}")
                continue
        
        # Extract images
        for i, picture in enumerate(getattr(docling_doc, 'pictures', []), 1):
            prov = getattr(picture, 'prov', [])
            if prov:
                # Fixed caption extraction - call the method instead of accessing it as a property
                caption = picture.caption_text(doc=docling_doc) if hasattr(picture, 'caption_text') and callable(picture.caption_text) else None
                
                # Fixed bounding box extraction - convert to dictionary
                bbox_obj = getattr(prov[0], 'bbox', None)
                bbox_dict = None
                if bbox_obj:
                    bbox_dict = {
                        "left": bbox_obj.l,
                        "top": bbox_obj.t,
                        "right": bbox_obj.r,
                        "bottom": bbox_obj.b,
                    }
                
                structure["images"].append({
                    "picture_number": i,  # Add picture_number field
                    "page": prov[0].page_no,
                    "caption": caption,
                    "bounding_box": bbox_dict  # Use the converted dictionary
                })
        
        return structure

    def process_uploaded_files(self, uploaded_files) -> Tuple[List[Document], List[Dict]]:
        """Process uploaded files with optimized Docling conversion."""
        documents = []
        docling_docs = []
        temp_dir = tempfile.mkdtemp()

        try:
            for uploaded_file in uploaded_files:
                file_hash = compute_file_hash(uploaded_file)
                if file_hash in st.session_state.get("processed_hashes", set()):
                    st.info(f"⏭️ Skipping duplicate: {uploaded_file.name}")
                    continue

                st.write(f"🔄 Processing {uploaded_file.name}...")
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                
                # Write file to temporary location
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                try:
                    # Convert document using Docling
                    conversion_result = self.converter.convert(temp_path)
                    docling_doc = conversion_result.document
                    
                    # Extract markdown content
                    markdown_content = docling_doc.export_to_markdown()
                    
                    # Extract metadata
                    metadata = self._extract_document_metadata(docling_doc)
                    metadata.update({
                        "filename": uploaded_file.name,
                        "file_type": uploaded_file.type,
                        "source": uploaded_file.name,
                        "file_hash": file_hash,
                    })
                    
                    # Extract structured content
                    structured_content = self._extract_structured_content(docling_doc)
                    
                    # Create LangChain Document
                    doc = Document(
                        page_content=markdown_content,
                        metadata=metadata
                    )
                    
                    documents.append(doc)
                    docling_docs.append({
                        "filename": uploaded_file.name,
                        "doc": docling_doc,
                        "structured_content": structured_content,
                        "metadata": metadata
                    })

                    # Mark as processed
                    st.session_state.processed_hashes.add(file_hash)
                    st.success(f"✅ Processed {uploaded_file.name}")

                except Exception as e:
                    st.error(f"❌ Failed to process {uploaded_file.name}: {str(e)}")
                    logger.error(f"Failed to process {uploaded_file.name}: {str(e)}")
                    
        finally:
            # Cleanup temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)

        return documents, docling_docs

    def process_document_stream(self, file_path: str) -> Optional[DoclingDocument]:
        """Process document using DocumentStream for large files."""
        try:
            with DocumentStream(file_path) as stream:
                conversion_result = self.converter.convert(stream)
                return conversion_result.document
        except Exception as e:
            st.error(f"Stream processing failed: {e}")
            logger.error(f"Stream processing failed: {e}")
            return None


# ===========================
# Vector Store Manager
# ===========================
class VectorStoreManager:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embedding_model = st.session_state.get("embedding_model", EMBEDDING_MODEL)
        self._verify_ollama_model(self.embedding_model)

        self.embeddings = OllamaEmbeddings(
            model=self.embedding_model,
            base_url=OLLAMA_BASE_URL
        )
                
        # Optimized text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]  # Better splitting
        )

    def _verify_ollama_model(self, model_name: str):
        try:
            resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                if model_name not in models:
                    st.warning(f"Embedding model '{model_name}' not found. Run: ollama pull {model_name}")
        except:
            pass  # Fail silently if Ollama unreachable

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk documents with metadata preservation."""
        chunks = self.text_splitter.split_documents(documents)
        
        # Enhance chunks with additional metadata
        for chunk in chunks:
            # Add chunk-specific metadata
            chunk.metadata.update({
                "chunk_size": len(chunk.page_content),
                "chunk_hash": hashlib.md5(chunk.page_content.encode()).hexdigest()
            })
        
        return chunks

    def create_vectorstore(self, chunks: List[Document]) -> Chroma:
        """Create persistent Chroma vector store."""
        try:
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name="documents",
                collection_metadata={"hnsw:space": "cosine"}  # Optimize for similarity search
            )
            return vectorstore
        except Exception as e:
            st.error(f"Error creating vector store: {e}")
            logger.error(f"Error creating vector store: {e}")
            raise

    def load_existing_vectorstore(self) -> Optional[Chroma]:
        """Load existing vector store with error handling."""
        if not os.path.exists(self.persist_directory):
            return None
        try:
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name="documents"
            )
        except Exception as e:
            st.warning(f"Could not load existing vector store: {e}")
            logger.warning(f"Could not load existing vector store: {e}")
            return None


# ===========================
# Enhanced Document Structure Visualizer
# ===========================
class DocumentStructureVisualizer:
    """Enhanced visualizer using direct DoclingDocument access."""

    def __init__(self, docling_document: DoclingDocument, structured_content: Dict[str, Any] = None):
        self.doc = docling_document
        self.structured_content = structured_content

    def get_document_summary(self) -> Dict[str, Any]:
        """Get comprehensive document summary."""
        pages = getattr(self.doc, "pages", {})
        texts = getattr(self.doc, "texts", [])
        tables = getattr(self.doc, "tables", [])
        pictures = getattr(self.doc, "pictures", [])

        # Analyze text types and content
        text_types = {}
        total_chars = 0
        
        for item in texts:
            label = getattr(item, "label", "unknown")
            text_content = getattr(item, "text", "")
            text_types[label] = text_types.get(label, 0) + 1
            total_chars += len(text_content)

        return {
            "name": getattr(self.doc, "name", "Unknown"),
            "num_pages": len(pages),
            "num_texts": len(texts),
            "num_tables": len(tables),
            "num_pictures": len(pictures),
            "total_characters": total_chars,
            "avg_text_length": total_chars / len(texts) if texts else 0,
            "text_types": text_types,
        }

    def get_document_hierarchy(self) -> List[Dict[str, Any]]:
        """Extract document hierarchy with enhanced information."""
        hierarchy = []

        if self.structured_content and "headings" in self.structured_content:
            # Use pre-extracted structured content if available
            for heading in self.structured_content["headings"]:
                hierarchy.append({
                    "type": heading["label"],
                    "text": heading["text"],
                    "page": heading["page"],
                    "level": self._infer_heading_level(heading["label"]),
                })
        else:
            # Fallback to direct extraction
            if not hasattr(self.doc, "texts") or not self.doc.texts:
                return hierarchy

            for item in self.doc.texts:
                label = getattr(item, "label", "")
                text = getattr(item, "text", "").strip()
                
                if text and ("heading" in label.lower() or "title" in label.lower()):
                    prov = getattr(item, "prov", [])
                    page_no = prov[0].page_no if prov else None

                    hierarchy.append({
                        "type": label,
                        "text": text,
                        "page": page_no,
                        "level": self._infer_heading_level(label),
                    })

        return sorted(hierarchy, key=lambda x: (x["page"] or 0, x["level"]))

    def _infer_heading_level(self, label: str) -> int:
        """Infer heading level from label with improved logic."""
        label_lower = label.lower()
        if "title" in label_lower or "h1" in label_lower:
            return 1
        elif "heading1" in label_lower or "chapter" in label_lower:
            return 1
        elif "heading2" in label_lower or "section" in label_lower:
            return 2
        elif "heading3" in label_lower or "subsection" in label_lower:
            return 3
        elif "heading4" in label_lower:
            return 4
        else:
            return 2  # Default level

    def get_tables_info(self) -> List[Dict[str, Any]]:
        """Extract table information with enhanced metadata."""
        tables_info = []

        if self.structured_content and "tables" in self.structured_content:
            # Use pre-extracted tables
            tables_info = self.structured_content["tables"]
            # Ensure each table has an is_empty key
            for table in tables_info:
                if "is_empty" not in table:
                    df = table.get("dataframe")
                    table["is_empty"] = df.empty if df is not None else True
        else:
            # Fallback to direct extraction
            if not hasattr(self.doc, "tables") or not self.doc.tables:
                return tables_info

            for i, table in enumerate(self.doc.tables, 1):
                try:
                    df = table.export_to_dataframe(doc=self.doc)
                    
                    # Handle duplicate column names
                    if df.columns.duplicated().any():
                        col_counts = {}
                        new_columns = []
                        for col in df.columns:
                            if col in col_counts:
                                col_counts[col] += 1
                                new_columns.append(f"{col}_{col_counts[col]}")
                            else:
                                col_counts[col] = 0
                                new_columns.append(col)
                        df.columns = new_columns
                    
                    prov = getattr(table, "prov", [])
                    page_no = prov[0].page_no if prov else None
                    
                    # Fixed caption extraction - call the method instead of accessing it as a property
                    caption = table.caption_text(self.doc) if hasattr(table, 'caption_text') and callable(table.caption_text) else None

                    tables_info.append({
                        "table_number": i,
                        "page": page_no,
                        "caption": caption,
                        "dataframe": df,
                        "shape": df.shape,
                        "is_empty": df.empty,  # Add is_empty field
                    })
                except Exception as e:
                    logger.warning(f"Warning: Could not process table {i}: {e}")
                    continue

        return tables_info

    def get_pictures_info(self) -> List[Dict[str, Any]]:
        """Extract picture information with enhanced details."""
        pictures_info = []

        if self.structured_content and "images" in self.structured_content:
            # Use pre-extracted images
            pictures_info = self.structured_content["images"]
            # Ensure each picture has a picture_number
            for i, pic in enumerate(pictures_info, 1):
                if "picture_number" not in pic:
                    pic["picture_number"] = i
                
                # Extract PIL image from DoclingDocument
                if "pil_image" not in pic:
                    doc_pictures = getattr(self.doc, "pictures", [])
                    if i-1 < len(doc_pictures):
                        picture = doc_pictures[i-1]
                        # Fixed PIL image extraction
                        pil_image = None
                        if hasattr(picture, 'image') and picture.image and hasattr(picture.image, 'pil_image'):
                            pil_image = picture.image.pil_image
                        pic["pil_image"] = pil_image
        else:
            # Fallback to direct extraction
            if not hasattr(self.doc, "pictures") or not self.doc.pictures:
                return pictures_info

            for i, pic in enumerate(self.doc.pictures, 1):
                prov = getattr(pic, "prov", [])
                if prov:
                    page_no = prov[0].page_no
                    bbox = prov[0].bbox
                    
                    # Fixed caption extraction - call the method instead of accessing it as a property
                    caption = pic.caption_text(self.doc) if hasattr(pic, 'caption_text') and callable(pic.caption_text) else None
                    
                    # Fixed PIL image extraction
                    pil_image = None
                    if hasattr(pic, 'image') and pic.image and hasattr(pic.image, 'pil_image'):
                        pil_image = pic.image.pil_image

                    # Fixed bounding box extraction - convert to dictionary
                    bbox_dict = None
                    if bbox:
                        bbox_dict = {
                            "left": bbox.l,
                            "top": bbox.t,
                            "right": bbox.r,
                            "bottom": bbox.b,
                        }

                    pictures_info.append({
                        "picture_number": i,
                        "page": page_no,
                        "caption": caption,
                        "pil_image": pil_image,
                        "bounding_box": bbox_dict,
                    })

        return pictures_info

    def get_content_statistics(self) -> Dict[str, Any]:
        """Get detailed content statistics."""
        summary = self.get_document_summary()
        hierarchy = self.get_document_hierarchy()
        tables = self.get_tables_info()
        pictures = self.get_pictures_info()

        return {
            "summary": summary,
            "hierarchy_levels": len(set(h["level"] for h in hierarchy)),
            "total_headings": len(hierarchy),
            "total_tables": len(tables),
            "total_images": len(pictures),
            "tables_with_data": sum(1 for t in tables if not t.get("is_empty", True)),  # Use get with default
            "images_with_captions": sum(1 for p in pictures if p.get("caption")),  # Use get with default
        }


# ===========================
# Tools & Agent (Fixed)
# ===========================
@tool
def search_documents(query: str) -> str:
    """Search the uploaded documents for relevant information."""
    try:
        vectorstore = st.session_state.get("vectorstore")
        if not vectorstore:
            return "No documents available. Please upload and process documents first."
            
        results = vectorstore.similarity_search(query, k=TOP_K_RESULTS)
        if not results:
            return "No relevant information found in the documents."
        
        parts = []
        for i, doc in enumerate(results, 1):
            src = doc.metadata.get("filename", "Unknown")
            parts.append(f"[Source {i}: {src}]\n{doc.page_content.strip()}\n")
        
        return "\n---\n".join(parts)
    except Exception as e:
        logger.error(f"Search error: {e}")
        return f"Search error: {e}"

# ===========================
# Tools & Agent (FIXED)
# ===========================
def create_documentation_agent():
    """Create agent with vectorstore access"""
    
    # Get references before creating tools
    vectorstore = st.session_state.get("vectorstore")
    
    if not vectorstore:
        raise ValueError("No vectorstore found. Please process documents first.")
    
    # Create tool with vectorstore closure
    @tool
    def search_documents(query: str) -> str:
        """Search the uploaded documents for relevant information.
        
        Args:
            query: The search query to find relevant information in documents
        """
        try:
            logger.info(f"TOOL CALLED with query: {query}")
            
            # Use the vectorstore from closure
            results = vectorstore.similarity_search(query, k=TOP_K_RESULTS)
            logger.info(f"Found {len(results)} results")
            
            if not results:
                return "No relevant information found in the documents for this query."
            
            parts = []
            for i, doc in enumerate(results, 1):
                src = doc.metadata.get("filename", "Unknown")
                content = doc.page_content.strip()
                parts.append(f"[Source {i}: {src}]\n{content}\n")
            
            return "\n---\n".join(parts)
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return f"Search error: {str(e)}"
    
    # Create LLM
    llm = ChatOllama(
        model=st.session_state.get("chat_model", CHAT_MODEL),
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
    )
    
    # Create agent with the tool
    tools = [search_documents]
    memory = MemorySaver()
    
    agent = create_react_agent(
        llm, 
        tools, 
        checkpointer=memory
    )
    
    return agent


# ===========================
# Streamlit App (Enhanced UI)
# ===========================
st.set_page_config(
    page_title="Document Intelligence Assistant", 
    page_icon="📚", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #4E79A7, #F28E2B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        margin-bottom: 1rem;
        color: #555;
    }
    
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        border-left: 4px solid #4E79A7;
    }
    
    .status-success {
        color: #28a745;
        font-weight: 500;
    }
    
    .status-warning {
        color: #ffc107;
        font-weight: 500;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: 500;
    }
    
    .chat-message {
        padding: 10px 15px;
        border-radius: 18px;
        margin-bottom: 10px;
        max-width: 80%;
        word-wrap: break-word;
    }
    
    .user-message {
        background-color: #f1f0f0;
        margin-left: auto;
        text-align: right;
    }
    
    .assistant-message {
        background-color: #e3f2fd;
    }
    
    .file-upload-container {
        border: 2px dashed #4E79A7;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
        background-color: #f8f9fa;
    }
    
    .progress-container {
        margin: 15px 0;
    }
    
    .tab-container {
        margin-top: 20px;
    }
    
    .document-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
        border-left: 4px solid #F28E2B;
    }
    
    .document-title {
        font-weight: 600;
        margin-bottom: 5px;
    }
    
    .document-meta {
        font-size: 0.9rem;
        color: #666;
    }
    
    /* Sidebar styles */
    .sidebar-content {
        padding: 15px;
    }
    
    .sidebar-section {
        margin-bottom: 25px;
    }
    
    .sidebar-title {
        font-weight: 600;
        margin-bottom: 10px;
        color: #333;
    }
    
    /* Table styles */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Image styles */
    .image-container {
        text-align: center;
        margin: 15px 0;
    }
    
    .image-caption {
        font-style: italic;
        color: #666;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    defaults = {
        "uploaded_files": [],
        "vectorstore": None,
        "agent": None,
        "messages": [],
        "processing_status": "not_started",
        "docling_docs": [],
        "processed_hashes": set(),
        "embedding_model": EMBEDDING_MODEL,
        "chat_model": CHAT_MODEL,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if "vs_manager" not in st.session_state:
        st.session_state.vs_manager = VectorStoreManager()

def process_and_index(uploaded_files):
    if not uploaded_files:
        st.warning("No files uploaded.")
        return

    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        with st.spinner("🔄 Processing documents with optimized Docling..."):
            status_text.text("Initializing document processor...")
            progress_bar.progress(10)
            
            processor = DocumentProcessor()
            progress_bar.progress(20)
            
            status_text.text("Extracting content from documents...")
            docs, docling_docs = processor.process_uploaded_files(uploaded_files)
            progress_bar.progress(60)
            
            if not docs:
                st.error("No new documents were processed.")
                return
                
            st.session_state.docling_docs.extend(docling_docs)
            progress_bar.progress(70)

        with st.spinner("✂️ Chunking & embedding..."):
            status_text.text("Splitting documents into chunks...")
            chunks = st.session_state.vs_manager.chunk_documents(docs)
            progress_bar.progress(80)

        with st.spinner("🏗️ Building vector database..."):
            status_text.text("Creating vector embeddings...")
            vectorstore = st.session_state.vs_manager.create_vectorstore(chunks)
            st.session_state.vectorstore = vectorstore
            progress_bar.progress(90)

        with st.spinner("🤖 Initializing agent..."):
            status_text.text("Setting up AI assistant...")
            st.session_state.agent = create_documentation_agent()
            progress_bar.progress(100)

        st.session_state.processing_status = "completed"
        st.success("🎉 All documents indexed! You can now chat.")
        logger.info("Document processing completed successfully")
        
    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        logger.error(f"Error during processing: {str(e)}")
    finally:
        # Clear the progress indicators
        progress_bar.empty()
        status_text.empty()

def render_sidebar():
    with st.sidebar:
        # Enhanced sidebar header
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-title">⚙️ Configuration</div>', unsafe_allow_html=True)
        
        # Model selection with enhanced UI
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                models = [m["name"] for m in response.json().get("models", [])]
            else:
                models = [EMBEDDING_MODEL, CHAT_MODEL]
        except:
            models = [EMBEDDING_MODEL, CHAT_MODEL]

        st.session_state.embedding_model = st.selectbox(
            "Embedding Model", 
            options=models, 
            index=models.index(EMBEDDING_MODEL) if EMBEDDING_MODEL in models else 0,
            help="Model used for creating embeddings from document text"
        )
        
        st.session_state.chat_model = st.selectbox(
            "Chat Model", 
            options=models, 
            index=models.index(CHAT_MODEL) if CHAT_MODEL in models else 0,
            help="Model used for generating responses to your questions"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        
        # Enhanced file upload section
        st.markdown('<div class="sidebar-title">📤 Document Upload</div>', unsafe_allow_html=True)
        
        # File upload with drag and drop
        with st.container():
            st.markdown('<div class="file-upload-container">', unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "Drag & drop files here or click to browse", 
                type=SUPPORTED_FORMATS, 
                accept_multiple_files=True,
                help="Supported formats: PDF, DOCX, PPTX, HTML",
                label_visibility="collapsed"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded:
            new_files = [f for f in uploaded if compute_file_hash(f) not in st.session_state.processed_hashes]
            if new_files:
                st.info(f"🆕 {len(new_files)} new file(s) to process")
                
                if st.button("🚀 Process & Index", type="primary", use_container_width=True):
                    process_and_index(new_files)
            else:
                st.info("✅ All files already processed")
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        
        # Reset button with confirmation
        if st.button("🗑️ Clear All & Reset", use_container_width=True):
            if st.session_state.get("confirm_reset", False):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
            else:
                st.session_state.confirm_reset = True
                st.warning("Click again to confirm reset")
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        
        # Enhanced status display
        st.markdown('<div class="sidebar-title">📊 System Status</div>', unsafe_allow_html=True)
        
        if st.session_state.vectorstore:
            st.markdown('<div class="status-success">✅ Documents indexed and ready</div>', unsafe_allow_html=True)
            st.write(f"📚 Documents: {len(st.session_state.docling_docs)}")
            
            # Show document list
            if st.session_state.docling_docs:
                with st.expander("View Documents"):
                    for doc in st.session_state.docling_docs:
                        st.markdown(f"""
                        <div class="document-card">
                            <div class="document-title">{doc["filename"]}</div>
                            <div class="document-meta">
                                Pages: {doc["metadata"].get("total_pages", "N/A")} | 
                                Tables: {doc["metadata"].get("total_tables", "N/A")} | 
                                Images: {doc["metadata"].get("total_images", "N/A")}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-warning">📥 Waiting for documents...</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def render_chat_interface():
    """Render the enhanced chat interface."""
    # Header for the chat interface
    st.markdown('<h1 class="main-header">Document Intelligence Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about your uploaded documents</p>', unsafe_allow_html=True)
    
    if not st.session_state.vectorstore:
        # Show a welcome message when no documents are loaded
        st.markdown("""
        <div class="card">
            <h3>Welcome to Document Intelligence Assistant</h3>
            <p>Please upload and process documents to start chatting. Use the sidebar to upload your files.</p>
            <p>Once your documents are processed, you can ask questions like:</p>
            <ul>
                <li>What are the main topics covered in the documents?</li>
                <li>Summarize the key findings from the reports.</li>
                <li>What tables are included in the documents?</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Chat history container
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages with enhanced styling
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {msg["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>Assistant:</strong> {msg["content"]}
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input with enhanced styling
    with st.container():
        # Create a form for better control of the input
        with st.form(key="chat_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                user_input = st.text_input("Ask a question about your documents:", key="user_input", label_visibility="collapsed")
            with col2:
                submit_button = st.form_submit_button("Send", type="primary")
            
            if submit_button and user_input:
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                # Display user message immediately
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {user_input}
                </div>
                """, unsafe_allow_html=True)
                
                # Generate and display assistant response
                with st.spinner("Thinking..."):
                    try:
                        # Use consistent thread_id for conversation continuity
                        config = {"configurable": {"thread_id": "streamlit_session"}}
                        
                        result = st.session_state.agent.invoke(
                            {"messages": [{"role": "user", "content": user_input}]},
                            config=config
                        )
                        
                        # Extract response
                        messages = result.get("messages", [])
                        full_response = messages[-1].content if messages else "No response."
                        
                        # Add assistant message to chat history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": full_response
                        })
                        
                        # Display assistant response
                        st.markdown(f"""
                        <div class="chat-message assistant-message">
                            <strong>Assistant:</strong> {full_response}
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.markdown(f"""
                        <div class="chat-message assistant-message">
                            <strong>Assistant:</strong> {error_msg}
                        </div>
                        """, unsafe_allow_html=True)
                        st.error(f"Detailed error: {e}")
                        logger.error(f"Chat error: {e}")

def render_structure_viz():
    """Render enhanced document structure visualization."""
    st.markdown('<h1 class="main-header">Document Structure Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Explore the structure and content of your documents</p>', unsafe_allow_html=True)

    if not st.session_state.docling_docs:
        st.info("👈 Please upload and process documents first!")
        return

    # Document selector with enhanced UI
    doc_names = [doc["filename"] for doc in st.session_state.docling_docs]
    selected_doc_name = st.selectbox("Select document:", doc_names)

    selected_doc_data = next(
        (doc for doc in st.session_state.docling_docs if doc["filename"] == selected_doc_name),
        None,
    )

    if not selected_doc_data:
        return

    # Use enhanced visualizer with structured content
    visualizer = DocumentStructureVisualizer(
        selected_doc_data["doc"],
        selected_doc_data.get("structured_content")
    )

    # Create tabs with enhanced styling
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Overview", "📑 Summary", "🏗️ Hierarchy", "📊 Tables", "🖼️ Images"])

    with tab1:
        st.markdown('<h2 class="sub-header">Document Overview</h2>', unsafe_allow_html=True)
        stats = visualizer.get_content_statistics()
        
        # Display metrics in a grid
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{stats["summary"]["num_pages"]}</h3>
                <p>Pages</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{stats["total_headings"]}</h3>
                <p>Headings</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{stats["total_tables"]}</h3>
                <p>Tables</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{stats["total_images"]}</h3>
                <p>Images</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Second row of metrics
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{stats["summary"]["num_texts"]}</h3>
                <p>Text Items</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col6:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{stats["summary"]["total_characters"]:,}</h3>
                <p>Characters</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col7:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{stats["tables_with_data"]}</h3>
                <p>Data Tables</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col8:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{stats["images_with_captions"]}</h3>
                <p>Captioned Images</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Content distribution
        st.markdown('<h3 class="sub-header">Content Distribution</h3>', unsafe_allow_html=True)
        text_types_df = pd.DataFrame(
            [{"Type": k, "Count": v} for k, v in sorted(stats["summary"]["text_types"].items(), key=lambda x: -x[1])]
        )
        st.dataframe(text_types_df, use_container_width=True)

    with tab2:
        st.markdown('<h2 class="sub-header">Document Summary</h2>', unsafe_allow_html=True)
        summary = visualizer.get_document_summary()
        
        # Display summary in an expandable JSON viewer
        st.json(summary, expanded=False)

    with tab3:
        st.markdown('<h2 class="sub-header">Document Hierarchy</h2>', unsafe_allow_html=True)
        hierarchy = visualizer.get_document_hierarchy()
        
        if hierarchy:
            # Display hierarchy with indentation based on level
            for item in hierarchy:
                indent = "  " * (item["level"] - 1)
                emoji = "🔹" * min(item["level"], 3)
                st.markdown(f"{indent}{emoji} **{item['text']}** _(Page {item['page']})_")
        else:
            st.info("No hierarchical structure detected")

    with tab4:
        st.markdown('<h2 class="sub-header">Document Tables</h2>', unsafe_allow_html=True)
        tables_info = visualizer.get_tables_info()
        
        if tables_info:
            for table_data in tables_info:
                with st.expander(f"Table {table_data['table_number']} (Page {table_data['page']})", expanded=False):
                    if table_data.get("caption"):
                        st.caption(f"**Caption:** {table_data['caption']}")
                    st.write(f"**Dimensions:** {table_data['shape'][0]} rows × {table_data['shape'][1]} columns")
                    
                    if not table_data.get("is_empty", True):
                        st.dataframe(table_data["dataframe"], use_container_width=True)
                    else:
                        st.info("Table is empty")
        else:
            st.info("No tables found")

    with tab5:
        st.markdown('<h2 class="sub-header">Document Images</h2>', unsafe_allow_html=True)
        pictures_info = visualizer.get_pictures_info()
        
        if pictures_info:
            cols = st.columns(2)
            for idx, pic_data in enumerate(pictures_info):
                col = cols[idx % 2]
                with col:
                    with st.expander(f"Image {pic_data['picture_number']} (Page {pic_data['page']})", expanded=False):
                        if pic_data.get("caption"):
                            st.caption(f"**Caption:** {pic_data['caption']}")
                        if pic_data.get("pil_image") is not None:
                            st.image(pic_data["pil_image"], use_container_width=True)
                        else:
                            st.info("Image preview not available")
                        if pic_data.get("bounding_box"):
                            bbox = pic_data["bounding_box"]
                            st.text(f"Position: ({bbox['left']:.1f}, {bbox['top']:.1f}) - ({bbox['right']:.1f}, {bbox['bottom']:.1f})")
        else:
            st.info("No images found")

def main():
    initialize_session_state()
    
    # Render the sidebar
    render_sidebar()
    
    # Create main content area with tabs
    tab1, tab2 = st.tabs(["💬 Chat", "📊 Document Structure"])
    
    with tab1:
        render_chat_interface()
    
    with tab2:
        render_structure_viz()

if __name__ == "__main__":
    main()