import time
import os
import uuid
from datetime import datetime
import logging
import json
import shutil
import re
import traceback
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from functools import lru_cache
import fitz  # PyMuPDF
from ollama import AsyncClient
from fastapi import (
    FastAPI, 
    File, 
    UploadFile,
    WebSocket, 
    WebSocketDisconnect,
    HTTPException,
    BackgroundTasks,
    Depends
)
from fastapi.responses import (
    HTMLResponse, 
    FileResponse, 
    JSONResponse
)
from fastapi.staticfiles import StaticFiles

# Added imports for hybrid retrieval
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.schema import Document, NodeWithScore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from docling_core.types.doc import DoclingDocument
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, RapidOcrOptions
from docling.document_converter import (
    ConversionResult,
    DocumentConverter,
    InputFormat,
    PdfFormatOption,
    DocumentStream,
)
from io import BytesIO
import nest_asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
BASE_DIR = Path(__file__).parent
UPLOAD_DIRECTORY = BASE_DIR / "uploads"
INDEXED_DIRECTORY = BASE_DIR / "indexed"
HTML_FILE_PATH = BASE_DIR / "index.html"
STATUS_FILE_PATH = BASE_DIR / "status.json"
DOCLING_ARTIFACTS_PATH = BASE_DIR / "artifacts"

# Ensure directories exist
UPLOAD_DIRECTORY.mkdir(exist_ok=True)
INDEXED_DIRECTORY.mkdir(exist_ok=True)
DOCLING_ARTIFACTS_PATH = Path.home() / ".cache" / "docling" / "models"

# Initialize global status tracking
class IndexingStatus:
    def __init__(self):
        self.is_indexing = False
        self.progress = 0
        self.total_files = 0
        self.processed_files = 0
        self.current_file = ""
        self.status_message = "Idle"
        self.error = None
        self.lock = threading.Lock()
    
    def update(self, **kwargs):
        with self.lock:
            for key, value in kwargs.items():
                setattr(self, key, value)
            self._persist_status()
    
    def _persist_status(self):
        try:
            with open(STATUS_FILE_PATH, 'w') as f:
                json.dump({
                    "is_indexing": self.is_indexing,
                    "progress": self.progress,
                    "total_files": self.total_files,
                    "processed_files": self.processed_files,
                    "current_file": self.current_file,
                    "status_message": self.status_message,
                    "error": self.error
                }, f)
        except Exception as e:
            logger.error(f"Failed to persist status: {e}")

indexing_status = IndexingStatus()

    

class StructureCache:
    """Cache for document structure to avoid reprocessing."""
    
    def __init__(self, cache_dir: Path = Path("cache/structures")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get(self, filename: str, content_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached structure if it exists and is valid."""
        cache_file = self.cache_dir / f"{self._sanitize_filename(filename)}_{content_hash}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8', errors='replace') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading cache: {e}")
        
        return None
    
    def set(self, filename: str, content_hash: str, structure: Dict[str, Any]):
        """Cache structure data."""
        cache_file = self.cache_dir / f"{self._sanitize_filename(filename)}_{content_hash}.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(structure, f, indent=2)
        except Exception as e:
            logger.error(f"Error writing cache: {e}")
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe caching."""
        return re.sub(r'[^\w\-_.]', '_', filename)


class GazetteDocumentChunker:
    """
    Advanced document chunking strategy specifically designed for gazette documents.
    Implements both semantic and structured chunking approaches.
    """
    
    def __init__(self, chunk_size=1000, chunk_overlap=100):
        """
        Initialize the gazette document chunker.
        
        Args:
            chunk_size (int): Maximum size of chunks in characters
            chunk_overlap (int): Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Regex patterns for gazette structure
        self.gazette_notice_pattern = re.compile(r'GAZETTE NOTICE NO\. (\d+)', re.IGNORECASE)
        self.section_header_pattern = re.compile(r'^[A-Z][A-Z\s]{5,}$', re.MULTILINE)
        self.appointment_pattern = re.compile(r'APPOINTMENT', re.IGNORECASE)
        self.notice_pattern = re.compile(r'NOTICE', re.IGNORECASE)
        
        # Initialize the sentence splitter for fallback
        self.sentence_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    def chunk_document(self, text, file_name):
        """
        Chunk a document using structured and semantic approaches.
        
        Args:
            text (str): The full text of the document
            file_name (str): Name of the source file
            
        Returns:
            List[Dict]: List of chunks with metadata
        """
        # First try structured chunking for gazette documents
        if self._is_gazette_document(text):
            return self._structured_chunk_gazette(text, file_name)
        else:
            # Fall back to semantic chunking
            return self._semantic_chunk(text, file_name)
    
    def _is_gazette_document(self, text):
        """Check if the document appears to be a gazette notice"""
        return bool(self.gazette_notice_pattern.search(text))
    
    def _structured_chunk_gazette(self, text, file_name):
        """
        Chunk gazette documents based on their structure.
        
        Args:
            text (str): The full text of the gazette document
            file_name (str): Name of the source file
            
        Returns:
            List[Dict]: List of chunks with metadata
        """
        chunks = []
        
        # Find all gazette notice numbers
        gazette_notices = list(self.gazette_notice_pattern.finditer(text))
        
        if not gazette_notices:
            # If no clear gazette notices, fall back to semantic chunking
            return self._semantic_chunk(text, file_name)
        
        # Extract each notice as a separate chunk
        for i, notice_match in enumerate(gazette_notices):
            start_pos = notice_match.start()
            end_pos = gazette_notices[i+1].start() if i+1 < len(gazette_notices) else len(text)
            
            notice_text = text[start_pos:end_pos].strip()
            notice_number = notice_match.group(1)
            
            # Create metadata for this notice
            metadata = {
                "document_name": file_name,
                "notice_number": notice_number,
                "chunk_type": "gazette_notice",
                "start_pos": start_pos,
                "end_pos": end_pos
            }
            
            # Further chunk the notice if it's too large
            if len(notice_text) > self.chunk_size:
                sub_chunks = self._chunk_large_text(notice_text, metadata)
                chunks.extend(sub_chunks)
            else:
                chunks.append({
                    "text": notice_text,
                    "metadata": metadata
                })
        
        return chunks
    
    def _semantic_chunk(self, text, file_name):
        """
        Chunk documents using semantic boundaries.
        
        Args:
            text (str): The full text of the document
            file_name (str): Name of the source file
            
        Returns:
            List[Dict]: List of chunks with metadata
        """
        # Use LlamaIndex's SentenceSplitter as a base for semantic chunking
        document = Document(text=text)
        nodes = self.sentence_splitter.get_nodes_from_documents([document])
        
        chunks = []
        for i, node in enumerate(nodes):
            metadata = {
                "document_name": file_name,
                "chunk_type": "semantic",
                "chunk_index": i
            }
            
            chunks.append({
                "text": node.text,
                "metadata": metadata
            })
        
        return chunks
    
    def _chunk_large_text(self, text, base_metadata):
        """
        Split a large text into smaller chunks while preserving context.
        
        Args:
            text (str): The text to chunk
            base_metadata (Dict): Base metadata to include with each chunk
            
        Returns:
            List[Dict]: List of smaller chunks
        """
        chunks = []
        
        # Use sentence splitter to break down the text
        document = Document(text=text)
        nodes = self.sentence_splitter.get_nodes_from_documents([document])
        
        for i, node in enumerate(nodes):
            metadata = base_metadata.copy()
            metadata.update({
                "sub_chunk_index": i,
                "total_sub_chunks": len(nodes)
            })
            
            chunks.append({
                "text": node.text,
                "metadata": metadata
            })
        
        return chunks
    

def download_rapidocr_models():
    """Download RapidOCR models from HuggingFace"""
    try:
        from huggingface_hub import snapshot_download
        
        # Create the models directory if it doesn't exist
        models_dir = Path.home() / ".cache" / "docling" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Download RapidOCR models
        print("Downloading RapidOCR models")
        download_path = snapshot_download(repo_id="SWHL/RapidOCR")
        
        # Create a symlink or copy to the expected location
        rapidocr_dir = models_dir / "RapidOcr"
        if rapidocr_dir.exists():
            shutil.rmtree(rapidocr_dir)
        
        # Copy the models to the expected location
        shutil.copytree(download_path, rapidocr_dir)
        
        print(f"RapidOCR models downloaded to {rapidocr_dir}")
        return rapidocr_dir
    except Exception as e:
        logger.error(f"Failed to download RapidOCR models: {e}")
        return None


#---- docling utility function
def download_docling_model():
    """Download Docling model if not available"""
    try:
        from huggingface_hub import hf_hub_download
        
        # Create the models directory if it doesn't exist
        models_dir = Path.home() / ".cache" / "docling" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Download the Docling model
        model_path = models_dir / "model.safetensors"
        if not model_path.exists():
            logger.info("Downloading Docling model...")
            hf_hub_download(
                repo_id="ds4sd/docling-model",
                filename="model.safetensors",
                local_dir=str(models_dir)
            )
            logger.info("Docling model downloaded successfully")
        
        return model_path
    except Exception as e:
        logger.error(f"Failed to download Docling model: {e}")
        return None

# Ensure_docling_models function to also download the Docling model (if missing)
def ensure_docling_models():
    """Ensure Docling models are downloaded and available."""
    try:
        # Check if models exist
        models_dir = Path.home() / ".cache" / "docling" / "models"
        rapidocr_dir = models_dir / "RapidOcr"
        
        # Download Docling model if needed
        download_docling_model()
        
        # Check if required RapidOCR model files exist
        det_model_path = rapidocr_dir / "PP-OCRv4" / "en_PP-OCRv3_det_infer.onnx"
        rec_model_path = rapidocr_dir / "PP-OCRv4" / "ch_PP-OCRv4_rec_server_infer.onnx"
        cls_model_path = rapidocr_dir / "PP-OCRv3" / "ch_ppocr_mobile_v2.0_cls_train.onnx"
        
        if not rapidocr_dir.exists() or not det_model_path.exists() or not rec_model_path.exists():
            logger.info("Downloading Docling models...")
            # Download the models
            download_rapidocr_models()
            logger.info("Models downloaded successfully")
        else:
            logger.info(f"Models directory exists at {models_dir}")
            
        return models_dir
    except Exception as e:
        logger.warning(f"Failed to ensure models: {e}")
        return None

class PDFTextExtractor:
    @staticmethod
    def extract_structure_from_pdf(pdf_path, output_dir, artifacts_path=None):
        """Extract RICH structure from PDF using Docling document object"""
        os.makedirs(output_dir, exist_ok=True)
        pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
        
        try:
            # Configure pipeline
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True
            pipeline_options.generate_picture_images = True
            pipeline_options.images_scale = 2.0
            
            # Create converter
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            
            # Convert document - KEEP THE DOCUMENT OBJECT with explicit type
            result = converter.convert(pdf_path)
            doc: DoclingDocument = result.document  # ← The RICH object with explicit type
            
            # ✅ Extract structure using the RICH document API
            structure = {
                "filename": pdf_basename,
                "summary": PDFTextExtractor._extract_summary_from_docling(doc),
                "hierarchy": PDFTextExtractor._extract_hierarchy_from_docling(doc),
                "tables": PDFTextExtractor._extract_tables_from_docling(doc),
                "images": PDFTextExtractor._extract_images_from_docling(doc),
                "full_markdown": doc.export_to_markdown()
            }

            # Validate and clean structure before saving
            structure = PDFTextExtractor._validate_and_clean_for_json(structure)
            
            # Save structure
            structure_path = os.path.join(output_dir, f"{pdf_basename}_structure.json")
            with open(structure_path, 'w', encoding='utf-8') as f:
                json.dump(structure, f, indent=2, ensure_ascii=False)
            
            return structure_path
            
        except Exception as e:
            logger.error(f"Docling structure extraction failed: {e}")
            return None
        
        
    @staticmethod
    def _validate_and_enrich_structure(structure_data: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """Validate and enrich document structure data."""
        
        # Quick validation check
        if not isinstance(structure_data, dict):
            return {
                "filename": filename,
                "timestamp": datetime.now().isoformat(),
                "validation_status": "invalid_format",
                "validation_warnings": ["Structure data is not a dictionary"]
            }
        
        # Start with clean validated structure
        validated = structure_data.copy()
        validated.update({
            "timestamp": datetime.now().isoformat(),
            "validation_status": "validated",
            "validation_warnings": []
        })
        
        # Ensure required fields exist
        required_fields = {
            "filename": filename,
            "summary": {},
            "hierarchy": [],
            "tables": [],
            "images": [],
            "full_markdown": "",
            "sections": []
        }
        
        for field, default_value in required_fields.items():
            if field not in validated or not validated[field]:
                validated[field] = default_value
                if field != "filename":  # Don't warn about filename
                    validated["validation_warnings"].append(f"Missing or empty field: {field}")
        
        # Generate sections from hierarchy if missing
        if not validated["sections"] and validated["hierarchy"]:
            validated["sections"] = [
                {
                    "heading": heading,
                    "paragraphs": [],
                    "level": heading.get("level", 1),
                    "generated": True
                }
                for heading in sorted(validated["hierarchy"], key=lambda x: x.get("page", 0))
            ]
            validated["validation_warnings"].append("Generated sections from hierarchy")
        
        # Update validation status if there are warnings
        if validated["validation_warnings"]:
            validated["validation_status"] = "validated_with_warnings"
        
        return validated
    

    @staticmethod
    def _build_sections_from_items(headings: List[Dict], paragraphs: List[Dict]) -> List[Dict[str, Any]]:
        """
        Build sections by associating paragraphs with their parent headings.
        
        Args:
            headings: List of heading dictionaries sorted by page
            paragraphs: List of paragraph dictionaries sorted by page
            
        Returns:
            List of section dictionaries with headings and associated paragraphs
        """
        sections = []
        
        if not headings:
            return sections
        
        for i, heading in enumerate(headings):
            # Determine the boundary for this section (until next heading)
            next_heading_page = headings[i + 1]["page"] if i + 1 < len(headings) else float('inf')
            
            # Find paragraphs that belong to this section
            section_paragraphs = []
            for paragraph in paragraphs:
                para_page = paragraph.get("page")
                heading_page = heading.get("page")
                
                # Include paragraph if it's between this heading and the next
                if para_page is not None and heading_page is not None:
                    if heading_page <= para_page < next_heading_page:
                        section_paragraphs.append(paragraph)
            
            # Validate paragraphs in section
            for paragraph in section_paragraphs:
                if not paragraph.get("text"):
                    paragraph["validation_warnings"].append("Empty paragraph text")
                    paragraph["validation_status"] = "validated_with_warnings"
            
            # Create section
            section = {
                "heading": heading,
                "paragraphs": section_paragraphs,
                "level": heading.get("level", 1),
                "validation_status": "validated",
                "validation_warnings": []
            }
            
            # Validate section
            if not section_paragraphs:
                section["validation_warnings"].append("No paragraphs associated with this heading")
                section["validation_status"] = "validated_with_warnings"
            
            sections.append(section)
        
        return sections


    @staticmethod
    def _handle_orphaned_content(sections: List[Dict], headings: List[Dict], paragraphs: List[Dict]) -> List[Dict[str, Any]]:
        """
        Handle paragraphs that don't belong to any heading (orphaned content).
        
        Args:
            sections: Existing sections built from headings
            headings: List of all headings in document
            paragraphs: List of all paragraphs in document
            
        Returns:
            Updated sections list including orphaned content sections
        """
        # Case 1: No headings at all - create a single section with all paragraphs
        if not headings and paragraphs:
            sections.append({
                "heading": {
                    "text": "Untitled Document",
                    "page": paragraphs[0].get("page", 0),
                    "level": 1,
                    "label": "generated",
                    "type": "heading",
                    "validation_status": "generated",
                    "validation_warnings": ["Generated heading for document with no headings"]
                },
                "paragraphs": paragraphs,
                "level": 1,
                "validation_status": "generated",
                "validation_warnings": ["Generated section for document with no headings"]
            })
            return sections
        
        # Case 2: Headings exist - find paragraphs after the last heading
        if headings and paragraphs:
            last_heading_page = headings[-1].get("page")
            
            if last_heading_page is not None:
                orphaned_paragraphs = [
                    p for p in paragraphs 
                    if p.get("page") is not None and p["page"] > last_heading_page
                ]
                
                if orphaned_paragraphs:
                    sections.append({
                        "heading": {
                            "text": "Untitled Section",
                            "page": orphaned_paragraphs[0].get("page", 0),
                            "level": 1,
                            "label": "generated",
                            "type": "heading",
                            "validation_status": "generated",
                            "validation_warnings": ["Generated heading for orphaned paragraphs"]
                        },
                        "paragraphs": orphaned_paragraphs,
                        "level": 1,
                        "validation_status": "generated",
                        "validation_warnings": ["Generated section for orphaned paragraphs"]
                    })
        
        # Case 3: Paragraphs that appear before the first heading
        if headings and paragraphs:
            first_heading_page = headings[0].get("page")
            
            if first_heading_page is not None:
                pre_heading_paragraphs = [
                    p for p in paragraphs 
                    if p.get("page") is not None and p["page"] < first_heading_page
                ]
                
                if pre_heading_paragraphs:
                    # Insert at the beginning
                    sections.insert(0, {
                        "heading": {
                            "text": "Introduction",
                            "page": pre_heading_paragraphs[0].get("page", 0),
                            "level": 1,
                            "label": "generated",
                            "type": "heading",
                            "validation_status": "generated",
                            "validation_warnings": ["Generated heading for content before first heading"]
                        },
                        "paragraphs": pre_heading_paragraphs,
                        "level": 1,
                        "validation_status": "generated",
                        "validation_warnings": ["Generated section for content before first heading"]
                    })
        
        return sections
        
    @staticmethod
    def _extract_sections_with_context(doc: DoclingDocument) -> List[Dict[str, Any]]:
        """Extract document sections with paragraphs."""
        
        # Extract and sort content
        headings, paragraphs = PDFTextExtractor._extract_text_items(doc)
        
        # Build sections
        sections = PDFTextExtractor._build_sections_from_items(headings, paragraphs)
        
        # Handle orphaned content
        sections = PDFTextExtractor._handle_orphaned_content(sections, headings, paragraphs)
        
        return sections
    
    @staticmethod
    def _extract_text_items(doc: DoclingDocument) -> Tuple[List[Dict], List[Dict]]:
        """Extract and sort headings and paragraphs from document."""
        headings = []
        paragraphs = []
        
        if not hasattr(doc, "texts"):
            return headings, paragraphs
        
        for item in doc.texts:
            label = getattr(item, "label", None)
            text = getattr(item, "text", "")
            prov = getattr(item, "prov", [])
            page_no = prov[0].page_no if prov else None
            
            base_item = {
                "text": text,
                "page": page_no,
                "label": label or "unlabeled",
                "validation_status": "validated",
                "validation_warnings": []
            }
            
            if label and "header" in label.lower():
                headings.append({
                    **base_item,
                    "level": PDFTextExtractor._infer_heading_level(label),
                    "type": "heading"
                })
            else:
                paragraphs.append({
                    **base_item,
                    "type": "paragraph"
                })
                if not label:
                    paragraphs[-1]["validation_warnings"].append("Unlabeled text")
        
        # Sort by page
        headings.sort(key=lambda x: x["page"] or 0)
        paragraphs.sort(key=lambda x: x["page"] or 0)
        
        return headings, paragraphs


    @staticmethod
    def _extract_summary_from_docling(doc: DoclingDocument) -> Dict[str, Any]:
        """Extract summary directly from Docling document object"""
        pages = getattr(doc, "pages", {})
        texts = getattr(doc, "texts", [])
        tables = getattr(doc, "tables", [])
        pictures = getattr(doc, "pictures", [])
        
        text_types = {}
        for item in texts:
            label = getattr(item, "label", "unknown")
            text_types[label] = text_types.get(label, 0) + 1
        
        return {
            "num_pages": len(pages) if pages else 0,
            "num_texts": len(texts),
            "num_tables": len(tables),
            "num_pictures": len(pictures),
            "text_types": text_types
        }

    @staticmethod
    def _extract_hierarchy_from_docling(doc: DoclingDocument) -> List[Dict[str, Any]]:
        """Extract document hierarchy from Docling document"""
        hierarchy = []
        
        if not hasattr(doc, "texts") or not doc.texts:
            return hierarchy
        
        for item in doc.texts:
            label = getattr(item, "label", None)
            
            if label and "header" in label.lower():
                text = getattr(item, "text", "")
                prov = getattr(item, "prov", [])
                page_no = prov[0].page_no if prov else None
                
                hierarchy.append({
                    "type": label,
                    "text": text,
                    "page": page_no,
                    "level": PDFTextExtractor._infer_heading_level(label)
                })
        
        return hierarchy

    @staticmethod
    def _extract_images_from_docling(doc: DoclingDocument) -> List[Dict[str, Any]]:
        images_info = []
        
        if not hasattr(doc, "pictures") or not doc.pictures:
            return images_info
        
        for i, picture in enumerate(doc.pictures, 1):
            try:
                prov = getattr(picture, "prov", [])
                page_no = prov[0].page_no if prov else None
                
                # FIX: Pass doc parameter to caption_text method
                caption_attr = getattr(picture, "caption_text", None)
                if callable(caption_attr):
                    try:
                        caption = caption_attr(doc)  # ← Pass doc parameter
                    except TypeError:
                        # Fallback if signature is different
                        caption = caption_attr() if callable(caption_attr) else str(caption_attr)
                else:
                    caption = str(caption_attr) if caption_attr else None
                
                images_info.append({
                    "image_number": i,
                    "page": int(page_no) if page_no is not None else None,
                    "caption": str(caption) if caption else None
                })
            except Exception as e:
                logger.warning(f"Could not process image {i}: {e}")
                continue
        
        return images_info
    
    @staticmethod
    def _extract_tables_from_docling(doc: DoclingDocument) -> List[Dict[str, Any]]:
        tables_info = []
        
        if not hasattr(doc, "tables") or not doc.tables:
            return tables_info
        
        # Extract all headings to establish context
        headings = []
        if hasattr(doc, "texts"):
            for item in doc.texts:
                label = getattr(item, "label", None)
                if label and "header" in label.lower():
                    text = getattr(item, "text", "")
                    prov = getattr(item, "prov", [])
                    page_no = prov[0].page_no if prov else None
                    
                    headings.append({
                        "text": text,
                        "page": page_no,
                        "level": PDFTextExtractor._infer_heading_level(label),
                        "label": label
                    })
        
        # Sort headings by page number to establish document order
        headings.sort(key=lambda x: x["page"] if x["page"] is not None else 0)
        
        for i, table in enumerate(doc.tables, 1):
            try:
                df = table.export_to_dataframe(doc=doc)
                prov = getattr(table, "prov", [])
                page_no = prov[0].page_no if prov else None
                
                # Find the nearest preceding heading
                context_heading = None
                contextual_label = None
                
                if page_no is not None:
                    # Find headings on the same or previous pages
                    for heading in reversed(headings):
                        if heading["page"] is not None and heading["page"] <= page_no:
                            context_heading = heading["text"]
                            contextual_label = heading["label"]
                            break
                
                # FIX: Pass doc parameter to caption_text method
                caption_attr = getattr(table, "caption_text", None)
                if callable(caption_attr):
                    try:
                        caption = caption_attr(doc)  # ← Pass doc parameter
                    except TypeError:
                        # Fallback if signature is different
                        caption = caption_attr() if callable(caption_attr) else str(caption_attr)
                else:
                    caption = str(caption_attr) if caption_attr else None
                
                # FIX: Convert all values to strings for JSON serialization
                table_data = {
                    "table_number": i,
                    "page": int(page_no) if page_no is not None else None,
                    "caption": str(caption) if caption else None,
                    "headers": [str(h) for h in df.columns.tolist()],
                    "rows": [[str(cell) for cell in row] for row in df.values.tolist()],
                    "shape": [int(s) for s in df.shape],
                    "markdown": df.to_markdown(),
                    # NEW FIELDS
                    "context_heading": context_heading,
                    "contextual_label": contextual_label,
                    # Validation fields
                    "validation_status": "validated",
                    "validation_warnings": []
                }
                
                # Validate table data
                if not table_data["context_heading"]:
                    table_data["validation_warnings"].append("No context heading found")
                    table_data["validation_status"] = "validated_with_warnings"
                
                if not table_data["caption"]:
                    table_data["validation_warnings"].append("No caption found")
                    table_data["validation_status"] = "validated_with_warnings"
                
                tables_info.append(table_data)
                
            except Exception as e:
                logger.warning(f"Could not process table {i}: {e}")
                # Add a placeholder table with error information
                tables_info.append({
                    "table_number": i,
                    "page": None,
                    "caption": None,
                    "headers": [],
                    "rows": [],
                    "shape": [0, 0],
                    "markdown": f"Error processing table: {str(e)}",
                    "context_heading": None,
                    "contextual_label": None,
                    "validation_status": "error",
                    "validation_warnings": [f"Error processing table: {str(e)}"]
                })
                continue
        
        return tables_info


    @staticmethod
    def _extract_tables_from_docling(doc: DoclingDocument) -> List[Dict[str, Any]]:
        tables_info = []
        
        if not hasattr(doc, "tables") or not doc.tables:
            return tables_info
        
        # Extract all headings to establish context
        headings = []
        if hasattr(doc, "texts"):
            for item in doc.texts:
                label = getattr(item, "label", None)
                if label and "header" in label.lower():
                    text = getattr(item, "text", "")
                    prov = getattr(item, "prov", [])
                    page_no = prov[0].page_no if prov else None
                    
                    headings.append({
                        "text": text,
                        "page": page_no,
                        "level": PDFTextExtractor._infer_heading_level(label),
                        "label": label
                    })
        
        # Sort headings by page number
        headings.sort(key=lambda x: x["page"] if x["page"] is not None else 0)
        
        for i, table in enumerate(doc.tables, 1):
            # Initialize table_data with defaults to avoid scope issues
            table_data = {
                "table_number": i,
                "page": None,
                "caption": None,
                "headers": [],
                "rows": [],
                "shape": [0, 0],
                "markdown": "",
                "context_heading": None,
                "contextual_label": None,
                "validation_status": "error",
                "validation_warnings": []
            }
            
            try:
                df = table.export_to_dataframe(doc=doc)
                prov = getattr(table, "prov", [])
                page_no = prov[0].page_no if prov else None
                
                # Find the nearest preceding heading
                context_heading = None
                contextual_label = None
                
                if page_no is not None:
                    for heading in reversed(headings):
                        if heading["page"] is not None and heading["page"] <= page_no:
                            context_heading = heading["text"]
                            contextual_label = heading["label"]
                            break
                
                # Handle caption safely
                caption_attr = getattr(table, "caption_text", None)
                caption = None
                if callable(caption_attr):
                    try:
                        caption = caption_attr(doc)
                    except (TypeError, AttributeError):
                        try:
                            caption = caption_attr()
                        except:
                            caption = str(caption_attr) if caption_attr else None
                else:
                    caption = str(caption_attr) if caption_attr else None
                
                # Update table_data with successful extraction
                table_data.update({
                    "page": int(page_no) if page_no is not None else None,
                    "caption": str(caption) if caption else None,
                    "headers": [str(h) for h in df.columns.tolist()],
                    "rows": [[str(cell) for cell in row] for row in df.values.tolist()],
                    "shape": [int(s) for s in df.shape],
                    "markdown": df.to_markdown(),
                    "context_heading": context_heading,
                    "contextual_label": contextual_label,
                    "validation_status": "validated",
                    "validation_warnings": []
                })
                
                # Validate
                if not table_data["context_heading"]:
                    table_data["validation_warnings"].append("No context heading found")
                    table_data["validation_status"] = "validated_with_warnings"
                
                if not table_data["caption"]:
                    table_data["validation_warnings"].append("No caption found")
                    table_data["validation_status"] = "validated_with_warnings"
                    
            except Exception as e:
                logger.warning(f"Could not process table {i}: {e}")
                table_data["markdown"] = f"Error processing table: {str(e)}"
                table_data["validation_warnings"] = [f"Error processing table: {str(e)}"]
            
            tables_info.append(table_data)
        
        return tables_info

    @staticmethod
    def _infer_heading_level(label: str) -> int:
        """Infer heading level from Docling label"""
        if "title" in label.lower():
            return 1
        elif "section" in label.lower():
            return 2
        elif "subsection" in label.lower():
            return 3
        else:
            return 4
        

    @staticmethod
    def _validate_structure_completeness(structure_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the completeness of the structure data.
        
        Args:
            structure_data: The structure data to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "is_complete": True,
            "missing_elements": [],
            "validation_warnings": []
        }
        
        # Check for required top-level elements
        required_elements = ["summary", "hierarchy", "tables", "images", "full_markdown", "sections"]
        for element in required_elements:
            if element not in structure_data:
                validation_result["is_complete"] = False
                validation_result["missing_elements"].append(element)
                validation_result["validation_warnings"].append(f"Missing required element: {element}")
        
        # Check for empty elements
        for element in required_elements:
            if element in structure_data and not structure_data[element]:
                validation_result["validation_warnings"].append(f"Empty element: {element}")
        
        # Check for table context
        if "tables" in structure_data and structure_data["tables"]:
            for i, table in enumerate(structure_data["tables"]):
                if "context_heading" not in table or not table["context_heading"]:
                    validation_result["validation_warnings"].append(f"Table {i+1} missing context heading")
        
        # Check for section associations
        if "sections" in structure_data and structure_data["sections"]:
            for i, section in enumerate(structure_data["sections"]):
                if "paragraphs" not in section or not section["paragraphs"]:
                    validation_result["validation_warnings"].append(f"Section {i+1} has no associated paragraphs")
        
        return validation_result

    @staticmethod
    def _regenerate_structure_if_needed(filename: str, base_name: str) -> Optional[str]:
        """
        Regenerate the structure file if it's missing or corrupted.
        
        Args:
            filename: The original filename
            base_name: The base name for the structure file
            
        Returns:
            Path to the regenerated structure file or None if regeneration failed
        """
        try:
            # Check if the original PDF exists
            pdf_path = UPLOAD_DIRECTORY / filename
            if not pdf_path.exists():
                logger.error(f"Original PDF not found for {filename}")
                return None
            
            # Try to regenerate the structure
            structure_path = PDFTextExtractor.extract_structure_from_pdf(
                pdf_path,
                INDEXED_DIRECTORY,
                artifacts_path=app_state.artifacts_path
            )
            
            if structure_path:
                logger.info(f"Successfully regenerated structure for {filename}")
                return structure_path
            else:
                logger.error(f"Failed to regenerate structure for {filename}")
                return None
        except Exception as e:
            logger.error(f"Error regenerating structure for {filename}: {e}")
            return None
        

    @staticmethod
    def _validate_and_clean_for_json(obj, path="root"):
        """Recursively validate and clean objects for JSON serialization"""
        try:
            if obj is None or isinstance(obj, (str, int, float, bool)):
                return obj
            elif callable(obj):
                logger.warning(f"Removing callable object at {path}")
                return None
            elif isinstance(obj, dict):
                return {k: PDFTextExtractor._validate_and_clean_for_json(v, f"{path}.{k}") 
                        for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [PDFTextExtractor._validate_and_clean_for_json(item, f"{path}[{i}]") 
                        for i, item in enumerate(obj)]
            elif hasattr(obj, 'tolist'):  # numpy arrays
                return obj.tolist()
            elif hasattr(obj, '__dict__'):
                logger.warning(f"Converting complex object to string at {path}")
                return str(obj)
            else:
                # Try to serialize it
                json.dumps(obj)
                return obj
        except (TypeError, ValueError) as e:
            logger.error(f"Non-serializable object at {path}: {e}")
            return str(obj)  # Convert to string as fallback
        
    @staticmethod
    def extract_text_from_stream(file_stream, filename, output_dir, use_docling=True, artifacts_path=None, max_pages=200):
        """
        Extract text from PDF stream - simplified to just create full markdown file.
        """
        os.makedirs(output_dir, exist_ok=True)
        pdf_basename = os.path.splitext(os.path.basename(filename))[0]
        
        if use_docling:
            try:
                # Configure pipeline
                pipeline_options = PdfPipelineOptions()
                pipeline_options.do_ocr = True
                pipeline_options.do_table_structure = True
                
                if artifacts_path:
                    pipeline_options.artifacts_path = artifacts_path
                
                converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                    }
                )
                
                # Convert document from stream
                source = DocumentStream(name=filename, stream=file_stream)
                result = converter.convert(source)
                
                # Export to markdown
                markdown_content = result.document.export_to_markdown()
                
                # Save full markdown document ONLY
                md_path = os.path.join(output_dir, f"{pdf_basename}.md")
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                
                # Return just the main markdown file path
                return {f"{pdf_basename}.md": md_path}
                
            except Exception as e:
                logger.warning(f"Docling extraction failed, falling back to PyMuPDF: {e}")
        
        # Fallback to PyMuPDF extraction (simplified)
        file_stream.seek(0)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
            temp_pdf.write(file_stream.read())
            temp_pdf_path = temp_pdf.name
        
        try:
            pdf_document = fitz.open(temp_pdf_path)
            
            # Combine all pages into single markdown
            combined_content = ""
            total_pages = min(len(pdf_document), max_pages)
            
            for page_num in range(total_pages):
                page = pdf_document.load_page(page_num)
                text = page.get_text()
                combined_content += f"--- Page {page_num + 1} ---\n\n{text}\n\n"
            
            # Save as single markdown file
            md_path = os.path.join(output_dir, f"{pdf_basename}.md")
            with open(md_path, 'w', encoding='utf-8') as file:
                file.write(combined_content)
            
            pdf_document.close()
            return {f"{pdf_basename}.md": md_path}
            
        finally:
            os.unlink(temp_pdf_path)
        
    @staticmethod
    def extract_structure_from_stream(file_stream, filename, output_dir, artifacts_path=None):
        """Extract rich structure from PDF stream using Docling document object with robust error handling"""
        os.makedirs(output_dir, exist_ok=True)
        pdf_basename = os.path.splitext(os.path.basename(filename))[0]
        
        try:
            # Configure pipeline
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True
            pipeline_options.generate_picture_images = True
            pipeline_options.images_scale = 2.0
            # Add this to ensure paragraph extraction
            pipeline_options.do_code_enrichment = True
            pipeline_options.do_formula_enrichment = True
            
            # Create converter
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            
            # Convert document from stream - KEEP THE DOCUMENT OBJECT with explicit type
            source = DocumentStream(name=filename, stream=file_stream)
            result = converter.convert(source)
            doc: DoclingDocument = result.document  # ← The RICH object with explicit type
            
            # ✅ ALWAYS export to markdown and include it in structure
            full_markdown = doc.export_to_markdown()
            
            # ✅ Extract structure using the RICH document API
            structure = {
                "filename": pdf_basename,
                "summary": PDFTextExtractor._extract_summary_from_docling(doc),
                "hierarchy": PDFTextExtractor._extract_hierarchy_from_docling(doc),
                "tables": PDFTextExtractor._extract_tables_from_docling(doc),
                "images": PDFTextExtractor._extract_images_from_docling(doc),
                "full_markdown": full_markdown,  # ← ALWAYS include this
                # NEW FIELD
                "sections": PDFTextExtractor._extract_sections_with_context(doc),
                # Validation fields
                "validation_status": "validated",
                "validation_warnings": []
            }
            
            # Validate the structure
            structure = PDFTextExtractor._validate_and_enrich_structure(structure, filename)

            # validate structure before saving
            structure = PDFTextExtractor._validate_and_clean_for_json(structure)
            
            # Also save the full markdown as a separate file for compatibility
            md_path = os.path.join(output_dir, f"{pdf_basename}.md")
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(full_markdown)
            
            # Save structure
            structure_path = os.path.join(output_dir, f"{pdf_basename}_structure.json")
            with open(structure_path, 'w', encoding='utf-8') as f:
                json.dump(structure, f, indent=2, ensure_ascii=False)
            
            return structure_path
            
        except Exception as e:
            logger.error(f"Docling structure extraction from stream failed: {e}")
            
            # Create a minimal structure with error information
            error_structure = {
                "filename": pdf_basename,
                "summary": {"error": str(e)},
                "hierarchy": [],
                "tables": [],
                "images": [],
                "full_markdown": f"Error extracting structure: {str(e)}",
                "sections": [],
                "validation_status": "error",
                "validation_warnings": [f"Error extracting structure: {str(e)}"]
            }
            
            # Save the error structure
            structure_path = os.path.join(output_dir, f"{pdf_basename}_structure.json")
            try:
                with open(structure_path, 'w', encoding='utf-8') as f:
                    json.dump(error_structure, f, indent=2, ensure_ascii=False)
                return structure_path
            except Exception as save_error:
                logger.error(f"Failed to save error structure: {save_error}")
                return None

    @staticmethod
    def get_file_hash(file_path):
        """Calculate SHA256 hash of a file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    @staticmethod
    def get_stream_hash(file_stream):
        """Calculate SHA256 hash of a file stream"""
        # Remember current position
        current_position = file_stream.tell()
        
        # Reset to beginning
        file_stream.seek(0)
        
        # Calculate hash
        hash_sha256 = hashlib.sha256()
        for chunk in iter(lambda: file_stream.read(4096), b""):
            hash_sha256.update(chunk)
        
        # Reset to original position
        file_stream.seek(current_position)
        
        return hash_sha256.hexdigest()

        

class KnowledgeBaseIndexer:
    def __init__(self, 
                 collection_name="pdf_documents", 
                 ollama_url="http://localhost:11434/api/embeddings", 
                 indexed_directory=INDEXED_DIRECTORY,
                 max_workers=4,  # Number of parallel workers
                 artifacts_path=None,  # Path to model artifacts
                 max_file_size=50 * 1024 * 1024,  # 50MB default limit
                 max_pages=200):  # Default page limit
        
        """
        Initialize LlamaIndex components for indexing documents
        
        Args:
            collection_name (str): Name of the collection (used for persistence)
            ollama_url (str): URL of the Ollama server
            indexed_directory (str): Directory containing indexed text files
            max_workers (int): Maximum number of parallel workers for indexing
            artifacts_path (str): Path to model artifacts for offline/controlled environments
            max_file_size (int): Maximum file size in bytes (default: 50MB)
            max_pages (int): Maximum number of pages to process (default: 200)
        """
        # Track PDF hashes to avoid re-extraction
        self.pdf_hashes_file = BASE_DIR / "pdf_hashes.json"
        self.pdf_hashes = self._load_pdf_hashes()
        
        # Store configuration
        self.collection_name = collection_name
        self.ollama_url = ollama_url
        self.indexed_directory = indexed_directory
        self.max_workers = max_workers
        self.artifacts_path = artifacts_path
        self.max_file_size = max_file_size
        self.max_pages = max_pages
        
        # Initialize LlamaIndex components for hybrid retrieval
        self._init_llama_index()
        
        # Apply nest_asyncio for QueryFusionRetriever
        nest_asyncio.apply()
    
    def _load_pdf_hashes(self) -> Dict[str, str]:
        """Load previously processed PDF hashes"""
        if self.pdf_hashes_file.exists():
            try:
                with open(self.pdf_hashes_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_pdf_hashes(self):
        """Save PDF hashes to file"""
        try:
            with open(self.pdf_hashes_file, 'w') as f:
                json.dump(self.pdf_hashes, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save PDF hashes: {e}")

    def _init_llama_index(self):
        """Initialize LlamaIndex components for hybrid retrieval"""
        try:
            # Create docstore and simple vector store
            self.docstore = SimpleDocumentStore()
            self.vector_store = SimpleVectorStore()
            
            # Create storage context
            self.storage_context = StorageContext.from_defaults(
                docstore=self.docstore, 
                vector_store=self.vector_store
            )
            
            # Set up HuggingFace embedding model
            embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en-v1.5"
            )
            
            # Set up Ollama LLM
            llm = Ollama(model="granite3.1-moe:1b", request_timeout=120.0)
            
            # Set the embedding model and LLM in LlamaIndex settings
            Settings.embed_model = embed_model
            Settings.llm = llm
            
            # Initialize index as None - will be created when first documents are indexed
            self.index = None
            
            # Initialize node parser for deterministic chunking
            self.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
            
            # Initialize BM25 retriever (will be populated during indexing)
            self.bm25_retriever = None
            
            # Initialize hybrid retriever (will be created after indexing)
            self.hybrid_retriever = None
            
            logger.info("Successfully initialized LlamaIndex components")
        except Exception as e:
            logger.error(f"Failed to initialize LlamaIndex components: {e}")
            raise

    def _process_docling_structure(self, structure_data, file_name, file_hash=None, file_size=None):
        """Process the extracted structure (not raw markdown!)"""
        nodes = []
        
        # Sanitize metadata for all nodes
        def sanitize_metadata(metadata):
            """Ensure metadata is JSON serializable and clean"""
            sanitized = {}
            for key, value in metadata.items():
                if value is None:
                    continue
                if isinstance(value, (str, int, float, bool)):
                    sanitized[key] = value
                elif isinstance(value, dict):
                    sanitized[key] = json.dumps(value)
                else:
                    sanitized[key] = str(value)
            return sanitized
        
        # Process hierarchy
        for item in structure_data.get("hierarchy", []):
            metadata = sanitize_metadata({
                "file_name": file_name,
                "document_type": "heading",
                "heading_level": item.get("level", 1),
                "page": item.get("page", 0),
                "file_hash": file_hash
            })
            
            heading_node = Document(
                text=item.get("text", ""),
                metadata=metadata
            )
            nodes.append(heading_node)
        
        # Process tables
        for table_data in structure_data.get("tables", []):
            metadata = sanitize_metadata({
                "file_name": file_name,
                "document_type": "table",
                "table_number": table_data.get("table_number", 0),
                "page": table_data.get("page", 0),
                "caption": table_data.get("caption", ""),
                "shape": table_data.get("shape", ""),
                "file_hash": file_hash
            })
            
            table_node = Document(
                text=table_data.get("markdown", ""),
                metadata=metadata
            )
            nodes.append(table_node)
        
        # Process full text (chunked)
        full_text = structure_data.get("full_markdown", "")
        if full_text:
            metadata = sanitize_metadata({
                "file_name": file_name,
                "document_type": "full_text",
                "file_hash": file_hash
            })
            
            text_node = Document(
                text=full_text,
                metadata=metadata
            )
            nodes.append(text_node)
        
        return nodes

    def _process_and_index_document(self, file_path_or_stream, filename, is_stream=False, use_docling=True):
        """Unified method to process and index a document"""
        indexed_count = 0
        
        try:
            # Get file hash and size
            if is_stream:
                file_size = len(file_path_or_stream.getvalue())
                file_hash = PDFTextExtractor.get_stream_hash(file_path_or_stream)
            else:
                file_size = os.path.getsize(file_path_or_stream)
                file_hash = PDFTextExtractor.get_file_hash(file_path_or_stream)
            
            # Check if file is too large
            if file_size > self.max_file_size:
                logger.warning(f"Skipping large file: {filename} ({file_size / (1024*1024):.2f}MB > {self.max_file_size / (1024*1024):.2f}MB)")
                return 0
            
            # Check if this exact file content was already indexed AND if docstore has documents for this file
            file_already_indexed = (
                filename in self.pdf_hashes and 
                self.pdf_hashes[filename] == file_hash and
                any(node.metadata.get("file_name") == filename for node in self.docstore.docs.values())
            )
            
            if file_already_indexed:
                logger.info(f"Skipping already indexed file: {filename}")
                return 0
            
            nodes = []
            
            if use_docling:
                # Extract structure using Docling
                if is_stream:
                    structure_path = PDFTextExtractor.extract_structure_from_stream(
                        file_path_or_stream,
                        filename,
                        INDEXED_DIRECTORY,
                        artifacts_path=self.artifacts_path
                    )
                else:
                    structure_path = PDFTextExtractor.extract_structure_from_pdf(
                        file_path_or_stream,
                        INDEXED_DIRECTORY,
                        artifacts_path=self.artifacts_path
                    )
                
                if structure_path:
                    # Load the structured data
                    with open(structure_path, 'r', encoding='utf-8', errors='replace') as f:
                        structure_data = json.load(f)
                    
                    # Process the structure to create nodes
                    nodes = self._process_docling_structure(
                        structure_data, 
                        filename, 
                        file_hash, 
                        file_size
                    )
            else:
                # Fallback to text extraction
                if is_stream:
                    # Reset stream position
                    file_path_or_stream.seek(0)
                    
                    # Extract text from stream
                    extracted_pages = PDFTextExtractor.extract_text_from_stream(
                        file_path_or_stream,
                        filename,
                        INDEXED_DIRECTORY,
                        use_docling=False,
                        artifacts_path=self.artifacts_path,
                        max_pages=self.max_pages
                    )
                else:
                    # Extract text from file
                    extracted_pages = PDFTextExtractor.extract_text_from_pdf(
                        file_path_or_stream,
                        INDEXED_DIRECTORY,
                        use_docling=False,
                        artifacts_path=self.artifacts_path,
                        max_pages=self.max_pages
                    )
                
                # Process extracted pages
                for page_filename, page_path in extracted_pages.items():
                    chunks = self._process_file_safe(page_path, page_filename)
                    if chunks:
                        for chunk in chunks:
                            chunk["metadata"]["file_hash"] = file_hash
                            chunk["metadata"]["file_size"] = file_size
                        
                        # Convert chunks to nodes
                        for chunk in chunks:
                            metadata = chunk["metadata"]
                            text = chunk["text"]
                            node = Document(text=text, metadata=metadata)
                            nodes.append(node)
            
            # Ensure we have nodes to index
            if not nodes:
                logger.warning(f"No content extracted from {filename}")
                return 0
            
            # Use deterministic node parsing
            nodes = self.node_parser.get_nodes_from_documents(nodes)
            
            # Add nodes to docstore
            self.docstore.add_documents(nodes)
            
            # Update or create index
            if self.index is None:
                self.index = VectorStoreIndex(nodes=nodes, storage_context=self.storage_context)
            else:
                self.index.insert_nodes(nodes)
            
            indexed_count = len(nodes)
            
            # Update PDF hash
            self.pdf_hashes[filename] = file_hash
            self._save_pdf_hashes()
            
            return indexed_count
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            traceback.print_exc()
            return 0

    def index_documents(self, document_paths: Dict[str, str] = None, document_name: Optional[str] = None, use_docling=True):
        """Index documents with enhanced Docling structure preservation"""
        try:
            # Use provided directory or default
            document_paths = document_paths or self.indexed_directory
            
            # Validate directory exists
            if not os.path.isdir(document_paths):
                logger.error(f"Directory not found: {document_paths}")
                return 0
            
            # Update status
            indexing_status.update(
                is_indexing=True,
                status_message="Preparing to index documents",
                progress=0,
                error=None
            )
            
            # List all PDF files for Docling processing
            pdf_files = [f for f in os.listdir(UPLOAD_DIRECTORY) if f.endswith('.pdf')]
            
            # List all text files for fallback
            text_files = [f for f in os.listdir(document_paths) if f.endswith('.md')]
            
            if not pdf_files and not text_files:
                logger.warning("No files found for indexing")
                indexing_status.update(
                    is_indexing=False,
                    status_message="No files to index"
                )
                return 0
            
            # Check if we have any files that need processing
            files_to_process = []
            
            # Check PDF files
            for pdf_file in pdf_files:
                pdf_path = os.path.join(UPLOAD_DIRECTORY, pdf_file)
                file_hash = PDFTextExtractor.get_file_hash(pdf_path)
                
                # Only process if not already indexed or hash doesn't match
                if pdf_file not in self.pdf_hashes or self.pdf_hashes[pdf_file] != file_hash:
                    files_to_process.append((pdf_path, pdf_file))
            
            # Check text files
            for text_file in text_files:
                text_path = os.path.join(document_paths, text_file)
                file_hash = PDFTextExtractor.get_file_hash(text_path)
                
                # Only process if not already indexed or hash doesn't match
                if text_file not in self.pdf_hashes or self.pdf_hashes[text_file] != file_hash:
                    files_to_process.append((text_path, text_file))
            
            # If no files need processing, exit early
            if not files_to_process:
                logger.info("No new or modified files to index")
                indexing_status.update(
                    is_indexing=False,
                    status_message="No new or modified files to index"
                )
                return 0
            
            # Update status with file count
            indexing_status.update(
                total_files=len(files_to_process),
                processed_files=0,
                status_message=f"Found {len(files_to_process)} files to index"
            )
            
            indexed_count = 0
            
            # Process files using the unified method
            for file_path, filename in files_to_process:
                try:
                    # Update status
                    indexing_status.update(
                        current_file=filename,
                        status_message=f"Processing {filename}"
                    )
                    
                    # Use the unified method to process and index the file
                    count = self._process_and_index_document(
                        file_path_or_stream=file_path,
                        filename=filename,
                        is_stream=False,
                        use_docling=use_docling
                    )
                    
                    indexed_count += count
                    
                    # Update progress
                    indexing_status.update(
                        processed_files=indexing_status.processed_files + 1,
                        progress=int(((indexing_status.processed_files + 1) / len(files_to_process)) * 100),
                        status_message=f"Successfully processed {filename}"
                    )
                    
                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")
                    # Continue with next file instead of failing completely
                    indexing_status.update(
                        processed_files=indexing_status.processed_files + 1,
                        progress=int(((indexing_status.processed_files + 1) / len(files_to_process)) * 100),
                        status_message=f"Error processing {filename}: {str(e)}"
                    )
            
            # Create BM25 retriever only if we have documents
            try:
                if self.docstore.docs:
                    self.bm25_retriever = BM25Retriever.from_defaults(
                        docstore=self.docstore, 
                        similarity_top_k=5
                    )
                else:
                    logger.warning("No documents in docstore, skipping BM25 retriever creation")
                    self.bm25_retriever = None
            except Exception as bm25_error:
                logger.error(f"Failed to create BM25 retriever: {bm25_error}")
                self.bm25_retriever = None
            
            # Create hybrid retriever
            self.hybrid_retriever = self._create_hybrid_retriever()
            
            # Update final status
            indexing_status.update(
                is_indexing=False,
                status_message=f"Indexing complete. {indexed_count} chunks indexed from {len(files_to_process)} files.",
                progress=100
            )
            
            logger.info(f"Indexing complete. Total chunks indexed: {indexed_count}")
            return indexed_count
            
        except Exception as e:
            logger.error(f"Error in index_documents: {e}")
            # Reset status on error
            indexing_status.update(
                is_indexing=False,
                status_message=f"Indexing failed: {str(e)}",
                error=str(e)
            )
            return 0
    
    def index_document_from_stream(self, file_stream, filename, use_docling=True):
        """Index a document from a file stream with enhanced Docling structure preservation"""
        # Update status
        indexing_status.update(
            is_indexing=True,
            status_message=f"Processing {filename} from stream",
            progress=0,
            error=None
        )
        
        try:
            # Check file size before processing
            file_size = len(file_stream.getvalue())
            
            if file_size > self.max_file_size:
                logger.warning(f"Skipping large file: {filename} ({file_size / (1024*1024):.2f}MB > {self.max_file_size / (1024*1024):.2f}MB)")
                indexing_status.update(
                    is_indexing=False,
                    status_message=f"Skipping large file: {filename}",
                    error=f"File too large: {file_size / (1024*1024):.2f}MB"
                )
                return 0
            
            # Use the unified method to process and index the stream
            indexed_count = self._process_and_index_document(
                file_path_or_stream=file_stream,
                filename=filename,
                is_stream=True,
                use_docling=use_docling
            )
            
            if indexed_count > 0:
                # Create BM25 retriever
                try:
                    self.bm25_retriever = BM25Retriever.from_defaults(
                        docstore=self.docstore, 
                        similarity_top_k=5
                    )
                except Exception as bm25_error:
                    logger.error(f"Failed to create BM25 retriever: {bm25_error}")
                    self.bm25_retriever = None
                
                # Create hybrid retriever
                self.hybrid_retriever = self._create_hybrid_retriever()
                
                # Update status
                indexing_status.update(
                    is_indexing=False,
                    status_message=f"Successfully indexed {filename} from stream with Docling structure",
                    progress=100
                )
            else:
                # If nothing was indexed (e.g., file was already indexed)
                indexing_status.update(
                    is_indexing=False,
                    status_message=f"No new content to index from {filename}",
                    progress=100
                )
            
            return indexed_count
            
        except Exception as e:
            logger.error(f"Error processing {filename} from stream: {e}")
            # Ensure status is reset on error
            indexing_status.update(
                is_indexing=False,
                status_message=f"Error processing {filename}",
                error=str(e)
            )
            return 0
    
    def _create_hybrid_retriever(self):
        """Create a hybrid retriever that combines BM25 and vector search"""
        try:
            # Create retrievers
            vector_retriever = self.index.as_retriever(similarity_top_k=5)
            
            # Check if docstore has documents before creating BM25 retriever
            if not self.docstore.docs:
                logger.warning("No documents in docstore, skipping BM25 retriever creation")
                return vector_retriever
                
            # Create BM25 retriever
            if self.bm25_retriever is None:
                self.bm25_retriever = BM25Retriever.from_defaults(
                    docstore=self.docstore, 
                    similarity_top_k=5
                )
                
            # Create hybrid retriever using QueryFusionRetriever with explicit LLM
            hybrid_retriever = QueryFusionRetriever(
                [vector_retriever, self.bm25_retriever],
                num_queries=1,
                use_async=True,
                similarity_top_k=5,
                llm=Settings.llm
            )
            
            logger.info("Successfully created hybrid retriever")
            return hybrid_retriever
        except Exception as e:
            logger.error(f"Failed to create hybrid retriever: {e}")
            # Return vector retriever as fallback
            try:
                return self.index.as_retriever(similarity_top_k=5)
            except Exception as fallback_error:
                logger.error(f"Failed to create fallback retriever: {fallback_error}")
                return None
    
    async def index_documents_async(self, document_paths: Dict[str, str] = None, document_name: Optional[str] = None) -> int:
        """Async wrapper for index_documents to run in a thread pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.index_documents, document_paths, document_name
        )
    
    def search_documents(self, query: str, n_results: int = 3) -> List[Dict]:
        """
        Search indexed documents for relevant context using hybrid retrieval
        
        Args:
            query (str): User's query
            n_results (int): Number of top results to return
        
        Returns:
            List[Dict]: Relevant document contexts with metadata
        """
        # Validate query and results count
        if not query or not isinstance(query, str):
            logger.warning("Invalid query provided")
            return []
        
        n_results = max(1, min(n_results, 10))  # Limit results between 1 and 10
        
        # Check if index is empty
        if self.index is None or not self.docstore.docs:
            logger.warning("No documents have been indexed yet")
            return []
        
        try:
            # Use hybrid retriever if available
            if self.hybrid_retriever:
                # Perform hybrid search
                nodes = self.hybrid_retriever.retrieve(query)
                
                # Convert nodes to the expected format
                context_results = []
                for node in nodes[:n_results]:
                    context_results.append({
                        "text": node.text,
                        "metadata": node.metadata or {},
                        "relevance_score": node.score if hasattr(node, 'score') else 1.0
                    })
                
                return context_results
            else:
                # Fallback to vector search
                retriever = self.index.as_retriever(similarity_top_k=n_results)
                nodes = retriever.retrieve(query)
                
                # Convert nodes to the expected format
                context_results = []
                for node in nodes:
                    context_results.append({
                        "text": node.text,
                        "metadata": node.metadata or {},
                        "relevance_score": node.score if hasattr(node, 'score') else 1.0
                    })
                
                return context_results
        
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            traceback.print_exc()
            return []
    
    def get_collection_info(self) -> Dict:
        """
        Get information about the current collection
        
        Returns:
            Dict: Information about the collection
        """
        try:
            return {
                "collection_name": self.collection_name,
                "total_documents": len(self.docstore.docs) if self.docstore.docs else 0,
                "document_metadata": list(self.docstore.docs.keys()) if self.docstore.docs else [],
                "has_hybrid_retriever": self.hybrid_retriever is not None
            }
        except Exception as e:
            logger.error(f"Error retrieving collection info: {e}")
            return {}
        
    def search_documents_by_entity(self, entity_name: str, n_results: int = 10) -> List[Dict]:
        """
        Search for documents containing a specific entity
        
        Args:
            entity_name (str): Name of the entity to search for
            n_results (int): Number of top results to return
        
        Returns:
            List[Dict]: Relevant document contexts with metadata
        """
        # Validate entity name and results count
        if not entity_name or not isinstance(entity_name, str):
            logger.warning("Invalid entity name provided")
            return []
        
        n_results = max(1, min(n_results, 20))  # Limit results between 1 and 20
        
        # Check if index is empty
        if self.index is None or not self.docstore.docs:
            logger.warning("No documents have been indexed yet")
            return []
        
        try:
            # Use hybrid retriever if available
            if self.hybrid_retriever:
                # Perform hybrid search with entity-focused query
                query = f"Find information about {entity_name}"
                nodes = self.hybrid_retriever.retrieve(query)
                
                # Filter and convert nodes to the expected format
                context_results = []
                for node in nodes[:n_results]:
                    # Only include nodes that mention the entity
                    if entity_name.lower() in node.text.lower():
                        context_results.append({
                            "text": node.text,
                            "metadata": node.metadata or {},
                            "relevance_score": node.score if hasattr(node, 'score') else 1.0
                        })
                
                return context_results
            else:
                # Fallback to vector search
                query = f"Find information about {entity_name}"
                retriever = self.index.as_retriever(similarity_top_k=n_results)
                nodes = retriever.retrieve(query)
                
                # Filter and convert nodes to the expected format
                context_results = []
                for node in nodes:
                    # Only include nodes that mention the entity
                    if entity_name.lower() in node.text.lower():
                        context_results.append({
                            "text": node.text,
                            "metadata": node.metadata or {},
                            "relevance_score": node.score if hasattr(node, 'score') else 1.0
                        })
                
                return context_results
        
        except Exception as e:
            logger.error(f"Error searching documents by entity: {e}")
            traceback.print_exc()
            return []
    
    def search_by_document_type(self, doc_type: str, query: str = None, n_results: int = 5):
        """
        Search documents filtered by type (table, heading, paragraph).
        """
        # Validate inputs
        if not doc_type or not isinstance(doc_type, str):
            logger.warning("Invalid document type provided")
            return []
        
        n_results = max(1, min(n_results, 20))  # Limit results between 1 and 20
        
        # Check if index is empty
        if self.index is None or not self.docstore.docs:
            logger.warning("No documents have been indexed yet")
            return []
        
        try:
            # Create a filter for document type
            docstore_filter = lambda doc: doc.metadata.get("document_type") == doc_type
            
            # Create retriever with filter
            retriever = self.index.as_retriever(
                similarity_top_k=n_results,
                docstore_filter=docstore_filter
            )
            
            # Perform search
            if query:
                nodes = retriever.retrieve(query)
            else:
                # If no query, just get all documents of this type
                nodes = []
                for node_id, node in self.docstore.docs.items():
                    if node.metadata.get("document_type") == doc_type:
                        nodes.append(node)
            
            # Convert nodes to the expected format
            context_results = []
            for node in nodes[:n_results]:
                context_results.append({
                    "text": node.text,
                    "metadata": node.metadata or {},
                    "relevance_score": node.score if hasattr(node, 'score') else 1.0
                })
            
            return context_results
        
        except Exception as e:
            logger.error(f"Error searching by document type: {e}")
            traceback.print_exc()
            return []
    
    def search_tables_with_structure(self, query: str, n_results: int = 5):
        """
        Search specifically for tables with cell-level metadata.
        """
        # Validate inputs
        n_results = max(1, min(n_results, 20))  # Limit results between 1 and 20
        
        # Check if index is empty
        if self.index is None or not self.docstore.docs:
            logger.warning("No documents have been indexed yet")
            return []
        
        try:
            # Create a filter for table documents
            docstore_filter = lambda doc: doc.metadata.get("document_type") == "table"
            
            # Create retriever with filter
            retriever = self.index.as_retriever(
                similarity_top_k=n_results,
                docstore_filter=docstore_filter
            )
            
            # Perform search
            nodes = retriever.retrieve(query) if query else []
            
            # If no query, get all table documents
            if not query:
                nodes = []
                for node_id, node in self.docstore.docs.items():
                    if node.metadata.get("document_type") == "table":
                        nodes.append(node)
            
            # Parse cell_data for richer results
            formatted = []
            for node in nodes[:n_results]:
                metadata = node.metadata or {}
                cell_data = metadata.get("cell_data", "{}")
                
                try:
                    cells = json.loads(cell_data) if isinstance(cell_data, str) else cell_data
                except:
                    cells = []
                
                formatted.append({
                    "text": node.text,
                    "metadata": metadata,
                    "cells": cells,
                    "relevance_score": node.score if hasattr(node, 'score') else 1.0
                })
            
            return formatted
        
        except Exception as e:
            logger.error(f"Error searching tables with structure: {e}")
            traceback.print_exc()
            return []
    
    def get_document_outline(self, filename: str):
        """
        Get document structure outline from Docling headings.
        """
        try:
            base_name = filename.replace('.pdf', '').replace('.md', '')
            structure_file = INDEXED_DIRECTORY / f"{base_name}_structure.json"
            
            if not structure_file.exists():
                return []
            
            # Load Docling structure
            with open(structure_file, 'r', encoding='utf-8') as f:
                structure_data = json.load(f)
            
            # Extract hierarchy from Docling structure
            hierarchy = structure_data.get("hierarchy", [])
            
            # Convert to outline format
            outline = []
            for item in hierarchy:
                outline.append({
                    "text": item.get("text", ""),
                    "level": item.get("level", 1),
                    "page": item.get("page", 0),
                    "type": item.get("type", "heading")
                })
            
            return sorted(outline, key=lambda x: (x["page"], x["level"]))
            
        except Exception as e:
            logger.error(f"Error getting document outline: {e}")
            return []
    
    def _process_file_safe(self, file_path: str, file_name: str) -> Optional[List[Dict]]:
        """Process a single file with safe encoding handling"""
        try:
            # Calculate file hash
            file_hash = PDFTextExtractor.get_file_hash(file_path)
            
            # Check if this exact file content was already indexed
            if file_name in self.pdf_hashes and self.pdf_hashes[file_name] == file_hash:
                logger.info(f"Skipping already indexed file: {file_name}")
                return None
            
            # Read file content using safe reader
            text = safe_read_file(Path(file_path)).strip()
            
            if not text:
                logger.warning(f"Skipping empty document: {file_name}")
                return None
            
            # Use advanced chunking
            chunker = GazetteDocumentChunker(chunk_size=800, chunk_overlap=50)
            chunks = chunker.chunk_document(text, file_name)
            
            # Add metadata to each chunk
            for chunk in chunks:
                chunk["metadata"].update({
                    "file_name": file_name,
                    "file_hash": file_hash,
                    "timestamp": str(datetime.now())
                })
                chunk["id"] = str(uuid.uuid4())
            
            return chunks
        
        except Exception as e:
            logger.error(f"Error processing file {file_name}: {e}")
            traceback.print_exc()
            return None
        
class ChatbotService:
    def __init__(self, knowledge_base=None):
        """
        Initialize chatbot with an optional knowledge base
        
        Args:
            knowledge_base (KnowledgeBaseIndexer, optional): Indexed knowledge base
        """
        self.knowledge_base = knowledge_base
        self.llm_client = AsyncClient()
        self.logger = logging.getLogger(__name__)
    
    async def generate_response(self, query):
        """
        Generate a response using retrieved context and LLM
        
        Args:
            query (str): User's query
        
        Returns:
            str: Generated response
        """
        try:
            # If knowledge base exists, try to retrieve relevant context
            context_results = []
            if self.knowledge_base:
                try:
                    context_results = self.knowledge_base.search_documents(query)
                except Exception as context_error:
                    print(f"Context retrieval error: {context_error}")
            
            # Prepare the prompt
            if context_results:
                # If context is found, create an enhanced prompt
                context_str = "\n\n".join([
                    f"[{result['metadata'].get('document_name', 'Unknown Document')}] "
                    f"Relevance: {result['relevance_score']:.2f}\n"
                    f"{result['text']}"
                    for result in context_results
                ])
                
                enhanced_prompt = f"""
                Context from indexed documents:
                {context_str}
                User Query: {query}
                You are a helpful document intelligence assistant. You have access to documents that have been uploaded and processed.
                GUIDELINES:
                - Use the search_documents tool to find relevant information
                - Be efficient: one well-crafted search is usually sufficient
                - Only search again if the first results are clearly incomplete
                - Provide clear, accurate answers based on the document contents
                - Always cite your sources with filenames
                - If information isn't found, say so clearly
                - Be concise but thorough
                - IMPORTANT: Only use information from the search results provided by the tool. Do not make up information.

                When answering:
                1. Search the documents with a focused query
                2. Synthesize a clear answer from the results
                3. Include source citations (filenames)
                4. Only search again if absolutely necessary
                5. If the search results don't contain relevant information, clearly state that the information was not found
                """
            else:
                # If no context, use the query directly
                enhanced_prompt = query
            
            # Generate response using Ollama
            response = await self.llm_client.chat(
                model="granite3.1-moe:1b",
                messages=[{"role": "user", "content": enhanced_prompt}],
                stream=False
            )
            
            return response["message"]["content"]
        
        except Exception as e:
            # Fallback to a general response if any error occurs
            print(f"Error in generate_response: {e}")
            try:
                # Try to generate a response without context
                response = await self.llm_client.chat(
                    model="granite3.1-moe:1b",
                    messages=[{"role": "user", "content": query}],
                    stream=False
                )
                return response["message"]["content"]
            except Exception as fallback_error:
                return f"I'm sorry, but I'm having trouble processing your query. Error: {str(fallback_error)}"
            
    async def generate_structured_response(self, query):
        """
        Generate a structured response using entity extraction
        
        Args:
            query (str): User's query
        
        Returns:
            str: Generated response in structured format
        """
        try:
            # If knowledge base exists, try to retrieve relevant context
            context_results = []
            if self.knowledge_base:
                try:
                    context_results = self.knowledge_base.search_documents(query)
                except Exception as context_error:
                    print(f"Context retrieval error: {context_error}")
            
            # Prepare the structured prompt
            if context_results:
                # If context is found, create a structured prompt
                context_str = "\n\n".join([
                    f"[{result['metadata'].get('document_name', 'Unknown Document')}] "
                    f"Relevance: {result['relevance_score']:.2f}\n"
                    f"{result['text']}"
                    for result in context_results
                ])
                
                structured_prompt = f"""
                Context from indexed documents:
                {context_str}
                
                User Query: {query}
                
                You are an expert legal document analyst. Based ONLY on the provided context, perform the following tasks and output the result in JSON format:
                
                Task 1: Summary - Provide a one-sentence summary of the notice.
                Task 2: Identification - State the exact Gazette Notice number.
                Task 3: Entity Extraction - Identify all appointed members and their respective roles/titles from the list.
                Task 4: Key Details - Extract any important dates, terms of reference, or other significant information.
                
                Output format:
                {{
                "Gazette_Notice_ID": "GAZETTE NOTICE NO. XXXXX",
                "Summary": "One-sentence summary",
                "Exact_Location_Page": page_number,
                "Appointed_Entities": [
                    {{"Role": "Title", "Name": "Full Name"}},
                    // ... (rest of the list)
                ],
                "Key_Details": [
                    {{"Category": "Date", "Value": "Date information"}},
                    {{"Category": "Terms", "Value": "Terms of reference or other key information"}}
                    // ... (other important details)
                ]
                }}
                
                If the information is not found in the provided context, clearly state that in the respective fields.
                """
            else:
                # If no context, use the query directly
                structured_prompt = f"""
                User Query: {query}
                
                You are an expert legal document analyst. Based on your knowledge, perform the following tasks and output the result in JSON format:
                
                Task 1: Summary - Provide a one-sentence summary of what the user is asking about.
                Task 2: Entity Extraction - Identify key entities related to the query.
                
                Output format:
                {{
                "Query": "{query}",
                "Summary": "One-sentence summary of the topic",
                "Related_Entities": [
                    {{"Type": "Entity Type", "Name": "Entity Name", "Description": "Brief description"}}
                    // ... (related entities)
                ]
                }}
                
                If you don't have information about the query, clearly state that.
                """
            
            # Generate response using Ollama with structured output
            response = await self.llm_client.chat(
                model="granite3.1-moe:1b",
                messages=[{"role": "user", "content": structured_prompt}],
                stream=False
            )
            
            return response["message"]["content"]
        
        except Exception as e:
            # Fallback to a general response if any error occurs
            print(f"Error in generate_structured_response: {e}")
            try:
                # Try to generate a response without context
                response = await self.llm_client.chat(
                    model="granite3.1-moe:1b",
                    messages=[{"role": "user", "content": query}],
                    stream=False
                )
                return response["message"]["content"]
            except Exception as fallback_error:
                return f"I'm sorry, but I'm having trouble processing your query. Error: {str(fallback_error)}"




# Application state management
class AppState:
    def __init__(self, artifacts_path=None):
        self.knowledge_base_indexer = None
        self.initialized = False
        
        # Ensure models are available
        if artifacts_path:
            self.artifacts_path = Path(artifacts_path)
        else:
            # Try to ensure models are downloaded
            models_dir = ensure_docling_models()
            self.artifacts_path = models_dir if models_dir else (Path.home() / ".cache" / "docling" / "models")
    
    def initialize(self):
        """Initialize the knowledge base indexer"""
        if not self.initialized:
            self.knowledge_base_indexer = KnowledgeBaseIndexer(
                artifacts_path=self.artifacts_path
            )
            self.initialized = True
    
    def get_knowledge_base(self) -> KnowledgeBaseIndexer:
        """Get the knowledge base indexer, initializing if necessary"""
        if not self.initialized:
            self.initialize()
        return self.knowledge_base_indexer

# Global application state with artifacts path from constant
app_state = AppState(artifacts_path=DOCLING_ARTIFACTS_PATH)


structure_cache = StructureCache()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background indexing on application startup"""
    # Initialize knowledge base
    app_state.initialize()
    
    # Start indexing in background without blocking startup
    def background_index():
        try:
            logger.info("Starting background indexing")
            app_state.knowledge_base_indexer.index_documents()
            logger.info("Background indexing completed")
        except Exception as e:
            logger.error(f"Background indexing failed: {e}")
            indexing_status.update(
                is_indexing=False,
                status_message=f"Indexing failed: {str(e)}",
                error=str(e)
            )
    
    # Start indexing in a daemon thread
    indexing_thread = threading.Thread(target=background_index, daemon=True)
    indexing_thread.start()
    
    yield
    
    # Cleanup code (if needed)
    logger.info("Application shutting down")

app = FastAPI(
    lifespan=lifespan, 
    title="RAG Chatbot API", 
    version="1.0.0",
    description="Document Intelligence API with enhanced Docling support"
)


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Connection might be closed
                pass

manager = ConnectionManager()



def safe_read_file(file_path: Path, encoding: str = 'utf-8') -> str:
    """
    Safely read a file with multiple encoding fallbacks.
    
    Args:
        file_path: Path to the file
        encoding: Initial encoding to try (default: utf-8)
        
    Returns:
        str: File content
        
    Raises:
        Exception: If file cannot be read with any encoding
    """
    encodings = [encoding, 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc, errors='replace') as f:
                content = f.read()
                logger.debug(f"Successfully read {file_path} with encoding {enc}")
                return content
        except Exception as e:
            logger.debug(f"Failed to read {file_path} with encoding {enc}: {e}")
            continue
    
    # Last resort: read as binary and decode with ignore
    try:
        with open(file_path, 'rb') as f:
            content = f.read().decode('utf-8', errors='ignore')
            logger.warning(f"Read {file_path} as binary with ignored errors")
            return content
    except Exception as e:
        logger.error(f"Could not read {file_path} with any method: {e}")
        raise Exception(f"Failed to read file {file_path}")


#DONE
@app.post("/upload/")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...), use_docling: bool = True):
    """Upload and process PDF file with hash checking and Docling support"""
    try:
        if not file.filename.lower().endswith('.pdf'):
            return JSONResponse(
                content={"status": "error", "message": "Only PDF files are supported"},
                status_code=400
            )
        
        # Get knowledge base
        knowledge_base = app_state.get_knowledge_base()
        
        # Read file content into memory
        file_content = await file.read()
        
        # Check file size
        if len(file_content) > knowledge_base.max_file_size:
            return JSONResponse(
                content={
                    "status": "error", 
                    "message": f"File too large: {len(file_content) / (1024*1024):.2f}MB > {knowledge_base.max_file_size / (1024*1024):.2f}MB"
                },
                status_code=400
            )
        
        # Calculate file hash
        file_hash = hashlib.sha256(file_content).hexdigest()
        
        # Check if this PDF was already processed
        if file.filename in knowledge_base.pdf_hashes and knowledge_base.pdf_hashes[file.filename] == file_hash:
            logger.info(f"PDF already processed: {file.filename}")
            
            # Get existing page count
            pdf_basename = os.path.splitext(file.filename)[0]
            existing_pages = [f for f in os.listdir(INDEXED_DIRECTORY) 
                            if f.startswith(f"{pdf_basename}_page_") and f.endswith('.md')]
            
            return JSONResponse({
                "status": "success",
                "message": f"PDF already processed: {file.filename}",
                "pages_extracted": len(existing_pages),
                "already_indexed": True
            })
        
        # Save uploaded file to disk for compatibility
        file_location = UPLOAD_DIRECTORY / file.filename
        with open(file_location, "wb") as file_object:
            file_object.write(file_content)
        
        # Process using stream processing
        try:
            # Extract text and structure using Docling from stream
            if use_docling:
                # Create a fresh BytesIO object for structure extraction
                structure_stream = BytesIO(file_content)
                
                # Extract structure using Docling
                structure_path = PDFTextExtractor.extract_structure_from_stream(
                    structure_stream,
                    file.filename,
                    INDEXED_DIRECTORY,
                    artifacts_path=knowledge_base.artifacts_path
                )
                
                # Create another fresh BytesIO object for text extraction
                text_stream = BytesIO(file_content)
                
                # Extract text for compatibility
                extracted_pages = PDFTextExtractor.extract_text_from_stream(
                    text_stream,
                    file.filename,
                    INDEXED_DIRECTORY,
                    use_docling=False,  # Use fallback for text extraction
                    artifacts_path=knowledge_base.artifacts_path,
                    max_pages=knowledge_base.max_pages
                )
            else:
                # Create a fresh BytesIO object for text extraction
                text_stream = BytesIO(file_content)
                
                # Use standard text extraction
                extracted_pages = PDFTextExtractor.extract_text_from_stream(
                    text_stream,
                    file.filename,
                    INDEXED_DIRECTORY,
                    use_docling=False,
                    artifacts_path=knowledge_base.artifacts_path,
                    max_pages=knowledge_base.max_pages
                )
                structure_path = None
                
            # Store PDF hash
            knowledge_base.pdf_hashes[file.filename] = file_hash
            knowledge_base._save_pdf_hashes()
            
            # Index in background with better error handling
            def index_in_background():
                try:
                    # Create yet another fresh BytesIO object for indexing
                    index_stream = BytesIO(file_content)
                    
                    indexed_count = knowledge_base.index_document_from_stream(
                        index_stream,
                        file.filename,
                        use_docling=use_docling
                    )
                    
                    asyncio.run(manager.broadcast(json.dumps({
                        "type": "indexing_complete",
                        "filename": file.filename,
                        "pages_extracted": len(extracted_pages),
                        "documents_indexed": indexed_count,
                        "structure_path": structure_path
                    })))
                except Exception as e:
                    logger.error(f"Background indexing failed: {e}")
                    asyncio.run(manager.broadcast(json.dumps({
                        "type": "error",
                        "message": f"Indexing failed: {str(e)}"
                    })))
                    # Reset status on error
                    indexing_status.update(
                        is_indexing=False,
                        status_message=f"Indexing failed: {str(e)}",
                        error=str(e)
                    )
            
            background_tasks.add_task(index_in_background)
            
            return JSONResponse({
                "status": "processing", 
                "message": f"PDF uploaded successfully: {file.filename}",
                "pages_extracted": len(extracted_pages),
                "pages": list(extracted_pages.keys()),
                "structure_path": structure_path
            })
            
        except Exception as extract_error:
            logger.error(f"Error extracting content from PDF: {extract_error}")
            traceback.print_exc()
            
            # Reset status on error
            indexing_status.update(
                is_indexing=False,
                status_message=f"Error extracting content: {str(extract_error)}",
                error=str(extract_error)
            )
            
            return JSONResponse(
                content={
                    "status": "error", 
                    "message": f"Failed to extract text from PDF: {str(extract_error)}"
                },
                status_code=400
            )
        
    except Exception as e:
        logger.error(f"Unexpected error processing upload: {e}")
        traceback.print_exc()
        
        # Reset status on error
        indexing_status.update(
            is_indexing=False,
            status_message=f"Error processing upload: {str(e)}",
            error=str(e)
        )
        
        return JSONResponse(
            content={"status": "error", "message": f"Error: {str(e)}"},
            status_code=500
        )

# Load HTML content
try:
    with open(HTML_FILE_PATH, "r", encoding="utf-8", errors='replace') as file:
        html_content = file.read()
except FileNotFoundError:
    html_content = "<html><body><h1>Home page not found</h1></body></html>"
    logger.warning(f"HTML file not found at {HTML_FILE_PATH}")

@app.get("/")
async def get():
    return HTMLResponse(html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat and status updates"""
    await manager.connect(websocket)
    conversation = []
    
    # Initialize chatbot with knowledge base
    try:
        knowledge_base = app_state.get_knowledge_base()
        chatbot = ChatbotService(knowledge_base)
    except Exception as init_error:
        logger.error(f"Error initializing ChatbotService: {init_error}")
        chatbot = ChatbotService()
    
    try:
        # Send initial status
        await websocket.send_text(json.dumps({
            "type": "status",
            "data": {
                "is_indexing": indexing_status.is_indexing,
                "progress": indexing_status.progress,
                "total_files": indexing_status.total_files,
                "processed_files": indexing_status.processed_files,
                "current_file": indexing_status.current_file,
                "status_message": indexing_status.status_message,
                "error": indexing_status.error
            }
        }))
        
        while True:
            # Receive user message
            data = await websocket.receive_text()
            
            try:
                # Try to parse as JSON for structured commands
                message = json.loads(data)
                
                if message.get("type") == "query":
                    # Process as a query
                    chat_time = datetime.now()
                    response = await chatbot.generate_response(message.get("content", ""))
                    
                    # Store conversation
                    conversation.append({
                        "timestamp": chat_time.strftime('%B %d, %Y %I:%M %p'),
                        "user_message": message.get("content", ""),
                        "bot_response": response
                    })
                    
                    # Send response back
                    await websocket.send_text(json.dumps({
                        "type": "response",
                        "content": response
                    }))
                
                elif message.get("type") == "status_request":
                    # Send current status
                    await websocket.send_text(json.dumps({
                        "type": "status",
                        "data": {
                            "is_indexing": indexing_status.is_indexing,
                            "progress": indexing_status.progress,
                            "total_files": indexing_status.total_files,
                            "processed_files": indexing_status.processed_files,
                            "current_file": indexing_status.current_file,
                            "status_message": indexing_status.status_message,
                            "error": indexing_status.error
                        }
                    }))
                
                else:
                    # Unknown message type
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Unknown message type"
                    }))
            
            except json.JSONDecodeError:
                # Not JSON, treat as plain text query
                chat_time = datetime.now()
                response = await chatbot.generate_response(data)
                
                # Store conversation
                conversation.append({
                    "timestamp": chat_time.strftime('%B %d, %Y %I:%M %p'),
                    "user_message": data,
                    "bot_response": response
                })
                
                # Send response back
                await websocket.send_text(json.dumps({
                    "type": "response",
                    "content": response
                }))
    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        error_message = f"Error processing your request: {str(e)}"
        logger.error(error_message)
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": error_message
            }))
        except:
            pass  # Connection might be closed
    
    finally:
        manager.disconnect(websocket)
        
        # Log conversation (with error handling)
        try:
            with open("history.md", 'a', encoding='utf-8', errors='replace') as f:
                json.dump(conversation, f, indent=2)
                f.write('\n')
        except Exception as log_error:
            logger.error(f"Error logging conversation: {log_error}")

#DONE 
@app.post("/structured-query/")
async def structured_query_endpoint(query: str):
    # Add input validation
    if not query or len(query.strip()) == 0:
        return JSONResponse(
            content={"error": "Query cannot be empty"},
            status_code=400
        )
    
    try:
        knowledge_base = app_state.get_knowledge_base()
        chatbot = ChatbotService(knowledge_base)
        result = await chatbot.generate_structured_response(query)
        
        # Try to parse the response as JSON
        try:
            structured_result = json.loads(result)
            return JSONResponse(content=structured_result)
        except json.JSONDecodeError:
            # If not valid JSON, return as is
            return JSONResponse(content={"response": result})
    
    except Exception as e:
        # Log the full error for debugging
        logging.error(f"Structured query error: {str(e)}", exc_info=True)
        
        return JSONResponse(
            content={
                "error": "An error occurred during document search",
                "details": str(e)
            },
            status_code=500
        )


#DONE    
@app.post("/entity-search/")
async def entity_search_endpoint(entity_name: str, n_results: int = 10):
    # Add input validation
    if not entity_name or len(entity_name.strip()) == 0:
        return JSONResponse(
            content={"error": "Entity name cannot be empty"},
            status_code=400
        )
    
    try:
        knowledge_base = app_state.get_knowledge_base()
        results = knowledge_base.search_documents_by_entity(entity_name, n_results)
        
        # Ensure result is serializable
        if results is None:
            return JSONResponse(content={"results": []})
        
        return JSONResponse(content={"results": results})
    
    except Exception as e:
        # Log the full error for debugging
        logging.error(f"Entity search error: {str(e)}", exc_info=True)
        
        return JSONResponse(
            content={
                "error": "An error occurred during entity search",
                "details": str(e)
            },
            status_code=500
        )


#DONE
@app.get("/collection-info/")
async def get_collection_info():
    """
    Endpoint to retrieve current collection information
    """
    try:
        knowledge_base = app_state.get_knowledge_base()
        collection_info = knowledge_base.get_collection_info()
        
        # Enhanced Docling support information
        collection_info["docling_enabled"] = True
        collection_info["docling_structures"] = []
        collection_info["docling_metrics"] = {
            "total_structures": 0,
            "total_tables": 0,
            "total_headings": 0,
            "documents_with_tables": 0
        }
        
        # Check for Docling structure files and gather metrics
        for f in os.listdir(INDEXED_DIRECTORY):
            if f.endswith("_structure.json"):
                collection_info["docling_structures"].append(f)
                
                # Gather metrics from structure files
                try:
                    structure_file = INDEXED_DIRECTORY / f
                    with open(structure_file, 'r', encoding='utf-8', errors='replace') as sf:
                        structure_data = json.load(sf)
                    
                    collection_info["docling_metrics"]["total_structures"] += 1
                    collection_info["docling_metrics"]["total_tables"] += len(structure_data.get("tables", []))
                    collection_info["docling_metrics"]["total_headings"] += len(structure_data.get("hierarchy", []))
                    
                    if len(structure_data.get("tables", [])) > 0:
                        collection_info["docling_metrics"]["documents_with_tables"] += 1
                        
                except Exception as e:
                    logger.error(f"Error reading structure file {f}: {e}")
        
        return collection_info
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error", 
                "message": str(e)
            }
        )
    

#DONE
@app.get("/list-documents/")
async def list_documents():
    """
    List all documents with proper PDF-to-markdown mapping.
    Returns documents sorted by last modified date (newest first).
    """
    try:
        # Validate directories exist
        if not UPLOAD_DIRECTORY.exists():
            logger.error(f"Upload directory does not exist: {UPLOAD_DIRECTORY}")
            return JSONResponse(content={"error": "Upload directory not found"}, status_code=500)
        
        if not INDEXED_DIRECTORY.exists():
            logger.error(f"Indexed directory does not exist: {INDEXED_DIRECTORY}")
            return JSONResponse(content={"error": "Indexed directory not found"}, status_code=500)
        
        documents = []
        
        # Get all PDFs
        pdf_files = [f for f in os.listdir(UPLOAD_DIRECTORY) if f.endswith('.pdf')]
        
        # Build lookup dictionaries for markdown and structure files
        md_files_by_pdf = {}
        structure_files_by_pdf = {}
        
        # Map markdown files (single file per PDF: <basename>.md)
        for md_file in os.listdir(INDEXED_DIRECTORY):
            if md_file.endswith('.md'):
                base_name = os.path.splitext(md_file)[0]
                md_files_by_pdf[base_name] = md_file
        
        # Map structure files (<basename>_structure.json)
        for struct_file in os.listdir(INDEXED_DIRECTORY):
            if struct_file.endswith("_structure.json"):
                match = re.match(r"^(.+?)_structure\.json$", struct_file)
                if match:
                    base_name = match.group(1)
                    structure_files_by_pdf[base_name] = struct_file
        
        # Build document list
        for pdf_file in pdf_files:
            pdf_path = UPLOAD_DIRECTORY / pdf_file
            base_name = os.path.splitext(pdf_file)[0]
            
            # Get associated markdown file (if any)
            md_file = md_files_by_pdf.get(base_name)
            
            # Check for Docling structure
            has_docling_structure = base_name in structure_files_by_pdf
            
            # Initialize document data with defaults
            doc_data = {
                "name": pdf_file,
                "pages": 1,
                "size": pdf_path.stat().st_size if pdf_path.exists() else 0,
                "last_modified": os.path.getmtime(pdf_path) if pdf_path.exists() else time.time(),
                "preview": "Preview not available",
                "md_files": [md_file] if md_file else [],  # Keep as list for API consistency
                "has_content": bool(md_file or has_docling_structure),
                "has_docling_structure": has_docling_structure,
                "structure_file": structure_files_by_pdf.get(base_name),
                "structure_details": {},
                "processing_method": "docling" if has_docling_structure else "legacy"
            }
            
            # Load and process structure data if available
            structure_data = None
            if has_docling_structure:
                try:
                    structure_file_path = INDEXED_DIRECTORY / structure_files_by_pdf[base_name]
                    with open(structure_file_path, 'r', encoding='utf-8', errors='replace') as f:
                        structure_data = json.load(f)
                    
                    # Extract structure metrics
                    doc_data["structure_details"] = {
                        "tables_count": len(structure_data.get("tables", [])),
                        "headings_count": len(structure_data.get("hierarchy", [])),
                        "images_count": len(structure_data.get("images", [])),
                        "has_full_markdown": "full_markdown" in structure_data
                    }
                    
                    # Get page count from structure
                    doc_data["pages"] = structure_data.get("page_count") or \
                                       structure_data.get("summary", {}).get("num_pages", 1)
                    
                    # Get preview from full_markdown in structure
                    if structure_data.get("full_markdown"):
                        full_markdown = structure_data["full_markdown"]
                        # Truncate at word boundary for cleaner preview
                        if len(full_markdown) > 200:
                            preview_text = full_markdown[:200]
                            last_space = preview_text.rfind(' ')
                            if last_space > 150:  # Only truncate at space if it's not too far back
                                preview_text = preview_text[:last_space]
                            doc_data["preview"] = preview_text + "..."
                        else:
                            doc_data["preview"] = full_markdown
                            
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in structure file for {pdf_file}: {e}")
                    doc_data["structure_details"]["error"] = "Invalid JSON format"
                except Exception as e:
                    logger.error(f"Error reading structure details for {pdf_file}: {e}")
                    doc_data["structure_details"]["error"] = str(e)
            
            # Fall back to markdown file if structure data not available or incomplete
            if md_file and (not structure_data or not doc_data.get("preview") or doc_data["preview"] == "Preview not available"):
                try:
                    md_file_path = INDEXED_DIRECTORY / md_file
                    with open(md_file_path, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                    
                    # Get preview (truncate at word boundary)
                    if len(content) > 200:
                        preview_text = content[:200]
                        last_space = preview_text.rfind(' ')
                        if last_space > 150:
                            preview_text = preview_text[:last_space]
                        doc_data["preview"] = preview_text + "..."
                    else:
                        doc_data["preview"] = content
                    
                    # Estimate page count if not already set from structure
                    if doc_data["pages"] == 1:  # Default wasn't changed
                        page_breaks = content.count('\n--- Page ')
                        if page_breaks > 0:
                            doc_data["pages"] = page_breaks + 1
                        else:
                            # Estimate based on content length (2000 chars ≈ 1 page)
                            doc_data["pages"] = max(1, len(content) // 2000)
                            
                except Exception as e:
                    logger.error(f"Error reading markdown for {pdf_file}: {e}")
                    # Keep default preview and page count
            
            documents.append(doc_data)
        
        # Sort by last modified (newest first)
        documents.sort(key=lambda x: x['last_modified'], reverse=True)
        
        return JSONResponse(content=documents)
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}", exc_info=True)
        return JSONResponse(
            content={"error": f"Failed to list documents: {str(e)}"},
            status_code=500
        )
    
#DONE
@app.get("/document-structure/{filename}")
async def get_document_structure(filename: str, regenerate: bool = False):
    """
    Get comprehensive document structure analysis from Docling with robust error handling.
    
    Args:
        filename: The filename of the document
        regenerate: Whether to regenerate the structure if it's missing or corrupted
    """
    try:
        # Sanitize filename
        safe_filename = os.path.basename(filename)
        base_name = safe_filename.replace('.pdf', '').replace('.md', '')
        
        # Look for Docling structure file
        structure_file = None
        for f in os.listdir(INDEXED_DIRECTORY):
            if f == f"{base_name}_structure.json":
                structure_file = INDEXED_DIRECTORY / f
                break
        
        # If structure file doesn't exist or regeneration is requested
        if not structure_file or not structure_file.exists() or regenerate:
            logger.info(f"No structure file found for {safe_filename} or regeneration requested")
            
            # Try to regenerate the structure
            structure_path = PDFTextExtractor._regenerate_structure_if_needed(safe_filename, base_name)
            
            if structure_path:
                structure_file = Path(structure_path)
            else:
                return JSONResponse(
                    content={"error": f"Failed to regenerate structure for {safe_filename}"},
                    status_code=404
                )
        
        # Load the structured data with proper encoding and validation
        try:
            with open(structure_file, 'r', encoding='utf-8', errors='replace') as f:
                structure_data = json.load(f)
        except json.JSONDecodeError as json_error:
            logger.error(f"Invalid JSON in structure file for {safe_filename}: {json_error}")
            
            # Try to regenerate the structure
            if not regenerate:
                logger.info(f"Attempting to regenerate structure for {safe_filename} due to JSON error")
                return await get_document_structure(filename, regenerate=True)
            else:
                return JSONResponse(
                    content={"error": f"Invalid JSON in structure file for {safe_filename} and regeneration failed: {str(json_error)}"},
                    status_code=500
                )
        except Exception as file_error:
            logger.error(f"Error reading structure file for {safe_filename}: {file_error}")
            
            # Try to regenerate the structure
            if not regenerate:
                logger.info(f"Attempting to regenerate structure for {safe_filename} due to file error")
                return await get_document_structure(filename, regenerate=True)
            else:
                return JSONResponse(
                    content={"error": f"Error reading structure file for {safe_filename} and regeneration failed: {str(file_error)}"},
                    status_code=500
                )
        
        # Validate and enrich the structure data
        structure_data = PDFTextExtractor._validate_and_enrich_structure(structure_data, safe_filename)
        
        # Validate structure completeness
        completeness_check = PDFTextExtractor._validate_structure_completeness(structure_data)
        structure_data["completeness_check"] = completeness_check
        
        # If structure is incomplete and regeneration is not requested, suggest regeneration
        if not completeness_check["is_complete"] and not regenerate:
            structure_data["regeneration_suggested"] = True
            structure_data["regeneration_reason"] = "Structure is incomplete"
        
        return JSONResponse(content=structure_data)
        
    except Exception as e:
        logger.error(f"Error retrieving Docling structure: {e}")
        return JSONResponse(
            content={"error": f"Failed to retrieve structure: {str(e)}"},
            status_code=500
        )
    
#DONE
@app.get("/document-outline/{filename}")
async def get_document_outline(filename: str):
    """
    Get hierarchical outline of document structure.
    """
    try:
        knowledge_base = app_state.get_knowledge_base()
        outline = knowledge_base.get_document_outline(filename)
        
        return JSONResponse(content={
            "filename": filename,
            "outline": outline
        })
    except Exception as e:
        logger.error(f"Error getting outline: {e}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )
    
#DONE
@app.get("/document-markdown/{filename}")
async def get_document_markdown(filename: str):
    """
    Get document in markdown format with preserved formatting from Docling.
    """
    try:
        # Sanitize filename
        safe_filename = os.path.basename(filename)
        base_name = safe_filename.replace('.pdf', '').replace('.md', '')
        
        # First try to get markdown from Docling structure
        structure_file = INDEXED_DIRECTORY / f"{base_name}_structure.json"
        if structure_file.exists():
            try:
                with open(structure_file, 'r', encoding='utf-8', errors='replace') as f:
                    structure_data = json.load(f)
                
                # Get markdown from Docling structure
                if "full_markdown" in structure_data:
                    return JSONResponse(content={
                        "filename": filename,
                        "format": "markdown",
                        "content": structure_data["full_markdown"],
                        "has_formatting": True,
                        "source": "docling"
                    })
            except Exception as e:
                logger.error(f"Error reading Docling structure file: {e}")
        
        # Fall back to individual markdown file
        md_path = INDEXED_DIRECTORY / f"{base_name}.md"
        if md_path.exists():
            try:
                with open(md_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                
                return JSONResponse(content={
                    "filename": filename,
                    "format": "markdown",
                    "content": content,
                    "has_formatting": True,
                    "source": "markdown_file"
                })
            except Exception as e:
                logger.error(f"Error reading markdown file {md_path}: {e}")
        
        return JSONResponse(
            content={"error": f"No markdown content found for {filename}"},
            status_code=404
        )
        
    except Exception as e:
        logger.error(f"Error in get_document_markdown: {e}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )
    

#DONE
@app.post("/table-search/")
async def table_search_endpoint(query: str, n_results: int = 10):
    """
    Search specifically for tables in the indexed documents with robust error handling.
    """
    try:
        if not query or len(query.strip()) == 0:
            return JSONResponse(
                content={"error": "Query cannot be empty"},
                status_code=400
            )
        
        knowledge_base = app_state.get_knowledge_base()
        
        # Search for documents with document_type="table"
        results = knowledge_base.collection.query(
            query_texts=[query],
            n_results=n_results,
            where={"document_type": "table"},
            include=["documents", "metadatas", "distances"]
        )
        
        # Prepare results
        table_results = []
        documents = results.get('documents', [[]])
        metadatas = results.get('metadatas', [[]])
        distances = results.get('distances', [[]])
        
        # Safely flatten and combine results
        for doc_group, meta_group, dist_group in zip(documents, metadatas, distances):
            for doc, meta, distance in zip(doc_group, meta_group, dist_group):
                try:
                    # Initialize table result with default values
                    table_result = {
                        "table_id": meta.get("table_id", "unknown"),
                        "file_name": meta.get("file_name", "unknown"),
                        "page": meta.get("page", 0),
                        "caption": meta.get("caption", ""),
                        "content": "",
                        "headers": [],
                        "row_count": 0,
                        "column_count": 0,
                        "relevance_score": 1 / (1 + (distance or 0)),
                        "format": "unknown",
                        "context_heading": "",
                        "contextual_label": "",
                        "validation_status": "unknown",
                        "validation_warnings": []
                    }
                    
                    # Try to parse as JSON first (old format)
                    try:
                        table_data = json.loads(doc)
                        
                        # Handle new Docling table format
                        if "markdown" in table_data:
                            # New format: use the markdown table directly
                            table_result["content"] = table_data["markdown"]
                            table_result["headers"] = table_data.get("headers", [])
                            table_result["row_count"] = meta.get("shape", [0, 0])[0] if "shape" in meta else len(table_data.get("rows", []))
                            table_result["column_count"] = meta.get("shape", [0, 0])[1] if "shape" in meta else len(table_data.get("headers", []))
                            # NEW FIELDS
                            table_result["context_heading"] = table_data.get("context_heading", "")
                            table_result["contextual_label"] = table_data.get("contextual_label", "")
                            table_result["validation_status"] = table_data.get("validation_status", "unknown")
                            table_result["validation_warnings"] = table_data.get("validation_warnings", [])
                            table_result["format"] = "docling"
                        elif "cells" in table_data:
                            # Old format: convert cells to markdown
                            table_result["content"] = "Table content:\n"
                            for row in table_data["cells"]:
                                table_result["content"] += " | ".join([cell.get("text", "") for cell in row]) + "\n"
                            table_result["row_count"] = len(table_data["cells"])
                            table_result["column_count"] = max(len(row) for row in table_data["cells"]) if table_data["cells"] else 0
                            table_result["format"] = "legacy"
                            table_result["validation_warnings"].append("Legacy table format")
                        else:
                            # Unknown format, use raw text
                            table_result["content"] = doc
                            table_result["format"] = "unknown"
                            table_result["validation_warnings"].append("Unknown table format")
                    except json.JSONDecodeError:
                        # If it's not JSON, it's likely the new markdown table format
                        table_result["content"] = doc
                        table_result["headers"] = meta.get("headers", [])
                        table_result["row_count"] = meta.get("shape", [0, 0])[0] if "shape" in meta else 0
                        table_result["column_count"] = meta.get("shape", [0, 0])[1] if "shape" in meta else 0
                        table_result["format"] = "markdown"
                        table_result["validation_warnings"].append("Markdown table format without JSON structure")
                    
                    # Validate table result
                    if not table_result["content"]:
                        table_result["validation_warnings"].append("Empty table content")
                        table_result["validation_status"] = "error"
                    
                    if not table_result["context_heading"]:
                        table_result["validation_warnings"].append("No context heading found")
                        if table_result["validation_status"] == "validated":
                            table_result["validation_status"] = "validated_with_warnings"
                    
                    table_results.append(table_result)
                    
                except Exception as table_error:
                    logger.error(f"Error processing table result: {table_error}")
                    # Add a placeholder table with error information
                    table_results.append({
                        "table_id": meta.get("table_id", "unknown"),
                        "file_name": meta.get("file_name", "unknown"),
                        "page": meta.get("page", 0),
                        "caption": meta.get("caption", ""),
                        "content": f"Error processing table: {str(table_error)}",
                        "headers": [],
                        "row_count": 0,
                        "column_count": 0,
                        "relevance_score": 1 / (1 + (distance or 0)),
                        "format": "error",
                        "context_heading": "",
                        "contextual_label": "",
                        "validation_status": "error",
                        "validation_warnings": [f"Error processing table: {str(table_error)}"]
                    })
        
        return JSONResponse(content={
            "query": query,
            "table_count": len(table_results),
            "tables": table_results
        })
    
    except Exception as e:
        logging.error(f"Table search error: {str(e)}", exc_info=True)
        return JSONResponse(
            content={
                "error": "An error occurred during table search",
                "details": str(e)
            },
            status_code=500
        )
    
# NOT YET UPDATED FROM it's parsing old table format. IT should handle the new Docling table structure.
# SOME MAY BE REDUNDANT, OR REPLACED BY OTHER ENDPOINTS

@app.get("/pdf/{file_name}")
async def get_pdf(file_name: str):
    # Ensure file_name is sanitized to prevent directory traversal
    safe_file_name = os.path.basename(file_name)
    file_path = UPLOAD_DIRECTORY / safe_file_name
    
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File {safe_file_name} not found")
    
    return FileResponse(path=str(file_path), filename=safe_file_name)


@app.post("/query/")
async def query_endpoint(query: str):
    # Add input validation
    if not query or len(query.strip()) == 0:
        return JSONResponse(
            content={"error": "Query cannot be empty"},
            status_code=400
        )
    
    try:
        knowledge_base = app_state.get_knowledge_base()
        result = knowledge_base.search_documents(query)
        
        # Ensure result is serializable
        if result is None:
            return JSONResponse(content={"result": []})
        
        return JSONResponse(content={"result": result})
    
    except Exception as e:
        # Log the full error for debugging
        logging.error(f"Query error: {str(e)}", exc_info=True)
        
        return JSONResponse(
            content={
                "error": "An error occurred during document search",
                "details": str(e)
            },
            status_code=500
        )
    
    
@app.post("/reindex/")
async def reindex_documents(background_tasks: BackgroundTasks):
    """
    Endpoint to trigger reindexing of documents
    
    Uses background tasks to prevent blocking the request
    """
    try:
        knowledge_base = app_state.get_knowledge_base()
        
        # Run indexing in background
        background_tasks.add_task(knowledge_base.index_documents)
        
        return {
            "status": "Reindexing started",
            "message": "Documents will be reindexed in the background"
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error", 
                "message": str(e)
            }
        )
    


@app.get("/status/")
async def get_status():
    """
    Endpoint to retrieve current indexing status
    """
    try:
        # Read status from file if it exists
        if os.path.exists(STATUS_FILE_PATH):
            with open(STATUS_FILE_PATH, 'r', encoding='utf-8', errors='replace') as f:
                return json.load(f)
        
        # Return default status if file doesn't exist
        return {
            "is_indexing": False,
            "progress": 0,
            "total_files": 0,
            "processed_files": 0,
            "current_file": "",
            "status_message": "Idle",
            "error": None
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error", 
                "message": str(e)
            }
        )

@app.get("/conversation-history/")
async def get_conversation_history():
    """
    Get conversation history from stored conversations
    
    Returns:
        JSONResponse: List of conversation history items
    """
    try:
        conversation_history = []
        
        # Try to load from history file
        history_file = "history.md"
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8', errors='replace') as f:
                    # Read all conversation entries
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                conversation = json.loads(line)
                                if isinstance(conversation, list) and len(conversation) > 0:
                                    # Get the latest message from this conversation
                                    latest_msg = conversation[-1]
                                    conversation_history.append({
                                        "id": str(hash(line)),  # Simple ID from content hash
                                        "title": latest_msg.get("user_message", "Conversation")[:30] + ("..." if len(latest_msg.get("user_message", "")) > 30 else ""),
                                        "preview": latest_msg.get("user_message", "")[:100] + ("..." if len(latest_msg.get("user_message", "")) > 100 else ""),
                                        "timestamp": latest_msg.get("timestamp", datetime.now().isoformat())
                                    })
                            except json.JSONDecodeError:
                                continue
            except Exception as e:
                logger.error(f"Error reading conversation history: {e}")
        
        # If no history found, return sample data
        if not conversation_history:
            conversation_history = [
                {
                    "id": "1",
                    "title": "Welcome to GovKenyaGazette AI",
                    "preview": "Start a conversation about your documents",
                    "timestamp": datetime.now().isoformat()
                }
            ]
        
        # Sort by timestamp (newest first)
        conversation_history.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return JSONResponse(content=conversation_history)
        
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {e}")
        return JSONResponse(
            content={"error": f"Failed to retrieve conversation history: {str(e)}"},
            status_code=500
        )
    

@app.get("/document-tables/{filename}")
async def get_document_tables(filename: str, use_docling: bool = True):
    """Get all tables from document with enhanced Docling support."""
    try:
        # Sanitize filename
        safe_filename = os.path.basename(filename)
        base_name = safe_filename.replace('.pdf', '').replace('.md', '')
        
        # If Docling is requested, try to get tables from structure first
        if use_docling:
            # Look for Docling structure file
            structure_file = None
            for f in os.listdir(INDEXED_DIRECTORY):
                if f == f"{base_name}_structure.json":
                    structure_file = INDEXED_DIRECTORY / f
                    break
            
            if structure_file and structure_file.exists():
                # Load the structured data
                with open(structure_file, 'r', encoding='utf-8', errors='replace') as f:
                    structure_data = json.load(f)
                
                # Extract tables from Docling structure
                tables = []
                doc_items = structure_data.get("doc_items", [])
                
                for i, item in enumerate(doc_items):
                    if item.get("label") == "table":
                        # Extract table data
                        table_data = {
                            "table_id": f"docling_table_{i}",
                            "source": "docling",
                            "page": item.get("prov", [{}])[0].get("page_no", 0)
                        }
                        
                        # Extract table content
                        if "cells" in item:
                            # Convert to a more readable format
                            table_content = []
                            for row in item["cells"]:
                                table_content.append([cell.get("text", "") for cell in row])
                            
                            table_data["content"] = table_content
                            table_data["rows"] = len(table_content)
                            table_data["columns"] = len(table_content[0]) if table_content else 0
                        else:
                            # Fallback to raw text
                            table_data["content"] = str(item)
                            table_data["rows"] = 0
                            table_data["columns"] = 0
                        
                        tables.append(table_data)
                
                if tables:
                    return JSONResponse(content={
                        "filename": safe_filename,
                        "tables": tables,
                        "table_count": len(tables),
                        "source": "docling"
                    })
        
        # Fall back to regular structure extraction
        structure_response = await get_document_structure(filename)
        if isinstance(structure_response, JSONResponse):
            structure_data = json.loads(structure_response.body.decode())
            if "error" in structure_data:
                return structure_response
            
            return JSONResponse(content={
                "filename": safe_filename,
                "tables": structure_data.get("tables", []),
                "table_count": len(structure_data.get("tables", [])),
                "source": "text_extraction"
            })
        
        return JSONResponse(
            content={"error": "Failed to retrieve tables"},
            status_code=500
        )
        
    except Exception as e:
        logger.error(f"Error extracting tables: {e}")
        return JSONResponse(
            content={"error": f"Failed to extract tables: {str(e)}"},
            status_code=500
        )
    

@app.get("/document-export/{filename}")
async def export_document_structure(filename: str, format: str = "json", use_docling: bool = True):
    """
    Export document structure in various formats with enhanced Docling support.
    Supports: json, csv (for tables), docling (raw Docling structure)
    """
    try:
        # Sanitize filename
        safe_filename = os.path.basename(filename)
        base_name = safe_filename.replace('.pdf', '').replace('.md', '')
        
        # If Docling format is requested, try to get Docling structure first
        if format == "docling" and use_docling:
            # Look for Docling structure file
            structure_file = None
            for f in os.listdir(INDEXED_DIRECTORY):
                if f == f"{base_name}_structure.json":
                    structure_file = INDEXED_DIRECTORY / f
                    break
            
            if structure_file and structure_file.exists():
                # Load the structured data
                with open(structure_file, 'r', encoding='utf-8', errors='replace') as f:
                    structure_data = json.load(f)
                
                return JSONResponse(content={
                    "format": "docling",
                    "data": structure_data
                })
        
        # If CSV format is requested, try to get tables from Docling
        if format == "csv" and use_docling:
            # Look for Docling structure file
            structure_file = None
            for f in os.listdir(INDEXED_DIRECTORY):
                if f == f"{base_name}_structure.json":
                    structure_file = INDEXED_DIRECTORY / f
                    break
            
            if structure_file and structure_file.exists():
                # Load the structured data
                with open(structure_file, 'r', encoding='utf-8', errors='replace') as f:
                    structure_data = json.load(f)
                
                # Extract tables and convert to CSV
                csv_data = []
                doc_items = structure_data.get("doc_items", [])
                
                for item in doc_items:
                    if item.get("label") == "table" and "cells" in item:
                        # Convert table to CSV format
                        for row in item["cells"]:
                            csv_row = [cell.get("text", "") for cell in row]
                            csv_data.append(csv_row)
                
                if csv_data:
                    return JSONResponse(content={
                        "format": "csv",
                        "data": csv_data,
                        "source": "docling"
                    })
        
        # Get structure using standard method
        structure_response = await get_document_structure(filename)
        if isinstance(structure_response, JSONResponse):
            structure_data = json.loads(structure_response.body.decode())
            
            if "error" in structure_data:
                return structure_response
            
            if format == "json":
                return JSONResponse(content=structure_data)
            
            elif format == "csv" and structure_data.get("tables"):
                # Export tables as CSV
                csv_data = []
                for table in structure_data["tables"]:
                    csv_data.append(table["columns"])
                    csv_data.extend(table["rows"])
                
                return JSONResponse(content={
                    "format": "csv",
                    "data": csv_data,
                    "source": "text_extraction"
                })
            
            else:
                return JSONResponse(
                    content={"error": f"Unsupported format: {format}"},
                    status_code=400
                )
        
        return JSONResponse(
            content={"error": "Failed to export structure"},
            status_code=500
        )
        
    except Exception as e:
        logger.error(f"Error exporting structure: {e}")
        return JSONResponse(
            content={"error": f"Export failed: {str(e)}"},
            status_code=500
        )


@app.get("/config/")
async def get_config():
    """
    Get current configuration
    """
    try:
        knowledge_base = app_state.get_knowledge_base()
        
        return JSONResponse(content={
            "artifacts_path": str(knowledge_base.artifacts_path),  # Convert to string for JSON serialization
            "max_file_size": knowledge_base.max_file_size,
            "max_pages": knowledge_base.max_pages,
            "max_workers": knowledge_base.max_workers,
            "collection_name": knowledge_base.collection_name
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error", 
                "message": str(e)
            }
        )

@app.post("/config/")
async def update_config(
    artifacts_path: Optional[str] = None,
    max_file_size: Optional[int] = None,
    max_pages: Optional[int] = None,
    max_workers: Optional[int] = None
):
    """
    Update configuration (requires restart to take effect)
    """
    try:
        # Note: These changes will only take effect after a restart
        # This is just for viewing what would be changed
        
        config_updates = {}
        
        if artifacts_path is not None:
            config_updates["artifacts_path"] = artifacts_path
            # Update the constant
            global DOCLING_ARTIFACTS_PATH
            DOCLING_ARTIFACTS_PATH = Path(artifacts_path)
            DOCLING_ARTIFACTS_PATH.mkdir(exist_ok=True)
        
        if max_file_size is not None:
            config_updates["max_file_size"] = max_file_size
        
        if max_pages is not None:
            config_updates["max_pages"] = max_pages
        
        if max_workers is not None:
            config_updates["max_workers"] = max_workers
        
        return JSONResponse(content={
            "status": "success",
            "message": "Configuration updated. Restart required for changes to take effect.",
            "updates": config_updates
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error", 
                "message": str(e)
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)