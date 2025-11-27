import time
import os
import uuid
from datetime import datetime
import logging
import json
import re
import traceback
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Set
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

import chromadb
from chromadb import HttpClient
from chromadb.config import Settings
from chromadb.utils import embedding_functions
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
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.schema import Document, NodeWithScore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
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

# Ensure directories exist
UPLOAD_DIRECTORY.mkdir(exist_ok=True)
INDEXED_DIRECTORY.mkdir(exist_ok=True)

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

class PDFTextExtractor:
    @staticmethod
    def extract_text_from_pdf(pdf_path, output_dir):
        """
        Extract text from each page of a PDF with naming convention:
        <pdf_basename>_page_<n>.txt
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get PDF base name without extension
        pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
        
        pdf_document = fitz.open(pdf_path)
        extracted_pages = {}
        
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text = page.get_text()
            
            # Use consistent naming: <basename>_page_<n>.txt
            txt_filename = f"{pdf_basename}_page_{page_num + 1}.txt"
            txt_file_path = os.path.join(output_dir, txt_filename)
            
            with open(txt_file_path, 'w', encoding='utf-8') as file:
                file.write(text)
            
            extracted_pages[txt_filename] = txt_file_path
        
        pdf_document.close()
        return extracted_pages

    @staticmethod
    def get_file_hash(file_path):
        """Calculate SHA256 hash of a file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    

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
    

class KnowledgeBaseIndexer:
    def __init__(self, 
                 collection_name="pdf_documents", 
                 ollama_url="http://localhost:11434/api/embeddings", 
                 chroma_host="localhost", 
                 chroma_port=8081, 
                 indexed_directory=INDEXED_DIRECTORY,
                 max_workers=4):  # Number of parallel workers
        
        """
        Initialize ChromaDB collection for indexing documents
        
        Args:
            collection_name (str): Name of the ChromaDB collection
            ollama_url (str): URL of the Ollama server
            chroma_host (str): ChromaDB server host
            chroma_port (int): ChromaDB server port
            indexed_directory (str): Directory containing indexed text files
            max_workers (int): Maximum number of parallel workers for indexing
        """
        # Track PDF hashes to avoid re-extraction
        self.pdf_hashes_file = BASE_DIR / "pdf_hashes.json"
        self.pdf_hashes = self._load_pdf_hashes()
        # Store configuration
        self.collection_name = collection_name
        self.ollama_url = ollama_url
        self.indexed_directory = indexed_directory
        self.max_workers = max_workers
        
        # Initialize ChromaDB client
        try:
            self.chroma_client = chromadb.HttpClient(
                host=chroma_host, 
                port=chroma_port,
                settings=Settings(
                    chroma_server_host=chroma_host,
                    chroma_server_http_port=chroma_port
                )
            )
            logger.info("Successfully initialized ChromaDB HTTP Client")
        except Exception as http_client_error:
            logger.warning(f"HTTP Client initialization failed: {http_client_error}")
            try:
                # Fallback to Persistent Client
                self.chroma_client = chromadb.PersistentClient(path="./chroma_db_new")
                logger.info("Fallback to Persistent Client")
            except Exception as persistent_client_error:
                logger.error(f"Persistent Client initialization failed: {persistent_client_error}")
                raise
        
        # Initialize Ollama embedding function 
        try:
            from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
            self.embedding_function = OllamaEmbeddingFunction(
                model_name="granite-embedding:278m",
                url=ollama_url
            )
            logger.info("Successfully initialized Ollama embedding function")
        except Exception as embedding_error:
            logger.error(f"Failed to initialize embedding function: {embedding_error}")
            raise
        
        # Create or get collection
        try:
            self.collection = self.chroma_client.get_or_create_collection(name=collection_name)
            logger.info(f"Successfully created/retrieved collection: {collection_name}")
        except Exception as collection_error:
            logger.error(f"Collection creation failed: {collection_error}")
            raise
        
        # Track metadata for indexed documents
        self.document_metadata = {}
        
        # Load existing file hashes to avoid re-indexing
        self.indexed_files = self._load_indexed_files()
        
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
            # Create docstore and vector store
            self.docstore = SimpleDocumentStore()
            
            # Create Chroma vector store
            chroma_collection = self.chroma_client.get_or_create_collection(f"{self.collection_name}_llama")
            self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            
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
            
            # Create index (will be populated during indexing)
            self.index = VectorStoreIndex(nodes=[], storage_context=self.storage_context)
            
            # Initialize node parser for chunking
            self.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
            
            # Initialize BM25 retriever (will be populated during indexing)
            self.bm25_retriever = None
            
            # Initialize hybrid retriever (will be created after indexing)
            self.hybrid_retriever = None
            
            logger.info("Successfully initialized LlamaIndex components")
        except Exception as e:
            logger.error(f"Failed to initialize LlamaIndex components: {e}")
            raise


    def _load_indexed_files(self) -> Dict[str, str]:
        """Load information about already indexed files"""
        try:
            # Get all existing documents in the collection
            result = self.collection.get(include=["metadatas"])
            indexed_files = {}
            
            for metadata in result.get("metadatas", []):
                if metadata and "file_hash" in metadata and "file_path" in metadata:
                    indexed_files[metadata["file_path"]] = metadata["file_hash"]
            
            logger.info(f"Loaded {len(indexed_files)} previously indexed files")
            return indexed_files
        except Exception as e:
            logger.error(f"Error loading indexed files: {e}")
            return {}

    def _process_file(self, file_path: str, file_name: str) -> Optional[List[Dict]]:
        """Process a single file"""
        try:
            # Calculate file hash
            file_hash = PDFTextExtractor.get_file_hash(file_path)
            
            # Check if this exact file content was already indexed
            if file_path in self.indexed_files and self.indexed_files[file_path] == file_hash:
                logger.info(f"Skipping already indexed file: {file_name}")
                return None
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                text = file.read().strip()
            
            if not text:
                logger.warning(f"Skipping empty document: {file_name}")
                return None
            
            # Use advanced chunking
            chunker = GazetteDocumentChunker(chunk_size=800, chunk_overlap=50)
            chunks = chunker.chunk_document(text, file_name)
            
            # Add metadata to each chunk
            for chunk in chunks:
                chunk["metadata"].update({
                    "file_path": file_path,
                    "file_hash": file_hash,
                    "timestamp": str(datetime.now())
                })
                chunk["id"] = str(uuid.uuid4())
            
            return chunks
        
        except Exception as e:
            logger.error(f"Error processing file {file_name}: {e}")
            return None


    def index_documents(self, document_paths: Dict[str, str] = None, document_name: Optional[str] = None) -> int:
        """Index documents with advanced chunking and parallel processing"""
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
        
        # List all text files in the directory
        text_files = [f for f in os.listdir(document_paths) if f.endswith('.txt')]
        
        if not text_files:
            logger.warning("No text files found for indexing")
            indexing_status.update(
                is_indexing=False,
                status_message="No files to index"
            )
            return 0
        
        # Update status with file count
        indexing_status.update(
            total_files=len(text_files),
            processed_files=0,
            status_message=f"Found {len(text_files)} files to index"
        )
        
        indexed_count = 0
        all_chunks = []
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for file_name in text_files:
                file_path = os.path.join(document_paths, file_name)
                futures.append(executor.submit(self._process_file, file_path, file_name))
            
            # Collect results
            for future in as_completed(futures):
                chunks = future.result()
                if chunks:
                    all_chunks.extend(chunks)
                    
                    # Update progress
                    processed = indexing_status.processed_files + 1
                    progress = int((processed / len(text_files)) * 100)
                    indexing_status.update(
                        processed_files=processed,
                        current_file=chunks[0]["metadata"]["document_name"],
                        progress=progress,
                        status_message=f"Processing {chunks[0]['metadata']['document_name']}"
                    )
        
        # Create LlamaIndex documents from chunks
        llama_docs = [Document(text=chunk["text"], metadata=chunk["metadata"]) for chunk in all_chunks]
        
        # Parse documents into nodes
        nodes = self.node_parser.get_nodes_from_documents(llama_docs)
        
        # Add nodes to docstore
        self.docstore.add_documents(nodes)
        
        # Add to Chroma vector store
        self.index = VectorStoreIndex(nodes=nodes, storage_context=self.storage_context)
        
        # Create BM25 retriever
        self.bm25_retriever = BM25Retriever.from_defaults(
            docstore=self.docstore, 
            similarity_top_k=5
        )
        
        # Create hybrid retriever
        self.hybrid_retriever = self._create_hybrid_retriever()
        
        # Add documents to Chroma collection in batches
        if all_chunks:
            batch_size = 100  # Adjust based on performance
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i + batch_size]
                
                try:
                    self.collection.add(
                        documents=[chunk["text"] for chunk in batch],
                        ids=[chunk["id"] for chunk in batch],
                        metadatas=[chunk["metadata"] for chunk in batch]
                    )
                    
                    # Update indexed files cache
                    for chunk in batch:
                        file_path = chunk["metadata"]["file_path"]
                        file_hash = chunk["metadata"]["file_hash"]
                        self.indexed_files[file_path] = file_hash
                        self.document_metadata[chunk["id"]] = {
                            "text": chunk["text"],
                            "metadata": chunk["metadata"]
                        }
                    
                    indexed_count += len(batch)
                    logger.info(f"Indexed batch of {len(batch)} chunks")
                except Exception as e:
                    logger.error(f"Error adding batch to collection: {e}")
        
        # Update final status
        indexing_status.update(
            is_indexing=False,
            status_message=f"Indexing complete. {indexed_count} chunks indexed from {len(text_files)} files.",
            progress=100
        )
        
        logger.info(f"Indexing complete. Total chunks indexed: {indexed_count}")
        return indexed_count
    
    def _create_hybrid_retriever(self):
        """Create a hybrid retriever that combines BM25 and vector search"""
        try:
            # Create retrievers
            vector_retriever = self.index.as_retriever(similarity_top_k=5)
            
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
        
        # Check if collection is empty
        if self.collection.count() == 0:
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
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    include=["documents", "metadatas", "distances"]
                )
                
                # Prepare results with full context and robust extraction
                context_results = []
                
                documents = results.get('documents', [[]])
                metadatas = results.get('metadatas', [[]])
                distances = results.get('distances', [[]])
                
                # Safely flatten and combine results
                for doc_group, meta_group, dist_group in zip(documents, metadatas, distances):
                    for doc, meta, distance in zip(doc_group, meta_group, dist_group):
                        # Ensure all required fields are present
                        context_results.append({
                            "text": doc or "",
                            "metadata": meta or {},
                            "relevance_score": 1 / (1 + (distance or 0))
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
                "total_documents": self.collection.count(),
                "document_metadata": list(self.document_metadata.keys()),
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
        
        # Check if collection is empty
        if self.collection.count() == 0:
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
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    include=["documents", "metadatas", "distances"]
                )
                
                # Prepare results with full context and robust extraction
                context_results = []
                
                documents = results.get('documents', [[]])
                metadatas = results.get('metadatas', [[]])
                distances = results.get('distances', [[]])
                
                # Safely flatten and combine results
                for doc_group, meta_group, dist_group in zip(documents, metadatas, distances):
                    for doc, meta, distance in zip(doc_group, meta_group, dist_group):
                        # Only include documents that mention the entity
                        if entity_name.lower() in doc.lower():
                            # Ensure all required fields are present
                            context_results.append({
                                "text": doc or "",
                                "metadata": meta or {},
                                "relevance_score": 1 / (1 + (distance or 0))
                            })
                
                return context_results
        
        except Exception as e:
            logger.error(f"Error searching documents by entity: {e}")
            traceback.print_exc()
            return []


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
    def __init__(self):
        self.knowledge_base_indexer = None
        self.initialized = False
    
    def initialize(self):
        """Initialize the knowledge base indexer"""
        if not self.initialized:
            self.knowledge_base_indexer = KnowledgeBaseIndexer()
            self.initialized = True
    
    def get_knowledge_base(self) -> KnowledgeBaseIndexer:
        """Get the knowledge base indexer, initializing if necessary"""
        if not self.initialized:
            self.initialize()
        return self.knowledge_base_indexer

# Global application state
app_state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background indexing on application startup"""
    # Initialize the knowledge base
    app_state.initialize()
    
    # Start indexing in background without blocking startup
    def background_index():
        try:
            logger.info("Starting background indexing")
            app_state.knowledge_base_indexer.index_documents()
            logger.info("Background indexing completed")
        except Exception as e:
            logger.error(f"Background indexing failed: {e}")
            indexing_status.update(error=str(e))
    
    # Start indexing in a daemon thread
    indexing_thread = threading.Thread(target=background_index, daemon=True)
    indexing_thread.start()
    
    yield
    
    # Cleanup code (if needed)
    logger.info("Application shutting down")

app = FastAPI(lifespan=lifespan, title="RAG Chatbot API", version="1.0.0")

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

@app.post("/upload/")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload and process PDF file with hash checking"""
    try:
        if not file.filename.lower().endswith('.pdf'):
            return JSONResponse(
                content={"status": "error", "message": "Only PDF files are supported"},
                status_code=400
            )
        
        # Save uploaded file
        file_location = UPLOAD_DIRECTORY / file.filename
        with open(file_location, "wb") as file_object:
            content = await file.read()
            file_object.write(content)
        
        # Calculate PDF hash
        pdf_hash = PDFTextExtractor.get_file_hash(str(file_location))
        
        # Check if this PDF was already processed
        knowledge_base = app_state.get_knowledge_base()
        
        if file.filename in knowledge_base.pdf_hashes and knowledge_base.pdf_hashes[file.filename] == pdf_hash:
            logger.info(f"PDF already processed: {file.filename}")
            
            # Get existing page count
            pdf_basename = os.path.splitext(file.filename)[0]
            existing_pages = [f for f in os.listdir(INDEXED_DIRECTORY) 
                            if f.startswith(f"{pdf_basename}_page_") and f.endswith('.txt')]
            
            return JSONResponse({
                "status": "success",
                "message": f"PDF already processed: {file.filename}",
                "pages_extracted": len(existing_pages),
                "already_indexed": True
            })
        
        # Extract text from PDF
        try:
            extracted_pages = PDFTextExtractor.extract_text_from_pdf(
                str(file_location), 
                str(INDEXED_DIRECTORY)
            )
        except Exception as extract_error:
            return JSONResponse(
                content={
                    "status": "error", 
                    "message": f"Failed to extract text from PDF: {str(extract_error)}"
                },
                status_code=400
            )
        
        # Store PDF hash
        knowledge_base.pdf_hashes[file.filename] = pdf_hash
        knowledge_base._save_pdf_hashes()
        
        # Index in background
        def index_in_background():
            try:
                indexed_count = knowledge_base.index_documents(
                    document_paths=str(INDEXED_DIRECTORY), 
                    document_name=file.filename
                )
                
                asyncio.run(manager.broadcast(json.dumps({
                    "type": "indexing_complete",
                    "filename": file.filename,
                    "pages_extracted": len(extracted_pages),
                    "documents_indexed": indexed_count
                })))
            except Exception as e:
                logger.error(f"Background indexing failed: {e}")
                asyncio.run(manager.broadcast(json.dumps({
                    "type": "error",
                    "message": f"Indexing failed: {str(e)}"
                })))
        
        background_tasks.add_task(index_in_background)
        
        return JSONResponse({
            "status": "processing", 
            "message": f"PDF uploaded successfully: {file.filename}",
            "pages_extracted": len(extracted_pages),
            "pages": list(extracted_pages.keys())
        })
        
    except Exception as e:
        logger.error(f"Unexpected error processing upload: {e}")
        return JSONResponse(
            content={"status": "error", "message": f"Error: {str(e)}"},
            status_code=500
        )
    
# Load HTML content
try:
    with open(HTML_FILE_PATH, "r", encoding="utf-8") as file:
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
            with open("history.txt", 'a', encoding='utf-8') as f:
                json.dump(conversation, f, indent=2)
                f.write('\n')
        except Exception as log_error:
            logger.error(f"Error logging conversation: {log_error}")

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
            with open(STATUS_FILE_PATH, 'r') as f:
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

@app.get("/collection-info/")
async def get_collection_info():
    """
    Endpoint to retrieve current collection information
    """
    try:
        knowledge_base = app_state.get_knowledge_base()
        collection_info = knowledge_base.get_collection_info()
        return collection_info
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error", 
                "message": str(e)
            }
        )
    

@app.get("/list-documents/")
async def list_documents():
    """
    List all documents with proper PDF-to-text-files mapping.
    """
    try:
        documents = []
        
        # Get all PDFs
        pdf_files = [f for f in os.listdir(UPLOAD_DIRECTORY) if f.endswith('.pdf')]
        
        # Get all text files and group by base name
        text_files_by_pdf = {}
        for txt_file in os.listdir(INDEXED_DIRECTORY):
            if not txt_file.endswith('.txt'):
                continue
            
            # Extract base name from pattern: <basename>_page_<n>.txt
            match = re.match(r"^(.+?)_page_\d+\.txt$", txt_file)
            if match:
                base_name = match.group(1)
                if base_name not in text_files_by_pdf:
                    text_files_by_pdf[base_name] = []
                text_files_by_pdf[base_name].append(txt_file)
        
        # Build document list
        for pdf_file in pdf_files:
            pdf_path = UPLOAD_DIRECTORY / pdf_file
            base_name = os.path.splitext(pdf_file)[0]
            
            # Get associated text files
            text_files = sorted(text_files_by_pdf.get(base_name, []))
            
            # Get preview from first page
            preview = "Preview not available"
            if text_files:
                try:
                    first_page_path = INDEXED_DIRECTORY / text_files[0]
                    with open(first_page_path, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                        preview = content[:200] + ("..." if len(content) > 200 else "")
                except Exception as e:
                    logger.error(f"Error reading preview for {pdf_file}: {e}")
            
            documents.append({
                "name": pdf_file,
                "pages": len(text_files),
                "size": pdf_path.stat().st_size if pdf_path.exists() else 0,
                "last_modified": os.path.getmtime(pdf_path) if pdf_path.exists() else time.time(),
                "preview": preview,
                "text_files": text_files,
                "has_content": len(text_files) > 0
            })
        
        # Sort by last modified (newest first)
        documents.sort(key=lambda x: x['last_modified'], reverse=True)
        
        return JSONResponse(content=documents)
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return JSONResponse(
            content={"error": f"Failed to list documents: {str(e)}"},
            status_code=500
        )


@app.get("/document-content/{filename}")
async def get_document_content(filename: str):
    """
    Return merged text content for a PDF by finding all its page files.
    Handles both "filename.pdf" and just "filename" inputs.
    """
    try:
        # Sanitize filename
        safe_filename = os.path.basename(filename)
        
        # Remove .pdf extension if present to get base name
        base_name = safe_filename.replace('.pdf', '').replace('.txt', '')
        
        # Find all page files for this document
        # Pattern: <base_name>_page_<n>.txt
        page_files = []
        for f in os.listdir(INDEXED_DIRECTORY):
            # Match: exact_basename_page_number.txt
            match = re.match(rf"^{re.escape(base_name)}_page_(\d+)\.txt$", f)
            if match:
                page_num = int(match.group(1))
                page_files.append((page_num, f))
        
        if not page_files:
            logger.error(f"No text content found for document: {base_name}")
            return JSONResponse(
                content={"error": f"No text content found for document {base_name}.pdf"},
                status_code=404
            )
        
        # Sort by page number
        page_files.sort(key=lambda x: x[0])
        
        # Combine all pages
        combined_content = ""
        for page_num, page_file in page_files:
            page_path = INDEXED_DIRECTORY / page_file
            try:
                with open(page_path, 'r', encoding='utf-8', errors='replace') as f:
                    combined_content += f"--- Page {page_num} ---\n\n"
                    combined_content += f.read()
                    combined_content += "\n\n"
            except Exception as e:
                logger.error(f"Error reading page file {page_file}: {e}")
                combined_content += f"--- Page {page_num} (ERROR) ---\n\n"
        
        return JSONResponse(content={
            "filename": f"{base_name}.pdf",
            "content": combined_content,
            "pages": len(page_files),
            "file_size": len(combined_content),
            "text_files": [f for _, f in page_files]
        })
        
    except Exception as e:
        logger.error(f"Error retrieving document content for {filename}: {e}")
        return JSONResponse(
            content={"error": f"Failed to retrieve document: {str(e)}"},
            status_code=500
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
        history_file = "history.txt"
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
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
                    "title": "Welcome to DocuMind AI",
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
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)