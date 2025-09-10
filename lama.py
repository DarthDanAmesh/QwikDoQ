import time
import os
import uuid
from datetime import datetime
import logging
import json
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
        Extract text from each page of a PDF and save as separate text files
        
        Args:
            pdf_path (str): Path to the input PDF file
            output_dir (str): Directory to save extracted text files
        
        Returns:
            Dict[str, str]: Dictionary of page numbers to file paths
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Open the PDF file
        pdf_document = fitz.open(pdf_path)
        
        # Extract text from each page
        extracted_pages = {}
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text = page.get_text()
            
            # Define the filename for each page
            txt_file_path = os.path.join(output_dir, f"page_{page_num + 1}.txt")
            
            # Save the extracted text to a file
            with open(txt_file_path, 'w', encoding='utf-8') as file:
                file.write(text)
            
            extracted_pages[f"page_{page_num + 1}"] = txt_file_path
        
        # Close the PDF file
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
                model_name="embeddinggemma:latest",
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

    def _process_file(self, file_path: str, file_name: str) -> Optional[Dict]:
        """Process a single file for indexing"""
        try:
            # Calculate file hash to check if it's already indexed
            file_hash = PDFTextExtractor.get_file_hash(file_path)
            
            # Skip if file is already indexed with the same hash
            if file_path in self.indexed_files and self.indexed_files[file_path] == file_hash:
                logger.info(f"Skipping already indexed file: {file_name}")
                return None
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                text = file.read().strip()
            
            # Skip empty documents
            if not text:
                logger.warning(f"Skipping empty document: {file_name}")
                return None
            
            # Generate unique ID
            doc_id = str(uuid.uuid4())
            
            # Prepare metadata
            metadata = {
                "document_name": file_name,
                "file_path": file_path,
                "file_hash": file_hash,
                "timestamp": str(datetime.now())
            }
            
            return {
                "id": doc_id,
                "text": text,
                "metadata": metadata
            }
        except Exception as e:
            logger.error(f"Error processing file {file_name}: {e}")
            return None

    def index_documents(self, document_paths: Dict[str, str] = None, document_name: Optional[str] = None) -> int:
        """Index documents with parallel processing"""
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
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for file_name in text_files:
                file_path = os.path.join(document_paths, file_name)
                futures.append(executor.submit(self._process_file, file_path, file_name))
            
            # Collect results
            documents_to_add = []
            for future in as_completed(futures):
                result = future.result()
                if result:
                    documents_to_add.append(result)
                    
                    # Update progress
                    processed = indexing_status.processed_files + 1
                    progress = int((processed / len(text_files)) * 100)
                    indexing_status.update(
                        processed_files=processed,
                        current_file=result["metadata"]["document_name"],
                        progress=progress,
                        status_message=f"Processing {result['metadata']['document_name']}"
                    )
            
            # Add documents to collection in batches
            if documents_to_add:
                batch_size = 100  # Adjust based on performance
                for i in range(0, len(documents_to_add), batch_size):
                    batch = documents_to_add[i:i + batch_size]
                    
                    try:
                        self.collection.add(
                            documents=[doc["text"] for doc in batch],
                            ids=[doc["id"] for doc in batch],
                            metadatas=[doc["metadata"] for doc in batch]
                        )
                        
                        # Update indexed files cache
                        for doc in batch:
                            self.indexed_files[doc["metadata"]["file_path"]] = doc["metadata"]["file_hash"]
                            self.document_metadata[doc["id"]] = {
                                "text": doc["text"],
                                "metadata": doc["metadata"]
                            }
                        
                        indexed_count += len(batch)
                        logger.info(f"Indexed batch of {len(batch)} documents")
                    except Exception as e:
                        logger.error(f"Error adding batch to collection: {e}")
        
        # Update final status
        indexing_status.update(
            is_indexing=False,
            status_message=f"Indexing complete. {indexed_count} documents indexed.",
            progress=100
        )
        
        logger.info(f"Indexing complete. Total documents indexed: {indexed_count}")
        return indexed_count
    
    async def index_documents_async(self, document_paths: Dict[str, str] = None, document_name: Optional[str] = None) -> int:
        """Async wrapper for index_documents to run in a thread pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.index_documents, document_paths, document_name
        )
    
    def search_documents(self, query: str, n_results: int = 3) -> List[Dict]:
        """
        Search indexed documents for relevant context
        
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
            # Perform query with error handling
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
                "document_metadata": list(self.document_metadata.keys())
            }
        except Exception as e:
            logger.error(f"Error retrieving collection info: {e}")
            return {}

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
                Please provide a comprehensive answer based on the given context. 
                If the context doesn't fully answer the query, use your general knowledge 
                to supplement the response and indicate what additional information might be needed.
                """
            else:
                # If no context, use the query directly
                enhanced_prompt = query
            
            # Generate response using Ollama
            response = await self.llm_client.chat(
                model="qwen2:0.5b",
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
                    model="qwen2:0.5b",
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
    """Upload and process PDF file"""
    try:
        # Validate file type
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
        
        # Index the extracted documents in the background
        knowledge_base = app_state.get_knowledge_base()
        
        def index_in_background():
            try:
                indexed_count = knowledge_base.index_documents(
                    document_paths=str(INDEXED_DIRECTORY), 
                    document_name=file.filename
                )
                
                # Broadcast completion message
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
            "message": f"PDF uploaded and text extraction completed. Indexing in progress: {file.filename}",
            "pages_extracted": len(extracted_pages),
            "pages": list(extracted_pages.keys())
        })
        
    except Exception as e:
        logger.error(f"Unexpected error processing upload: {e}")
        return JSONResponse(
            content={
                "status": "error", 
                "message": f"Unexpected error processing upload: {str(e)}"
            },
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)