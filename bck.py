import time
import os
import uuid
from datetime import datetime
import logging
import json
import traceback
from pathlib import Path

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
    BackgroundTasks
)

from fastapi.responses import (
    HTMLResponse, 
    FileResponse, 
    JSONResponse
)
from fastapi.staticfiles import StaticFiles

# Initialize Ollama and ChromaDB
llm = AsyncClient()
chroma_client = chromadb.PersistentClient(path="./chroma_db_new")


CHROMA_HOST = "localhost"
CHROMA_PORT = 8081

# Constants
UPLOAD_DIRECTORY = r"C:\Users\danie\OneDrive\Pictures\OSINT\TwitterAPI\Sentiment\uploads"
#prefix r before path works with tryindexing.py file
INDEXED_DIRECTORY = r"C:\Users\danie\OneDrive\Pictures\OSINT\TwitterAPI\Sentiment\indexed"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(INDEXED_DIRECTORY, exist_ok=True)


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



from typing import Dict, List, Optional

class KnowledgeBaseIndexer:
    #https://cookbook.chromadb.dev/integrations/ollama/embeddings/#basic-usage -- python section: for the PersistentClient
    def __init__(self, 
                 collection_name="pdf_documents", 
                 ollama_url="http://localhost:11434/api/embeddings", 
                 chroma_host="localhost", 
                 chroma_port=8081, 
                 indexed_directory=INDEXED_DIRECTORY):
        
        """
        Initialize ChromaDB collection for indexing documents
        
        Args:
            collection_name (str): Name of the ChromaDB collection
            ollama_url (str): URL of the Ollama server
            chroma_host (str): ChromaDB server host
            chroma_port (int): ChromaDB server port
            indexed_directory (str): Directory containing indexed text files
        """
        
        
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Store configuration
        self.collection_name = collection_name
        self.ollama_url = ollama_url
        self.indexed_directory = indexed_directory
        
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
            self.logger.info("Successfully initialized ChromaDB HTTP Client")
        except Exception as http_client_error:
            self.logger.warning(f"HTTP Client initialization failed: {http_client_error}")
            try:
                # Fallback to Persistent Client
                self.chroma_client = chromadb.PersistentClient(path="./chroma_db_new")
                self.logger.info("Fallback to Persistent Client")
            except Exception as persistent_client_error:
                self.logger.error(f"Persistent Client initialization failed: {persistent_client_error}")
                raise
        
        # Initialize Ollama embedding function 

        try:
            from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
            self.embedding_function = OllamaEmbeddingFunction(
                model_name="qwen2:0.5b",
                #model_name="mxbai-embed-large",
                url=ollama_url
            )
            self.logger.info("Successfully initialized Ollama embedding function")
        except Exception as embedding_error:
            self.logger.error(f"Failed to initialize embedding function: {embedding_error}")
            raise
        
        # Create or get collection
        # create_collection(name="docs")
        try:
            self.collection = self.chroma_client.get_or_create_collection(name=collection_name)
            self.logger.info(f"Successfully created/retrieved collection: {collection_name}")
        except Exception as collection_error:
            self.logger.error(f"Collection creation failed: {collection_error}")
            raise
        
        # Track metadata for indexed documents
        self.document_metadata = {}


    def index_documents(self, document_paths: Dict[str, str] = None, document_name: Optional[str] = None) -> int:

    # Use provided directory or default
        document_paths = document_paths or self.indexed_directory
        
        # Validate directory exists
        if not os.path.isdir(document_paths):
            self.logger.warning("No documents provided for indexing")
            self.logger.error(f"Directory not found: {document_paths}")
            return 0
        
        # Clear existing documents if needed
        if self.collection.count() > 0:
            existing_ids = [str(id) for id in range(self.collection.count())]
            self.collection.delete(ids=existing_ids)
            self.logger.info("Cleared previous collection contents")

        indexed_count = 0
        
        # List all text files in the directory
        text_files = [f for f in os.listdir(document_paths) if f.endswith('.txt')]
        
        self.logger.info(f"Attempting to index from directory: {document_paths}")
        self.logger.info(f"Files found: {text_files}")
        
        for file_name in text_files:
            file_path = os.path.join(document_paths, file_name)
            
            try:
                # Read file content
                with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                    text = file.read().strip()
                    print("PRINTING THE length text", len(text))
                
                # Skip empty documents
                if not text:
                    self.logger.warning(f"Skipping empty document: {file_name}")
                    continue
                
                # Generate unique ID
                doc_id = str(uuid.uuid4())
                
                # Prepare metadata
                metadata = {
                    "document_name": file_name,
                    "file_path": file_path,
                    "timestamp": str(datetime.now())
                }
                
                # Add document to collection
                self.collection.add(
                    documents=[text],
                    ids=[doc_id],
                    metadatas=[metadata]
                )
                
                # Track metadata
                self.document_metadata[doc_id] = {
                    "text": text,
                    "metadata": metadata
                }
                # Add to collection with retry mechanism
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        self.collection.add(
                            documents=[text], ids=[doc_id], metadatas=[metadata]
                        )
                        indexed_count += 1
                        self.logger.info(f"Indexed document: {file_name}")
                        break
                    except Exception as add_error:
                        if attempt < max_retries - 1:
                            self.logger.warning(f"Retrying to add document {file_name}: {add_error}")
                            time.sleep(2 ** attempt)  # Exponential backoff
                        else:
                            self.logger.error(f"Error adding document {file_name} to collection: {add_error}")
                            # Log full details of the problematic data
                            self.logger.error(f"Document ID: {doc_id}")
                            self.logger.error(f"Document Text Length: {len(text)}")
                            self.logger.error(f"Metadata: {metadata}")
                            break
                           
            except Exception as file_error:
                self.logger.error(f"Error processing {file_name}: {file_error}")
            except PermissionError:
                self.logger.error(f"Permission denied reading {file_path}")
            except FileNotFoundError:
                self.logger.error(f"File not found: {file_path}")
            except Exception as e:
                self.logger.error(f"Error during indexing: {e}")
                import traceback
                traceback.print_exc()            
        self.logger.info(f"Indexing complete. Total documents indexed: {indexed_count}")
        return indexed_count
          
    
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
            self.logger.warning("Invalid query provided")
            return []
        
        n_results = max(1, min(n_results, 10))  # Limit results between 1 and 10
        
        # Check if collection is empty
        if self.collection.count() == 0:
            self.logger.warning("No documents have been indexed yet")
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
            self.logger.error(f"Error searching documents: {e}")
            # Log full traceback for debugging
            
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
            self.logger.error(f"Error retrieving collection info: {e}")
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
                    f"[{result['metadata'].get('page_name', 'Unknown Page')}] "
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
                #model="mxbai-embed-large",
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
                    #model="mxbai-embed-large",
                    messages=[{"role": "user", "content": query}],
                    stream=False
                )
                return response["message"]["content"]
            except Exception as fallback_error:
                return f"I'm sorry, but I'm having trouble processing your query. Error: {str(fallback_error)}"


# Global knowledge base
knowledge_base_indexer = KnowledgeBaseIndexer()

# FastAPI Application commented out

#app = FastAPI()

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Automatically index documents on application startup
    """
    try:
        document_paths = INDEXED_DIRECTORY
        indexed_count = knowledge_base_indexer.index_documents(document_paths)
        logging.info(f"Indexed {indexed_count} documents on startup")
        yield
    except Exception as e:
        logging.error(f"Failed to index documents on startup: {e}")
        yield
    finally:
        # Any cleanup code can go here
        pass

app = FastAPI(lifespan=lifespan)

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and process PDF file
    """
    # Save uploaded file
    file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(file_location, "wb+") as file_object:
        file_object.write(await file.read())
    
    # Extract text from PDF
    extracted_pages = PDFTextExtractor.extract_text_from_pdf(
        file_location, 
        INDEXED_DIRECTORY
    )
    
    # Index the extracted documents
    global knowledge_base_indexer
    knowledge_base_indexer.index_documents(
        extracted_pages, 
        document_name=file.filename
    )
    
    return {
        "status": "success", 
        "message": f"PDF processed: {file.filename}",
        "pages_extracted": len(extracted_pages)
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    conversation = []
    
    # Use global knowledge base
    global knowledge_base_indexer
    chatbot = ChatbotService(knowledge_base_indexer)
    
    while True:
        # Receive user message
        data = await websocket.receive_text()
        chat_time = str(datetime.now())

        try:
            # Generate response
            response = await chatbot.generate_response(data)
            
            # Persisting the chat
            conversation.append((chat_time, data, response))
            
            # Send response back
            await websocket.send_text(response)
        
        except Exception as e:
            await websocket.send_text(f"Error processing your request: {str(e)}")
        
        with open("history.txt", 'a') as f:
            f.write(str(conversation) + "\n")




#ADDED LATER
clients = set()

#app.mount("/qwikdoq/static", StaticFiles(directory="qwikdoq/static"), name="static") #windows: C:\Users\danie\OneDrive\Documents\Qwik\QwikDoQ\res\static\ for example
html_file_path = Path(__file__).parent / "home.html" #windows: C:\Users\danie\OneDrive\Documents\Qwik\QwikDoQ\index.html
with open(html_file_path, "r") as file:
    html= file.read()


@app.get("/")
async def get():
    return HTMLResponse(html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    conversation = []
    
    # Use global knowledge base (if available)
    global knowledge_base_indexer
    
    # Initialize chatbot with optional knowledge base
    try:
        chatbot = ChatbotService(knowledge_base_indexer)
    except Exception as init_error:
        print(f"Error initializing ChatbotService: {init_error}")
        # Fallback to ChatbotService without knowledge base
        chatbot = ChatbotService()
    
    while True:
        try:
            # Receive user message
            data = await websocket.receive_text()
            chat_time = str(datetime.now())

            # Generate response
            response = await chatbot.generate_response(data)
            
            # Persisting the chat
            conversation.append((chat_time.strftime('%B %m, %Y %I:%M %p'), data, response))
            
            # Send response back
            await websocket.send_text(response)
        
        except WebSocketDisconnect:
            print("WebSocket disconnected")
            break
        except Exception as e:
            error_message = f"Error processing your request: {str(e)}"
            print(error_message)
            await websocket.send_text(error_message)
        
        # Optionally log conversation (with error handling)
        try:
            with open("history.txt", 'a') as f:
                print(conversation, end="", file=f)
        except Exception as log_error:
            print(f"Error logging conversation: {log_error}")


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Ensure upload directory exists
        os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
        os.makedirs(INDEXED_DIRECTORY, exist_ok=True)
        
        # Save uploaded file
        file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
        with open(file_location, "wb+") as file_object:
            file_object.write(await file.read())
        
        # Extract text from PDF
        try:
            extracted_pages = PDFTextExtractor.extract_text_from_pdf(
                file_location, 
                INDEXED_DIRECTORY
            )
        except Exception as extract_error:
            return JSONResponse(
                content={
                    "status": "error", 
                    "message": f"Failed to extract text from PDF: {str(extract_error)}"
                },
                status_code=400
            )
        
        # Index the extracted documents
        global knowledge_base_indexer
        try:
            # Pass the INDEXED_DIRECTORY instead of extracted_pages
            knowledge_base_indexer.index_documents(
                document_paths=INDEXED_DIRECTORY, 
                document_name=file.filename
            )
        except Exception as index_error:
            return JSONResponse(
                content={
                    "status": "error", 
                    "message": f"Failed to index document: {str(index_error)}"
                },
                status_code=400
            )
        
        return JSONResponse({
            "status": "success", 
            "message": f"PDF processed successfully: {file.filename}",
            "pages_extracted": len(extracted_pages),
            "pages": list(extracted_pages.keys())
        })
    
    except Exception as e:
        return JSONResponse(
            content={
                "status": "error", 
                "message": f"Unexpected error processing upload: {str(e)}"
            },
            status_code=500
        )



@app.get("/pdf/{file_name}")
async def get_pdf(file_name: str):
    # Ensure file_name is sanitized to prevent directory traversal
    safe_file_name = os.path.basename(file_name)
    file_path = os.path.join(UPLOAD_DIRECTORY, safe_file_name)
    
    # More robust path checking
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail=f"File {safe_file_name} not found")
    
    return FileResponse(path=file_path, filename=safe_file_name)



@app.post("/query/")
async def query_endpoint(query: str):
    # Add input validation
    if not query or len(query.strip()) == 0:
        return JSONResponse(
            content={"error": "Query cannot be empty"},
            status_code=400
        )
    
    try:
        result = knowledge_base_indexer.search_documents(query)
        
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
        # Run indexing in background
        background_tasks.add_task(knowledge_base_indexer.index_documents)
        
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

@app.get("/collection-info/")
async def get_collection_info():
    """
    Endpoint to retrieve current collection information
    """
    try:
        collection_info = knowledge_base_indexer.get_collection_info()
        return collection_info
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error", 
                "message": str(e)
            }
        )
    
async def generate_response(self, query):
    try:
        # Verify model is available
        available_models = await self.llm_client.list()
        if "qwen2:0.5b" not in available_models:
            raise ValueError("Model qwen2:0.5b not available. Please pull the model.")

        # Rest of your existing generate_response method remains the same
    except Exception as e:
        print(f"Model availability error: {e}")
        # Fallback logic

if __name__ == "__main__":
    knowledge_base_indexer.index_documents(INDEXED_DIRECTORY)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    #chroma run --host localhost --port 8081