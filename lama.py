import os
import uuid
from datetime import datetime
import logging
import json
import traceback
from pathlib import Path
import time

import chromadb
from chromadb import HttpClient
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import fitz  # PyMuPDF
import pandas as pd
from ollama import AsyncClient
from fastapi import (
    FastAPI,
    File,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
    BackgroundTasks,
)
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

# Constants
CHROMA_HOST = "localhost"
CHROMA_PORT = 8081
OLLAMA_URL = "http://localhost:11434/api/embeddings"
UPLOAD_DIRECTORY = r"C:\Users\danie\OneDrive\Pictures\OSINT\TwitterAPI\Sentiment\uploads"
INDEXED_DIRECTORY = r"C:\Users\danie\OneDrive\Pictures\OSINT\TwitterAPI\Sentiment\indexed"
SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".xlsx", ".xls"}

# Ensure directories exist
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(INDEXED_DIRECTORY, exist_ok=True)

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize Ollama
llm = AsyncClient()


class FileExtractor:
    """Extract text from PDF, CSV, and Excel files and save as text files."""

    @staticmethod
    def extract_text_from_pdf(pdf_path: str, output_dir: str) -> Dict[str, str]:
        """Extract text from each page of a PDF and save as separate text files."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            pdf_document = fitz.open(pdf_path)
            extracted_pages = {}
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                text = page.get_text()
                txt_file_path = os.path.join(output_dir, f"{Path(pdf_path).stem}_page_{page_num + 1}.txt")
                with open(txt_file_path, "w", encoding="utf-8") as file:
                    file.write(text)
                extracted_pages[f"page_{page_num + 1}"] = txt_file_path
            pdf_document.close()
            logger.info(f"Extracted {len(extracted_pages)} pages from {pdf_path}")
            return extracted_pages
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            raise

    @staticmethod
    def extract_text_from_csv(csv_path: str, output_dir: str) -> Dict[str, str]:
        """Extract text from a CSV file and save as a single text file."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            df = pd.read_csv(csv_path, encoding="utf-8", errors="ignore")
            text = df.to_string(index=False)
            file_name = Path(csv_path).stem
            txt_file_path = os.path.join(output_dir, f"{file_name}.txt")
            with open(txt_file_path, "w", encoding="utf-8") as file:
                file.write(text)
            logger.info(f"Extracted text from CSV {csv_path}")
            return {file_name: txt_file_path}
        except Exception as e:
            logger.error(f"Error extracting text from CSV {csv_path}: {e}")
            raise

    @staticmethod
    def extract_text_from_excel(excel_path: str, output_dir: str) -> Dict[str, str]:
        """Extract text from an Excel file (all sheets) and save as text files."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            xl = pd.ExcelFile(excel_path)
            extracted_sheets = {}
            for sheet_name in xl.sheet_names:
                df = pd.read_excel(excel_path, sheet_name=sheet_name)
                text = df.to_string(index=False)
                safe_sheet_name = sheet_name.replace(" ", "_").replace("/", "_")
                txt_file_path = os.path.join(output_dir, f"{Path(excel_path).stem}_{safe_sheet_name}.txt")
                with open(txt_file_path, "w", encoding="utf-8") as file:
                    file.write(text)
                extracted_sheets[sheet_name] = txt_file_path
            logger.info(f"Extracted {len(extracted_sheets)} sheets from Excel {excel_path}")
            return extracted_sheets
        except Exception as e:
            logger.error(f"Error extracting text from Excel {excel_path}: {e}")
            raise

    def extract_text(self, file_path: str, output_dir: str) -> Dict[str, str]:
        """Extract text from a file based on its extension."""
        file_ext = Path(file_path).suffix.lower()
        if file_ext == ".pdf":
            return self.extract_text_from_pdf(file_path, output_dir)
        elif file_ext == ".csv":
            return self.extract_text_from_csv(file_path, output_dir)
        elif file_ext in (".xlsx", ".xls"):
            return self.extract_text_from_excel(file_path, output_dir)
        else:
            logger.error(f"Unsupported file extension: {file_ext}")
            raise ValueError(f"Unsupported file type: {file_ext}")


class KnowledgeBaseIndexer:
    def __init__(
        self,
        collection_name: str = "documents",
        ollama_url: str = OLLAMA_URL,
        chroma_host: str = CHROMA_HOST,
        chroma_port: int = CHROMA_PORT,
        indexed_directory: str = INDEXED_DIRECTORY,
    ):
        """Initialize ChromaDB collection for indexing documents."""
        self.collection_name = collection_name
        self.ollama_url = ollama_url
        self.indexed_directory = indexed_directory
        self.document_metadata = {}

        # Initialize ChromaDB client
        try:
            self.chroma_client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=Settings(chroma_server_host=chroma_host, chroma_server_http_port=chroma_port),
            )
            logger.info("Initialized ChromaDB HTTP Client")
        except Exception as e:
            logger.warning(f"HTTP Client initialization failed: {e}")
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db_new")
            logger.info("Fallback to Persistent Client")

        # Initialize Ollama embedding function
        try:
            self.embedding_function = embedding_functions.OllamaEmbeddingFunction(
                model_name="qwen2:0.5b", url=ollama_url
            )
            logger.info("Initialized Ollama embedding function")
        except Exception as e:
            logger.error(f"Failed to initialize embedding function: {e}")
            raise

        # Create or get collection
        try:
            self.collection = self.chroma_client.get_or_create_collection(
                name=collection_name, embedding_function=self.embedding_function
            )
            logger.info(f"Created/retrieved collection: {collection_name}")
        except Exception as e:
            logger.error(f"Collection creation failed: {e}")
            raise

    def index_documents(self, document_paths: str = None, document_name: Optional[str] = None) -> int:
        """Index text documents in the specified directory."""
        document_paths = document_paths or self.indexed_directory
        if not os.path.isdir(document_paths):
            logger.error(f"Directory not found: {document_paths}")
            return 0

        # Clear existing collection
        if self.collection.count() > 0:
            self.collection.delete(ids=[str(i) for i in range(self.collection.count())])
            logger.info("Cleared previous collection contents")

        indexed_count = 0
        text_files = [f for f in os.listdir(document_paths) if f.endswith(".txt")]
        logger.info(f"Found {len(text_files)} files to index in {document_paths}")

        for file_name in text_files:
            file_path = os.path.join(document_paths, file_name)
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as file:
                    text = file.read().strip()
                if not text:
                    logger.warning(f"Skipping empty document: {file_name}")
                    continue

                doc_id = str(uuid.uuid4())
                metadata = {
                    "document_name": document_name or file_name,
                    "file_path": file_path,
                    "timestamp": str(datetime.now()),
                    "source_file": document_name or Path(file_name).stem,
                }

                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        self.collection.add(documents=[text], ids=[doc_id], metadatas=[metadata])
                        self.document_metadata[doc_id] = {"text": text, "metadata": metadata}
                        indexed_count += 1
                        logger.info(f"Indexed document: {file_name}")
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            logger.warning(f"Retrying to add document {file_name}: {e}")
                            time.sleep(2**attempt)
                        else:
                            logger.error(f"Failed to index {file_name}: {e}")
                            break
            except Exception as e:
                logger.error(f"Error processing {file_name}: {e}")

        logger.info(f"Indexed {indexed_count} documents")
        return indexed_count

    def search_documents(self, query: str, n_results: int = 3) -> List[Dict]:
        """Search indexed documents for relevant context."""
        if not query or not isinstance(query, str):
            logger.warning("Invalid query provided")
            return []

        n_results = max(1, min(n_results, 10))
        if self.collection.count() == 0:
            logger.warning("No documents indexed")
            return []

        try:
            results = self.collection.query(
                query_texts=[query], n_results=n_results, include=["documents", "metadatas", "distances"]
            )
            context_results = []
            for doc, meta, dist in zip(
                results.get("documents", [[]])[0],
                results.get("metadatas", [[]])[0],
                results.get("distances", [[]])[0],
            ):
                context_results.append(
                    {"text": doc or "", "metadata": meta or {}, "relevance_score": 1 / (1 + (dist or 0))}
                )
            return context_results
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

    def get_collection_info(self) -> Dict:
        """Retrieve information about the current collection."""
        try:
            return {
                "collection_name": self.collection_name,
                "total_documents": self.collection.count(),
                "document_metadata": list(self.document_metadata.keys()),
            }
        except Exception as e:
            logger.error(f"Error retrieving collection info: {e}")
            return {}


class ChatbotService:
    def __init__(self, knowledge_base=None):
        """Initialize chatbot with an optional knowledge base."""
        self.knowledge_base = knowledge_base
        self.llm_client = AsyncClient()
        logger.info("Initialized ChatbotService")

    async def generate_response(self, query: str) -> str:
        """Generate a response using retrieved context and LLM."""
        if not query or not isinstance(query, str):
            logger.warning("Invalid query provided")
            return "Error: Query cannot be empty."

        try:
            # Verify model availability
            available_models = await self.llm_client.list()
            if "qwen2:0.5b" not in [model["name"] for model in available_models.get("models", [])]:
                raise ValueError("Model qwen2:0.5b not available. Please pull the model.")

            # Retrieve context if knowledge base is available
            context_results = []
            if self.knowledge_base:
                try:
                    context_results = self.knowledge_base.search_documents(query)
                except Exception as e:
                    logger.error(f"Context retrieval error: {e}")

            # Prepare prompt
            if context_results:
                context_str = "\n\n".join(
                    [
                        f"[{result['metadata'].get('source_file', 'Unknown Document')}] "
                        f"Relevance: {result['relevance_score']:.2f}\n{result['text']}"
                        for result in context_results
                    ]
                )
                enhanced_prompt = (
                    f"Context from indexed documents:\n{context_str}\n\n"
                    f"User Query: {query}\n\n"
                    f"Provide a comprehensive answer based on the given context. "
                    f"If the context doesn't fully answer the query, use your general knowledge to supplement the response."
                )
            else:
                enhanced_prompt = query

            # Generate response
            response = await self.llm_client.chat(
                model="qwen2:0.5b", messages=[{"role": "user", "content": enhanced_prompt}], stream=False
            )
            return response["message"]["content"]

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            try:
                # Fallback to direct query
                response = await self.llm_client.chat(
                    model="qwen2:0.5b", messages=[{"role": "user", "content": query}], stream=False
                )
                return response["message"]["content"]
            except Exception as fallback_error:
                logger.error(f"Fallback failed: {fallback_error}")
                return f"Error: Unable to process query. {str(fallback_error)}"


# Initialize FastAPI app
knowledge_base_indexer = KnowledgeBaseIndexer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    try:
        indexed_count = knowledge_base_indexer.index_documents(INDEXED_DIRECTORY)
        logger.info(f"Indexed {indexed_count} documents on startup")
        yield
    except Exception as e:
        logger.error(f"Startup indexing failed: {e}")
        yield


app = FastAPI(lifespan=lifespan)
#app.mount("/qwikdoq/static", StaticFiles(directory="qwikdoq/static"), name="static")

# Serve HTML
html_file_path = Path(__file__).parent / "index.html"
try:
    with open(html_file_path, "r") as file:
        html = file.read()
except FileNotFoundError:
    logger.error(f"HTML file not found: {html_file_path}")
    html = "<html><body><h1>Error: index.html not found</h1></body></html>"


@app.get("/")
async def get_root():
    return HTMLResponse(html)


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process PDF, CSV, or Excel file."""
    try:
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in SUPPORTED_EXTENSIONS:
            return JSONResponse(
                content={"status": "error", "message": f"Unsupported file type: {file_ext}"}, status_code=400
            )

        file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
        with open(file_location, "wb+") as file_object:
            file_object.write(await file.read())

        try:
            extracted_files = FileExtractor().extract_text(file_location, INDEXED_DIRECTORY)
        except Exception as extract_error:
            return JSONResponse(
                content={"status": "error", "message": f"Failed to extract text: {str(extract_error)}"},
                status_code=400,
            )

        try:
            knowledge_base_indexer.index_documents(INDEXED_DIRECTORY, document_name=file.filename)
        except Exception as index_error:
            return JSONResponse(
                content={"status": "error", "message": f"Failed to index document: {str(index_error)}"},
                status_code=400,
            )

        return JSONResponse(
            {
                "status": "success",
                "message": f"File processed successfully: {file.filename}",
                "files_extracted": len(extracted_files),
                "extracted": list(extracted_files.keys()),
            }
        )
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        return JSONResponse(
            content={"status": "error", "message": f"Unexpected error processing upload: {str(e)}"},
            status_code=500,
        )


@app.get("/file/{file_name}")
async def get_file(file_name: str):
    """Serve an uploaded file."""
    safe_file_name = os.path.basename(file_name)
    file_path = os.path.join(UPLOAD_DIRECTORY, safe_file_name)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail=f"File {safe_file_name} not found")
    return FileResponse(path=file_path, filename=safe_file_name)


@app.post("/query/")
async def query_endpoint(query: str):
    """Search documents based on query."""
    if not query.strip():
        return JSONResponse(content={"error": "Query cannot be empty"}, status_code=400)
    try:
        result = knowledge_base_indexer.search_documents(query)
        return JSONResponse(content={"result": result})
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        return JSONResponse(
            content={"error": "An error occurred during document search", "details": str(e)}, status_code=500
        )


@app.post("/reindex/")
async def reindex_documents(background_tasks: BackgroundTasks):
    """Trigger reindexing of documents."""
    try:
        background_tasks.add_task(knowledge_base_indexer.index_documents, INDEXED_DIRECTORY)
        return {"status": "Reindexing started", "message": "Documents will be reindexed in the background"}
    except Exception as e:
        logger.error(f"Reindexing error: {e}")
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)


@app.get("/collection-info/")
async def get_collection_info():
    """Retrieve current collection information."""
    try:
        return knowledge_base_indexer.get_collection_info()
    except Exception as e:
        logger.error(f"Collection info error: {e}")
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections for chat."""
    await websocket.accept()
    chatbot = ChatbotService(knowledge_base_indexer)
    conversation = []

    try:
        while True:
            data = await websocket.receive_text()
            chat_time = datetime.now().strftime("%B %d, %Y %I:%M %p")
            response = await chatbot.generate_response(data)
            conversation.append((chat_time, data, response))
            await websocket.send_text(response)

            try:
                with open("history.txt", "a") as f:
                    print(conversation[-1], file=f)
            except Exception as e:
                logger.error(f"Error logging conversation: {e}")
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_text(f"Error: {str(e)}")


if __name__ == "__main__":
    knowledge_base_indexer.index_documents(INDEXED_DIRECTORY)
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)