"""
FastAPI Backend Endpoints for GovKenyaGazette AI Document Analysis System

This module identifies all required endpoints based on frontend calls.
Focus: Perfect extraction using Docling and LlamaIndex with proper structural integrity.
"""

from fastapi import FastAPI, File, UploadFile, WebSocket, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json

app = FastAPI(title="GovKenyaGazette AI Backend")

# ============================================================================
# ENDPOINT 1: WebSocket Connection (/ws)
# ============================================================================
# Purpose: Real-time communication for queries, status updates, and streaming responses
# Frontend calls: ws = new WebSocket(`${protocol}//${window.location.host}/ws`)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time communication.
    
    Message Types (Received from Frontend):
    - type: 'query' - User query for document analysis
    - type: 'status_request' - Request for indexing status
    
    Message Types (Sent to Frontend):
    - type: 'response' - AI response to query
    - type: 'status' - Current indexing status
    - type: 'indexing_complete' - Indexing finished notification
    - type: 'error' - Error message
    """
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get('type')
            
            if message_type == 'query':
                content = data.get('content')
                # Process query with LlamaIndex
                response = await process_query(content)
                await websocket.send_json({
                    'type': 'response',
                    'content': response
                })
            
            elif message_type == 'status_request':
                status = await get_indexing_status()
                await websocket.send_json({
                    'type': 'status',
                    'data': status
                })
    
    except Exception as e:
        await websocket.send_json({
            'type': 'error',
            'message': str(e)
        })


# ============================================================================
# ENDPOINT 2: File Upload (/upload/)
# ============================================================================
# Purpose: Upload and process documents with Docling
# Frontend calls: fetch('/upload/', {method: 'POST', body: formData})

@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process document using Docling for structure extraction.
    
    Returns:
    {
        "status": "success" | "processing" | "error",
        "message": str,
        "pages_extracted": int,
        "filename": str,
        "has_docling_structure": bool
    }
    
    Key Requirements:
    - Use Docling to extract complete document structure
    - Preserve heading hierarchy and contextual relationships
    - Extract tables with proper labeling based on nearest headers
    - Validate structural parsing of all text elements
    - Ensure correct paragraph association with headings
    """
    # Docling processing logic here
    result = await process_with_docling(file)
    
    return JSONResponse({
        "status": "success",
        "message": "Document uploaded successfully",
        "pages_extracted": result.get('pages', 1),
        "filename": file.filename,
        "has_docling_structure": True
    })


# ============================================================================
# ENDPOINT 3: Document Markdown (/document-markdown/{filename})
# ============================================================================
# Purpose: Retrieve document content in markdown format
# Frontend calls: fetch(`/document-markdown/${encodeURIComponent(filename)}`)

@app.get("/document-markdown/{filename}")
async def get_document_markdown(filename: str):
    """
    Retrieve document content in markdown format.
    
    Returns:
    {
        "content": str,  # Markdown formatted document content
        "error": Optional[str]
    }
    
    Critical Requirements:
    - Maintain heading hierarchy (# ## ###)
    - Preserve table structure in markdown format
    - Include all text with proper formatting
    - Ensure contextual labeling is intact
    """
    try:
        content = await load_document_markdown(filename)
        return JSONResponse({
            "content": content
        })
    except Exception as e:
        return JSONResponse({
            "error": str(e)
        }, status_code=404)


# ============================================================================
# ENDPOINT 4: Document Structure (/document-structure/{filename})
# ============================================================================
# Purpose: Get comprehensive document structure analysis
# Frontend calls: fetch(`/document-structure/${encodeURIComponent(filename)}`)

@app.get("/document-structure/{filename}")
async def get_document_structure(filename: str):
    """
    Get complete document structure analysis using Docling.
    
    Returns:
    {
        "summary": {
            "total_pages": int,
            "total_words": int,
            "total_lines": int,
            "total_paragraphs": int,
            "total_sentences": int,
            "headings_count": int,
            "tables_count": int,
            "document_type": str,
            "estimated_reading_time": str,
            "avg_sentence_length": float,
            "avg_word_length": float,
            "complexity_score": float
        },
        "hierarchy": List[{
            "level": int,
            "type": str,
            "text": str,
            "page": int
        }],
        "tables": List[{
            "format": "markdown" | "docling",
            "source": str,
            "content": List[List[str]],  # For structured tables
            "markdown": str,  # For markdown tables
            "rows": int,
            "columns": int,
            "page": int,
            "start_line": int,
            "end_line": int,
            "row_count": int,
            "column_count": int,
            "has_merged_cells": bool
        }],
        "entities": {
            "organizations": List[str],
            "people": List[str],
            "dates": List[str],
            "emails": List[str],
            "phones": List[str],
            "references": List[str]
        },
        "sections": List[{
            "title": str,
            "level": int,
            "content_preview": str,
            "word_count": int,
            "start_line": int,
            "type": str
        }]
    }
    
    Critical Requirements:
    - Use Docling to extract all structural elements
    - Correctly identify heading levels
    - Extract tables with contextual labeling from nearest headers
    - Validate paragraph-heading associations
    - Preserve document hierarchy integrity
    """
    try:
        structure = await analyze_document_structure(filename)
        return JSONResponse(structure)
    except Exception as e:
        return JSONResponse({
            "error": str(e)
        }, status_code=404)


# ============================================================================
# ENDPOINT 5: Document Outline (/document-outline/{filename})
# ============================================================================
# Purpose: Get document outline/table of contents
# Frontend calls: fetch(`/document-outline/${encodeURIComponent(filename)}`)

@app.get("/document-outline/{filename}")
async def get_document_outline(filename: str):
    """
    Get document outline based on heading structure.
    
    Returns:
    {
        "outline": List[{
            "level": int,
            "text": str,
            "page": int
        }]
    }
    
    Requirements:
    - Extract all headings with correct level hierarchy
    - Associate each heading with correct page number
    - Maintain document structure order
    """
    try:
        outline = await extract_document_outline(filename)
        return JSONResponse({
            "outline": outline
        })
    except Exception as e:
        return JSONResponse({
            "error": str(e)
        }, status_code=404)


# ============================================================================
# ENDPOINT 6: List Documents (/list-documents/)
# ============================================================================
# Purpose: List all uploaded documents
# Frontend calls: fetch('/list-documents/')

@app.get("/list-documents/")
async def list_documents():
    """
    List all uploaded and processed documents.
    
    Returns: List[{
        "name": str,
        "pages": int,
        "has_docling_structure": bool,
        "upload_date": str
    }]
    """
    documents = await get_all_documents()
    return JSONResponse(documents)


# ============================================================================
# ENDPOINT 7: Collection Info (/collection-info/)
# ============================================================================
# Purpose: Get information about the document collection
# Frontend calls: fetch('/collection-info/')

@app.get("/collection-info/")
async def get_collection_info():
    """
    Get collection statistics.
    
    Returns:
    {
        "total_documents": int,
        "total_chunks": int,
        "error": Optional[str]
    }
    """
    try:
        info = await get_collection_statistics()
        return JSONResponse(info)
    except Exception as e:
        return JSONResponse({
            "error": str(e)
        })


# ============================================================================
# ENDPOINT 8: Conversation History (/conversation-history/)
# ============================================================================
# Purpose: Retrieve conversation history
# Frontend calls: fetch('/conversation-history/')

@app.get("/conversation-history/")
async def get_conversation_history():
    """
    Get conversation history.
    
    Returns: List[{
        "title": str,
        "preview": str,
        "timestamp": str
    }] | {"error": str}
    """
    try:
        history = await load_conversation_history()
        return JSONResponse(history)
    except Exception as e:
        return JSONResponse({
            "error": str(e)
        })


# ============================================================================
# ENDPOINT 9: Entity Search (/entity-search/)
# ============================================================================
# Purpose: Search for specific entities in documents
# Frontend calls: fetch('/entity-search/', {method: 'POST', body: {entity_name, n_results}})

class EntitySearchRequest(BaseModel):
    entity_name: str
    n_results: int = 10

@app.post("/entity-search/")
async def search_entities(request: EntitySearchRequest):
    """
    Search for specific entities across documents using LlamaIndex.
    
    Returns:
    {
        "results": List[{
            "text": str,
            "relevance_score": float,
            "metadata": {
                "document_name": str,
                "id": str
            }
        }],
        "error": Optional[str]
    }
    
    Requirements:
    - Use LlamaIndex for semantic search
    - Return results with relevance scores
    - Include document context
    """
    try:
        results = await search_for_entity(request.entity_name, request.n_results)
        return JSONResponse({
            "results": results
        })
    except Exception as e:
        return JSONResponse({
            "error": str(e)
        }, status_code=500)


# ============================================================================
# ENDPOINT 10: Structured Query (/structured-query/)
# ============================================================================
# Purpose: Execute structured queries on documents
# Frontend calls: fetch('/structured-query/', {method: 'POST', body: {query}})

class StructuredQueryRequest(BaseModel):
    query: str

@app.post("/structured-query/")
async def structured_query(request: StructuredQueryRequest):
    """
    Execute structured queries to extract specific information.
    
    Returns: Structured data matching query pattern
    Example for Gazette Notice:
    {
        "Gazette_Notice_ID": str,
        "Summary": str,
        "Exact_Location_Page": int,
        "Appointed_Entities": List[{
            "Name": str,
            "Role": str
        }],
        "Key_Details": List[{
            "Category": str,
            "Value": str
        }],
        "error": Optional[str]
    }
    
    Requirements:
    - Parse query intent
    - Extract structured data from documents
    - Return formatted response matching expected structure
    """
    try:
        result = await execute_structured_query(request.query)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({
            "error": str(e)
        }, status_code=500)


# ============================================================================
# ENDPOINT 11: Table Search (/table-search/)
# ============================================================================
# Purpose: Search for tables in documents
# Frontend calls: fetch('/table-search/', {method: 'POST', body: {query, n_results}})

class TableSearchRequest(BaseModel):
    query: str
    n_results: int = 10

@app.post("/table-search/")
async def search_tables(request: TableSearchRequest):
    """
    Search for tables matching query criteria.
    
    Returns:
    {
        "query": str,
        "table_count": int,
        "tables": List[{
            "table_id": str,
            "file_name": str,
            "page": int,
            "relevance_score": float,
            "format": "markdown" | "docling",
            "markdown": Optional[str],
            "content": Optional[List[List[str]]],
            "cells": Optional[List[{
                "row": int,
                "col": int,
                "text": str
            }]],
            "rows": int,
            "columns": int,
            "row_count": int,
            "column_count": int,
            "has_merged_cells": bool
        }]
    }
    
    Critical Requirements:
    - Search tables extracted by Docling
    - Tables must be labeled based on nearest preceding header (contextual association)
    - Return table structure preserving formatting
    - Include rich cell data from Docling extraction
    - Handle merged cells properly
    - Ensure correct paragraph-table associations
    """
    try:
        results = await search_tables_in_documents(request.query, request.n_results)
        return JSONResponse(results)
    except Exception as e:
        return JSONResponse({
            "error": str(e)
        }, status_code=500)


# ============================================================================
# HELPER FUNCTIONS (To be implemented with Docling + LlamaIndex)
# ============================================================================

async def process_with_docling(file: UploadFile) -> Dict[str, Any]:
    """
    Process document with Docling for complete structure extraction.
    
    Key Implementation Requirements:
    1. Use Docling DocumentConverter to parse PDF/DOCX
    2. Extract heading hierarchy correctly (recognize all levels)
    3. Extract tables with contextual labeling from nearest headers
    4. Associate paragraphs correctly with their headings
    5. Preserve document structure integrity
    6. Handle merged cells in tables
    7. Extract rich metadata
    
    Classes to Use:
    - docling.document_converter import DocumentConverter
    - docling.datamodel.base_models import InputFormat
    - docling.datamodel.pipeline_options import PdfPipelineOptions
    """
    pass

async def process_query(query: str) -> str:
    """
    Process user query using LlamaIndex RAG.
    
    Key Implementation Requirements:
    1. Use LlamaIndex VectorStoreIndex for semantic search
    2. Query both text and table indices
    3. Use appropriate retrieval strategy
    4. Format response appropriately
    
    Classes to Use:
    - llama_index.core import VectorStoreIndex, StorageContext
    - llama_index.core.query_engine import RetrieverQueryEngine
    """
    pass

async def analyze_document_structure(filename: str) -> Dict[str, Any]:
    """
    Analyze document structure using Docling parsed data.
    
    Key Implementation Requirements:
    1. Load Docling-parsed document
    2. Extract all structural elements:
       - Headings with correct hierarchy
       - Tables with contextual labels
       - Paragraphs with correct heading associations
    3. Validate structural integrity
    4. Generate comprehensive statistics
    """
    pass

async def search_tables_in_documents(query: str, n_results: int) -> Dict[str, Any]:
    """
    Search tables with contextual awareness.
    
    Key Implementation Requirements:
    1. Search table content semantically
    2. Ensure tables are labeled from nearest preceding header
    3. Return table structure preserving Docling formatting
    4. Include context about table location and purpose
    """
    pass

async def get_indexing_status() -> Dict[str, Any]:
    """Get current indexing status."""
    pass

async def load_document_markdown(filename: str) -> str:
    """Load document content in markdown format."""
    pass

async def extract_document_outline(filename: str) -> List[Dict[str, Any]]:
    """Extract document outline from headings."""
    pass

async def get_all_documents() -> List[Dict[str, Any]]:
    """Get list of all documents."""
    pass

async def get_collection_statistics() -> Dict[str, Any]:
    """Get collection statistics."""
    pass

async def load_conversation_history() -> List[Dict[str, Any]]:
    """Load conversation history."""
    pass

async def search_for_entity(entity_name: str, n_results: int) -> List[Dict[str, Any]]:
    """Search for specific entity."""
    pass

async def execute_structured_query(query: str) -> Dict[str, Any]:
    """Execute structured query."""
    pass

async def search_tables_in_documents(query: str, n_results: int) -> Dict[str, Any]:
    """Search for tables."""
    pass


# ============================================================================
# SUMMARY OF CRITICAL REQUIREMENTS
# ============================================================================
"""
Based on the frontend requirements document, these are the critical implementation points:

1. CORRECT DATA AND STRUCTURAL INTEGRITY
   - All extracted tables must be labeled contextually based on nearest preceding header
   - Use Docling to properly recognize all heading levels
   - Ensure paragraphs are associated with and placed under correct headings
   - Discard any extracted table data not under current context

2. TEXT DATA EXTRACTION (HIERARCHY/SECTIONS)
   - Recognize ALL heading levels correctly
   - Capture associated body text correctly
   - Validate structural parsing of all text elements

3. HEADING RECOGNITION
   - Correctly identify and output headings
   - Maintain hierarchy levels

4. PARAGRAPH ASSOCIATION
   - Ensure all descriptive paragraph text is correctly associated
   - Place paragraphs immediately under corresponding heading

5. TABLE EXTRACTION
   - Label tables based on nearest preceding header (NO arbitrary numerical index)
   - Preserve table structure including merged cells
   - Extract table content accurately
   - Maintain contextual relationship with document sections

6. DOCLING + LLAMAINDEX INTEGRATION
   - Use Docling for document parsing and structure extraction
   - Use LlamaIndex for indexing and querying
   - Maintain structural integrity through the pipeline
   - Enable semantic search while preserving document structure
"""