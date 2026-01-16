# tools/strategy_document_ingestion.py
# Strategy Document Ingestion Utility
# Ingest procurement strategy documents into Pinecone vector store

import os
import logging
import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ============================================================================
# DOCUMENT PROCESSORS
# ============================================================================

def extract_vendor_references(text: str) -> List[str]:
    """Extract vendor names from text."""
    # Common vendor patterns
    vendor_patterns = [
        r'\b(Honeywell|Emerson|Siemens|ABB|Yokogawa|Rosemount|Endress\+Hauser|'
        r'E\+H|Foxboro|Fisher|Masoneilan|Flowserve|Valvtechnologies|'
        r'Velan|Cameron|Metso|Samson|Schneider|Rockwell|GE|Krohne|'
        r'VEGA|Pepperl\+Fuchs|WIKA|Ashcroft|Dwyer|Fluke|Omega)\b'
    ]
    
    vendors = set()
    for pattern in vendor_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        vendors.update([m.title() for m in matches])
    
    return list(vendors)


def extract_policy_references(text: str) -> List[str]:
    """Extract policy references from text."""
    policy_patterns = [
        r'\b(single[-\s]?source|multi[-\s]?source|dual[-\s]?source)\b',
        r'\b(preferred\s+vendor|forbidden\s+vendor|approved\s+vendor)\b',
        r'\b(lifecycle\s+cost|total\s+cost\s+of\s+ownership|TCO)\b',
        r'\b(lead\s+time|delivery\s+time)\b',
        r'\b(sustainability|green\s+procurement|environmental)\b',
        r'\b(local\s+content|regional\s+preference)\b',
        r'\b(quality\s+certification|ISO\s+\d+)\b',
        r'\b(spare\s+parts|standardization)\b'
    ]
    
    policies = set()
    for pattern in policy_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        policies.update([m.lower().replace(' ', '_') for m in matches])
    
    return list(policies)


def determine_document_type(text: str, filename: str) -> str:
    """Determine the type of strategy document."""
    text_lower = text.lower()
    filename_lower = filename.lower()
    
    if any(kw in text_lower or kw in filename_lower for kw in ['vendor', 'supplier', 'manufacturer']):
        return 'vendor_strategy'
    elif any(kw in text_lower or kw in filename_lower for kw in ['policy', 'guideline', 'procedure']):
        return 'policy_document'
    elif any(kw in text_lower or kw in filename_lower for kw in ['contract', 'agreement', 'term']):
        return 'contract_terms'
    elif any(kw in text_lower or kw in filename_lower for kw in ['cost', 'price', 'budget']):
        return 'cost_strategy'
    elif any(kw in text_lower or kw in filename_lower for kw in ['quality', 'certification', 'compliance']):
        return 'quality_requirements'
    else:
        return 'general_strategy'


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence end within last 100 chars
            search_start = max(start + chunk_size - 100, start)
            last_period = text.rfind('.', search_start, end)
            if last_period > start:
                end = last_period + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
    
    return chunks


def generate_doc_id(content: str, filename: str) -> str:
    """Generate unique document ID."""
    hash_input = f"{filename}:{content[:500]}"
    return hashlib.md5(hash_input.encode()).hexdigest()


# ============================================================================
# INGESTION FUNCTIONS
# ============================================================================

def ingest_strategy_document(
    content: str,
    filename: str,
    metadata: Dict[str, Any] = None,
    chunk_size: int = 1000,
    overlap: int = 200
) -> Dict[str, Any]:
    """
    Ingest a single strategy document into the vector store.
    
    Args:
        content: Document text content
        filename: Original filename
        metadata: Additional metadata
        chunk_size: Size of text chunks
        overlap: Overlap between chunks
    
    Returns:
        Ingestion result with document IDs
    """
    try:
        from agentic.vector_store import get_vector_store
        
        vector_store = get_vector_store()
        
        # Extract metadata from content
        vendor_refs = extract_vendor_references(content)
        policy_refs = extract_policy_references(content)
        doc_type = determine_document_type(content, filename)
        
        # Chunk the document
        chunks = chunk_text(content, chunk_size, overlap)
        
        logger.info(f"Ingesting '{filename}': {len(chunks)} chunks, {len(vendor_refs)} vendors, {len(policy_refs)} policies")
        
        # Prepare base metadata
        base_metadata = {
            'filename': filename,
            'document_type': doc_type,
            'vendor_references': vendor_refs,
            'policy_references': policy_refs,
            'ingestion_timestamp': datetime.utcnow().isoformat(),
            'chunk_count': len(chunks)
        }
        
        # Merge with provided metadata
        if metadata:
            base_metadata.update(metadata)
        
        # Ingest each chunk
        doc_ids = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = base_metadata.copy()
            chunk_metadata['chunk_index'] = i
            chunk_metadata['is_first_chunk'] = (i == 0)
            chunk_metadata['is_last_chunk'] = (i == len(chunks) - 1)
            
            doc_id = generate_doc_id(chunk, f"{filename}_chunk_{i}")
            
            vector_store.add_document(
                collection_type="strategy",  # Maps to strategy_documents namespace
                content=chunk,
                metadata=chunk_metadata,
                doc_id=doc_id
            )
            
            doc_ids.append(doc_id)
        
        logger.info(f"Successfully ingested '{filename}': {len(doc_ids)} chunks added to strategy vector store")
        
        return {
            'success': True,
            'filename': filename,
            'chunks_ingested': len(doc_ids),
            'document_ids': doc_ids,
            'document_type': doc_type,
            'vendor_references': vendor_refs,
            'policy_references': policy_refs
        }
        
    except Exception as e:
        logger.error(f"Error ingesting document '{filename}': {e}", exc_info=True)
        return {
            'success': False,
            'filename': filename,
            'error': str(e)
        }


def ingest_strategy_file(
    file_path: str,
    encoding: str = 'utf-8'
) -> Dict[str, Any]:
    """
    Ingest a strategy document from file.
    
    Supports: .txt, .md, .json, .csv
    
    Args:
        file_path: Path to the file
        encoding: File encoding
    
    Returns:
        Ingestion result
    """
    try:
        filename = os.path.basename(file_path)
        ext = os.path.splitext(filename)[1].lower()
        
        if ext == '.json':
            with open(file_path, 'r', encoding=encoding) as f:
                data = json.load(f)
                if isinstance(data, dict):
                    content = json.dumps(data, indent=2)
                elif isinstance(data, list):
                    content = '\n\n'.join([json.dumps(item, indent=2) for item in data])
                else:
                    content = str(data)
        elif ext == '.csv':
            import csv
            with open(file_path, 'r', encoding=encoding) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                content = '\n\n'.join([
                    ', '.join([f"{k}: {v}" for k, v in row.items()]) 
                    for row in rows
                ])
        else:  # .txt, .md, or other text files
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
        
        return ingest_strategy_document(
            content=content,
            filename=filename,
            metadata={'source_path': file_path}
        )
        
    except Exception as e:
        logger.error(f"Error reading file '{file_path}': {e}")
        return {
            'success': False,
            'filename': file_path,
            'error': str(e)
        }


def ingest_strategy_directory(
    directory_path: str,
    extensions: List[str] = None,
    recursive: bool = True
) -> Dict[str, Any]:
    """
    Ingest all strategy documents from a directory.
    
    Args:
        directory_path: Path to directory
        extensions: File extensions to include (default: ['.txt', '.md', '.json', '.csv'])
        recursive: Whether to process subdirectories
    
    Returns:
        Summary of ingestion results
    """
    if extensions is None:
        extensions = ['.txt', '.md', '.json', '.csv', '.pdf', '.docx']
    
    results = {
        'success_count': 0,
        'failure_count': 0,
        'files_processed': [],
        'errors': []
    }
    
    # Walk directory
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext in extensions:
                file_path = os.path.join(root, filename)
                
                logger.info(f"Processing: {file_path}")
                result = ingest_strategy_file(file_path)
                
                if result['success']:
                    results['success_count'] += 1
                    results['files_processed'].append({
                        'filename': filename,
                        'chunks': result.get('chunks_ingested', 0),
                        'vendors': result.get('vendor_references', [])
                    })
                else:
                    results['failure_count'] += 1
                    results['errors'].append({
                        'filename': filename,
                        'error': result.get('error', 'Unknown error')
                    })
        
        if not recursive:
            break
    
    logger.info(f"Directory ingestion complete: {results['success_count']} success, {results['failure_count']} failures")
    
    return results


def ingest_csv_strategy(
    csv_path: str,
    content_column: str = 'strategy',
    metadata_columns: List[str] = None
) -> Dict[str, Any]:
    """
    Ingest strategy data from CSV file.
    
    Useful for structured strategy data like the instrumentation_procurement_strategy.csv
    
    Args:
        csv_path: Path to CSV file
        content_column: Column containing strategy text
        metadata_columns: Columns to include as metadata
    
    Returns:
        Ingestion results
    """
    import csv
    
    try:
        from agentic.vector_store import get_vector_store
        
        vector_store = get_vector_store()
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        if metadata_columns is None:
            # Use all columns except content column as metadata
            metadata_columns = [c for c in reader.fieldnames if c != content_column]
        
        ingested = 0
        errors = []
        
        for row in rows:
            # Build content from all relevant fields
            content_parts = []
            for key, value in row.items():
                if value and value.strip():
                    content_parts.append(f"{key}: {value}")
            
            content = '\n'.join(content_parts)
            
            # Extract metadata
            metadata = {col: row.get(col, '') for col in metadata_columns if col in row}
            metadata['source_file'] = os.path.basename(csv_path)
            
            # Add vendor and policy references
            metadata['vendor_references'] = extract_vendor_references(content)
            metadata['policy_references'] = extract_policy_references(content)
            
            # Generate doc ID
            doc_id = generate_doc_id(content, f"{csv_path}_{ingested}")
            
            try:
                vector_store.add_document(
                    collection_type="strategy",
                    content=content,
                    metadata=metadata,
                    doc_id=doc_id
                )
                ingested += 1
            except Exception as e:
                errors.append({'row': ingested, 'error': str(e)})
        
        logger.info(f"CSV ingestion complete: {ingested} rows ingested from {csv_path}")
        
        return {
            'success': True,
            'rows_ingested': ingested,
            'errors': errors,
            'source_file': csv_path
        }
        
    except Exception as e:
        logger.error(f"Error ingesting CSV '{csv_path}': {e}")
        return {
            'success': False,
            'error': str(e),
            'source_file': csv_path
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def ingest_default_strategy():
    """
    Ingest the default instrumentation_procurement_strategy.csv into vector store.
    
    This populates the strategy RAG with the existing CSV data.
    """
    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        'instrumentation_procurement_strategy.csv'
    )
    
    if os.path.exists(csv_path):
        logger.info(f"Ingesting default strategy CSV: {csv_path}")
        return ingest_csv_strategy(
            csv_path=csv_path,
            metadata_columns=['category', 'subcategory', 'vendor', 'strategy']
        )
    else:
        logger.warning(f"Default strategy CSV not found: {csv_path}")
        return {
            'success': False,
            'error': f'File not found: {csv_path}'
        }


def get_strategy_store_stats() -> Dict[str, Any]:
    """Get statistics about the strategy vector store."""
    try:
        from agentic.vector_store import get_vector_store
        
        vector_store = get_vector_store()
        stats = vector_store.get_collection_stats()
        
        # Get strategy-specific stats if available
        return {
            'success': True,
            'stats': stats,
            'strategy_namespace': 'strategy_documents'
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Strategy Document Ingestion')
    parser.add_argument('--file', help='Single file to ingest')
    parser.add_argument('--directory', help='Directory to ingest')
    parser.add_argument('--csv', help='CSV file to ingest')
    parser.add_argument('--default', action='store_true', help='Ingest default CSV')
    parser.add_argument('--stats', action='store_true', help='Show store statistics')
    
    args = parser.parse_args()
    
    if args.stats:
        stats = get_strategy_store_stats()
        print(json.dumps(stats, indent=2))
    elif args.default:
        result = ingest_default_strategy()
        print(json.dumps(result, indent=2))
    elif args.file:
        result = ingest_strategy_file(args.file)
        print(json.dumps(result, indent=2))
    elif args.directory:
        result = ingest_strategy_directory(args.directory)
        print(json.dumps(result, indent=2))
    elif args.csv:
        result = ingest_csv_strategy(args.csv)
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()
