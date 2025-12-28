"""
PDF Ingest Script

Script for batch ingestion of PDF documents into the RAG system.
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_core import get_rag_system, RAGConfig
from services.pdf_loader import PDFLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_pdf_files(directory: str) -> List[str]:
    """
    Find all PDF files in a directory recursively.
    
    Args:
        directory: Directory to search
        
    Returns:
        List of PDF file paths
    """
    pdf_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    return pdf_files


def ingest_single_pdf(rag_system, file_path: str) -> Dict[str, Any]:
    """
    Ingest a single PDF file.
    
    Args:
        rag_system: RAGSystem instance
        file_path: Path to PDF file
        
    Returns:
        Dictionary with ingestion results
    """
    try:
        logger.info(f"Ingesting: {file_path}")
        
        # Use the async method via asyncio
        import asyncio
        result = asyncio.run(rag_system.ingest_document(file_path))
        
        if result["success"]:
            logger.info(f"  ✓ Success: {result['chunks_count']} chunks")
        else:
            logger.error(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"  ✗ Error ingesting {file_path}: {e}")
        return {
            "success": False,
            "file_path": file_path,
            "error": str(e)
        }


def batch_ingest_pdfs(directory: str, rag_system = None) -> Dict[str, Any]:
    """
    Batch ingest all PDFs in a directory.
    
    Args:
        directory: Directory containing PDFs
        rag_system: Optional RAGSystem instance
        
    Returns:
        Dictionary with batch ingestion results
    """
    start_time = time.time()
    
    # Initialize RAG system if not provided
    if rag_system is None:
        rag_system = get_rag_system()
    
    # Find PDF files
    pdf_files = find_pdf_files(directory)
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {directory}")
        return {
            "success": False,
            "error": "No PDF files found",
            "directory": directory
        }
    
    logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
    
    # Ingest each PDF
    results = []
    successful = 0
    failed = 0
    total_chunks = 0
    
    for i, pdf_file in enumerate(pdf_files, 1):
        logger.info(f"Processing file {i}/{len(pdf_files)}: {os.path.basename(pdf_file)}")
        
        result = ingest_single_pdf(rag_system, pdf_file)
        results.append(result)
        
        if result["success"]:
            successful += 1
            total_chunks += result.get("chunks_count", 0)
        else:
            failed += 1
    
    processing_time = time.time() - start_time
    
    # Get collection info
    import asyncio
    collection_info = asyncio.run(rag_system.get_collection_info())
    
    return {
        "success": True,
        "directory": directory,
        "total_files": len(pdf_files),
        "successful": successful,
        "failed": failed,
        "total_chunks": total_chunks,
        "processing_time": processing_time,
        "collection_info": collection_info,
        "results": results
    }


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Batch ingest PDF documents into RAG system"
    )
    
    parser.add_argument(
        "directory",
        help="Directory containing PDF files to ingest"
    )
    
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear collection before ingestion"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be ingested without actually ingesting"
    )
    
    parser.add_argument(
        "--output",
        choices=["json", "text", "summary"],
        default="summary",
        help="Output format"
    )
    
    args = parser.parse_args()
    
    # Check if directory exists
    if not os.path.isdir(args.directory):
        logger.error(f"Directory not found: {args.directory}")
        sys.exit(1)
    
    # Initialize RAG system
    try:
        rag_system = get_rag_system()
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        sys.exit(1)
    
    # Clear collection if requested
    if args.clear:
        logger.info("Clearing collection...")
        import asyncio
        success = asyncio.run(rag_system.clear_collection())
        if success:
            logger.info("Collection cleared")
        else:
            logger.warning("Failed to clear collection")
    
    # Dry run mode
    if args.dry_run:
        pdf_files = find_pdf_files(args.directory)
        logger.info(f"Dry run: Found {len(pdf_files)} PDF files in {args.directory}")
        
        for pdf_file in pdf_files:
            file_size = os.path.getsize(pdf_file) / 1024 / 1024  # MB
            logger.info(f"  {os.path.basename(pdf_file)} ({file_size:.2f} MB)")
        
        sys.exit(0)
    
    # Perform batch ingestion
    logger.info(f"Starting batch ingestion from {args.directory}")
    
    try:
        results = batch_ingest_pdfs(args.directory, rag_system)
        
        # Output results
        if args.output == "json":
            import json
            print(json.dumps(results, indent=2, default=str))
        
        elif args.output == "text":
            print(f"\n{'='*60}")
            print(f"Batch Ingestion Results")
            print(f"{'='*60}")
            print(f"Directory: {results['directory']}")
            print(f"Total files: {results['total_files']}")
            print(f"Successful: {results['successful']}")
            print(f"Failed: {results['failed']}")
            print(f"Total chunks: {results['total_chunks']}")
            print(f"Processing time: {results['processing_time']:.2f} seconds")
            print(f"\nCollection info:")
            print(f"  Name: {results['collection_info'].get('name')}")
            print(f"  Count: {results['collection_info'].get('count')}")
            
            if results['failed'] > 0:
                print(f"\nFailed files:")
                for result in results['results']:
                    if not result['success']:
                        print(f"  {result['file_path']}: {result.get('error')}")
        
        else:  # summary
            print(f"\nSummary:")
            print(f"  ✓ {results['successful']} files ingested successfully")
            print(f"  ✗ {results['failed']} files failed")
            print(f"  📊 {results['total_chunks']} total chunks created")
            print(f"  ⏱️  {results['processing_time']:.2f} seconds")
            print(f"  🗄️  Collection now has {results['collection_info'].get('count', 0)} documents")
        
        # Cleanup
        import asyncio
        asyncio.run(rag_system.close())
        
    except Exception as e:
        logger.error(f"Batch ingestion failed: {e}")
        sys.exit(1)


# Interactive mode
if __name__ == "__main__":
    # If no arguments provided, run in interactive mode
    if len(sys.argv) == 1:
        print("PDF Ingest Script")
        print("=" * 60)
        
        directory = input("Enter directory path containing PDFs: ").strip()
        
        if not directory or not os.path.isdir(directory):
            print(f"Error: Directory not found: {directory}")
            sys.exit(1)
        
        clear_collection = input("Clear collection before ingestion? (y/N): ").strip().lower() == 'y'
        
        # Set arguments and call main
        sys.argv = [sys.argv[0], directory]
        if clear_collection:
            sys.argv.append("--clear")
        
        main()
    else:
        main()
