"""
PDF Loader Service

Handles loading, parsing, and text extraction from PDF documents.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a loaded document."""
    
    content: str
    metadata: Dict[str, Any]
    pages: List[Dict[str, Any]]
    

def extract_text_with_pymupdf(
    pdf_path: str,
    extract_images: bool = False,
    extract_tables: bool = True
) -> Document:
    """
    Extract text from PDF using PyMuPDF.
    
    Args:
        pdf_path: Path to PDF file
        extract_images: Whether to extract image information
        extract_tables: Whether to attempt table extraction
        
    Returns:
        Document object with text and metadata
    """
    doc = fitz.open(pdf_path)
    
    metadata = {
        "source": pdf_path,
        "filename": Path(pdf_path).name,
        "total_pages": len(doc),
        "author": doc.metadata.get("author", ""),
        "title": doc.metadata.get("title", ""),
        "subject": doc.metadata.get("subject", ""),
        "creator": doc.metadata.get("creator", ""),
        "producer": doc.metadata.get("producer", ""),
        "creation_date": doc.metadata.get("creationDate", ""),
        "modification_date": doc.metadata.get("modDate", "")
    }
    
    full_text = ""
    pages = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # Extract text
        text = page.get_text()
        
        # Extract images if requested
        images = []
        if extract_images:
            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                images.append({
                    "index": img_index,
                    "bbox": img[1:5] if len(img) > 5 else None,
                    "size": img[2] if len(img) > 2 else None
                })
        
        # Extract tables if requested
        tables = []
        if extract_tables:
            # PyMuPDF doesn't have built-in table extraction
            # We can add table extraction logic here if needed
            pass
        
        page_info = {
            "page_number": page_num + 1,
            "text": text,
            "images": images,
            "tables": tables,
            "dimensions": page.rect
        }
        
        pages.append(page_info)
        full_text += f"\n--- Page {page_num + 1} ---\n{text}\n"
    
    doc.close()
    
    return Document(
        content=full_text,
        metadata=metadata,
        pages=pages
    )


class PDFLoader:
    """PDF document loader and processor."""
    
    def __init__(self, extract_images: bool = False, extract_tables: bool = True):
        """
        Initialize PDF loader.
        
        Args:
            extract_images: Whether to extract image information
            extract_tables: Whether to attempt table extraction
        """
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        
        logger.info(
            f"PDFLoader initialized: extract_images={extract_images}, "
            f"extract_tables={extract_tables}"
        )
    
    def load_pdf(self, file_path: str) -> Document:
        """
        Load and parse a PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Document object
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a PDF
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not file_path.lower().endswith('.pdf'):
            raise ValueError(f"File is not a PDF: {file_path}")
        
        logger.info(f"Loading PDF: {file_path}")
        
        try:
            document = extract_text_with_pymupdf(
                file_path,
                extract_images=self.extract_images,
                extract_tables=self.extract_tables
            )
            
            logger.info(
                f"Successfully loaded PDF: {file_path}, "
                f"pages: {document.metadata['total_pages']}, "
                f"text length: {len(document.content)} characters"
            )
            
            return document
            
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            raise
    
    def extract_text(self, document: Document) -> str:
        """
        Extract clean text from document.
        
        Args:
            document: Document object
            
        Returns:
            Cleaned text
        """
        # Basic text cleaning
        text = document.content
        
        # Remove excessive whitespace
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Remove page markers if they exist
                if line.startswith('--- Page') and line.endswith('---'):
                    continue
                cleaned_lines.append(line)
        
        # Join with single newlines
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Remove multiple consecutive newlines
        while '\n\n\n' in cleaned_text:
            cleaned_text = cleaned_text.replace('\n\n\n', '\n\n')
        
        logger.debug(f"Extracted text: {len(cleaned_text)} characters")
        
        return cleaned_text
    
    def batch_load_pdfs(self, directory_path: str) -> List[Document]:
        """
        Load all PDFs from a directory.
        
        Args:
            directory_path: Path to directory containing PDFs
            
        Returns:
            List of Document objects
        """
        if not os.path.isdir(directory_path):
            raise ValueError(f"Not a directory: {directory_path}")
        
        pdf_files = []
        for file in os.listdir(directory_path):
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(directory_path, file))
        
        logger.info(f"Found {len(pdf_files)} PDF files in {directory_path}")
        
        documents = []
        for pdf_file in pdf_files:
            try:
                document = self.load_pdf(pdf_file)
                documents.append(document)
            except Exception as e:
                logger.error(f"Failed to load {pdf_file}: {e}")
        
        return documents


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        
        loader = PDFLoader()
        
        try:
            doc = loader.load_pdf(pdf_path)
            print(f"Loaded PDF: {pdf_path}")
            print(f"Metadata: {doc.metadata}")
            print(f"Pages: {len(doc.pages)}")
            print(f"\nFirst 500 characters of text:")
            print(doc.content[:500] + "...")
            
            cleaned = loader.extract_text(doc)
            print(f"\nCleaned text length: {len(cleaned)} characters")
            
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python pdf_loader.py <path_to_pdf>")
