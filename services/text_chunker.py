"""
Text Chunker Service

Handles splitting text into overlapping chunks for embedding.
"""

import logging
import re
from typing import List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a text chunk."""
    
    text: str
    start_index: int
    end_index: int
    metadata: dict


class TextChunker:
    """Text chunking utility with various strategies."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize text chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
            separators: List of separators to split on (in order of preference)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if separators is None:
            self.separators = [
                "\n\n",  # Double newlines (paragraphs)
                "\n",    # Single newlines
                ". ",     # Sentences
                "! ",     # Exclamations
                "? ",     # Questions
                ", ",     # Commas
                " ",      # Spaces
                "",       # No separator (character level)
            ]
        else:
            self.separators = separators
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"Chunk overlap ({chunk_overlap}) must be smaller than "
                f"chunk size ({chunk_size})"
            )
        
        logger.info(
            f"TextChunker initialized: chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap}"
        )
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        logger.debug(f"Chunking text of length {len(text)} characters")
        
        # Clean the text first
        text = self._clean_text(text)
        
        # Use recursive splitting
        chunks = self._recursive_split(text)
        
        # Apply overlap
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._apply_overlap(chunks)
        
        logger.info(f"Created {len(chunks)} chunks from text")
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters except newline and tab
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Normalize newlines
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Trim whitespace
        text = text.strip()
        
        return text
    
    def _recursive_split(self, text: str) -> List[str]:
        """Recursively split text using separators."""
        # Base case: text is short enough
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        
        # Try each separator in order
        for separator in self.separators:
            if separator:
                # Split by separator
                parts = text.split(separator)
                
                # Check if any split would produce chunks that are too large
                valid_split = True
                current_chunk = ""
                chunks = []
                
                for i, part in enumerate(parts):
                    # Add separator back except for the last part
                    if i < len(parts) - 1:
                        part_with_sep = part + separator
                    else:
                        part_with_sep = part
                    
                    # If adding this part would exceed chunk size, split here
                    if len(current_chunk) + len(part_with_sep) > self.chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk)
                            current_chunk = part_with_sep
                        else:
                            # Single part is too large, need to split differently
                            valid_split = False
                            break
                    else:
                        current_chunk += part_with_sep
                
                # Add the last chunk
                if current_chunk:
                    chunks.append(current_chunk)
                
                if valid_split and len(chunks) > 1:
                    # Recursively split any chunks that are still too large
                    final_chunks = []
                    for chunk in chunks:
                        if len(chunk) > self.chunk_size:
                            final_chunks.extend(self._recursive_split(chunk))
                        else:
                            final_chunks.append(chunk)
                    
                    return final_chunks
            
        # If no separator works, split by character count
        return self._split_by_characters(text)
    
    def _split_by_characters(self, text: str) -> List[str]:
        """Split text by character count when no separator works."""
        chunks = []
        start = 0
        
        while start < len(text):
            # Find a good break point near chunk_size
            end = start + self.chunk_size
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Try to break at a word boundary
            break_point = text.rfind(' ', start, end)
            if break_point == -1 or break_point < start + self.chunk_size // 2:
                # No good word boundary, break at exact position
                break_point = end
            
            chunks.append(text[start:break_point].strip())
            start = break_point
        
        return chunks
    
    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """Apply overlap between consecutive chunks."""
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = []
        
        for i in range(len(chunks)):
            current_chunk = chunks[i]
            
            # Add overlap from previous chunk
            if i > 0:
                prev_chunk = chunks[i - 1]
                overlap_start = max(0, len(prev_chunk) - self.chunk_overlap)
                overlap_text = prev_chunk[overlap_start:]
                
                if overlap_text:
                    current_chunk = overlap_text + "\n" + current_chunk
            
            # Add overlap from next chunk
            if i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                overlap_end = min(self.chunk_overlap, len(next_chunk))
                overlap_text = next_chunk[:overlap_end]
                
                if overlap_text:
                    current_chunk = current_chunk + "\n" + overlap_text
            
            overlapped_chunks.append(current_chunk)
        
        return overlapped_chunks
    
    def chunk_with_metadata(self, text: str) -> List[Chunk]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of Chunk objects with metadata
        """
        chunks_text = self.chunk_text(text)
        chunks = []
        
        current_position = 0
        for i, chunk_text in enumerate(chunks_text):
            # Find the position of this chunk in the original text
            start_index = text.find(chunk_text[:50], current_position)  # Look for first 50 chars
            if start_index == -1:
                start_index = current_position
            
            end_index = start_index + len(chunk_text)
            current_position = end_index
            
            chunk = Chunk(
                text=chunk_text,
                start_index=start_index,
                end_index=end_index,
                metadata={
                    "chunk_index": i,
                    "total_chunks": len(chunks_text),
                    "length": len(chunk_text)
                }
            )
            chunks.append(chunk)
        
        return chunks


# Alternative chunking strategies
class SentenceChunker(TextChunker):
    """Chunker that prioritizes sentence boundaries."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[". ", "! ", "? ", "\n", ", ", " ", ""]
        )


class ParagraphChunker(TextChunker):
    """Chunker that prioritizes paragraph boundaries."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
        )


# Example usage
if __name__ == "__main__":
    import sys
    
    # Example text
    sample_text = """\
    This is a sample document. It contains multiple paragraphs.
    
    Each paragraph should be treated as a separate unit when possible.
    The chunker will try to keep paragraphs together.
    
    However, if a paragraph is too long, it will be split at sentence boundaries.
    And if sentences are too long, they will be split at word boundaries.
    """
    
    if len(sys.argv) > 1:
        # Read from file
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            sample_text = f.read()
    
    print(f"Original text length: {len(sample_text)} characters\n")
    
    # Test different chunkers
    chunkers = [
        ("Default", TextChunker(chunk_size=200, chunk_overlap=50)),
        ("Sentence", SentenceChunker(chunk_size=200, chunk_overlap=50)),
        ("Paragraph", ParagraphChunker(chunk_size=200, chunk_overlap=50)),
    ]
    
    for name, chunker in chunkers:
        print(f"\n=== {name} Chunker ===")
        chunks = chunker.chunk_text(sample_text)
        
        print(f"Created {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i+1}: {len(chunk)} chars")
            print(f"    Preview: {chunk[:80]}..." if len(chunk) > 80 else f"    {chunk}")
