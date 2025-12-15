"""
Text processing utilities cho tiếng Việt
"""

import re
from typing import List
import logging

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean Vietnamese text
    
    Args:
        text: Raw text
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing spaces
    text = text.strip()
    
    # Remove special characters (nếu cần)
    # text = re.sub(r'[^\w\s.,!?;:()-]', '', text)
    
    return text


def split_into_sentences(text: str) -> List[str]:
    """
    Chia text tiếng Việt thành câu
    
    Vietnamese sentence enders: . ! ? ; 
    
    Args:
        text: Full text
        
    Returns:
        List of sentences
    """
    # Clean trước
    text = clean_text(text)
    
    # Pattern cho Vietnamese sentence boundaries
    # Chia theo . ! ? nhưng không chia số thập phân (3.14)
    pattern = r'(?<!\d)[.!?;]+(?=\s+[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]|\s*$)'
    
    # Split và giữ delimiter
    sentences = re.split(pattern, text)
    
    # Clean và filter empty
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Nếu không split được gì (text không có dấu câu), return as single sentence
    if not sentences and text:
        sentences = [text]
    
    return sentences


def truncate_to_max_tokens(text: str, max_tokens: int = 512) -> str:
    """
    Truncate text về max tokens (approximation)
    
    Vietnamese: ~1 token = 1-2 words
    Conservative estimate: 1 token = 5 characters
    
    Args:
        text: Input text
        max_tokens: Maximum number of tokens
        
    Returns:
        Truncated text
    """
    # Rough approximation
    max_chars = max_tokens * 5
    
    if len(text) <= max_chars:
        return text
    
    # Truncate
    truncated = text[:max_chars]
    
    # Cố gắng cut ở sentence boundary
    last_period = max(
        truncated.rfind('.'),
        truncated.rfind('!'),
        truncated.rfind('?'),
        truncated.rfind(';')
    )
    
    if last_period > max_chars * 0.8:  # Nếu gần max
        truncated = truncated[:last_period + 1]
    
    return truncated.strip()


def count_words(text: str) -> int:
    """
    Đếm số từ trong text tiếng Việt
    
    Args:
        text: Input text
        
    Returns:
        Number of words
    """
    # Split by whitespace
    words = text.split()
    return len(words)


def calculate_compression_ratio(original: str, compressed: str) -> float:
    """
    Tính compression ratio
    
    Args:
        original: Original text
        compressed: Compressed text
        
    Returns:
        Compression ratio (0-1, 0 = fully compressed, 1 = no compression)
    """
    if not original:
        return 0.0
    
    return len(compressed) / len(original)