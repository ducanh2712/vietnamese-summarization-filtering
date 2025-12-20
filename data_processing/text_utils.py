"""
Text utilities with standardized output format
"""

import re
import logging
from typing import List
import unicodedata

logger = logging.getLogger(__name__)


import re
import unicodedata

def clean_text(text: str) -> str:
    if not text:
        return ""

    # 1. Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # 2. Normalize special spaces
    text = text.replace("\u00A0", " ").replace("\u200B", "")

    # 3. Normalize punctuation variants
    punct_map = {
        "，": ",", "．": ".", "！": "!", "？": "?",
        "：": ":", "；": ";",
        "“": '"', "”": '"', "‘": "'", "’": "'",
        "«": '"', "»": '"',
        "–": "-", "—": "-", "−": "-",
        "…": "..."
    }
    for k, v in punct_map.items():
        text = text.replace(k, v)

    # 4. Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # 5. Remove space before punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)

    # 6. Ensure space after punctuation
    text = re.sub(r'([.,!?;:])([^\s])', r'\1 \2', text)

    return text.strip()



def split_into_sentences(text: str) -> List[str]:
    """
    Split text thành sentences
    Simple implementation - có thể cải thiện với thư viện NLP
    """
    # Split by common punctuation
    sentences = re.split(r'[.!?]\s+', text)
    
    # Clean và filter empty
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def truncate_to_max_tokens(text: str, max_tokens: int = 512) -> str:
    """
    Truncate text to max tokens
    Approximate: 1 token ≈ 4 characters for Vietnamese
    """
    max_chars = max_tokens * 4
    
    if len(text) <= max_chars:
        return text
    
    # Truncate và tìm điểm cắt tốt
    truncated = text[:max_chars]
    
    # Tìm space cuối cùng để không cắt giữa từ
    last_space = truncated.rfind(' ')
    if last_space > 0:
        truncated = truncated[:last_space]
    
    return truncated


def count_words(text: str) -> int:
    """Đếm số từ trong text"""
    return len(text.split())


def calculate_compression_ratio(original: str, compressed: str) -> float:
    """Tính tỷ lệ nén (compressed length / original length)"""
    if len(original) == 0:
        return 0.0
    
    return len(compressed) / len(original)


def create_training_pair(
    document: str,
    summary: str,
    approach_name: str = "baseline",
    metadata: dict = None
) -> dict:
    """
    Tạo training pair với format đơn giản
    
    Format:
    {
        'document': str,
        'summary': str,
        'metadata': dict (optional - for tracking)
    }
    
    Args:
        document: Văn bản đã xử lý (input)
        summary: Tóm tắt (target)
        approach_name: Tên approach đã sử dụng
        metadata: Thông tin metadata (optional)
        
    Returns:
        Training pair dict
    """
    # Build training pair
    pair = {
        'document': clean_text(document),
        'summary': clean_text(summary)
    }
    
    # Thêm metadata nếu cần (để tracking)
    if metadata:
        metadata['approach'] = approach_name
        pair['metadata'] = metadata
    
    return pair


def check_duplicates(pairs: List[dict], mode: str = 'document') -> dict:
    """
    Kiểm tra duplicate trong list of training pairs
    
    Args:
        pairs: List of training pairs
        mode: 'document', 'summary', hoặc 'both'
        
    Returns:
        Dict with duplicate statistics
    """
    if not pairs:
        return {'total': 0, 'unique': 0, 'duplicates': 0, 'duplicate_rate': 0.0}
    
    seen = set()
    duplicates = []
    duplicate_indices = []
    
    for i, pair in enumerate(pairs):
        if mode == 'document':
            key = pair['document']
        elif mode == 'summary':
            key = pair['summary']
        elif mode == 'both':
            key = f"{pair['document']}|||{pair['summary']}"
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        if key in seen:
            duplicates.append(pair)
            duplicate_indices.append(i)
        else:
            seen.add(key)
    
    stats = {
        'total': len(pairs),
        'unique': len(seen),
        'duplicates': len(duplicates),
        'duplicate_rate': len(duplicates) / len(pairs) * 100 if pairs else 0.0,
        'duplicate_indices': duplicate_indices,
        'mode': mode
    }
    
    return stats


def remove_duplicates(pairs: List[dict], mode: str = 'document', keep: str = 'first') -> List[dict]:
    """
    Xóa duplicate trong list of training pairs
    
    Args:
        pairs: List of training pairs
        mode: 'document', 'summary', hoặc 'both'
        keep: 'first' hoặc 'last' - giữ bản nào khi gặp duplicate
        
    Returns:
        List of training pairs without duplicates
    """
    if not pairs:
        return []
    
    seen = set()
    unique_pairs = []
    
    # Nếu keep='last', reverse list
    if keep == 'last':
        pairs = list(reversed(pairs))
    
    for pair in pairs:
        if mode == 'document':
            key = pair['document']
        elif mode == 'summary':
            key = pair['summary']
        elif mode == 'both':
            key = f"{pair['document']}|||{pair['summary']}"
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        if key not in seen:
            seen.add(key)
            unique_pairs.append(pair)
    
    # Nếu đã reverse, reverse lại
    if keep == 'last':
        unique_pairs = list(reversed(unique_pairs))
    
    return unique_pairs


def get_duplicate_stats(pairs: List[dict]) -> dict:
    """
    Tính tổng hợp statistics về duplicate (document, summary, both)
    
    Args:
        pairs: List of training pairs
        
    Returns:
        Dict with comprehensive duplicate statistics
    """
    doc_stats = check_duplicates(pairs, mode='document')
    sum_stats = check_duplicates(pairs, mode='summary')
    both_stats = check_duplicates(pairs, mode='both')
    
    return {
        'total_pairs': len(pairs),
        'document_duplicates': {
            'count': doc_stats['duplicates'],
            'rate': doc_stats['duplicate_rate']
        },
        'summary_duplicates': {
            'count': sum_stats['duplicates'],
            'rate': sum_stats['duplicate_rate']
        },
        'exact_duplicates': {
            'count': both_stats['duplicates'],
            'rate': both_stats['duplicate_rate']
        },
        'unique_documents': doc_stats['unique'],
        'unique_summaries': sum_stats['unique'],
        'unique_pairs': both_stats['unique']
    }