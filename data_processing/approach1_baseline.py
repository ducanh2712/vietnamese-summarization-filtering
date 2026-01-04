"""
Approach 1: Baseline - Full text without filtering
Output format: {'document', 'summary'}
"""

import logging
from typing import Dict, List
from .text_utils import (
    clean_text,
    truncate_to_max_tokens,
    count_words,
    create_training_pair
)

logger = logging.getLogger(__name__)


def process_single_article(article: Dict, max_tokens: int = 512) -> Dict:
    """
    APPROACH 1: BASELINE
    
    Sử dụng toàn bộ văn bản gốc, không filtering
    Output format: {'document', 'summary'}
    
    Args:
        article: Dict with 'document', 'summary'
        max_tokens: Maximum token length
        
    Returns:
        Training pair dict
    """
    # Lấy full text
    full_text = article['document']
    summary = article['summary']
    
    # Clean
    full_text = clean_text(full_text)
    summary = clean_text(summary)
    
    # Truncate nếu cần
    input_text = truncate_to_max_tokens(full_text, max_tokens)
    
    # Check truncation
    was_truncated = len(input_text) < len(full_text)
    
    # Build metadata
    metadata = {
        'approach': 'baseline',
        'filtered': False,
        'original_length': len(full_text),
        'input_length': len(input_text),
        'target_length': len(summary),
        'original_words': count_words(full_text),
        'input_words': count_words(input_text),
        'target_words': count_words(summary),
        'truncated': was_truncated
    }
    
    # Tạo training pair
    return create_training_pair(
        document=input_text,
        summary=summary,
        approach_name='baseline',
        metadata=metadata
    )


def process_dataset(articles: List[Dict], max_tokens: int = 512) -> List[Dict]:
    """
    Process toàn bộ dataset với Approach 1
    
    Args:
        articles: List of article dicts
        max_tokens: Max token length
        
    Returns:
        List of training pairs
    """
    pairs = []
    errors = 0
    
    logger.info(f"🚀 Starting Approach 1 processing for {len(articles)} articles...")
    
    for i, article in enumerate(articles):
        try:
            pair = process_single_article(article, max_tokens)
            
            # Add article_id vào metadata
            if 'metadata' in pair:
                pair['metadata']['article_id'] = i
            
            pairs.append(pair)
            
            # Progress logging
            if (i + 1) % 100 == 0:
                logger.info(f"  Processed {i + 1}/{len(articles)} articles...")
                
        except Exception as e:
            errors += 1
            logger.error(f"⚠️ Error processing article {i}: {e}")
    
    logger.info(f"✅ Completed Approach 1: {len(pairs)}/{len(articles)} successful, {errors} errors")
    
    return pairs


def calculate_statistics(pairs: List[Dict]) -> Dict:
    """
    Tính statistics cho Approach 1
    
    Args:
        pairs: List of training pairs
        
    Returns:
        Statistics dict
    """
    if not pairs:
        return {}
    
    # Extract metadata
    metadatas = [p.get('metadata', {}) for p in pairs]
    
    stats = {
        'total_pairs': len(pairs),
        'avg_document_length': sum(len(p['document']) for p in pairs) / len(pairs),
        'avg_summary_length': sum(len(p['summary']) for p in pairs) / len(pairs),
        'avg_document_words': sum(m.get('input_words', 0) for m in metadatas) / len(pairs),
        'avg_summary_words': sum(m.get('target_words', 0) for m in metadatas) / len(pairs),
        'num_truncated': sum(1 for m in metadatas if m.get('truncated', False)),
        'truncation_rate': sum(1 for m in metadatas if m.get('truncated', False)) / len(pairs)
    }
    
    if stats['avg_summary_length'] > 0:
        stats['document_summary_ratio'] = stats['avg_document_length'] / stats['avg_summary_length']
    
    return stats