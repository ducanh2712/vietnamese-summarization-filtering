"""
Approach 3: Discourse-Aware Selection - Lead, Middle, Tail
Output format: {'document', 'summary'}
"""

import logging
from typing import Dict, List
from .text_utils import (
    clean_text,
    split_into_sentences,
    truncate_to_max_tokens,
    count_words,
    calculate_compression_ratio,
    create_training_pair
)

logger = logging.getLogger(__name__)


def process_single_article(article: Dict, max_tokens: int = 512) -> Dict:
    """
    APPROACH 3: DISCOURSE-AWARE SELECTION
    
    Chọn 3 câu: Lead (đầu), Middle (giữa), Tail (cuối)
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
    
    # Split thành sentences
    sentences = split_into_sentences(full_text)
    n = len(sentences)
    
    # Selection logic
    if n >= 3:
        # Chọn: first, middle, last
        lead_idx = 0
        middle_idx = n // 2
        tail_idx = n - 1
        
        selected_sentences = [
            sentences[lead_idx],
            sentences[middle_idx],
            sentences[tail_idx]
        ]
        
        selected_indices = [lead_idx, middle_idx, tail_idx]
        
    elif n == 2:
        # Nếu chỉ có 2 câu, lấy cả 2
        selected_sentences = sentences
        selected_indices = [0, 1]
        
    elif n == 1:
        # Nếu chỉ có 1 câu, lấy câu đó
        selected_sentences = sentences
        selected_indices = [0]
        
    else:
        # Edge case: không có câu nào
        logger.warning("Article has no sentences after splitting")
        selected_sentences = [full_text]  # Fallback
        selected_indices = [0]
    
    # Ghép lại
    input_text = ' '.join(selected_sentences)
    
    # Clean
    input_text = clean_text(input_text)
    
    # Truncate nếu cần (hiếm khi với 3 câu)
    input_text = truncate_to_max_tokens(input_text, max_tokens)
    
    # Calculate compression
    compression = calculate_compression_ratio(full_text, input_text)
    
    # Build metadata
    metadata = {
        'approach': 'discourse_aware',
        'filtered': True,
        'selection_strategy': 'lead_middle_tail',
        'num_sentences_original': n,
        'num_sentences_selected': len(selected_sentences),
        'selected_indices': selected_indices,
        'original_length': len(full_text),
        'input_length': len(input_text),
        'target_length': len(summary),
        'original_words': count_words(full_text),
        'input_words': count_words(input_text),
        'target_words': count_words(summary),
        'compression_ratio': compression,
        'reduction_percentage': (1 - compression) * 100
    }
    
    # Tạo training pair
    return create_training_pair(
        document=input_text,
        summary=summary,
        approach_name='discourse_aware',
        metadata=metadata
    )


def process_dataset(articles: List[Dict], max_tokens: int = 512) -> List[Dict]:
    """
    Process toàn bộ dataset với Approach 3
    
    Args:
        articles: List of article dicts
        max_tokens: Max token length
        
    Returns:
        List of training pairs
    """
    pairs = []
    errors = 0
    
    logger.info(f"🚀 Starting Approach 3 processing for {len(articles)} articles...")
    
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
            logger.error(f"âš ï¸  Error processing article {i}: {e}")
    
    logger.info(f"âœ… Completed Approach 3: {len(pairs)}/{len(articles)} successful, {errors} errors")
    
    return pairs


def calculate_statistics(pairs: List[Dict]) -> Dict:
    """
    Tính statistics cho Approach 3
    
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
        'avg_num_sentences_original': sum(m.get('num_sentences_original', 0) for m in metadatas) / len(pairs),
        'avg_num_sentences_selected': sum(m.get('num_sentences_selected', 0) for m in metadatas) / len(pairs),
        'avg_document_length': sum(len(p['document']) for p in pairs) / len(pairs),
        'avg_summary_length': sum(len(p['summary']) for p in pairs) / len(pairs),
        'avg_document_words': sum(m.get('input_words', 0) for m in metadatas) / len(pairs),
        'avg_summary_words': sum(m.get('target_words', 0) for m in metadatas) / len(pairs),
        'avg_compression_ratio': sum(m.get('compression_ratio', 0) for m in metadatas) / len(pairs),
        'avg_reduction_percentage': sum(m.get('reduction_percentage', 0) for m in metadatas) / len(pairs)
    }
    
    if stats['avg_summary_length'] > 0:
        stats['document_summary_ratio'] = stats['avg_document_length'] / stats['avg_summary_length']
    
    return stats