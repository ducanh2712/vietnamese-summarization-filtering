"""
Approach 3: Discourse-Aware Selection - Lead, Middle, Tail
"""

import logging
from typing import Dict, List
from .text_utils import (
    clean_text,
    split_into_sentences,
    truncate_to_max_tokens,
    count_words,
    calculate_compression_ratio
)

logger = logging.getLogger(__name__)


def process_single_article(article: Dict, max_tokens: int = 512) -> Dict:
    """
    APPROACH 3: DISCOURSE-AWARE SELECTION
    
    Chọn 3 câu: Lead (đầu), Middle (giữa), Tail (cuối)
    
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
    
    return {
        'input': input_text,
        'target': summary,
        'metadata': {
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
    }


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
            pair['metadata']['article_id'] = i
            pairs.append(pair)
            
            # Progress logging
            if (i + 1) % 100 == 0:
                logger.info(f"  Processed {i + 1}/{len(articles)} articles...")
                
        except Exception as e:
            errors += 1
            logger.error(f"⚠️  Error processing article {i}: {e}")
    
    logger.info(f"✅ Completed Approach 3: {len(pairs)}/{len(articles)} successful, {errors} errors")
    
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
    
    stats = {
        'total_pairs': len(pairs),
        'avg_num_sentences_original': sum(p['metadata']['num_sentences_original'] for p in pairs) / len(pairs),
        'avg_num_sentences_selected': sum(p['metadata']['num_sentences_selected'] for p in pairs) / len(pairs),
        'avg_input_length': sum(p['metadata']['input_length'] for p in pairs) / len(pairs),
        'avg_target_length': sum(p['metadata']['target_length'] for p in pairs) / len(pairs),
        'avg_input_words': sum(p['metadata']['input_words'] for p in pairs) / len(pairs),
        'avg_target_words': sum(p['metadata']['target_words'] for p in pairs) / len(pairs),
        'avg_compression_ratio': sum(p['metadata']['compression_ratio'] for p in pairs) / len(pairs),
        'avg_reduction_percentage': sum(p['metadata']['reduction_percentage'] for p in pairs) / len(pairs)
    }
    
    stats['input_target_ratio'] = stats['avg_input_length'] / stats['avg_target_length']
    
    return stats