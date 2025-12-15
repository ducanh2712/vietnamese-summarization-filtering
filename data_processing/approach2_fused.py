"""
Approach 2: Fused Sentences - Remove paragraph structure
"""

import logging
from typing import Dict, List
from .text_utils import (
    clean_text, 
    split_into_sentences, 
    truncate_to_max_tokens,
    count_words
)

logger = logging.getLogger(__name__)


def process_single_article(article: Dict, max_tokens: int = 512) -> Dict:
    """
    APPROACH 2: FUSED SENTENCES
    
    Remove paragraph structure, ghép thành continuous text
    
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
    
    # Fuse: Ghép tất cả với single space
    fused_text = ' '.join(sentences)
    
    # Clean lại
    fused_text = clean_text(fused_text)
    
    # Truncate nếu cần
    input_text = truncate_to_max_tokens(fused_text, max_tokens)
    
    # Check truncation
    was_truncated = len(input_text) < len(fused_text)
    
    return {
        'input': input_text,
        'target': summary,
        'metadata': {
            'approach': 'fused_sentences',
            'filtered': False,
            'fused': True,
            'num_sentences_original': len(sentences),
            'original_length': len(full_text),
            'fused_length': len(fused_text),
            'input_length': len(input_text),
            'target_length': len(summary),
            'original_words': count_words(full_text),
            'input_words': count_words(input_text),
            'target_words': count_words(summary),
            'truncated': was_truncated
        }
    }


def process_dataset(articles: List[Dict], max_tokens: int = 512) -> List[Dict]:
    """
    Process toàn bộ dataset với Approach 2
    
    Args:
        articles: List of article dicts
        max_tokens: Max token length
        
    Returns:
        List of training pairs
    """
    pairs = []
    errors = 0
    
    logger.info(f"🚀 Starting Approach 2 processing for {len(articles)} articles...")
    
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
    
    logger.info(f"✅ Completed Approach 2: {len(pairs)}/{len(articles)} successful, {errors} errors")
    
    return pairs


def calculate_statistics(pairs: List[Dict]) -> Dict:
    """
    Tính statistics cho Approach 2
    
    Args:
        pairs: List of training pairs
        
    Returns:
        Statistics dict
    """
    if not pairs:
        return {}
    
    stats = {
        'total_pairs': len(pairs),
        'avg_num_sentences': sum(p['metadata']['num_sentences_original'] for p in pairs) / len(pairs),
        'avg_input_length': sum(p['metadata']['input_length'] for p in pairs) / len(pairs),
        'avg_target_length': sum(p['metadata']['target_length'] for p in pairs) / len(pairs),
        'avg_input_words': sum(p['metadata']['input_words'] for p in pairs) / len(pairs),
        'avg_target_words': sum(p['metadata']['target_words'] for p in pairs) / len(pairs),
        'num_truncated': sum(1 for p in pairs if p['metadata']['truncated']),
        'truncation_rate': sum(1 for p in pairs if p['metadata']['truncated']) / len(pairs)
    }
    
    stats['input_target_ratio'] = stats['avg_input_length'] / stats['avg_target_length']
    
    return stats