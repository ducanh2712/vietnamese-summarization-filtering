"""
Main data processor - tích hợp tất cả approaches
Output format: document, summary
"""

import logging
from pathlib import Path
from datasets import load_dataset
import json
from datetime import datetime
from util.dataset_analysis import load_dataset_8opt
from .text_utils import get_duplicate_stats, remove_duplicates as dedup_pairs
from . import approach1_baseline, approach2_fused, approach3_discourse


def setup_logger(name='data_processor'):
    """Setup logger đơn giản"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def save_training_pairs(pairs, output_path, approach_name, include_metadata=False):
    """
    Save training pairs to JSONL
    
    Args:
        pairs: List of training pairs
        output_path: Output directory path
        approach_name: Name of approach
        include_metadata: Whether to include metadata in output
        
    Returns:
        Path to output file
    """
    output_file = Path(output_path) / f"{approach_name}_training_pairs.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in pairs:
            # Tạo output dict đơn giản: chỉ document và summary
            output = {
                'document': pair['document'],
                'summary': pair['summary']
            }
            
            # Thêm metadata nếu cần (cho debugging/tracking)
            if include_metadata and 'metadata' in pair:
                output['metadata'] = pair['metadata']
            
            f.write(json.dumps(output, ensure_ascii=False) + '\n')
    
    return output_file


def save_statistics(stats_dict, output_dir):
    """Save statistics to JSON"""
    stats_file = Path(output_dir) / 'statistics' / f'stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_dict, f, ensure_ascii=False, indent=2)
    
    return stats_file


def remove_duplicate_articles(articles, mode='both', keep='first'):
    """
    Remove duplicate articles TRƯỚC KHI processing
    
    Args:
        articles: List of article dicts với 'document' và 'summary'
        mode: 'document', 'summary', hoặc 'both'
        keep: 'first' hoặc 'last'
        
    Returns:
        List of deduplicated articles
    """
    logger = logging.getLogger(__name__)
    
    # Chuyển articles thành format giống training pairs để dùng hàm dedup hiện có
    pairs_format = [
        {
            'document': article['document'],
            'summary': article['summary']
        }
        for article in articles
    ]
    
    # Get duplicate stats trước khi remove
    dup_stats = get_duplicate_stats(pairs_format)
    
    logger.info(f"\n📊 Duplicate check on RAW dataset:")
    logger.info(f"  Total articles: {dup_stats['total_pairs']}")
    logger.info(f"  Document duplicates: {dup_stats['document_duplicates']['count']} ({dup_stats['document_duplicates']['rate']:.2f}%)")
    logger.info(f"  Summary duplicates: {dup_stats['summary_duplicates']['count']} ({dup_stats['summary_duplicates']['rate']:.2f}%)")
    logger.info(f"  Exact duplicates: {dup_stats['exact_duplicates']['count']} ({dup_stats['exact_duplicates']['rate']:.2f}%)")
    
    # Remove duplicates
    original_count = len(pairs_format)
    deduplicated = dedup_pairs(pairs_format, mode=mode, keep=keep)
    removed = original_count - len(deduplicated)
    
    logger.info(f"\n🔧 Deduplication (mode={mode}, keep={keep}):")
    logger.info(f"  Original: {original_count} articles")
    logger.info(f"  Removed: {removed} articles ({removed/original_count*100:.2f}%)")
    logger.info(f"  Remaining: {len(deduplicated)} articles")
    
    return deduplicated


def process_dataset_all_approaches(
    dataset_name: str,
    approaches: list = ['baseline'],
    max_tokens: int = 512,
    output_dir: str = './data/processed',
    limit: int = None,
    include_metadata: bool = False,
    remove_duplicates: bool = True,
    duplicate_mode: str = 'document'
):
    """
    Process dataset với tất cả approaches được chỉ định
    Output format: document, summary
    
    Args:
        dataset_name: HuggingFace dataset name
        approaches: List of approaches to run
        max_tokens: Max token length
        output_dir: Output directory
        limit: Limit samples (for testing)
        include_metadata: Include metadata in JSONL output
        remove_duplicates: Remove duplicates TRƯỚC KHI processing
        duplicate_mode: 'document', 'summary', hoặc 'both' (for deduplication)
        
    Returns:
        Dict of results
    """
    logger = setup_logger()
    
    logger.info("="*70)
    logger.info("VIETNAMESE SUMMARIZATION DATA PROCESSING")
    logger.info("Output format: document, summary")
    if remove_duplicates:
        logger.info(f"Deduplication: ENABLED (mode={duplicate_mode}) - BEFORE processing")
    logger.info("="*70)
    
    # Load dataset
    logger.info(f"\n📦 Loading dataset: {dataset_name}")
    try:
        dataset = load_dataset_8opt(dataset_name)
        train_data = dataset['train']
        logger.info(f"✅ Dataset loaded: {len(train_data)} samples")
        
        if limit:
            train_data = train_data.select(range(min(limit, len(train_data))))
            logger.info(f"   Limited to: {len(train_data)} samples")
        
        articles = [
            {
                'document': item['document'], 
                'summary': item['summary']
            }
            for item in train_data
        ]
        
    except Exception as e:
        logger.error(f"❌ Error loading dataset: {e}")
        raise
    
    # ===== REMOVE DUPLICATES TRƯỚC KHI PROCESSING =====
    if remove_duplicates:
        logger.info("\n" + "="*70)
        logger.info("🧹 REMOVING DUPLICATES FROM RAW DATASET")
        logger.info("="*70)
        articles = remove_duplicate_articles(articles, mode=duplicate_mode, keep='first')
        logger.info(f"\n✅ Will process {len(articles)} unique articles")
    
    # Process approaches
    if 'all' in approaches:
        approaches = ['baseline', 'fused_sentences', 'discourse_aware']
    
    logger.info(f"\n🎯 Processing approaches: {', '.join(approaches)}")
    logger.info(f"📄 Include metadata in output: {include_metadata}")
    
    results = {}
    all_stats = {}
    
    # Baseline
    if 'baseline' in approaches:
        logger.info("\n" + "="*70)
        logger.info("APPROACH 1: BASELINE")
        logger.info("="*70)
        
        pairs = approach1_baseline.process_dataset(articles, max_tokens)
        stats = approach1_baseline.calculate_statistics(pairs)
        
        output_file = save_training_pairs(
            pairs,
            Path(output_dir) / 'baseline',
            'baseline',
            include_metadata
        )
        logger.info(f"💾 Saved to: {output_file}")
        
        results['baseline'] = {
            'num_pairs': len(pairs),
            'output_file': str(output_file)
        }
        all_stats['baseline'] = stats
    
    # Fused
    if 'fused_sentences' in approaches:
        logger.info("\n" + "="*70)
        logger.info("APPROACH 2: FUSED SENTENCES")
        logger.info("="*70)
        
        pairs = approach2_fused.process_dataset(articles, max_tokens)
        stats = approach2_fused.calculate_statistics(pairs)
        
        output_file = save_training_pairs(
            pairs,
            Path(output_dir) / 'fused_sentences',
            'fused_sentences',
            include_metadata
        )
        logger.info(f"💾 Saved to: {output_file}")
        
        results['fused_sentences'] = {
            'num_pairs': len(pairs),
            'output_file': str(output_file)
        }
        all_stats['fused_sentences'] = stats
    
    # Discourse-aware
    if 'discourse_aware' in approaches:
        logger.info("\n" + "="*70)
        logger.info("APPROACH 3: DISCOURSE-AWARE")
        logger.info("="*70)
        
        pairs = approach3_discourse.process_dataset(articles, max_tokens)
        stats = approach3_discourse.calculate_statistics(pairs)
        
        output_file = save_training_pairs(
            pairs,
            Path(output_dir) / 'discourse_aware',
            'discourse_aware',
            include_metadata
        )
        logger.info(f"💾 Saved to: {output_file}")
        
        results['discourse_aware'] = {
            'num_pairs': len(pairs),
            'output_file': str(output_file)
        }
        all_stats['discourse_aware'] = stats
    
    # Save statistics
    stats_file = save_statistics(all_stats, output_dir)
    logger.info(f"\n📊 Statistics saved to: {stats_file}")
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("📊 SUMMARY STATISTICS")
    logger.info("="*70)
    
    for approach_name, stats in all_stats.items():
        logger.info(f"\n{approach_name.upper()}:")
        logger.info(f"  Total pairs: {stats.get('total_pairs', 0)}")
        logger.info(f"  Avg document length: {stats.get('avg_document_length', 0):.1f} chars")
        logger.info(f"  Avg summary length: {stats.get('avg_summary_length', 0):.1f} chars")
        logger.info(f"  Avg document words: {stats.get('avg_document_words', 0):.1f}")
        logger.info(f"  Avg summary words: {stats.get('avg_summary_words', 0):.1f}")
        
        if 'document_summary_ratio' in stats:
            logger.info(f"  Document/Summary ratio: {stats['document_summary_ratio']:.2f}x")
        
        if 'avg_compression_ratio' in stats:
            logger.info(f"  Avg compression: {(1 - stats['avg_compression_ratio']) * 100:.1f}%")
    
    logger.info("\n" + "="*70)
    logger.info("✅ PROCESSING COMPLETE")
    logger.info("="*70)
    
    return results