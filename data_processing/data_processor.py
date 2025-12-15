"""
Main data processor - tích hợp tất cả approaches
"""

import logging
from pathlib import Path
from datasets import load_dataset
import json
from datetime import datetime
from util.dataset_analysis import load_dataset_8opt
from .text_utils import clean_text
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


def save_training_pairs(pairs, output_path, approach_name):
    """Save training pairs to JSONL"""
    output_file = Path(output_path) / f"{approach_name}_training_pairs.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    return output_file


def save_statistics(stats_dict, output_dir):
    """Save statistics to JSON"""
    stats_file = Path(output_dir) / 'statistics' / f'stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_dict, f, ensure_ascii=False, indent=2)
    
    return stats_file


def process_dataset_all_approaches(
    dataset_name: str,
    approaches: list = ['baseline'],
    max_tokens: int = 512,
    output_dir: str = './data/processed',
    limit: int = None
):
    """
    Process dataset với tất cả approaches được chỉ định
    
    Args:
        dataset_name: HuggingFace dataset name
        approaches: List of approaches to run
        max_tokens: Max token length
        output_dir: Output directory
        limit: Limit samples (for testing)
        
    Returns:
        Dict of results
    """
    logger = setup_logger()
    
    logger.info("="*70)
    logger.info("VIETNAMESE SUMMARIZATION DATA PROCESSING")
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
            {'document': item['document'], 'summary': item['summary']}
            for item in train_data
        ]
        
    except Exception as e:
        logger.error(f"❌ Error loading dataset: {e}")
        raise
    
    # Process approaches
    if 'all' in approaches:
        approaches = ['baseline', 'fused_sentences', 'discourse_aware']
    
    logger.info(f"\n🎯 Processing approaches: {', '.join(approaches)}")
    
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
            'baseline'
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
            'fused_sentences'
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
            'discourse_aware'
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
        logger.info(f"  Avg input length: {stats.get('avg_input_length', 0):.1f} chars")
        logger.info(f"  Avg target length: {stats.get('avg_target_length', 0):.1f} chars")
        logger.info(f"  Input/Target ratio: {stats.get('input_target_ratio', 0):.2f}x")
        
        if 'avg_compression_ratio' in stats:
            logger.info(f"  Avg compression: {(1 - stats['avg_compression_ratio']) * 100:.1f}%")
    
    logger.info("\n" + "="*70)
    logger.info("✅ PROCESSING COMPLETE")
    logger.info("="*70)
    
    return results