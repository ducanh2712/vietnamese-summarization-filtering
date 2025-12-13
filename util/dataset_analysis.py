from datasets import load_dataset
import pandas as pd
from collections import Counter

from util import preprocess_test_cleaning


def load_dataset_8opt(dataset_name: str, clean: bool = True):
    print("=" * 70)
    print(f"LOADING DATASET: {dataset_name}")
    print("=" * 70)

    try:
        dataset = load_dataset(dataset_name)

        if clean:
            dataset = preprocess_test_cleaning.clean_dataset(dataset)

        print("✓ Successfully loaded!")
        return dataset

    except Exception as e:
        print(f"✗ Error: {e}")
        return None




def basic_info(dataset):
    """Display basic information about the dataset"""
    print("\n" + "="*70)
    print("1. BASIC INFORMATION")
    print("="*70)
    
    # Available splits
    print(f"\nAvailable splits: {list(dataset.keys())}")
    
    for split_name in dataset.keys():
        split_data = dataset[split_name]
        print(f"\n[{split_name.upper()}]")
        print(f"  - Number of samples: {len(split_data)}")
        print(f"  - Features (columns): {list(split_data.features.keys())}")
        print(f"  - Feature types:")
        for feat_name, feat_type in split_data.features.items():
            print(f"      {feat_name}: {feat_type}")


def analyze_text_lengths(dataset, split="train"):
    """Analyze text lengths"""
    print("\n" + "="*70)
    print(f"2. TEXT LENGTH ANALYSIS - Split: {split}")
    print("="*70)
    
    split_data = dataset[split]
    features = list(split_data.features.keys())
    
    # Find text columns
    text_columns = []
    for col in features:
        if isinstance(split_data[0][col], str):
            text_columns.append(col)
    
    print(f"\nText columns: {text_columns}\n")
    
    for col in text_columns:
        lengths = [len(text) if text else 0 for text in split_data[col]]
        word_counts = [len(text.split()) if text else 0 for text in split_data[col]]
        
        print(f"\n📊 {col.upper()}:")
        print(f"  Characters:")
        print(f"    - Min: {min(lengths):,}")
        print(f"    - Max: {max(lengths):,}")
        print(f"    - Average: {sum(lengths)/len(lengths):,.1f}")
        print(f"    - Median: {sorted(lengths)[len(lengths)//2]:,}")
        
        print(f"  Words:")
        print(f"    - Min: {min(word_counts):,}")
        print(f"    - Max: {max(word_counts):,}")
        print(f"    - Average: {sum(word_counts)/len(word_counts):,.1f}")
        print(f"    - Median: {sorted(word_counts)[len(word_counts)//2]:,}")


def show_samples(dataset, split="train", n=3):
    """Display sample data"""
    print("\n" + "="*70)
    print(f"3. DATA SAMPLES - Split: {split}")
    print("="*70)
    
    split_data = dataset[split]
    
    for i in range(min(n, len(split_data))):
        print(f"\n{'─'*70}")
        print(f"SAMPLE #{i+1}")
        print(f"{'─'*70}")
        
        sample = split_data[i]
        
        for key, value in sample.items():
            print(f"\n[{key}]")
            if isinstance(value, str):
                # Display first 300 characters
                display_text = value[:300] + "..." if len(value) > 300 else value
                print(f"{display_text}")
                print(f"(Total: {len(value)} chars, {len(value.split())} words)")
            else:
                print(f"{value}")


def check_data_quality(dataset, split="train"):
    """Check data quality"""
    print("\n" + "="*70)
    print(f"4. DATA QUALITY CHECK - Split: {split}")
    print("="*70)
    
    split_data = dataset[split]
    features = list(split_data.features.keys())
    
    print(f"\nTotal samples: {len(split_data)}")
    
    # Check missing/empty values
    print(f"\n📋 Checking null/empty values:")
    for col in features:
        null_count = sum(1 for x in split_data[col] if x is None or (isinstance(x, str) and len(x.strip()) == 0))
        if null_count > 0:
            print(f"  ⚠ {col}: {null_count} samples ({null_count/len(split_data)*100:.1f}%)")
        else:
            print(f"  ✓ {col}: OK")
    
    # Check duplicates
    print(f"\n📋 Checking duplicates:")
    for col in features:
        if isinstance(split_data[0][col], str):
            unique_count = len(set(split_data[col]))
            duplicate_count = len(split_data) - unique_count
            if duplicate_count > 0:
                print(f"  ⚠ {col}: {duplicate_count} duplicates ({duplicate_count/len(split_data)*100:.1f}%)")
            else:
                print(f"  ✓ {col}: No duplicates")


def analyze_vocab(dataset, split="train", n_common=20):
    """Analyze vocabulary"""
    print("\n" + "="*70)
    print(f"5. VOCABULARY ANALYSIS - Split: {split}")
    print("="*70)
    
    split_data = dataset[split]
    features = list(split_data.features.keys())
    
    # Get text columns
    text_columns = [col for col in features if isinstance(split_data[0][col], str)]
    
    for col in text_columns[:2]:  # Only analyze first 2 columns
        print(f"\n📚 {col.upper()}:")
        
        # Collect all words
        all_words = []
        for text in split_data[col]:
            if text:
                words = text.lower().split()
                all_words.extend(words)
        
        # Count
        word_counter = Counter(all_words)
        vocab_size = len(word_counter)
        
        print(f"  Total words: {len(all_words):,}")
        print(f"  Vocabulary size: {vocab_size:,}")
        print(f"  Words appearing once: {sum(1 for count in word_counter.values() if count == 1):,}")
        
        print(f"\n  Top {n_common} most common words:")
        for word, count in word_counter.most_common(n_common):
            print(f"    '{word}': {count:,}")


def export_to_csv(dataset, split="train", output_file="dataset_sample.csv", n=100):
    """Export samples to CSV for inspection"""
    print("\n" + "="*70)
    print(f"6. EXPORTING SAMPLE DATA")
    print("="*70)
    
    split_data = dataset[split]
    
    # Get first n samples
    sample_data = split_data.select(range(min(n, len(split_data))))
    
    # Convert to pandas
    df = pd.DataFrame(sample_data)
    
    # Save
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n✓ Exported {len(df)} samples to file: {output_file}")
    print(f"  Columns: {list(df.columns)}")
    print(f"\nYou can open this file with Excel or a text editor for detailed inspection.")


def summarization_ratio_analysis(dataset, split="train"):
    """Analyze compression ratio of summarization"""
    print("\n" + "="*70)
    print(f"7. COMPRESSION RATIO ANALYSIS")
    print("="*70)
    
    split_data = dataset[split]
    features = list(split_data.features.keys())
    
    # Find document and summary columns
    # Usually 'document'/'text'/'article' and 'summary'
    doc_col = None
    sum_col = None
    
    for col in features:
        col_lower = col.lower()
        if 'doc' in col_lower or 'text' in col_lower or 'article' in col_lower:
            doc_col = col
        if 'sum' in col_lower or 'abstract' in col_lower:
            sum_col = col
    
    if doc_col and sum_col:
        print(f"\nDocument column: {doc_col}")
        print(f"Summary column: {sum_col}\n")
        
        ratios = []
        for i in range(len(split_data)):
            doc = split_data[i][doc_col]
            summ = split_data[i][sum_col]
            
            if doc and summ:
                doc_len = len(doc.split())
                sum_len = len(summ.split())
                if doc_len > 0:
                    ratio = sum_len / doc_len
                    ratios.append(ratio)
        
        if ratios:
            print(f"Compression ratio (summary_len / doc_len):")
            print(f"  Min: {min(ratios):.3f}")
            print(f"  Max: {max(ratios):.3f}")
            print(f"  Average: {sum(ratios)/len(ratios):.3f}")
            print(f"  Median: {sorted(ratios)[len(ratios)//2]:.3f}")
    else:
        print("\n⚠ Could not find document/summary columns")
        
def run_dataset_exploration(
    dataset_name: str,
    split: str = "train",
    n_samples: int = 3,
    n_vocab: int = 15,
    export_n: int = 100
):
    """
    Run full exploration pipeline for a summarization dataset.
    """

    print("\n" + "🔍" * 35)
    print("VIETNAMESE SUMMARIZATION DATASET EXPLORER")
    print(f"Dataset: {dataset_name}")
    print("🔍" * 35 + "\n")

    # Load dataset
    dataset = load_dataset_8opt(dataset_name)

    if dataset is None:
        print("\n❌ Could not load dataset. Please check again.")
        return

    # Choose split
    available_splits = list(dataset.keys())
    split_to_analyze = split if split in available_splits else available_splits[0]

    print(f"Using split: {split_to_analyze}")

    # Run analyses
    basic_info(dataset)
    analyze_text_lengths(dataset, split=split_to_analyze)
    show_samples(dataset, split=split_to_analyze, n=n_samples)
    check_data_quality(dataset, split=split_to_analyze)
    analyze_vocab(dataset, split=split_to_analyze, n_common=n_vocab)
    summarization_ratio_analysis(dataset, split=split_to_analyze)
    export_to_csv(dataset, split=split_to_analyze, n=export_n)

    print("\n✅ ANALYSIS COMPLETED!")

def data_analysis():
    run_dataset_exploration(
        dataset_name="8Opt/vietnamese-summarization-dataset-0001",
        split="train",
        n_samples=3,
        n_vocab=15,
        export_n=100
    )