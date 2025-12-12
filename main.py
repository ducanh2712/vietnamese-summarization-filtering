# main.py
from util.dataset_analysis import (
    load_dataset_8opt,
    basic_info,
    analyze_text_lengths,
    show_samples,
    check_data_quality,
    analyze_vocab,
    export_to_csv,
    summarization_ratio_analysis
)

def main():
    print("\n" + "🔍"*35)
    print("VIETNAMESE SUMMARIZATION DATASET EXPLORER")
    print("Dataset: 8Opt/vietnamese-summarization-dataset-0001")
    print("🔍"*35 + "\n")
    
    # Load dataset
    dataset = load_dataset_8opt()
    
    if dataset is None:
        print("\n❌ Could not load dataset. Please check again.")
        return
    
    # Choose split to analyze
    available_splits = list(dataset.keys())
    split_to_analyze = "train" if "train" in available_splits else available_splits[0]
    
    # Run analyses
    basic_info(dataset)
    analyze_text_lengths(dataset, split=split_to_analyze)
    show_samples(dataset, split=split_to_analyze, n=3)
    check_data_quality(dataset, split=split_to_analyze)
    analyze_vocab(dataset, split=split_to_analyze, n_common=15)
    summarization_ratio_analysis(dataset, split=split_to_analyze)
    export_to_csv(dataset, split=split_to_analyze, n=100)
    
    print("\n✅ ANALYSIS COMPLETED!")

if __name__ == "__main__":
    main()
