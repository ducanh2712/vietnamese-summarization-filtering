from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from util.dataset_analysis import load_dataset_8opt

def run_inference(sample_index=0, dataset_name="8Opt/vietnamese-summarization-dataset-0001"):
    # Load dataset
    dataset = load_dataset_8opt(dataset_name)
    if dataset is None:
        print("❌ Dataset load failed.")
        return

    split = "test"
    sample = dataset[split][sample_index]

    document = sample["document"]
    gold_summary = sample["summary"]

    print("Original document:")
    print(document)
    print("-" * 70)

    # Load model
    model_name = "google/mt5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # ✅ IMPORTANT: task prefix
    input_text = "summarize: " + document

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    summary_ids = model.generate(
        **inputs,
        max_length=128,
        min_length=30,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )

    generated_summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    print("Generated summary (model output):")
    print(generated_summary)
    print("-" * 70)
    
    print("Gold summary (dataset ground truth):")
    print(gold_summary)
    print("-" * 70)