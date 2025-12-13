import os
import json
import yaml
import torch
import logging
from datetime import datetime
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from util.dataset_analysis import load_dataset_8opt
import evaluate

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def generate_summaries(model, tokenizer, dataset, config, split="test"):
    """Generate summaries for all samples in the dataset"""
    logger.info(f"Generating summaries for {split} split...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    predictions = []
    references = []
    documents = []
    
    task_prefix = config['dataset']['task_prefix']
    max_input_length = config['model']['max_input_length']
    
    batch_size = config['evaluation']['batch_size']
    samples = dataset[split]
    
    for i in tqdm(range(0, len(samples), batch_size), desc="Generating"):
        batch = samples[i:i+batch_size]
        
        # Prepare inputs
        input_texts = [task_prefix + doc for doc in batch["document"]]
        
        inputs = tokenizer(
            input_texts,
            return_tensors="pt",
            max_length=max_input_length,
            truncation=True,
            padding=True
        ).to(device)
        
        # Generate
        with torch.no_grad():
            summary_ids = model.generate(
                **inputs,
                max_length=config['generation']['max_length'],
                min_length=config['generation']['min_length'],
                num_beams=config['generation']['num_beams'],
                length_penalty=config['generation']['length_penalty'],
                early_stopping=config['generation']['early_stopping'],
                no_repeat_ngram_size=config['generation'].get('no_repeat_ngram_size', 3)
            )
        
        # Decode
        batch_predictions = tokenizer.batch_decode(
            summary_ids,
            skip_special_tokens=True
        )
        
        predictions.extend(batch_predictions)
        references.extend(batch["summary"])
        documents.extend(batch["document"])
    
    return predictions, references, documents

def compute_all_metrics(predictions, references):
    """Compute all evaluation metrics"""
    logger.info("Computing metrics...")
    
    # Load metrics
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")
    
    # Strip whitespace
    predictions = [pred.strip() for pred in predictions]
    references = [ref.strip() for ref in references]
    
    # Compute ROUGE
    logger.info("Computing ROUGE scores...")
    rouge_result = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=False
    )
    
    # Compute BLEU
    logger.info("Computing BLEU score...")
    bleu_result = bleu.compute(
        predictions=predictions,
        references=[[ref] for ref in references]
    )
    
    # Compute METEOR
    logger.info("Computing METEOR score...")
    meteor_result = meteor.compute(
        predictions=predictions,
        references=references
    )
    
    # Combine results
    metrics = {
        "rouge1": float(rouge_result["rouge1"]),
        "rouge2": float(rouge_result["rouge2"]),
        "rougeL": float(rouge_result["rougeL"]),
        "rougeLsum": float(rouge_result["rougeLsum"]),
        "bleu": float(bleu_result["bleu"]),
        "meteor": float(meteor_result["meteor"])
    }
    
    return metrics

def evaluate_model(model_path=None, config_path="config.yaml", split="test"):
    """
    Main evaluation function
    
    Args:
        model_path: Path to the fine-tuned model (if None, uses base model from config)
        config_path: Path to config file
        split: Dataset split to evaluate on
    """
    # Load configuration
    config = load_config(config_path)
    logger.info(f"Configuration loaded from {config_path}")
    
    # Create output directory
    os.makedirs("./output", exist_ok=True)
    
    # Load dataset
    logger.info(f"Loading dataset: {config['dataset']['name']}")
    dataset = load_dataset_8opt(config['dataset']['name'])
    if dataset is None:
        logger.error("Failed to load dataset")
        return
    
    # Load model and tokenizer
    if model_path:
        logger.info(f"Loading fine-tuned model from: {model_path}")
        model_name = model_path
        model_type = "fine-tuned"
    else:
        logger.info(f"Loading base model: {config['model']['name']}")
        model_name = config['model']['name']
        model_type = "base"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Generate summaries
    predictions, references, documents = generate_summaries(
        model, tokenizer, dataset, config, split
    )
    
    # Compute metrics
    metrics = compute_all_metrics(predictions, references)
    
    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info(f"Model type: {model_type}")
    logger.info(f"Dataset: {config['dataset']['name']}")
    logger.info(f"Split: {split}")
    logger.info(f"Number of samples: {len(predictions)}")
    logger.info("-" * 80)
    logger.info(f"ROUGE-1: {metrics['rouge1']:.4f}")
    logger.info(f"ROUGE-2: {metrics['rouge2']:.4f}")
    logger.info(f"ROUGE-L: {metrics['rougeL']:.4f}")
    logger.info(f"ROUGE-Lsum: {metrics['rougeLsum']:.4f}")
    logger.info(f"BLEU: {metrics['bleu']:.4f}")
    logger.info(f"METEOR: {metrics['meteor']:.4f}")
    logger.info("=" * 80)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save metrics to JSON
    results_file = f"./output/evaluation_results_{model_type}_{split}_{timestamp}.json"
    results_data = {
        "model": model_name,
        "model_type": model_type,
        "dataset": config['dataset']['name'],
        "split": split,
        "timestamp": timestamp,
        "num_samples": len(predictions),
        "metrics": metrics,
        "config": {
            "max_length": config['generation']['max_length'],
            "min_length": config['generation']['min_length'],
            "num_beams": config['generation']['num_beams'],
            "length_penalty": config['generation']['length_penalty']
        }
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to: {results_file}")
    
    # Save predictions to JSON
    predictions_file = f"./output/predictions_{model_type}_{split}_{timestamp}.json"
    predictions_data = {
        "model": model_name,
        "model_type": model_type,
        "timestamp": timestamp,
        "samples": [
            {
                "index": i,
                "document": doc,
                "prediction": pred,
                "reference": ref
            }
            for i, (doc, pred, ref) in enumerate(zip(documents, predictions, references))
        ]
    }
    
    with open(predictions_file, 'w', encoding='utf-8') as f:
        json.dump(predictions_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Predictions saved to: {predictions_file}")
    
    # Save detailed log
    log_file = f"./output/evaluation_log_{model_type}_{split}_{timestamp}.txt"
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("EVALUATION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Model type: {model_type}\n")
        f.write(f"Dataset: {config['dataset']['name']}\n")
        f.write(f"Split: {split}\n")
        f.write(f"Number of samples: {len(predictions)}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("METRICS\n")
        f.write("=" * 80 + "\n")
        f.write(f"ROUGE-1:    {metrics['rouge1']:.4f}\n")
        f.write(f"ROUGE-2:    {metrics['rouge2']:.4f}\n")
        f.write(f"ROUGE-L:    {metrics['rougeL']:.4f}\n")
        f.write(f"ROUGE-Lsum: {metrics['rougeLsum']:.4f}\n")
        f.write(f"BLEU:       {metrics['bleu']:.4f}\n")
        f.write(f"METEOR:     {metrics['meteor']:.4f}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("GENERATION CONFIG\n")
        f.write("=" * 80 + "\n")
        f.write(f"Max length: {config['generation']['max_length']}\n")
        f.write(f"Min length: {config['generation']['min_length']}\n")
        f.write(f"Num beams: {config['generation']['num_beams']}\n")
        f.write(f"Length penalty: {config['generation']['length_penalty']}\n")
        f.write(f"Early stopping: {config['generation']['early_stopping']}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("SAMPLE PREDICTIONS (first 5)\n")
        f.write("=" * 80 + "\n")
        for i in range(min(5, len(predictions))):
            f.write(f"\n--- Sample {i+1} ---\n")
            f.write(f"Document: {documents[i][:200]}...\n")
            f.write(f"Prediction: {predictions[i]}\n")
            f.write(f"Reference: {references[i]}\n")
    
    logger.info(f"Evaluation log saved to: {log_file}")
    logger.info("Evaluation completed successfully!")
    
    return metrics, predictions, references

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Vietnamese Summarization Model")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to fine-tuned model (if None, uses base model from config)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to evaluate"
    )
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model_path,
        config_path=args.config,
        split=args.split
    )