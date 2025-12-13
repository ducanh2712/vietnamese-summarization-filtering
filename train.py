import os
import yaml
import torch
import logging
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from util.dataset_analysis import load_dataset_8opt
import evaluate
import numpy as np

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

def preprocess_function(examples, tokenizer, config):
    """Preprocess dataset samples"""
    task_prefix = config['dataset']['task_prefix']
    max_input_length = config['model']['max_input_length']
    max_target_length = config['model']['max_target_length']
    
    # Add task prefix to input
    inputs = [task_prefix + doc for doc in examples["document"]]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding=False
    )
    
    # Tokenize targets
    labels = tokenizer(
        text_target=examples["summary"],
        max_length=max_target_length,
        truncation=True,
        padding=False
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_pred, tokenizer, config):
    """Compute evaluation metrics"""
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")
    
    predictions, labels = eval_pred
    
    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in labels (used for padding)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Strip whitespace
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    
    # Compute ROUGE
    rouge_result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=False
    )
    
    # Compute BLEU
    bleu_result = bleu.compute(
        predictions=decoded_preds,
        references=[[label] for label in decoded_labels]
    )
    
    # Compute METEOR
    meteor_result = meteor.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )
    
    # Combine results
    result = {
        "rouge1": rouge_result["rouge1"],
        "rouge2": rouge_result["rouge2"],
        "rougeL": rouge_result["rougeL"],
        "bleu": bleu_result["bleu"],
        "meteor": meteor_result["meteor"]
    }
    
    return result

def train_model(config_path="config.yaml"):
    """Main training function"""
    # Load configuration
    config = load_config(config_path)
    logger.info(f"Configuration loaded from {config_path}")
    
    # Create output directories
    os.makedirs(config['training']['output_dir'], exist_ok=True)
    os.makedirs(config['training']['logging_dir'], exist_ok=True)
    os.makedirs("./output", exist_ok=True)
    
    # Load dataset
    logger.info(f"Loading dataset: {config['dataset']['name']}")
    dataset = load_dataset_8opt(config['dataset']['name'])
    if dataset is None:
        logger.error("Failed to load dataset")
        return
    
    # Load tokenizer and model
    logger.info(f"Loading model: {config['model']['name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    model = AutoModelForSeq2SeqLM.from_pretrained(config['model']['name'])
    
    # Preprocess datasets
    logger.info("Preprocessing datasets...")
    tokenized_datasets = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, config),
        batched=True,
        remove_columns=dataset[config['dataset']['train_split']].column_names
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=config['training']['output_dir'],
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        warmup_steps=config['training']['warmup_steps'],
        
        eval_strategy=config['training']['eval_strategy'],
        eval_steps=config['training']['eval_steps'],
        save_strategy=config['training']['save_strategy'],
        save_steps=config['training']['save_steps'],
        save_total_limit=config['training']['save_total_limit'],
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        metric_for_best_model=config['training']['metric_for_best_model'],
        greater_is_better=config['training']['greater_is_better'],
        
        logging_dir=config['training']['logging_dir'],
        logging_steps=config['training']['logging_steps'],
        report_to=config['training']['report_to'],
        
        fp16=config['training']['fp16'],
        dataloader_num_workers=config['training']['dataloader_num_workers'],
        seed=config['training']['seed'],
        
        predict_with_generate=True,
        generation_max_length=config['generation']['max_length'],
        generation_num_beams=config['generation']['num_beams']
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets[config['dataset']['train_split']],
        eval_dataset=tokenized_datasets.get(config['dataset']['val_split']),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer, config)
    )
    
    # Train
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # Save final model
    final_model_path = os.path.join(config['training']['output_dir'], "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Log training summary
    log_file = os.path.join("./output", f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Model: {config['model']['name']}\n")
        f.write(f"Dataset: {config['dataset']['name']}\n")
        f.write(f"Training epochs: {config['training']['num_train_epochs']}\n")
        f.write(f"Learning rate: {config['training']['learning_rate']}\n")
        f.write(f"Batch size: {config['training']['per_device_train_batch_size']}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("TRAINING METRICS\n")
        f.write("=" * 80 + "\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    
    logger.info(f"Training log saved to {log_file}")
    logger.info("Training completed successfully!")
    
    return trainer, final_model_path

if __name__ == "__main__":
    # Train model
    trainer, model_path = train_model()
    
    # Run evaluation after training
    logger.info("\nStarting post-training evaluation...")
    from eval import evaluate_model
    evaluate_model(model_path=model_path)