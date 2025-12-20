# Vietnamese Summarization Project

A comprehensive system for fine-tuning and evaluating Vietnamese text summarization models using mT5, with multiple preprocessing approaches and automatic evaluation.

## 🌟 Features

- **Multiple Preprocessing Approaches**: Baseline, fused sentences, and discourse-aware processing
- **Automated Training Pipeline**: Easy-to-use Makefile for all operations
- **Comprehensive Evaluation**: ROUGE, BLEU, and METEOR metrics
- **Flexible Configuration**: YAML-based configuration system
- **GPU Support**: Automatic CUDA detection and utilization

## 📁 Directory Structure

```
.
├── Makefile                # Build automation and commands
├── main.py                 # Main entry point
├── train.py                # Training script
├── eval.py                 # Evaluation script
├── inference.py            # Inference script
├── config.yaml             # Configuration file
├── requirements.txt        # Python dependencies
├── data_processing/        # Data processing modules
│   └── data_processor.py
├── util/                   # Utility functions
│   └── dataset_analysis.py
├── data/                   # Data directory
│   └── processed/          # Processed datasets
└── output/                 # Results directory
    ├── checkpoints/        # Model checkpoints
    │   └── final_model/
    ├── logs/               # Training logs
    ├── evaluation_results_*.json
    ├── predictions_*.json
    ├── training_log_*.txt
    └── evaluation_log_*.txt
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment and install dependencies
make setup

# Verify installation
make check-env
make check-gpu
```

### 2. Data Processing

```bash
# Analyze dataset
make analysis

# Process data for all approaches
make process-data

# Or process specific approaches
make process-data-baseline    # Baseline approach only
make process-data-fused       # Fused sentences approach
make process-data-discourse   # Discourse-aware approach

# Test with limited samples
make process-data-test        # Process only 100 samples
```

### 3. Training

```bash
# Train model with default config
make train

# Quick training for testing
make train-quick
```

### 4. Evaluation

```bash
# Evaluate fine-tuned model on test set
make eval

# Evaluate base model (without fine-tuning)
make eval-base

# Evaluate on validation set
make eval-val
```

### 5. Inference

```bash
# Run inference on first sample
make inference

# Run inference on specific sample
make inference-random
```

## 📋 Makefile Commands

### Setup & Installation
- `make setup` - Create virtual environment and install all dependencies
- `make install` - Install dependencies from requirements.txt
- `make clean` - Remove virtual environment and cache files

### Data Processing & Analysis
- `make analysis` - Run dataset analysis
- `make process-data` - Process data for all approaches
- `make process-data-baseline` - Process baseline approach only
- `make process-data-fused` - Process fused sentences approach
- `make process-data-discourse` - Process discourse-aware approach
- `make process-data-test` - Process with limited samples (100)

### Training & Evaluation
- `make train` - Train model with default configuration
- `make train-quick` - Quick training for testing
- `make eval` - Evaluate trained model on test set
- `make eval-base` - Evaluate base model
- `make eval-val` - Evaluate on validation set

### Inference
- `make inference` - Run inference on sample 0
- `make inference-random` - Run inference on sample 42

### Utilities
- `make help` - Display all available commands
- `make check-env` - Check environment setup
- `make check-gpu` - Check CUDA/GPU availability
- `make list-outputs` - List all output files
- `make clean-outputs` - Clean output files (keep checkpoints)
- `make clean-all` - Clean everything
- `make test-all` - Run full test pipeline

## ⚙️ Configuration

Edit `config.yaml` to adjust parameters:

### Model Configuration
```yaml
model:
  name: "google/mt5-small"
  max_input_length: 512
  max_target_length: 128
```

### Dataset Configuration
```yaml
dataset:
  name: "8Opt/vietnamese-summarization-dataset-0001"
  train_split: "train"
  validation_split: "validation"
  test_split: "test"
  task_prefix: "summarize: "
```

### Training Configuration
```yaml
training:
  num_epochs: 3
  per_device_train_batch_size: 4
  per_device_eval_batch_size: 8
  learning_rate: 5.0e-5
  weight_decay: 0.01
  warmup_steps: 500
  gradient_accumulation_steps: 2
```

### Data Processing Configuration
```yaml
data_processing:
  approaches:
    - baseline
    - fused_sentences
    - discourse_aware
  max_tokens: 512
  output_dir: "./data/processed"
  limit_samples: null  # Set to number for testing
```

## 📊 Evaluation Metrics

The system calculates the following metrics:

- **ROUGE-1**: Unigram overlap between generated and reference summaries
- **ROUGE-2**: Bigram overlap for assessing fluency
- **ROUGE-L**: Longest common subsequence based similarity
- **ROUGE-Lsum**: ROUGE-L with sentence-level splitting
- **BLEU**: Precision-based metric for translation quality
- **METEOR**: Harmonic mean of precision and recall with synonym matching

### Output Files

Evaluation results are saved to:
- `evaluation_results_*.json` - Metrics summary with scores
- `predictions_*.json` - All predictions and references
- `evaluation_log_*.txt` - Detailed evaluation logs

## 🔧 Advanced Usage

### Manual Python Commands

If you prefer not to use Makefile:

```bash
# Activate virtual environment
source venv/bin/activate

# Run commands
python main.py --mode analysis --dataset 8Opt/vietnamese-summarization-dataset-0001
python main.py --mode process_data --config config.yaml
python main.py --mode train --config config.yaml
python main.py --mode eval --model_path output/checkpoints/final_model --split test
python main.py --mode inference --sample_index 0
```

### Custom Data Processing

```bash
# Process with specific approaches
python main.py --mode process_data \
    --config config.yaml \
    --approaches baseline fused_sentences \
    --limit 1000
```

### Custom Evaluation

```bash
# Evaluate on different splits
python main.py --mode eval \
    --model_path output/checkpoints/final_model \
    --split validation \
    --config config.yaml
```

## 📦 Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA (optional, for GPU acceleration)

See `requirements.txt` for full dependency list.

## 🐛 Troubleshooting

### Virtual Environment Issues
```bash
make clean
make setup
```

### NLTK Data Missing
```bash
source venv/bin/activate
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### GPU Not Detected
```bash
make check-gpu
```

### Clean Start
```bash
make clean-all
make setup
```

## 📈 Typical Workflow

```bash
# 1. Setup
make setup
make check-gpu

# 2. Data exploration
make analysis

# 3. Process data
make process-data

# 4. Train model
make train

# 5. Results are in output/ directory
make list-outputs
```

## 📚 Citation

If you use this project, please cite:

- **mT5**: Xue, L., et al. (2020). [mT5: A massively multilingual pre-trained text-to-text transformer](https://arxiv.org/abs/2010.11934)
- **ViT5**: Phan, L., et al. (2022). [ViT5: Pretrained Text-to-Text Transformer for Vietnamese Language Generation](https://arxiv.org/abs/2205.11001)
- **Dataset**: [8Opt/vietnamese-summarization-dataset-0001](https://huggingface.co/datasets/8Opt/vietnamese-summarization-dataset-0001)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face for the Transformers library
- Google Research for mT5
- VinAI Research for Vietnamese NLP resources
- Dataset contributors and maintainers