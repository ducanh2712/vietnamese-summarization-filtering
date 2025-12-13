# Vietnamese Summarization Project

A system for fine-tuning and evaluating Vietnamese text summarization models using mT5.

## Directory Structure

.
├── main.py                 # Main entry point
├── train.py                # Training script
├── eval.py                 # Evaluation script
├── inference.py            # Inference script
├── config.yaml             # Configuration file
├── pyproject.toml          # Poetry configuration
├── util/
│   └── dataset_analysis.py
└── output/                 # Directory for results
    ├── checkpoints/        # Model checkpoints
    ├── logs/               # Training logs
    ├── evaluation_results_*.json
    ├── predictions_*.json
    ├── training_log_*.txt
    └── evaluation_log_*.txt

## Installation

Install dependencies using Poetry:

# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install

Download NLTK data (required for METEOR metric):

poetry run python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

## Configuration

Edit `config.yaml` to adjust hyperparameters and paths:

- Model: Model name, max input/output length
- Dataset: Dataset name, splits, task prefix
- Training: Epochs, batch size, learning rate, evaluation strategy
- Generation: Beam search parameters
- Evaluation: Metrics, batch size, output paths

## Usage

### 1. Dataset Analysis

poetry run python main.py --mode analysis --dataset 8Opt/vietnamese-summarization-dataset-0001

### 2. Training

Train the model with automatic evaluation after training:

poetry run python main.py --mode train --config config.yaml

Results are saved to:

- output/checkpoints/final_model/ - Fine-tuned model
- output/training_log_*.txt - Training log
- output/evaluation_results_*.json - Evaluation metrics
- output/evaluation_log_*.txt - Detailed evaluation

### 3. Evaluation

Evaluate a fine-tuned model:

poetry run python main.py --mode eval \
    --model_path output/checkpoints/final_model \
    --split test

Evaluate the base model (without fine-tuning):

poetry run python main.py --mode eval --split test

Evaluate on a different split:

poetry run python main.py --mode eval \
    --model_path output/checkpoints/final_model \
    --split validation

### 4. Inference

Run inference on a single sample:

poetry run python main.py --mode inference --sample_index 0

## Metrics

The system calculates the following metrics:

- ROUGE-1: Unigram overlap
- ROUGE-2: Bigram overlap
- ROUGE-L: Longest common subsequence
- ROUGE-Lsum: ROUGE-L with sentence splitting
- BLEU: Precision-based metric
- METEOR: Harmonic mean of precision and recall

Evaluation results are saved in:

- evaluation_results_*.json – metrics summary
- predictions_*.json – all predictions and references
- evaluation_log_*.txt – detailed logs

## Citation

If you use this project, please cite:

- mT5: [Multilingual T5](https://arxiv.org/abs/2010.11934)
- ViT5: [Vietnamese T5](https://arxiv.org/abs/2205.11001)
- Dataset: 8Opt/vietnamese-summarization-dataset-0001
