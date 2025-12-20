# Vietnamese Summarization Project Makefile
# ===========================================

# Variables
PYTHON := python3
VENV := venv
VENV_BIN := $(VENV)/bin
PIP := $(VENV_BIN)/pip
PYTHON_VENV := $(VENV_BIN)/python
CONFIG := config.yaml
DATASET := 8Opt/vietnamese-summarization-dataset-0001

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Default target
.DEFAULT_GOAL := help

# Phony targets
.PHONY: help setup install clean analysis process-data train eval inference test-all

##@ Setup & Installation

setup: ## Tạo môi trường ảo và cài đặt dependencies
	@echo "$(GREEN)Creating virtual environment...$(NC)"
	@$(PYTHON) -m venv $(VENV)
	@echo "$(GREEN)Virtual environment created successfully!$(NC)"
	@$(MAKE) install

install: ## Cài đặt dependencies từ requirements.txt
	@echo "$(GREEN)Installing dependencies...$(NC)"
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@echo "$(GREEN)Downloading NLTK data...$(NC)"
	@$(PYTHON_VENV) -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
	@echo "$(GREEN)Installation completed!$(NC)"

clean: ## Xóa môi trường ảo và cache files
	@echo "$(YELLOW)Cleaning up...$(NC)"
	@rm -rf $(VENV)
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)Cleanup completed!$(NC)"

##@ Data Processing & Analysis

analysis: ## Chạy dataset analysis
	@echo "$(GREEN)Running dataset analysis...$(NC)"
	@$(PYTHON_VENV) main.py --mode analysis --dataset $(DATASET)

process-data: ## Xử lý dữ liệu cho tất cả approaches
	@echo "$(GREEN)Processing data for all approaches...$(NC)"
	@$(PYTHON_VENV) main.py --mode process_data --config $(CONFIG)

process-data-baseline: ## Xử lý dữ liệu cho baseline approach
	@echo "$(GREEN)Processing data for baseline approach...$(NC)"
	@$(PYTHON_VENV) main.py --mode process_data --config $(CONFIG) --approaches baseline

process-data-fused: ## Xử lý dữ liệu cho fused_sentences approach
	@echo "$(GREEN)Processing data for fused_sentences approach...$(NC)"
	@$(PYTHON_VENV) main.py --mode process_data --config $(CONFIG) --approaches fused_sentences

process-data-discourse: ## Xử lý dữ liệu cho discourse_aware approach
	@echo "$(GREEN)Processing data for discourse_aware approach...$(NC)"
	@$(PYTHON_VENV) main.py --mode process_data --config $(CONFIG) --approaches discourse_aware

process-data-test: ## Xử lý dữ liệu với limit 100 samples (for testing)
	@echo "$(GREEN)Processing data with limited samples...$(NC)"
	@$(PYTHON_VENV) main.py --mode process_data --config $(CONFIG) --limit 100

##@ Training & Evaluation

train: ## Train model với config mặc định
	@echo "$(GREEN)Starting training...$(NC)"
	@$(PYTHON_VENV) main.py --mode train --config $(CONFIG)

train-quick: ## Train nhanh với ít epochs (for testing)
	@echo "$(YELLOW)Starting quick training (for testing)...$(NC)"
	@$(PYTHON_VENV) main.py --mode train --config config_quick.yaml

eval: ## Evaluate model đã train trên test set
	@echo "$(GREEN)Evaluating trained model on test set...$(NC)"
	@$(PYTHON_VENV) main.py --mode eval \
		--model_path output/checkpoints/final_model \
		--split test \
		--config $(CONFIG)

eval-base: ## Evaluate base model (chưa fine-tune)
	@echo "$(GREEN)Evaluating base model...$(NC)"
	@$(PYTHON_VENV) main.py --mode eval --split test --config $(CONFIG)

eval-val: ## Evaluate model trên validation set
	@echo "$(GREEN)Evaluating model on validation set...$(NC)"
	@$(PYTHON_VENV) main.py --mode eval \
		--model_path output/checkpoints/final_model \
		--split validation \
		--config $(CONFIG)

##@ Inference

inference: ## Chạy inference trên sample đầu tiên
	@echo "$(GREEN)Running inference on sample 0...$(NC)"
	@$(PYTHON_VENV) main.py --mode inference --sample_index 0

inference-random: ## Chạy inference trên sample ngẫu nhiên
	@echo "$(GREEN)Running inference on random sample...$(NC)"
	@$(PYTHON_VENV) main.py --mode inference --sample_index 42

##@ Testing & Utilities

test-all: analysis process-data-test train-quick eval ## Chạy toàn bộ pipeline với test config
	@echo "$(GREEN)Full test pipeline completed!$(NC)"

check-env: ## Kiểm tra môi trường
	@echo "$(YELLOW)Checking environment...$(NC)"
	@if [ -d "$(VENV)" ]; then \
		echo "$(GREEN)✓ Virtual environment exists$(NC)"; \
		echo "Python version: $$($(PYTHON_VENV) --version)"; \
		echo "Pip version: $$($(PIP) --version)"; \
	else \
		echo "$(RED)✗ Virtual environment not found. Run 'make setup'$(NC)"; \
	fi

check-gpu: ## Kiểm tra CUDA/GPU availability
	@echo "$(YELLOW)Checking GPU availability...$(NC)"
	@$(PYTHON_VENV) -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

list-outputs: ## Liệt kê các file outputs
	@echo "$(YELLOW)Listing output files...$(NC)"
	@if [ -d "output" ]; then \
		ls -lah output/; \
		echo "\nCheckpoints:"; \
		ls -lah output/checkpoints/ 2>/dev/null || echo "No checkpoints found"; \
	else \
		echo "$(RED)No output directory found$(NC)"; \
	fi

clean-outputs: ## Xóa các file outputs (giữ lại model checkpoints)
	@echo "$(YELLOW)Cleaning output files (keeping checkpoints)...$(NC)"
	@find output -type f -name "*.json" -delete 2>/dev/null || true
	@find output -type f -name "*.txt" -delete 2>/dev/null || true
	@find output/logs -type f -delete 2>/dev/null || true
	@echo "$(GREEN)Output files cleaned!$(NC)"

clean-all: clean clean-outputs ## Xóa tất cả (bao gồm môi trường ảo, cache, outputs)
	@echo "$(YELLOW)Cleaning everything...$(NC)"
	@rm -rf output/
	@rm -rf data/processed/
	@echo "$(GREEN)Everything cleaned!$(NC)"
