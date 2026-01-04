"""
Visualize training data samples
Format: document, summary
"""

import os
import json
from transformers import AutoTokenizer

# ================= CONFIG =================
DATA_DIR = "./data/processed"
TARGET_INDEX = 10          # index cần kiểm tra (0-based)
MAX_PRINT_CHARS = 500
MODEL_NAME = "google/mt5-small"
# =========================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

jsonl_paths = []

# 🔍 tìm đệ quy tất cả file .jsonl
for root, _, files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith(".jsonl"):
            jsonl_paths.append(os.path.join(root, file))

print(f"📂 Found {len(jsonl_paths)} jsonl files")

printed_summary = False
saved_summary = None

for path in jsonl_paths:
    print("\n" + "=" * 80)
    print(f"📄 FILE: {path}")
    print("=" * 80)

    found = False
    total_lines = 0

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            total_lines += 1

            if i == TARGET_INDEX:
                sample = json.loads(line)

                # ========== KIỂM TRA SCHEMA ==========
                assert "document" in sample, "Missing 'document' field"
                assert "summary" in sample, "Missing 'summary' field"
                
                document_text = sample["document"]
                summary_text = sample["summary"]
                metadata = sample.get("metadata", {})

                print(f"\n===== DOCUMENT AT INDEX {TARGET_INDEX} =====")
                print("DOCUMENT (first 500 chars):")
                print(document_text[:MAX_PRINT_CHARS])
                if len(document_text) > MAX_PRINT_CHARS:
                    print("...")

                # Đếm token
                tokens = tokenizer(document_text, truncation=False)["input_ids"]
                print(f"\n📏 Token length (document): {len(tokens)}")
                
                # Đếm từ
                word_count = len(document_text.split())
                print(f"📏 Word count: {word_count}")
                print(f"📏 Character count: {len(document_text)}")

                # Hiển thị metadata nếu có
                if metadata:
                    print(f"\n--- Metadata ---")
                    print(f"Approach: {metadata.get('approach', 'N/A')}")
                    if 'filtered' in metadata:
                        print(f"Filtered: {metadata['filtered']}")
                    if 'num_sentences_original' in metadata:
                        print(f"Original sentences: {metadata['num_sentences_original']}")
                    if 'num_sentences_selected' in metadata:
                        print(f"Selected sentences: {metadata['num_sentences_selected']}")
                    if 'compression_ratio' in metadata:
                        print(f"Compression ratio: {metadata['compression_ratio']:.2%}")

                # Lưu summary để in sau (chỉ 1 lần)
                if not printed_summary:
                    saved_summary = summary_text
                    printed_summary = True

                found = True
                break

    if not found:
        print(
            f"⚠ File has only {total_lines} samples "
            f"(index {TARGET_INDEX} out of range)"
        )

# ================= PRINT SUMMARY ONCE =================
if saved_summary is not None:
    print("\n" + "=" * 80)
    print("🎯 SUMMARY (printed once)")
    print("=" * 80)
    print(saved_summary)
    
    # Đếm token và từ cho summary
    summary_tokens = tokenizer(saved_summary, truncation=False)["input_ids"]
    summary_words = len(saved_summary.split())
    print(f"\n📏 Summary token length: {len(summary_tokens)}")
    print(f"📏 Summary word count: {summary_words}")
    print(f"📏 Summary character count: {len(saved_summary)}")
else:
    print("\n⚠ No data was found at the given index in any file.")

print("\n" + "=" * 80)
print("✅ VISUALIZATION COMPLETE")
print("=" * 80)