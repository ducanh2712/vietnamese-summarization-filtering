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

printed_target = False
saved_target = None

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

                # kiểm tra schema
                assert "input" in sample, "Missing 'input'"
                assert "target" in sample, "Missing 'target'"

                input_text = sample["input"]
                target_text = sample["target"]

                print(f"\n===== INPUT AT INDEX {TARGET_INDEX} =====")
                print("INPUT (first 500 chars):")
                print(input_text[:MAX_PRINT_CHARS])

                tokens = tokenizer(input_text, truncation=False)["input_ids"]
                print(f"\nToken length (input): {len(tokens)}")

                # lưu target để in sau (chỉ 1 lần)
                if not printed_target:
                    saved_target = target_text
                    printed_target = True

                found = True
                break

    if not found:
        print(
            f"⚠ File has only {total_lines} samples "
            f"(index {TARGET_INDEX} out of range)"
        )

# ================= PRINT TARGET ONCE =================
if saved_target is not None:
    print("\n" + "=" * 80)
    print("🎯 TARGET (printed once)")
    print("=" * 80)
    print(saved_target)
else:
    print("\n⚠ No target was found at the given index in any file.")
