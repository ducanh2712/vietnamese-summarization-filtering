import re

def remove_pos_tags(text: str) -> str:
    """
    Remove POS-like tags (N, V, A, ...) from annotated Vietnamese text.
    """
    if not isinstance(text, str):
        return text

    # Remove standalone capital-letter tags
    cleaned = re.sub(r"\b[A-Z]\b", "", text)

    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned

def clean_dataset(dataset):
    def _clean_batch(batch):
        return {
            "document": [remove_pos_tags(x) for x in batch["document"]],
            "summary":  [remove_pos_tags(x) for x in batch["summary"]],
        }

    return dataset.map(
        _clean_batch,
        batched=True,
        desc="Removing POS tags"
    )
