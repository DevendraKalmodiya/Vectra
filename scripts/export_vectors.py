import json
import csv
import numpy as np
from pathlib import Path
import config


def export_to_json(output_file: str = "data/vectors_document.json", limit: int = None):
    """Exports vectors and document texts into a JSON document."""
    vector_path = config.VECTOR_STORE_PATH
    text_path = config.TEXT_STORE_PATH

    if not vector_path.exists() or not text_path.exists():
        print("❌ Error: Vector or text storage files not found in data/ directory.")
        return

    print("Loading vectors and document texts...")
    vectors = np.load(vector_path)
    with open(text_path, "r", encoding="utf-8") as f:
        texts = json.load(f)

    total = len(vectors)
    count = limit if limit else total

    print(f"Processing {count:,} of {total:,} document vectors...")

    document_data = []
    for vid in range(count):
        document_data.append({
            "vector_id": vid,
            "text": texts[vid] if vid < len(texts) else f"Vector ID #{vid}",
            "vector_dimensions": len(vectors[vid]),
            "vector": vectors[vid].tolist()  # Converts NumPy array to Python list
        })

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(document_data, f, indent=2)

    print(f"✓ Successfully saved document vectors to '{out_path}' ({out_path.stat().st_size / (1024 * 1024):.2f} MB)")


def export_to_csv(output_file: str = "data/vectors_document.csv", limit: int = 1000):
    """Exports vectors and document texts into a CSV spreadsheet."""
    vector_path = config.VECTOR_STORE_PATH
    text_path = config.TEXT_STORE_PATH

    if not vector_path.exists() or not text_path.exists():
        print("❌ Error: Files missing.")
        return

    vectors = np.load(vector_path)
    with open(text_path, "r", encoding="utf-8") as f:
        texts = json.load(f)

    count = min(limit, len(vectors)) if limit else len(vectors)
    out_path = Path(output_file)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Vector_ID", "Document_Text", "Vector_Embedding"])
        
        for vid in range(count):
            # Formats embedding vector as a compact string
            vec_str = "[" + ", ".join([f"{val:.4f}" for val in vectors[vid]]) + "]"
            writer.writerow([vid, texts[vid], vec_str])

    print(f"✓ Successfully exported CSV to '{out_path}'")


if __name__ == "__main__":
    # 1. Export all 50,000 document vectors to a single JSON file
    export_to_json(output_file="data/vectors_document.json")

    # 2. Optionally export a sample to CSV for viewing in Excel/Spreadsheets
    export_to_csv(output_file="data/vectors_sample.csv", limit=1000)