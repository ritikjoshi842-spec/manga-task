import os
import json
import random
import argparse
from src.config import Config
from src.data import find_dataset_root

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()
    
    config = Config.from_yaml(args.config)
    root = find_dataset_root(config.data_root)
    labels_path = os.path.join(root, "development", "labels.jsonl")
    
    # 1. Reproduce the split to find the 16 validation IDs
    with open(labels_path, "r", encoding="utf-8") as f:
        all_labels = [json.loads(line) for line in f if line.strip()]
        
    all_labels.sort(key=lambda x: x["sequence_id"])
    rng = random.Random(config.seed)
    rng.shuffle(all_labels)
    
    val_labels = all_labels[64:]
    val_ids = {item["sequence_id"] for item in val_labels}
    
    # 2. Extract only validation labels and predictions
    val_refs = [lbl for lbl in val_labels if lbl["sequence_id"] in val_ids]
    
    with open(args.predictions, "r", encoding="utf-8") as f:
        all_preds = [json.loads(line) for line in f if line.strip()]
        
    val_preds = [pred for pred in all_preds if pred["sequence_id"] in val_ids]
    
    # 3. Save them to temporary files so the official score.py can run
    os.makedirs("scratch", exist_ok=True)
    with open("scratch/val_refs.jsonl", "w", encoding="utf-8") as f:
        for ref in val_refs:
            f.write(json.dumps(ref) + "\n")
            
    with open("scratch/val_preds.jsonl", "w", encoding="utf-8") as f:
        for pred in val_preds:
            f.write(json.dumps(pred) + "\n")
            
    # 4. Run official score.py
    score_script = os.path.join(root, "score.py")
    cmd = f'python "{score_script}" --references scratch/val_refs.jsonl --predictions scratch/val_preds.jsonl --output scratch/val_scores.json'
    print(f"Running: {cmd}")
    os.system(cmd)
    
    # Print the score
    with open("scratch/val_scores.json", "r") as f:
        scores = json.load(f)
    print("\n=== Validation Set Scores (16 Sequences) ===")
    print(json.dumps(scores, indent=2))

if __name__ == "__main__":
    main()
