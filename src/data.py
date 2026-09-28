import os
import json
import glob
from dataclasses import dataclass
from typing import List

from .utils import setup_logger

logger = setup_logger("data")

@dataclass
class Sequence:
    sequence_id: str
    image_paths: List[str]

def find_dataset_root(config_data_root: str = None) -> str:
    if config_data_root and os.path.exists(config_data_root):
        return config_data_root
    
    if "MANGA_DATA_ROOT" in os.environ:
        env_root = os.environ["MANGA_DATA_ROOT"]
        if os.path.exists(env_root):
            return env_root

    # Scan /kaggle/input/*/
    kaggle_paths = glob.glob("/kaggle/input/*/")
    for path in kaggle_paths:
        if os.path.exists(os.path.join(path, "sequences.json")):
            return path
            
    # Maybe local dataset folder
    if os.path.exists("dataset/sequences.json"):
        return "dataset"
        
    raise FileNotFoundError("Could not auto-discover dataset root. Searched config, env var MANGA_DATA_ROOT, and /kaggle/input/*/")

def find_image_path(root: str, filename: str) -> str:
    # Search for the image path recursively if needed
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f == filename or f == os.path.basename(filename):
                return os.path.join(dirpath, f)
    # fallback
    return os.path.join(root, filename)

def load_sequences(config, split="test") -> List[Sequence]:
    root = find_dataset_root(config.data_root)
    logger.info(f"Using dataset root: {root}")
    
    seq_file = os.path.join(root, "sequences.json")
    with open(seq_file, 'r', encoding='utf-8') as f:
        raw_seqs = json.load(f)
        
    logger.info(f"Loaded sequences.json type: {type(raw_seqs)}")
    
    # normalize to dict
    seq_dict = {}
    if isinstance(raw_seqs, list):
        for item in raw_seqs:
            # guess the structure
            if "sequence_id" in item:
                seq_dict[item["sequence_id"]] = item.get("images", item.get("image_paths", []))
            elif len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], list): # e.g. [id, [imgs]]
                seq_dict[item[0]] = item[1]
    elif isinstance(raw_seqs, dict):
        seq_dict = raw_seqs
        
    # print first 2
    for i, (k, v) in enumerate(seq_dict.items()):
        if i < 2:
            logger.info(f"Sample sequence {k}: {v}")
            
    # find split IDs
    target_ids = []
    if split == "test":
        sample_sub = os.path.join(root, "sample_submission.jsonl")
        if not os.path.exists(sample_sub):
            logger.warning(f"sample_submission.jsonl not found at {sample_sub}. Processing all in sequences.json")
            target_ids = list(seq_dict.keys())
        else:
            with open(sample_sub, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        target_ids.append(obj["sequence_id"])
    elif split == "development":
        labels_file = os.path.join(root, "development", "labels.jsonl")
        if os.path.exists(labels_file):
            with open(labels_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        target_ids.append(obj["sequence_id"])
        else:
            logger.warning("development labels.jsonl not found.")
            target_ids = list(seq_dict.keys())
            
    logger.info(f"Target split '{split}' has {len(target_ids)} sequences.")
    
    sequences = []
    for seq_id in target_ids:
        if seq_id not in seq_dict:
            logger.warning(f"Sequence {seq_id} not found in sequences.json")
            continue
            
        imgs = seq_dict[seq_id]
        if isinstance(imgs, dict): # maybe dict of {"images": [...]}
            imgs = imgs.get("images", imgs.get("image_paths", []))
            
        if len(imgs) != 3:
            logger.warning(f"Sequence {seq_id} has {len(imgs)} images instead of 3. Pad/truncating.")
            imgs = (imgs + [imgs[-1]]*3)[:3] if imgs else []
            if not imgs: continue
            
        abs_paths = [find_image_path(root, img) for img in imgs]
        sequences.append(Sequence(sequence_id=seq_id, image_paths=abs_paths))
        
    return sequences
