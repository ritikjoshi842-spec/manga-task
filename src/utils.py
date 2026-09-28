import json
import logging
import random
from typing import Any, Dict

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def set_seed(seed: int):
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def write_jsonl_append(path: str, data: Dict[str, Any]):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data) + '\n')
