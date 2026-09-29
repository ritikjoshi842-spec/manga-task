import yaml
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    experiment_name: str
    model_id: str
    data_root: Optional[str]
    split: str
    mode: str
    dtype: str
    quantization: str
    device_map: str
    attn_implementation: str
    min_pixels: int
    max_pixels: int
    max_new_tokens: int
    do_sample: bool
    repetition_penalty: float
    max_retries: int
    seed: int
    prompt_version: str
    output_dir: str
    adapter_path: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls(**data)
