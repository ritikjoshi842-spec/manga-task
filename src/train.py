import os
import json
import torch
import random
import argparse
from dataclasses import dataclass
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset

from src.config import Config
from src.data import load_sequences, find_dataset_root
from src.prompts import SYSTEM_PROMPT
from qwen_vl_utils import process_vision_info

def collate_fn(batch, processor):
    input_ids = [item["input_ids"].squeeze(0) for item in batch]
    attention_mask = [item["attention_mask"].squeeze(0) for item in batch]
    
    from torch.nn.utils.rnn import pad_sequence
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    
    batch_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
    
    pixel_values = [item["pixel_values"] for item in batch if item.get("pixel_values") is not None]
    if pixel_values:
        batch_dict["pixel_values"] = torch.cat(pixel_values, dim=0)
    
    image_grid_thw = [item["image_grid_thw"] for item in batch if item.get("image_grid_thw") is not None]
    if image_grid_thw:
        batch_dict["image_grid_thw"] = torch.cat(image_grid_thw, dim=0)
        
    return batch_dict


class MangaDataset(Dataset):
    def __init__(self, config, processor, split="train"):
        root = find_dataset_root(config.data_root)
        labels_path = os.path.join(root, "development", "labels.jsonl")
        
        with open(labels_path, "r", encoding="utf-8") as f:
            all_labels = [json.loads(line) for line in f if line.strip()]
            
        all_labels.sort(key=lambda x: x["sequence_id"])
        rng = random.Random(config.seed)
        rng.shuffle(all_labels)
        
        # 64 train, 16 val split
        self.labels = all_labels[:64] if split == "train" else all_labels[64:]
        
        dev_seqs = load_sequences(config, split="development")
        seq_map = {s.sequence_id: s.image_paths for s in dev_seqs}
        
        self.data = []
        for l in self.labels:
            seq_id = l["sequence_id"]
            if seq_id in seq_map:
                self.data.append({
                    "sequence_id": seq_id,
                    "images": seq_map[seq_id],
                    "target_json": json.dumps({"pages": l["pages"]}, separators=(',', ':'))
                })
                
        self.processor = processor

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": []}
        ]
        
        for i, path in enumerate(item["images"]):
            messages[1]["content"].append({"type": "text", "text": f"Page {i+1}:"})
            messages[1]["content"].append({"type": "image", "image": f"file://{path}"})
            
        messages[1]["content"].append({"type": "text", "text": "Extract text for the 3 pages as a single JSON object."})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": "```json\n" + item["target_json"] + "\n```"}]})
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt"
        )
        return inputs


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    
    config = Config.from_yaml(args.config)
    
    processor = AutoProcessor.from_pretrained(
        config.model_id, 
        min_pixels=config.min_pixels, 
        max_pixels=config.max_pixels
    )
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    train_ds = MangaDataset(config, processor, "train")
    val_ds = MangaDataset(config, processor, "val")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model_id,
        device_map="auto",
        quantization_config=bnb_config,
        attn_implementation=config.attn_implementation
    )
    
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.config.use_cache = False
    
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        eval_strategy="no",
        save_strategy="epoch",
        logging_steps=5,
        optim="paged_adamw_8bit",
        fp16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    
    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lambda x: collate_fn(x, processor),
    )
    
    trainer.train()
    
    adapter_path = os.path.join(config.output_dir, "final_adapter")
    trainer.save_model(adapter_path)
    print(f"Model saved to {adapter_path}")

if __name__ == "__main__":
    train()
