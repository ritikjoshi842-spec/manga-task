import gc
from typing import List, Dict, Any

from .utils import setup_logger

logger = setup_logger("model")

class MockModel:
    def __init__(self, config):
        self.config = config
        logger.info("Initialized MockModel for dry-run")
        
    def generate(self, messages: List[Dict[str, Any]], max_pixels_override: int = None) -> str:
        # Return a fixed valid JSON string
        return '```json\n{"pages": [[{"speaker": "char1", "text": "Mock text"}], [], []]}\n```'
        
    def free_memory(self):
        pass

class QwenModel:
    def __init__(self, config):
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        
        self.config = config
        self.processor = AutoProcessor.from_pretrained(
            config.model_id, 
            min_pixels=config.min_pixels, 
            max_pixels=config.max_pixels
        )
        
        dtype = torch.float16 if config.dtype == "float16" else torch.float32
        
        model_kwargs = {
            "torch_dtype": dtype,
            "device_map": config.device_map,
            "attn_implementation": config.attn_implementation,
        }
        
        if config.quantization == "4bit":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs["quantization_config"] = quantization_config
            
        logger.info(f"Loading {config.model_id} with args: {model_kwargs}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.model_id,
            **model_kwargs
        )
        
        if getattr(config, "adapter_path", None):
            from peft import PeftModel
            logger.info(f"Loading LoRA adapter from {config.adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, config.adapter_path)
            
        self.model.eval()
        
    def generate(self, messages: List[Dict[str, Any]], max_pixels_override: int = None) -> str:
        from qwen_vl_utils import process_vision_info
        import torch
        
        processor = self.processor
        if max_pixels_override:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(
                self.config.model_id,
                min_pixels=self.config.min_pixels,
                max_pixels=max_pixels_override
            )
            
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        
        gen_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "repetition_penalty": self.config.repetition_penalty,
        }
        
        if self.config.do_sample:
            gen_kwargs["temperature"] = getattr(self.config, "temperature", 0.7)
            
        generated_ids = self.model.generate(**inputs, **gen_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]
        
    def free_memory(self):
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
