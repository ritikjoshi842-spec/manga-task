import argparse
import os
import time
import json
from tqdm import tqdm

from src.config import Config
from src.data import load_sequences
from src.utils import setup_logger, set_seed, write_jsonl_append
from src.prompts import get_messages_joint, get_messages_per_page
from src.parse import parse_model_output, normalize_parsed_json
from src.postprocess import postprocess_pages

logger = setup_logger("run_inference")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if args.split:
        config.split = args.split

    set_seed(config.seed)
    
    os.makedirs(config.output_dir, exist_ok=True)
    preds_file = os.path.join(config.output_dir, "predictions.jsonl")
    raw_file = os.path.join(config.output_dir, "raw_outputs.jsonl")
    log_file = os.path.join(config.output_dir, "run_log.txt")

    completed_ids = set()
    if args.resume and os.path.exists(preds_file):
        with open(preds_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    completed_ids.add(json.loads(line)["sequence_id"])
                    
    with open(log_file, "a") as f:
        f.write(f"Starting run with config: {config}\n")
        
    sequences = load_sequences(config, split=config.split)
    if args.limit:
        sequences = sequences[:args.limit]
        
    if args.dry_run:
        from src.model import MockModel
        model = MockModel(config)
    else:
        from src.model import QwenModel
        model = QwenModel(config)
        
    for seq in tqdm(sequences, desc="Processing sequences"):
        if seq.sequence_id in completed_ids:
            continue
            
        start_time = time.time()
        final_pages = []
        raw_outputs_data = {
            "sequence_id": seq.sequence_id,
            "mode": config.mode,
            "retries": 0,
            "parse_status": "success",
            "raw_texts": []
        }
        
        def attempt_generation(messages, max_pixels_override=None):
            for retry in range(config.max_retries + 1):
                try:
                    raw_text = model.generate(messages, max_pixels_override=max_pixels_override)
                    raw_outputs_data["raw_texts"].append(raw_text)
                    parsed = parse_model_output(raw_text)
                    if parsed:
                        pages = normalize_parsed_json(parsed)
                        if pages:
                            return pages
                            
                    # parsing failed
                    raw_outputs_data["retries"] += 1
                    messages.append({"role": "assistant", "content": [{"type": "text", "text": raw_text}]})
                    messages.append({"role": "user", "content": [{"type": "text", "text": "Your last answer was not valid JSON. Return ONLY the JSON object."}]})
                except Exception as e:
                    try:
                        import torch
                        if "CUDA out of memory" in str(e) or isinstance(e, torch.cuda.OutOfMemoryError):
                            raise e
                    except ImportError:
                        pass
                    logger.error(f"Generation error: {e}")
                    break
            raw_outputs_data["parse_status"] = "failed"
            return None

        try:
            if config.mode == "joint":
                messages = get_messages_joint(seq.image_paths)
                pages = attempt_generation(messages)
                if pages:
                    final_pages = pages
            elif config.mode == "per_page":
                raise ValueError("Force per_page")
                
        except Exception as e:
            is_oom = False
            try:
                import torch
                if isinstance(e, torch.cuda.OutOfMemoryError) or "CUDA out of memory" in str(e):
                    is_oom = True
            except ImportError:
                pass
                
            if is_oom:
                logger.warning(f"OOM on {seq.sequence_id}, retrying with reduced max_pixels")
                model.free_memory()
                try:
                    messages = get_messages_joint(seq.image_paths)
                    pages = attempt_generation(messages, max_pixels_override=int(config.max_pixels * 0.7))
                    if pages:
                        final_pages = pages
                except Exception as e2:
                    logger.warning(f"OOM again on {seq.sequence_id}, falling back to per_page")
                    model.free_memory()
                    config.mode = "per_page"
                    raw_outputs_data["mode"] = "per_page_fallback"
            
            if not final_pages and config.mode == "per_page" or "per_page" in raw_outputs_data["mode"]:
                char_roster = ""
                for i, img_path in enumerate(seq.image_paths):
                    messages = get_messages_per_page(img_path, i, char_roster)
                    try:
                        page_res = attempt_generation(messages)
                        if page_res and len(page_res) > 0:
                            final_pages.append(page_res[0])
                            char_roster += json.dumps(page_res[0]) + " "
                        else:
                            final_pages.append([])
                    except Exception as e3:
                        logger.error(f"OOM on single page {i} of {seq.sequence_id}")
                        final_pages.append([])
                        model.free_memory()

        if not final_pages:
            final_pages = [[], [], []]
            
        final_pages = postprocess_pages(final_pages)
        
        raw_outputs_data["time"] = time.time() - start_time
        write_jsonl_append(raw_file, raw_outputs_data)
        
        pred_data = {"sequence_id": seq.sequence_id, "pages": final_pages}
        write_jsonl_append(preds_file, pred_data)
        
        model.free_memory()

    # Validation summary
    total = 0
    empty_pages = 0
    with open(preds_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                total += 1
                for p in obj["pages"]:
                    if not p: empty_pages += 1
                    
    logger.info(f"Done. {total} sequences. {empty_pages} empty pages.")
    
    if os.path.exists("/kaggle/working/"):
        os.system(f"cp {preds_file} /kaggle/working/submission.jsonl")

if __name__ == "__main__":
    main()
