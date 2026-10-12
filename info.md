# Manga Task — Agent Build Brief (Experiment 1: Zero-Shot Baseline)

> **How to use this file:** Give this whole README to an AI coding agent as its task prompt. The agent must write all the code into a local repo folder. **Nothing is run locally.** The code will later be pushed to GitHub, cloned inside a Kaggle notebook (GPU), and executed there to produce a submission JSONL.

---

## 1. Your role

You are a senior ML engineer. Write a clean, robust, well-commented Python project that runs **Qwen/Qwen2.5-VL-7B-Instruct zero-shot** (no fine-tuning) on manga page sequences and outputs a JSONL file in the exact required format.

Because the code cannot be tested on the real GPU by you, it must be **defensive**: handle bad model output, out-of-memory errors, missing files, and resume after a crash. Include a CPU-only `--dry-run` mode with a mock model so the pipeline logic can be verified without a GPU.

**Do NOT** in this step: push to GitHub, set up Kaggle, fine-tune anything, or use the training/development data. Those come in later instructions.

---

## 2. The problem (task spec)

Build a system that reads **three consecutive English manga pages**, extracts the **story text in reading order**, and says **who spoke each line**.

- The same character must keep the **same label across all three pages** of a sequence. Real names are not needed: `char1`, `char2`, `tomato2`, etc. are fine. Labels may restart for the next sequence.
- Narrator text uses the exact label `NARRATION`.
- Thoughts belong to the **thinking character** (not NARRATION).

### 2.1 What counts as text (INCLUDE)
- Dialogue
- Internal thoughts
- Narration
- Clear spoken screams / grunts / vocalisations
- Punctuation-only speech/thought balloons such as `...`, `?!`, `!`

### 2.2 What to EXCLUDE
- Visual sound effects (SFX) and their translated captions
- Unclear tiny breath/reaction text drawn on the art
- Titles, logos, credits, page numbers, ads, watermarks
- Character intro labels
- Editor / scanlator notes
- Writing on signs, clothes, objects
- Text inside letters, diaries, phone messages, news pages, or other documents/interfaces

### 2.3 Reading order & text hygiene
- Usually **right-to-left, top-to-bottom** manga order, but follow panel layout and balloon connections when the page says otherwise.
- Keep stutters, repeats, and meaningful punctuation (`W-wait...`, `NO!!`).
- Do **not** keep simple printed line-wrapping (join wrapped lines of one balloon into one string with single spaces).
- Correctly empty pages should be `[]`. **Inventing text on an empty page is penalised.**

### 2.4 Output format (STRICT)
One JSON object per line (JSONL), one per sequence:

```json
{"sequence_id":"seq_001","pages":[
  [{"speaker":"char1","text":"Where are you going?"},{"speaker":"char2","text":"Home."}],
  [{"speaker":"char1","text":"Wait for me!"}],
  []
]}
```

(The example above is pretty-printed for readability; the real file must have **one object per line**.)

- `pages` always has **exactly 3 lists**, in the order of the sequence's images.
- Each list is in reading order. Empty list = no included text.
- Each item: `{"speaker": <str>, "text": <str>}`.

### 2.5 Scoring (for context; used later)
- `text_order_score` — how well text is recovered in reading order.
- `balanced_joint_f1` — how well each speaker's text is recovered and identity kept consistent.
- Scorer (later use): `python dataset/score.py --references dataset/development/labels.jsonl --predictions preds.jsonl --output scores.json`

---

## 3. Hard rules (must be respected by the code)

1. **Open-weight models only.** Use `Qwen/Qwen2.5-VL-7B-Instruct` loaded locally through Hugging Face `transformers`. **No hosted inference APIs** (no OpenAI/Anthropic/Gemini/etc. calls anywhere in the code).
2. Test predictions must be **generated automatically** from the supplied images. No manual editing, no hard-coded answers.
3. For this experiment: **zero-shot baseline only** (the task says a direct unmodified model call is a valid baseline). Design the code so later experiments (few-shot prompting, LoRA fine-tuning, post-processing tweaks) can be added **without rewriting the pipeline**.
4. Everything should be reproducible: fixed seed, greedy decoding (`do_sample=False`), all settings in a config file.

---

## 4. Runtime environment (assume this)

- **Where it runs:** Kaggle notebook, GPU accelerator (assume **2× T4 16 GB** or **1× P100 16 GB**), internet ON for downloading the model from Hugging Face.
- **Where code lives:** a GitHub repo cloned into `/kaggle/working/<repo>`.
- **Where data lives:** attached as a Kaggle dataset, typically under `/kaggle/input/<dataset-name>/`. Do **not** hardcode a single path — see §6.
- **Where output goes:** `/kaggle/working/outputs/`.
- **GPU constraints to respect:**
  - T4/P100 have **no bfloat16** → use `torch.float16`.
  - **No FlashAttention-2** on T4 → use `attn_implementation="sdpa"`.
  - 7B in fp16 ≈ 15–16 GB → must shard across 2 GPUs with `device_map="auto"`, or fall back to **4-bit (bitsandbytes NF4)** on a single GPU. Implement both, selected via config, with an automatic fallback to 4-bit on `torch.cuda.OutOfMemoryError`.
  - Manga pages are tall/large. Control visual token count with `min_pixels` / `max_pixels` in the processor (configurable; sensible default around `max_pixels = 1024*28*28` per page — tunable, because text must stay legible).

---

## 5. Repo structure to create

```
manga-task/
├── README.md                     # (short human README; this brief can live at docs/AGENT_BRIEF.md)
├── requirements.txt
├── configs/
│   └── exp1_zeroshot.yaml
├── src/
│   ├── __init__.py
│   ├── config.py                 # load/validate YAML → dataclass
│   ├── data.py                   # locate dataset, read sequences.json, resolve image paths
│   ├── prompts.py                # system + user prompt templates (versioned)
│   ├── model.py                  # Qwen2.5-VL loader + generate() wrapper (+ MockModel for dry-run)
│   ├── parse.py                  # robust JSON extraction/repair from model text
│   ├── postprocess.py            # normalise speakers/text, enforce schema
│   ├── run_inference.py          # main CLI entry point
│   └── utils.py                  # logging, seeding, timing, jsonl helpers
├── tests/
│   ├── test_parse.py
│   ├── test_postprocess.py
│   └── test_data.py              # uses tiny fake dataset built in tmp dir
├── notebooks/
│   └── kaggle_run.md             # plain-text list of the cells to paste into Kaggle (no execution needed now)
├── docs/
│   └── EXPERIMENTS.md            # experiment log template (see §11)
└── outputs/                      # gitignored
```

Also add a `.gitignore` (outputs, `__pycache__`, `.ipynb_checkpoints`, model caches).

---

## 6. Data handling (`data.py`)

Dataset layout (as described by the task; **verify defensively, don't assume**):

```
dataset/
├── sequences.json                # tells which three images belong together
├── sample_submission.jsonl       # 15 test sequence IDs, empty page lists
├── score.py
├── development/  (80 labelled sequences + labels.jsonl)   # NOT used in this step
└── test/         (15 unlabelled sequences)                # used in this step
```

Requirements:
- **Auto-discover the dataset root**: check `config.data_root`, then env var `MANGA_DATA_ROOT`, then scan `/kaggle/input/*/` for a folder containing `sequences.json`. Fail with a clear, actionable error listing what was searched.
- **`sequences.json` schema is unknown** (could be a dict keyed by sequence_id, a list of objects, with image filenames/paths). Write a tolerant loader that handles both shapes, resolves image paths relative to the dataset root (searching subfolders if needed), and **prints the detected structure and first 2 entries** to the log so it's easy to debug on Kaggle.
- Provide `load_sequences(split="test")` returning a list of `Sequence(sequence_id, image_paths[3])`. Split membership should come from `sample_submission.jsonl` IDs for `test` (so test = exactly the IDs in the sample submission), and later from `labels.jsonl` IDs for `development`.
- Preserve the **image order as given** in `sequences.json`. Assert exactly 3 images per sequence (warn, don't crash, if not; pad/truncate gracefully and log it).
- Output ordering must match `sample_submission.jsonl` exactly.
- Note: the user mentioned "16 panels" for testing; the task says **15 unlabelled test sequences**. Process **whatever the test split contains**, and log the count.

---

## 7. Model wrapper (`model.py`)

- Load with `Qwen2_5_VLForConditionalGeneration.from_pretrained` + `AutoProcessor.from_pretrained` (use `qwen_vl_utils.process_vision_info` for building inputs).
- Config-driven: `dtype`, `quantization: none|4bit`, `device_map`, `attn_implementation`, `min_pixels`, `max_pixels`, `max_new_tokens` (default 2048), `temperature=0 / do_sample=False`, `repetition_penalty` (default 1.05).
- `generate(messages) -> str` returns raw decoded text only (strip the prompt tokens).
- Free GPU memory between sequences (`torch.cuda.empty_cache()`), and catch `OutOfMemoryError`: retry once with reduced `max_pixels` (e.g. ×0.7), then fall back to per-page mode (§8.2), then to empty pages with an error logged.
- Implement `MockModel` with the same interface that returns a fixed valid JSON string, used by `--dry-run` (no torch/transformers import needed on that path).

---

## 8. Inference strategy (`run_inference.py`, `prompts.py`)

### 8.1 Default mode: `joint` (one call per sequence)
Send **all three pages in a single multi-image message**, labelled "Page 1", "Page 2", "Page 3", so the model can keep character identities consistent across pages. Ask for a single JSON output.

### 8.2 Fallback mode: `per_page` (config: `mode: per_page`, or automatic fallback)
Process pages one at a time; carry forward a **character roster** (label + one-line visual description the model produced for earlier pages) in the prompt of later pages so labels stay consistent. Merge into the 3-page output.

### 8.3 Prompt design (`prompts.py`)
Store prompts as versioned constants (`PROMPT_VERSION = "v1"`) so experiments are traceable. The system + user prompt must contain, in clear language:

1. **Role & goal:** you are reading 3 consecutive manga pages in order.
2. **Include / exclude rules** exactly as in §2.1–§2.2 (copy them in verbatim as bullet lists).
3. **Reading order rules:** right-to-left, top-to-bottom by default; follow panels and balloon connections.
4. **Speaker labelling rules:**
   - Use short labels `char1`, `char2`, `char3`, … assigned in order of first appearance in this sequence.
   - **The same character must keep the same label on all pages** — identify characters by face, hairstyle, clothing, and balloon tail direction.
   - Thoughts (cloud-shaped/dashed balloons) → the thinking character's label.
   - Narration boxes → `NARRATION`.
   - If the speaker truly can't be determined (off-panel voice), use a fresh consistent label such as `char_offscreen1` and reuse it, rather than guessing a wrong existing character.
5. **Text hygiene:** merge wrapped lines of one balloon into one string; keep stutters, repeats, `...`, `?!`; keep the original English wording, do not translate, correct, or paraphrase.
6. **Empty pages:** if a page has no includable text, return `[]` for it. Never invent text.
7. **Output contract:** respond with **only** valid JSON, no markdown fences, no commentary:

```json
{"pages":[[{"speaker":"char1","text":"..."}],[...],[...]]}
```

Add 1 tiny **abstract** illustrative example of the format (with placeholder text like `"<text>"`), not real manga content.

### 8.4 Robust retry logic
- If parsing fails → retry up to `max_retries` (default 2) with an appended instruction: "Your last answer was not valid JSON. Return ONLY the JSON object."
- After exhausting retries → use the best-effort partial parse if any, else 3 empty pages. Always log the failure and keep the raw text.

---

## 9. Parsing & post-processing (`parse.py`, `postprocess.py`)

**parse.py**
- Extract JSON from raw text: strip ``` fences, find the outermost `{...}`, try `json.loads`; on failure attempt light repair (trailing commas, single quotes, unterminated brackets by truncation to the last complete item).
- Accept alternative shapes the model might return (e.g. a top-level list of 3 lists, or `{"page1": [...], ...}`) and normalise to `pages: list[list[dict]]`.

**postprocess.py**
- Ensure exactly 3 pages; pad with `[]` / truncate extras.
- Each item must have string `speaker` and string `text`; drop items with empty/whitespace text (but **keep** punctuation-only text like `...`, `?!`).
- Collapse internal whitespace/newlines to single spaces; strip.
- Normalise speaker labels: strip, lowercase (except keep the exact string `NARRATION` for any case variant like "narration"/"Narrator" → `NARRATION`); replace spaces with underscores; then **re-map to consistent `char1..charN`** in order of first appearance across the three pages (so labels are tidy and consistent).
- Deduplicate accidental consecutive exact-duplicate items (only if identical speaker+text on the same page back-to-back **and** the text is longer than 3 chars — do not remove legitimate repeats like `No! No!` inside one string).
- Never fabricate content; post-processing only cleans.

---

## 10. CLI, outputs & resilience

```
python -m src.run_inference --config configs/exp1_zeroshot.yaml [--split test] [--dry-run] [--limit N] [--resume]
```

- **Outputs** (in `outputs/exp1_zeroshot/`):
  - `predictions.jsonl` — final submission file (`sequence_id`, `pages`), ordered like `sample_submission.jsonl`. Also copy to `/kaggle/working/submission.jsonl` if that directory exists.
  - `raw_outputs.jsonl` — per sequence: raw model text, number of retries, mode used, timings, parse status.
  - `run_log.txt` — config snapshot, package versions, GPU info, prompt version, git commit hash (if available), per-sequence status.
- **Write incrementally** (append + flush after each sequence) and support `--resume` to skip completed IDs.
- If a sequence fails for any reason, emit 3 empty pages for it and continue — the final file must always contain **all** sequence IDs.
- At the end, **validate** the final JSONL against the schema (all IDs present, 3 pages each, correct types) and print a summary (num sequences, num lines extracted, num empty pages, num parse failures).
- Show a `tqdm` progress bar and time per sequence.

### Config (`configs/exp1_zeroshot.yaml`) — provide with comments
```yaml
experiment_name: exp1_zeroshot
model_id: Qwen/Qwen2.5-VL-7B-Instruct
data_root: null            # auto-discover
split: test
mode: joint                # joint | per_page
dtype: float16
quantization: none         # none | 4bit
device_map: auto
attn_implementation: sdpa
min_pixels: 200704         # 256*28*28
max_pixels: 802816         # 1024*28*28
max_new_tokens: 2048
do_sample: false
repetition_penalty: 1.05
max_retries: 2
seed: 42
prompt_version: v1
output_dir: outputs/exp1_zeroshot
```

---

## 11. Documentation & honesty requirements

The task explicitly asks for **clear attribution of my own contribution**. So:
- Create `docs/EXPERIMENTS.md` with a template per experiment: *hypothesis, config, prompt version, what changed, result/observations, next step*. Pre-fill **Exp 1** as "zero-shot Qwen2.5-VL-7B baseline, prompt v1, no training".
- Add a `CREDITS` section in the human README: Qwen2.5-VL (Alibaba Qwen team, Hugging Face), `transformers`, `qwen-vl-utils`, `bitsandbytes`, and note that code was written with LLM assistance and orchestrated/decided by the author.
- Leave a placeholder section "My contributions" for the author to fill in.

---

## 12. Dependencies (`requirements.txt`)

Pin loosely and note Kaggle-specific caveats in comments:
```
transformers>=4.49.0
accelerate>=0.34.0
qwen-vl-utils>=0.0.8
bitsandbytes>=0.43.0
pillow
pyyaml
tqdm
```
(`torch` is preinstalled on Kaggle — do not pin/reinstall it. Do **not** require `flash-attn`.)

---

## 13. Tests & dry-run (must be included, must pass on CPU)

- `test_parse.py`: fenced JSON, trailing commas, extra prose around JSON, alternative shapes, truncated output.
- `test_postprocess.py`: label remapping across pages, `NARRATION` normalisation, punctuation-only text kept, empty page stays `[]`, always exactly 3 pages.
- `test_data.py`: builds a fake dataset in a temp dir with both possible `sequences.json` shapes.
- `--dry-run`: uses `MockModel`, runs the full pipeline on the (fake or real) sequences, and produces a schema-valid `predictions.jsonl`.

---

## 14. Acceptance checklist

- [ ] One command produces `predictions.jsonl` with all test sequence IDs, exactly 3 pages each, correct schema.
- [ ] Zero-shot Qwen2.5-VL-7B path implemented (fp16 sharded, 4-bit fallback, OOM handling).
- [ ] No hosted APIs anywhere; greedy decoding; fixed seed.
- [ ] Prompt encodes all include/exclude/reading-order/speaker rules; versioned.
- [ ] Robust JSON parsing, retries, and safe fallbacks; never crashes the whole run on one bad sequence.
- [ ] Incremental writes + `--resume`.
- [ ] Config-driven; easy to add few-shot / LoRA in later experiments (keep model loading and prompt building modular; leave a clearly marked `# EXTENSION POINT` for adapters and few-shot examples).
- [ ] Dry-run + unit tests pass on CPU.
- [ ] `docs/EXPERIMENTS.md` and credits section present.
- [ ] `notebooks/kaggle_run.md` lists the cells (clone repo → `pip install -r requirements.txt` → run command → show output head). **Do not** execute or push anything.

---

## 15. Explicitly out of scope right now

- Pushing to GitHub or configuring Kaggle (will be instructed later).
- Using the 80 development sequences (later: evaluation, few-shot, fine-tuning).
- Any manual correction of predictions.
- Panel/bounding-box extraction (optional internal step for future experiments only).

When done, output a concise summary of files created, key design decisions, and any assumptions (especially about `sequences.json` structure) that I should verify on the first Kaggle run.
