# Cells to paste into Kaggle for execution

## Cell 1 (Clone repo & install requirements)
```bash
!git clone <YOUR_REPO_URL> /kaggle/working/manga-task
%cd /kaggle/working/manga-task
!pip install -r requirements.txt
```

## Cell 2 (Run Inference)
```bash
!python -m src.run_inference --config configs/exp1_zeroshot.yaml --split test
```

## Cell 3 (Show Output Head)
```python
!head -n 2 outputs/exp1_zeroshot/predictions.jsonl
```
