import os
import json
import tempfile
from types import SimpleNamespace
from src.data import load_sequences

def test_load_sequences():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SimpleNamespace(data_root=tmpdir)
        
        for i in range(3):
            with open(os.path.join(tmpdir, f"img{i}.jpg"), "w") as f:
                f.write("dummy")
                
        seq_list = [
            {"sequence_id": "seq1", "images": ["img0.jpg", "img1.jpg", "img2.jpg"]}
        ]
        with open(os.path.join(tmpdir, "sequences.json"), "w") as f:
            json.dump(seq_list, f)
            
        with open(os.path.join(tmpdir, "sample_submission.jsonl"), "w") as f:
            f.write(json.dumps({"sequence_id": "seq1", "pages": []}) + "\n")
            
        seqs = load_sequences(config, split="test")
        assert len(seqs) == 1
        assert seqs[0].sequence_id == "seq1"
