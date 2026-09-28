from src.postprocess import postprocess_pages

def test_always_3_pages():
    assert len(postprocess_pages([])) == 3
    assert len(postprocess_pages([[], [], [], []])) == 3

def test_label_remapping():
    pages = [
        [{"speaker": "Bob", "text": "Hi"}],
        [{"speaker": "Alice", "text": "Hello"}, {"speaker": "Bob", "text": "Sup"}],
        []
    ]
    res = postprocess_pages(pages)
    assert res[0][0]["speaker"] == "char1"
    assert res[1][0]["speaker"] == "char2"
    assert res[1][1]["speaker"] == "char1"

def test_narration_and_punctuation():
    pages = [
        [{"speaker": "narrator", "text": "Once"}, {"speaker": "Bob", "text": "..."}],
    ]
    res = postprocess_pages(pages)
    assert res[0][0]["speaker"] == "NARRATION"
    assert res[0][1]["text"] == "..."
