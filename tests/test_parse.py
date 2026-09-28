from src.parse import parse_model_output, normalize_parsed_json

def test_parse_clean():
    out = parse_model_output('```json\n{"pages": [[{"speaker": "A", "text": "B"}]]}\n```')
    assert out == {"pages": [[{"speaker": "A", "text": "B"}]]}

def test_parse_trailing_comma():
    out = parse_model_output('{"pages": [[{"speaker": "A", "text": "B"},]],}')
    assert out == {"pages": [[{"speaker": "A", "text": "B"}]]}

def test_normalize():
    parsed = {"page1": [{"speaker": "A", "text": "B"}], "page2": []}
    pages = normalize_parsed_json(parsed)
    assert len(pages) == 2
