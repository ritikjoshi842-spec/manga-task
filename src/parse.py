import json
import re

def parse_model_output(text: str) -> dict:
    # Extract JSON from markdown fenced code block
    text = text.strip()
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        text = match.group(1).strip()
        
    # find outermost {}
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        text = text[start:end+1]
        
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
        
    # Try light repair
    # trailing commas
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
        
    return {}

def normalize_parsed_json(parsed: dict) -> list:
    if "pages" in parsed and isinstance(parsed["pages"], list):
        return parsed["pages"]
    # Alternative shapes
    # top level list of 3 lists
    if isinstance(parsed, list):
        if len(parsed) == 3 and all(isinstance(x, list) for x in parsed):
            return parsed
            
    # {"page1": [...], "page2": [...], "page3": [...]}
    pages = []
    for key in sorted(parsed.keys()):
        if "page" in key.lower() and isinstance(parsed[key], list):
            pages.append(parsed[key])
    if pages:
        return pages
        
    return []
