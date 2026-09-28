import re

def clean_text(text: str) -> str:
    # Collapse internal whitespace/newlines to single spaces; strip
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def normalize_speaker(speaker: str) -> str:
    if not speaker:
        return "char_unknown"
        
    s = speaker.strip()
    if s.lower() in ["narration", "narrator"]:
        return "NARRATION"
        
    s = s.lower().replace(" ", "_")
    return s

def postprocess_pages(pages: list) -> list:
    # Ensure exactly 3 pages
    if not isinstance(pages, list):
        pages = []
        
    while len(pages) < 3:
        pages.append([])
    if len(pages) > 3:
        pages = pages[:3]
        
    normalized_pages = []
    speaker_map = {}
    char_counter = 1
    
    for page in pages:
        norm_page = []
        if not isinstance(page, list):
            normalized_pages.append([])
            continue
            
        prev_speaker = None
        prev_text = None
        
        for item in page:
            if not isinstance(item, dict):
                continue
            
            speaker = item.get("speaker", "")
            text = item.get("text", "")
            
            if not isinstance(speaker, str) or not isinstance(text, str):
                continue
                
            text = clean_text(text)
            if not text:
                continue
                
            speaker = normalize_speaker(speaker)
            
            # Map to char1..charN
            if speaker != "NARRATION":
                if speaker not in speaker_map:
                    speaker_map[speaker] = f"char{char_counter}"
                    char_counter += 1
                speaker = speaker_map[speaker]
                
            # Deduplicate accidental consecutive exact-duplicate items
            if len(text) > 3 and speaker == prev_speaker and text == prev_text:
                continue
                
            norm_page.append({"speaker": speaker, "text": text})
            prev_speaker = speaker
            prev_text = text
            
        normalized_pages.append(norm_page)
        
    return normalized_pages
