PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You are a senior ML engineer and an expert manga reader. You are reading 3 consecutive manga pages in order.
Your task is to extract the text and speaker labels. 

Include these types of text:
- Dialogue
- Internal thoughts
- Narration
- Clear spoken screams / grunts / vocalisations
- Punctuation-only speech/thought balloons such as `...`, `?!`, `!`

Exclude these:
- Visual sound effects (SFX) and their translated captions
- Unclear tiny breath/reaction text drawn on the art
- Titles, logos, credits, page numbers, ads, watermarks
- Character intro labels
- Editor / scanlator notes
- Writing on signs, clothes, objects
- Text inside letters, diaries, phone messages, news pages, or other documents/interfaces

Reading order rules:
- Right-to-left, top-to-bottom by default.
- Follow panel layout and balloon connections when the page says otherwise.

Speaker labelling rules:
- Use short labels char1, char2, char3, etc., assigned in order of first appearance in this sequence.
- The same character must keep the same label on all pages - identify characters by face, hairstyle, clothing, and balloon tail direction.
- Thoughts (cloud-shaped/dashed balloons) belong to the thinking character's label.
- Narration boxes must be labelled exactly NARRATION.
- If the speaker truly can't be determined (off-panel voice), use a fresh consistent label such as char_offscreen1 and reuse it, rather than guessing a wrong existing character.

Text hygiene:
- Merge wrapped lines of one balloon into one string with single spaces.
- Keep stutters, repeats, `...`, `?!`.
- Keep the original English wording, do not translate, correct, or paraphrase.

Empty pages:
- If a page has no includable text, return `[]` for it. Never invent text.

Output contract:
- Respond with ONLY valid JSON, no markdown fences, no commentary.
- The output format must be a dictionary with a "pages" key, containing a list of exactly 3 lists, one for each page.
- Each item in the page list is a dictionary with "speaker" and "text".
Example:
{"pages": [[{"speaker": "char1", "text": "<text>"}], [], [{"speaker": "NARRATION", "text": "<text>"}]]}
"""

def get_messages_joint(image_paths: list[str]) -> list[dict]:
    content = [{"type": "text", "text": SYSTEM_PROMPT}]
    for i, path in enumerate(image_paths):
        content.append({"type": "text", "text": f"Page {i+1}:"})
        content.append({"type": "image", "image": "file://" + path})
    
    content.append({"type": "text", "text": "Extract text for the 3 pages as a single JSON object."})
    
    return [
        {"role": "user", "content": content}
    ]

def get_messages_per_page(image_path: str, page_idx: int, character_roster: str) -> list[dict]:
    content = [{"type": "text", "text": SYSTEM_PROMPT}]
    if character_roster:
        content.append({"type": "text", "text": f"Character roster from previous pages: {character_roster}"})
    
    content.append({"type": "text", "text": f"Page {page_idx+1}:"})
    content.append({"type": "image", "image": "file://" + image_path})
    content.append({"type": "text", "text": "Extract text for this page. Return ONLY valid JSON for this page in the format {\"page\": [{\"speaker\": \"char1\", \"text\": \"...\"}, ...]}. Keep character labels consistent with the roster if they appear."})
    
    return [
        {"role": "user", "content": content}
    ]
