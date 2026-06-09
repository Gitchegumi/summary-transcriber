from __future__ import annotations

import re


def clean_filler_words(text: str) -> str:
    if not text:
        return ""
        
    # 1. Pure fillers: um, uh, ah, er, hmm, mm-hmm
    pure_fillers = r"\b(um|uh|ah|er|hmm|mm-hmm)\b"
    text = re.sub(pure_fillers, "", text, flags=re.IGNORECASE)
    
    # 2. Phrasal fillers: you know
    text = re.sub(r"\b(you know)\b", "", text, flags=re.IGNORECASE)
    
    # 3. Conservative "like" and "I mean" only when set off by commas
    text = re.sub(r",\s*like\s*,", ", ", text, flags=re.IGNORECASE)
    text = re.sub(r",\s*I mean\s*,", ", ", text, flags=re.IGNORECASE)
    
    # Start of text followed by comma (e.g. "Like, we went..." -> "We went...")
    text = re.sub(r"^\b(like|I mean|yeah|yep|okay)\b\s*,\s*", "", text, flags=re.IGNORECASE)
    
    # 4. Accidental repeated starts (phrase starts: 1-word or 2-word starts)
    for _ in range(2):
        text = re.sub(r"\b(\w+(?:'\w+)?\s+\w+(?:'\w+)?)\s+\1\b", r"\1", text, flags=re.IGNORECASE)
    for _ in range(3):
        text = re.sub(r"\b(\w+(?:'\w+)?)\s+\1\b", r"\1", text, flags=re.IGNORECASE)
        
    # Clean up extra spaces and stray double punctuation
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s*\.\s*", ". ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    return text
