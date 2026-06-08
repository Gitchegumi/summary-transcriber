from __future__ import annotations

import re
import string
from collections import defaultdict

from src.config.manifest import SessionManifest


STOPWORDS = {
    "all", "alright", "and", "are", "but", "can", "cause", "cool", "did", "does",
    "exactly", "for", "from", "god", "good", "hey", "how", "huh", "nice", "nope",
    "not", "now", "ooh", "she", "sorry", "thank", "two", "wait", "was", "wow",
    "yeah", "yep", "yes", "you", "your", "you're", "i'm", "i'll", "i've", "he's",
    "it's", "let's", "we're", "what's", "that's", "there's", "mm-hmm", "about",
    "after", "again", "also", "because", "before", "being", "could", "every",
    "first", "going", "great", "have", "just", "like", "little", "maybe", "okay",
    "really", "right", "something", "the", "there", "their", "then", "they",
    "thing", "think", "this", "that", "well", "what", "when", "where", "which",
    "will", "with", "would", "you", "your", "our", "us", "them", "him", "her",
    "their", "his", "my", "mine", "ours", "yours", "hers", "its", "whose", "each",
    "either", "neither", "both", "any", "some", "many", "few", "several", "anyone",
    "anybody", "anything", "someone", "somebody", "something", "everyone",
    "everybody", "everything", "noone", "nobody", "nothing", "who", "whom",
    "whose", "which", "that", "what", "whatever", "whichever", "whoever",
    "whomever", "am", "is", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "shall", "will", "should", "would", "may", "might",
    "must", "can", "could", "a", "an", "the", "in", "on", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "of", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "only", "own", "same", "so", "than", "too", "very", "s", "t",
    "can", "will", "just", "don", "should", "shouldn", "now", "d", "ll", "m", "o",
    "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn",
    "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn",
    "weren", "won", "wouldn", "say", "says", "said", "go", "goes", "went", "gone",
    "get", "gets", "got", "gotten", "make", "makes", "made", "take", "takes",
    "took", "taken", "see", "sees", "saw", "seen", "come", "comes", "came", "back",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"
}

FILLER_WORDS = {
    "uh", "um", "ah", "like", "so", "well", "basically", "actually", "literally",
    "honestly", "mean", "you know", "kind of", "sort of", "alright", "okay",
    "hey", "huh", "oh", "ooh", "wow", "wait", "yeah", "yep", "yes", "nope",
    "mm-hmm", "mhm", "uh-huh", "sure", "cool", "fine", "sorry"
}

PROFANITY = {
    "fuck", "fucking", "shit", "damn", "god damn", "goddamn", "asshole", "bitch", "crap"
}

REAL_PEOPLE = {
    "craig", "giacomo", "james", "john", "brandon", "matt", "travis", "laura",
    "liam", "sam", "taliesin", "marisha", "ashley", "brian", "mercer", "gemini",
    "antigravity", "whisper", "faster-whisper", "whisperx", "nemo", "parakeet",
    "canary", "player", "dungeon master", "dm", "gm", "d&d", "dnd"
}

CONTEXT_CLUES = {
    "npc", "familiar", "spell", "city", "location", "faction", "item", "god",
    "deity", "patron", "cult", "ritual", "caravan", "library", "tower", "temple",
    "weapon", "armor", "potion", "map", "quest", "scroll", "artifact", "relic",
    "king", "queen", "lord", "lady", "prince", "princess", "duke", "baron",
    "village", "town", "castle", "fort", "keep", "dungeon", "cave", "forest",
    "mountain", "river", "lake", "sea", "ocean", "island", "continent", "world",
    "pantheon", "cleric", "paladin", "wizard", "sorcerer", "warlock", "druid",
    "ranger", "rogue", "fighter", "bard", "monk", "barbarian"
}


def build_glossary_candidates(
    manifest: SessionManifest,
    turns: list[dict],
    words: list[dict],
    corrections: list[dict],
) -> tuple[list[dict], list[dict]]:
    known_terms = {term.lower() for term in manifest.glossary.all_terms()}
    
    occurrences = defaultdict(lambda: {
        "term_variants": defaultdict(int),
        "example_turns": {},
        "confidence_values": [],
        "candidate_types": set(),
        "is_correction": False,
        "suggested_buckets": [],
        "sentence_start_count": 0,
        "total_occ": 0
    })

    # 1. Collect low-confidence words
    for word in words:
        confidence = word.get("confidence")
        word_text = word.get("word", "")
        term = _clean_term(word_text)
        if (
            confidence is None
            or confidence >= 0.75
            or not term
            or term.lower() in known_terms
        ):
            continue
            
        term_lower = term.lower()
        turn = _turn_by_id(turns, word.get("turn_id", ""))
        
        start_count, total_count = 0, 0
        if turn:
            start_count, total_count = check_sentence_starts(turn.get("text_cleaned", ""), term)
            occurrences[term_lower]["example_turns"][turn["turn_id"]] = turn
            
        occurrences[term_lower]["term_variants"][term] += max(1, total_count)
        occurrences[term_lower]["confidence_values"].append(confidence)
        occurrences[term_lower]["candidate_types"].add("low_confidence_term")
        occurrences[term_lower]["sentence_start_count"] += start_count
        occurrences[term_lower]["total_occ"] += max(1, total_count)

    # 2. Collect capitalized terms
    for turn in turns:
        text = turn.get("text_cleaned", "")
        for term in _capitalized_terms(text):
            if term.lower() in known_terms or not term:
                continue
            term_lower = term.lower()
            start_count, total_count = check_sentence_starts(text, term)
            
            occurrences[term_lower]["term_variants"][term] += max(1, total_count)
            occurrences[term_lower]["candidate_types"].add("possible_name_or_place")
            occurrences[term_lower]["example_turns"][turn["turn_id"]] = turn
            occurrences[term_lower]["sentence_start_count"] += start_count
            occurrences[term_lower]["total_occ"] += max(1, total_count)

    # 3. Collect corrections
    for correction in corrections:
        original, corrected = _parse_replacement(correction.get("reason", ""))
        if not original or original.lower() in known_terms:
            continue
        term = original
        term_lower = term.lower()
        turn = _turn_by_id(turns, correction.get("turn_id", ""))
        
        occurrences[term_lower]["term_variants"][term] += 1
        occurrences[term_lower]["candidate_types"].add("correction_source_phrase")
        occurrences[term_lower]["is_correction"] = True
        
        bucket = _bucket_for_term(corrected, manifest.glossary)
        occurrences[term_lower]["suggested_buckets"].append(bucket)
        
        if turn:
            occurrences[term_lower]["example_turns"][turn["turn_id"]] = turn
            start_count, _ = check_sentence_starts(turn.get("text_cleaned", ""), term)
            occurrences[term_lower]["sentence_start_count"] += start_count
        occurrences[term_lower]["total_occ"] += 1

    # Extract filter config
    filters = manifest.glossary_candidate_filters
    speaker_names = set()
    for sp in manifest.speakers:
        if sp.display_name:
            speaker_names.add(sp.display_name.lower())
        if sp.speaker_id:
            speaker_names.add(sp.speaker_id.lower())

    candidates = []
    noise_candidates = []

    for term_lower, info in occurrences.items():
        variants = info["term_variants"]
        canonical_term = max(variants.keys(), key=lambda k: (variants[k], k[0].isupper()))
        
        example_turns_list = list(info["example_turns"].values())
        example_texts = [t["text_cleaned"] for t in example_turns_list]
        turn_ids = [t["turn_id"] for t in example_turns_list]
        speaker_ids = {t["speaker_id"] for t in example_turns_list if t.get("speaker_id")}
        
        occurrence_count = info["total_occ"]
        is_correction = info["is_correction"]
        
        suggested_bucket = "custom_terms"
        if info["suggested_buckets"]:
            suggested_bucket = max(set(info["suggested_buckets"]), key=info["suggested_buckets"].count)
        else:
            suggested_bucket = _bucket_for_term(canonical_term, manifest.glossary)
            
        score, positive_signals, negative_signals = score_candidate(
            term=canonical_term,
            occurrence_count=occurrence_count,
            speaker_ids=speaker_ids,
            turn_ids=turn_ids,
            example_texts=example_texts,
            sentence_start_count=info["sentence_start_count"],
            confidence_values=info["confidence_values"],
            is_correction=is_correction,
            speaker_names=speaker_names,
            manifest=manifest,
        )
        
        candidate_type = get_classification(
            term=canonical_term,
            example_texts=example_texts,
            is_correction=is_correction,
            suggested_bucket=suggested_bucket,
            speaker_names=speaker_names,
            manifest=manifest,
        )
        
        # Determine if filtered
        filter_reason = None
        
        if len(canonical_term) < 3 and canonical_term.lower() not in {"dm", "gm", "pc", "npc"}:
            filter_reason = "Length less than 3 characters"
        elif is_entirely_stopword_or_filler(canonical_term):
            filter_reason = "Entirely stopword, filler, contraction, or profanity"
        elif has_repeated_tokens(canonical_term):
            filter_reason = "Contains repeated tokens or filler"
        elif is_numeric_or_number_word(canonical_term):
            filter_reason = "Numeric or number-word only"
        elif canonical_term.lower() in {t.lower() for t in filters.custom_noise_terms}:
            filter_reason = "Custom noise term specified in manifest"
        elif not filters.include_real_people and candidate_type == "possible_real_person":
            filter_reason = "Real person name excluded by filter configuration"
        elif occurrence_count < filters.min_occurrences and not is_correction:
            filter_reason = f"Occurrence count {occurrence_count} less than minimum {filters.min_occurrences}"
        elif score < filters.min_score and not is_correction:
            filter_reason = f"Score {score} less than minimum score {filters.min_score}"
        elif filters.filter_sentence_starters and info["sentence_start_count"] == occurrence_count and not is_correction:
            filter_reason = "Only appears at sentence starts"

        example_turn = example_turns_list[0] if example_turns_list else None
        
        cand_dict = {
            "session_id": manifest.session.id,
            "candidate_id": "",
            "term": canonical_term,
            "normalized_term": term_lower,
            "candidate_type": candidate_type,
            "suggested_glossary_bucket": suggested_bucket,
            "score": score,
            "positive_signals": ", ".join(positive_signals),
            "negative_signals": ", ".join(negative_signals),
            "reason": f"Correction term mapping to '{suggested_bucket}'" if is_correction else f"Score {score} candidate classified as {candidate_type}.",
            "example_text": example_turn.get("text_cleaned", "") if example_turn else "",
            "speaker_id": example_turn.get("speaker_id") if example_turn else None,
            "turn_id": example_turn.get("turn_id") if example_turn else None,
            "start_seconds": example_turn.get("start_seconds") if example_turn else None,
            "end_seconds": example_turn.get("end_seconds") if example_turn else None,
            "confidence": min(info["confidence_values"]) if info["confidence_values"] else None,
            "occurrence_count": occurrence_count,
            "speaker_count": len(speaker_ids),
            "example_turn_ids": ", ".join(turn_ids),
        }
        
        if filter_reason:
            cand_dict["reason_filtered"] = filter_reason
            cand_dict["candidate_type"] = "probably_noise"
            noise_candidates.append(cand_dict)
        else:
            candidates.append(cand_dict)

    # Sort final lists
    candidates.sort(key=lambda row: (row["candidate_type"], -row["score"], row["term"].lower()))
    noise_candidates.sort(key=lambda row: (row["reason_filtered"], row["term"].lower()))
    
    for index, row in enumerate(candidates, start=1):
        row["candidate_id"] = f"{manifest.session.id}-glossary-candidate-{index:05d}"
        
    return candidates, noise_candidates


def score_candidate(
    term: str,
    occurrence_count: int,
    speaker_ids: set[str],
    turn_ids: list[str],
    example_texts: list[str],
    sentence_start_count: int,
    confidence_values: list[float],
    is_correction: bool,
    speaker_names: set[str],
    manifest: SessionManifest,
) -> tuple[int, list[str], list[str]]:
    score = 0
    positive_signals = []
    negative_signals = []
    
    term_lower = term.lower()
    
    if is_correction:
        score += 5
        positive_signals.append("correction_source_phrase (+5)")
        
    if occurrence_count >= 2:
        score += 2
        positive_signals.append(f"repeated_term_occurrences_{occurrence_count} (+2)")
        
    mid_sentence_cap = occurrence_count - sentence_start_count
    if mid_sentence_cap > 0:
        score += 3
        positive_signals.append(f"mid_sentence_capitalized_count_{mid_sentence_cap} (+3)")
        
    found_clues = []
    context = " ".join(example_texts).lower()
    for clue in CONTEXT_CLUES:
        if clue in context:
            found_clues.append(clue)
    if found_clues:
        score += 2
        positive_signals.append(f"near_context_clues_{len(found_clues)} (+2)")
        
    known_terms = {t.lower() for t in manifest.glossary.all_terms()}
    resembles = False
    for kt in known_terms:
        if kt in term_lower or term_lower in kt:
            resembles = True
            break
    if resembles:
        score += 2
        positive_signals.append("resembles_existing_glossary_term (+2)")
        
    if len(turn_ids) > 1:
        score += 2
        positive_signals.append(f"appears_in_multiple_turns_{len(turn_ids)} (+2)")
        
    if len(speaker_ids) > 1:
        score += 2
        positive_signals.append(f"appears_from_multiple_speakers_{len(speaker_ids)} (+2)")
        
    if " " in term:
        score += 2
        positive_signals.append("multi_word_noun_phrase (+2)")
        
    fantasy_chars = {"z", "x", "q", "v"}
    has_fantasy_char = any(c in term_lower for c in fantasy_chars)
    ends_fantasy = term_lower[-1] in {"x", "z"} if term_lower else False
    if has_fantasy_char or ends_fantasy:
        score += 2
        positive_signals.append("unusual_fantasy_name_shape (+2)")
        
    low_conf_count = sum(1 for c in confidence_values if c < 0.75)
    if low_conf_count > 0:
        score += 2
        positive_signals.append(f"low_confidence_variants_count_{low_conf_count} (+2)")
        
    if term_lower in STOPWORDS:
        score -= 3
        negative_signals.append("common_english_word (-3)")
        
    if term_lower in FILLER_WORDS:
        score -= 5
        negative_signals.append("filler_word (-5)")
        
    if term_lower in PROFANITY:
        score -= 5
        negative_signals.append("profanity (-5)")
        
    if sentence_start_count == occurrence_count and occurrence_count > 0:
        score -= 4
        negative_signals.append("only_appears_at_sentence_starts (-4)")
        
    if occurrence_count > 10 and not found_clues:
        score -= 3
        negative_signals.append("frequent_table_chatter (-3)")
        
    acknowledgements = {"yeah", "yep", "yes", "okay", "alright", "sure", "nope"}
    if term_lower in acknowledgements:
        score -= 5
        negative_signals.append("acknowledgement_word (-5)")
        
    if term_lower in speaker_names or term_lower in REAL_PEOPLE:
        score -= 5
        negative_signals.append("real_person_or_table_participant_name (-5)")
        
    short_utterance = False
    for txt in example_texts:
        if len(txt.split()) <= 3:
            short_utterance = True
            break
    if short_utterance:
        score -= 2
        negative_signals.append("appears_in_short_utterances (-2)")
        
    return score, positive_signals, negative_signals


def get_classification(
    term: str,
    example_texts: list[str],
    is_correction: bool,
    suggested_bucket: str,
    speaker_names: set[str],
    manifest: SessionManifest,
) -> str:
    term_lower = term.lower()
    
    if term_lower in speaker_names or term_lower in REAL_PEOPLE:
        return "possible_real_person"
        
    known_terms = {t.lower() for t in manifest.glossary.all_terms()}
    if term_lower in known_terms:
        return "possible_alias"
        
    rule_keywords = {"spell", "rule", "cantrip", "guardians", "blast", "bolt", "silence", "save", "check", "attack", "roll", "action", "bonus"}
    if suggested_bucket == "rules_terms" or any(w in term_lower for w in rule_keywords):
        return "possible_spell_or_rule"
        
    context = " ".join(example_texts).lower()
    
    npc_titles = {"lady", "lord", "sir", "king", "queen", "father", "brother", "sister", "baron", "duke", "captain", "general"}
    words_in_term = term_lower.split()
    if words_in_term and words_in_term[0] in npc_titles:
        return "possible_npc"
    npc_keywords = {"npc", "familiar", "summon", "meet", "spoke to", "said", "talked", "tells", "told", "person", "character"}
    if any(w in context for w in npc_keywords):
        return "possible_npc"
        
    loc_prefixes = {"mount", "lake", "river", "fort", "castle", "keep", "tower", "temple", "forest", "mountain"}
    if words_in_term and words_in_term[0] in loc_prefixes:
        return "possible_location"
    loc_keywords = {"city", "town", "village", "location", "temple", "tower", "castle", "keep", "forest", "mountain", "river", "lake", "inn", "tavern", "shop", "place", "here", "there", "travel", "go to", "visit", "arrive"}
    if any(w in context for w in loc_keywords):
        return "possible_location"
        
    item_keywords = {"item", "weapon", "armor", "potion", "scroll", "ring", "sword", "shield", "amulet", "key", "map", "journal", "letter", "bag", "chest", "gold", "loot"}
    if any(w in context for w in item_keywords):
        return "possible_item"
        
    return "possible_campaign_entity"


def is_entirely_stopword_or_filler(term: str) -> bool:
    tokens = [t.strip(string.punctuation + "“”‘’").lower() for t in term.split()]
    return all(t in STOPWORDS or t in FILLER_WORDS or t in PROFANITY for t in tokens if t)


def is_numeric_or_number_word(term: str) -> bool:
    normalized = term.lower().strip()
    if normalized.isdigit():
        return True
    number_words = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                    "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
                    "eighty", "ninety", "hundred", "thousand", "million", "billion"}
    tokens = [t.strip(string.punctuation + "“”‘’") for t in normalized.split()]
    return all(t in number_words or t.isdigit() for t in tokens if t)


def has_repeated_tokens(term: str) -> bool:
    tokens = [t.strip(string.punctuation + "“”‘’").lower() for t in term.split()]
    for i in range(len(tokens) - 1):
        if tokens[i] == tokens[i+1]:
            return True
    return False


def is_at_sentence_start(text: str, match_start: int) -> bool:
    before = text[:match_start].rstrip()
    if not before:
        return True
    return before[-1] in {".", "?", "!"}


def check_sentence_starts(text: str, term: str) -> tuple[int, int]:
    term_lower = term.lower()
    text_lower = text.lower()
    
    start_count = 0
    total_count = 0
    
    pos = text_lower.find(term_lower)
    while pos != -1:
        total_count += 1
        if is_at_sentence_start(text, pos):
            start_count += 1
        pos = text_lower.find(term_lower, pos + len(term_lower))
        
    return start_count, total_count


def _clean_term(value: str) -> str:
    return value.strip().strip(string.punctuation + "“”‘’").strip()


def _capitalized_terms(text: str) -> list[str]:
    matches = re.findall(r"\b[A-Z][A-Za-z'’-]{2,}(?:\s+[A-Z][A-Za-z'’-]{2,}){0,2}\b", text)
    return [_clean_term(match) for match in matches]


def _parse_replacement(reason: str) -> tuple[str | None, str | None]:
    match = re.search(r":\s*(.+?)\s*->\s*(.+)$", reason)
    if not match:
        return None, None
    return match.group(1).strip(), match.group(2).strip()


def _turn_by_id(turns: list[dict], turn_id: str) -> dict | None:
    return next((turn for turn in turns if turn["turn_id"] == turn_id), None)


def _bucket_for_term(term: str, glossary: GlossaryConfig) -> str:
    term_lower = term.lower()
    if glossary and hasattr(glossary, "entries"):
        for bucket, entries in glossary.entries.items():
            for entry in entries:
                if entry.canonical.lower() == term_lower or any(alias.lower() == term_lower for alias in entry.aliases):
                    return bucket
    if "save" in term_lower or "check" in term_lower:
        return "rules_terms"
    return "custom_terms"
