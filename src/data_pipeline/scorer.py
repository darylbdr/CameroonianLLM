"""
Filtrage Linguistique & Formule Mathématique de Scoring
======================================================================
Formule implémentée :
    S = I(N >= 6) * D_cam * P_matrice

Seuils définis pour ne pas pénaliser les phrases longues :
  - Zone Haute ('high') : S >= 0.06 (au moins 1 marqueur fort sur 30 mots ou 1 marqueur moyen sur 16 mots)
  - Zone Grise ('grey') : 0.03 <= S < 0.06
  - Zone Basse ('low')  : S < 0.03
"""

import re
import json
from pathlib import Path
from difflib import SequenceMatcher
from typing import List, Dict, Tuple


def load_lexicon(lexicon_path: Path) -> Dict:
    with open(lexicon_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_and_tokenize(text: str) -> Tuple[str, List[str]]:
    if not text:
        return "", []
    cleaned = str(text)
    cleaned = re.sub(r'http\S+|www\.\S+', '', cleaned)
    cleaned = re.sub(r'@\s*[^\s@]+', '', cleaned)
    cleaned = re.sub(r'#\S+', '', cleaned)
    cleaned = re.sub(r'(.)\1{2,}', r'\1\1', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    words = re.findall(r"[a-zA-ZÀ-ÿ0-9']+", cleaned.lower())
    return cleaned, words


_LEXICON_CACHE = {}


def _get_compiled_lexicon(lexicon: Dict):
    key = (
        id(lexicon),
        len(lexicon.get("strong_markers", ())),
        len(lexicon.get("medium_markers", ())),
        len(lexicon.get("french_anchors", ())),
        len(lexicon.get("pidgin_english_anchors", ())),
    )
    cached = _LEXICON_CACHE.get(key)
    if cached is not None and cached["ref"] is lexicon:
        return cached

    w_strong = lexicon.get("strong_markers_weight", 2.0)
    w_medium = lexicon.get("medium_markers_weight", 1.0)
    strong_set = set(m.strip().lower() for m in lexicon["strong_markers"])
    medium_set = set(m.strip().lower() for m in lexicon["medium_markers"])
    french_set = set(m.strip().lower() for m in lexicon["french_anchors"])
    en_pid_set = set(a.strip().lower() for a in lexicon.get("pidgin_english_anchors", ()))
    strong_roots = [r for r in strong_set if len(r) >= 4]
    medium_roots = [r for r in medium_set if len(r) >= 4]

    cached = {
        "ref": lexicon,
        "w_strong": w_strong,
        "w_medium": w_medium,
        "strong_set": strong_set,
        "medium_set": medium_set,
        "french_set": french_set,
        "en_pid_set": en_pid_set,
        "strong_roots": strong_roots,
        "medium_roots": medium_roots,
        "word_weights": {},
    }
    _LEXICON_CACHE[key] = cached
    return cached


def get_word_camfranglais_weight(word: str, lexicon: Dict) -> float:
    comp = _get_compiled_lexicon(lexicon)
    word_weights = comp["word_weights"]
    if word in word_weights:
        return word_weights[word]

    w_strong = comp["w_strong"]
    w_medium = comp["w_medium"]

    if word in comp["french_set"]:
        weight = 0.0
    elif word in comp["strong_set"]:
        weight = w_strong
    elif word in comp["medium_set"]:
        weight = w_medium
    else:
        weight = 0.0
        lw = len(word)
        if lw >= 4:
            for root in comp["strong_roots"]:
                lr = len(root)
                if root in word or (2 * min(lw, lr) >= 0.84 * (lw + lr) and SequenceMatcher(None, word, root).ratio() >= 0.84):
                    weight = w_strong
                    break
            if weight == 0.0:
                for root in comp["medium_roots"]:
                    lr = len(root)
                    if root in word or (2 * min(lw, lr) >= 0.84 * (lw + lr) and SequenceMatcher(None, word, root).ratio() >= 0.84):
                        weight = w_medium
                        break

    word_weights[word] = weight
    return weight


def compute_d_cam(words: List[str], lexicon: Dict) -> float:
    if not words:
        return 0.0
    total_weight = sum(get_word_camfranglais_weight(w, lexicon) for w in words)
    return total_weight / len(words)


def compute_p_matrice(words: List[str], lexicon: Dict, epsilon: float = 1e-5) -> float:
    comp = _get_compiled_lexicon(lexicon)
    fr_anchors = comp["french_set"]
    en_pid_anchors = comp["en_pid_set"]

    c_fr = sum(1 for w in words if w in fr_anchors)
    c_en_pid = sum(1 for w in words if w in en_pid_anchors)

    return c_fr / (c_fr + c_en_pid + epsilon)


def score_camfranglais(
    text: str,
    lexicon: Dict,
    min_words: int = 6,
    high_threshold: float = 0.06,
    grey_threshold: float = 0.03
) -> Dict:
    cleaned_text, words = normalize_and_tokenize(text)
    n_words = len(words)
    d_cam = compute_d_cam(words, lexicon)
    p_matrice = compute_p_matrice(words, lexicon)
    length_indicator = 1.0 if n_words >= min_words else 0.0
    score = length_indicator * d_cam * p_matrice

    if score >= high_threshold:
        zone = "high"
    elif score >= grey_threshold:
        zone = "grey"
    else:
        zone = "low"

    return {
        "cleaned_text": cleaned_text,
        "N": n_words,
        "d_cam": round(d_cam, 4),
        "p_matrice": round(p_matrice, 4),
        "score": round(score, 4),
        "zone": zone
    }


def filter_sft_conversations(conversations_list: List[Dict], lexicon: Dict, threshold: float = 0.06) -> List[Dict]:
    valid_conversations = []

    for item in conversations_list:
        conv = item.get("conversations", [])
        if len(conv) < 2:
            continue
        user_msg = conv[0].get("content", "")
        assistant_msg = conv[1].get("content", "")

        clean_user, user_words = normalize_and_tokenize(user_msg)
        res_assistant = score_camfranglais(assistant_msg, lexicon, high_threshold=threshold)

        if len(user_words) >= 3 and res_assistant["score"] >= threshold:
            valid_conversations.append({
                "conversations": [
                    {"role": "user", "content": clean_user},
                    {"role": "assistant", "content": res_assistant["cleaned_text"]}
                ]
            })

    return valid_conversations
