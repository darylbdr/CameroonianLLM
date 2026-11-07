"""
Module 2 : Filtrage Linguistique & Formule Mathématique de Scoring (S)
======================================================================
Module Python natif complet correspondant au notebook :
`02_filtering_scoring/02_Filtering_and_Scoring.ipynb`

Formule implémentée :
    S = I(N >= 6) * D_cam * P_matrice
"""

import re
import json
from pathlib import Path
from difflib import SequenceMatcher
from typing import List, Dict, Tuple


def load_lexicon(lexicon_path: Path) -> Dict:
    """Charge le fichier config/lexicon.json."""
    with open(lexicon_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_and_tokenize(text: str) -> Tuple[str, List[str]]:
    """
    Nettoie le texte brut et retourne (cleaned_text, words_alphabetiques_minuscules).
    """
    if not text:
        return "", []
    cleaned = str(text)
    cleaned = re.sub(r'http\S+|www\.\S+', '', cleaned)
    cleaned = re.sub(r'@\s*[^\s@]+', '', cleaned)
    cleaned = re.sub(r'#\S+', '', cleaned)
    cleaned = re.sub(r'(.)\1{2,}', r'\1\1', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    words = re.findall(r"[a-zA-ZÀ-ÿ']+", cleaned.lower())
    return cleaned, words


def get_word_camfranglais_weight(word: str, lexicon: Dict) -> float:
    """Retourne le poids Camfranglais d'un mot (2.0 fort, 1.0 moyen, 0.0 sinon) avec tolérance phonétique."""
    w_strong = lexicon.get("strong_markers_weight", 2.0)
    w_medium = lexicon.get("medium_markers_weight", 1.0)
    strong_set = set(m.strip().lower() for m in lexicon["strong_markers"])
    medium_set = set(m.strip().lower() for m in lexicon["medium_markers"])
    french_set = set(m.strip().lower() for m in lexicon["french_anchors"])

    if word in french_set:
        return 0.0
    if word in strong_set:
        return w_strong
    if word in medium_set:
        return w_medium

    if len(word) >= 4:
        for root in strong_set:
            if len(root) >= 4 and (root in word or SequenceMatcher(None, word, root).ratio() >= 0.84):
                return w_strong
        for root in medium_set:
            if len(root) >= 4 and (root in word or SequenceMatcher(None, word, root).ratio() >= 0.84):
                return w_medium
    return 0.0


def compute_d_cam(words: List[str], lexicon: Dict) -> float:
    """
    Calcule la densité lexicale pondérée D_cam = (somme des w_i) / N.
    """
    if not words:
        return 0.0
    total_weight = sum(get_word_camfranglais_weight(w, lexicon) for w in words)
    return total_weight / len(words)


def compute_p_matrice(words: List[str], lexicon: Dict, epsilon: float = 1e-5) -> float:
    """
    Calcule P_matrice = C_fr / (C_fr + C_en_pid + epsilon).
    """
    fr_anchors = set(a.strip().lower() for a in lexicon["french_anchors"])
    en_pid_anchors = set(a.strip().lower() for a in lexicon["pidgin_english_anchors"])

    c_fr = sum(1 for w in words if w in fr_anchors)
    c_en_pid = sum(1 for w in words if w in en_pid_anchors)

    return c_fr / (c_fr + c_en_pid + epsilon)


def score_camfranglais(text: str, lexicon: Dict, min_words: int = 6) -> Dict:
    """
    Calcule S = I(N >= min_words) * D_cam * P_matrice et route dans ('high', 'grey', 'low').
    """
    cleaned_text, words = normalize_and_tokenize(text)
    n_words = len(words)
    d_cam = compute_d_cam(words, lexicon)
    p_matrice = compute_p_matrice(words, lexicon)
    length_indicator = 1.0 if n_words >= min_words else 0.0
    score = length_indicator * d_cam * p_matrice

    if score >= 0.15:
        zone = "high"
    elif score >= 0.05:
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


def filter_sft_conversations(conversations_list: List[Dict], lexicon: Dict, threshold: float = 0.15) -> List[Dict]:
    """
    Applique la Règle d'Or SFT (seul l'assistant doit avoir un score >= threshold).
    """
    valid_conversations = []

    for item in conversations_list:
        conv = item.get("conversations", [])
        if len(conv) < 2:
            continue
        user_msg = conv[0].get("content", "")
        assistant_msg = conv[1].get("content", "")

        clean_user, user_words = normalize_and_tokenize(user_msg)
        res_assistant = score_camfranglais(assistant_msg, lexicon)

        if len(user_words) >= 3 and res_assistant["score"] >= threshold:
            valid_conversations.append({
                "conversations": [
                    {"role": "user", "content": clean_user},
                    {"role": "assistant", "content": res_assistant["cleaned_text"]}
                ]
            })

    return valid_conversations
