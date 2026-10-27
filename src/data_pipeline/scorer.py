"""
Module 2 : Filtrage Linguistique & Formule Mathématique de Scoring (S)
======================================================================
Ce fichier est le module Python natif correspondant au notebook :
`02_filtering_scoring/02_Filtering_and_Scoring.ipynb`

Formule implémentée :
    S = I(N >= 6) * D_cam * P_matrice

Une fois tes fonctions validées dans le Notebook 02, reporte-les ici pour qu'elles
soient utilisables par `builder.py` (Étape 3) et `evaluator.py` (Étape 5) !
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
    TODO 1 (depuis le Notebook 02) :
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

    words = []
    # TODO 1 : Reporter ici l'extraction des mots alphabétiques validée dans le Notebook 02
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
    TODO 2 (depuis le Notebook 02) :
    Calcule la densité lexicale pondérée D_cam = (somme des w_i) / N.
    """
    if not words:
        return 0.0
    # TODO 2 : Reporter ton code validé dans le Notebook 02
    return 0.0


def compute_p_matrice(words: List[str], lexicon: Dict, epsilon: float = 1e-5) -> float:
    """
    TODO 3.1 (depuis le Notebook 02) :
    Calcule P_matrice = C_fr / (C_fr + C_en_pid + epsilon).
    """
    # TODO 3.1 : Reporter ton code validé dans le Notebook 02
    return 0.0


def score_camfranglais(text: str, lexicon: Dict, min_words: int = 6) -> Dict:
    """
    TODO 3.2 (depuis le Notebook 02) :
    Calcule S = I(N >= min_words) * D_cam * P_matrice et route dans ('high', 'grey', 'low').
    """
    cleaned_text, words = normalize_and_tokenize(text)
    n_words = len(words)
    d_cam = 0.0
    p_matrice = 0.0
    score = 0.0
    zone = "low"
    # TODO 3.2 : Reporter ton code validé dans le Notebook 02
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
    TODO 4 (depuis le Notebook 02) :
    Applique la Règle d'Or SFT (seul l'assistant doit avoir un score >= threshold).
    """
    valid_conversations = []
    # TODO 4 : Reporter ton code validé dans le Notebook 02
    return valid_conversations
