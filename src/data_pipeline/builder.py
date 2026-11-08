"""
Construction des Datasets (CPT & SFT), Rétro-Instruction & Push Hugging Face
"""

import re
import random
from typing import List, Dict, Tuple, Any

INTERROGATIVE_MARKERS = {
    "pourquoi", "comment", "qui", "quand", "où", "ou",
    "quel", "quelle", "quels", "quelles", "combien", "wanda"
}


def is_good_sft_dialogue_pair(
    user_text: str,
    assistant_text: str,
    assistant_score: float,
    min_score: float = 0.06,
    min_assistant_words: int = 6
) -> bool:
    user_words = re.findall(r"[a-zA-ZÀ-ÿ0-9']+", user_text.lower())
    assistant_words = re.findall(r"[a-zA-ZÀ-ÿ0-9']+", assistant_text.lower())

    if assistant_score < min_score or len(assistant_words) < min_assistant_words:
        return False

    has_question = ("?" in user_text) or any(w in INTERROGATIVE_MARKERS for w in user_words)
    is_long_enough = len(user_words) >= 5

    return has_question or is_long_enough


def build_backtranslation_prompt(camfranglais_response: str) -> str:
    return (
        "Voici une réponse écrite en camfranglais par un jeune Camerounais :\n"
        f"\"{camfranglais_response.strip()}\"\n\n"
        "Génère uniquement le message court d'un utilisateur (en français standard ou en camfranglais) "
        "qui aurait naturellement provoqué cette réponse dans une discussion entre amis, sans guillemets ni préambule."
    )


def create_sft_entry(user_message: str, assistant_response: str, source: str = "synthetic_backtranslation") -> Dict:
    return {
        "conversations": [
            {"role": "user", "content": user_message.strip()},
            {"role": "assistant", "content": assistant_response.strip()}
        ],
        "source": source
    }


def deduplicate_and_split_sft(
    sft_entries: List[Dict],
    test_ratio: float = 0.1,
    seed: int = 3407
) -> Tuple[List[Dict], List[Dict]]:
    unique_entries = []
    seen_responses = set()

    for entry in sft_entries:
        conv = entry.get("conversations", [])
        if len(conv) < 2:
            continue
        resp_key = conv[1].get("content", "").strip().lower()
        if resp_key and resp_key not in seen_responses:
            seen_responses.add(resp_key)
            unique_entries.append(entry)

    rng = random.Random(seed)
    rng.shuffle(unique_entries)

    if len(unique_entries) < 2:
        return unique_entries, []

    test_size = max(1, int(len(unique_entries) * test_ratio))
    sft_test = unique_entries[:test_size]
    sft_train = unique_entries[test_size:]

    return sft_train, sft_test


def build_hf_dataset_dict(
    cpt_texts: List[str],
    sft_train_list: List[Dict],
    sft_test_list: List[Dict]
) -> Dict[str, Any]:
    from datasets import Dataset

    cpt_rows = [{"text": t} if isinstance(t, str) else t for t in cpt_texts]
    return {
        "cpt_train": Dataset.from_list(cpt_rows),
        "sft_train": Dataset.from_list(sft_train_list),
        "sft_test": Dataset.from_list(sft_test_list)
    }
