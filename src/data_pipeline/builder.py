"""
Module 3 : Construction des Datasets (CPT & SFT), Rétro-Instruction & Push Hugging Face
=======================================================================================
Ce fichier est le module Python natif correspondant au notebook :
`03_dataset_builder/03_Dataset_Builder_and_HF_Push.ipynb`

Il importe directement `scorer.py` et assemble :
  - `cpt_train.jsonl`
  - `sft_train.jsonl`
  - `sft_test.jsonl`
"""

import re
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from datasets import Dataset, DatasetDict

INTERROGATIVE_MARKERS = {
    "pourquoi", "comment", "qui", "quand", "où", "ou",
    "quel", "quelle", "quels", "quelles", "combien", "wanda"
}


def is_good_sft_dialogue_pair(
    user_text: str,
    assistant_text: str,
    assistant_score: float,
    min_score: float = 0.15,
    min_assistant_words: int = 8
) -> bool:
    """
    TODO 1 (depuis le Notebook 03) :
    Vérifie si une paire (user_text, assistant_text) est pertinente pour le SFT.
    """
    # TODO 1 : Reporter ton code validé dans le Notebook 03
    return False


def build_backtranslation_prompt(camfranglais_response: str) -> str:
    """
    TODO 2.1 (depuis le Notebook 03) :
    Construit le prompt de Rétro-Instruction pour générer la question associée à un long commentaire CPT.
    """
    # TODO 2.1 : Reporter ton code validé dans le Notebook 03
    return ""


def create_sft_entry(user_message: str, assistant_response: str, source: str = "synthetic_backtranslation") -> Dict:
    """
    TODO 2.2 (depuis le Notebook 03) :
    Formate une paire au format {"conversations": [{"role": "user", ...}, {"role": "assistant", ...}]}.
    """
    # TODO 2.2 : Reporter ton code validé dans le Notebook 03
    return {}


def deduplicate_and_split_sft(
    sft_entries: List[Dict],
    test_ratio: float = 0.1,
    seed: int = 3407
) -> Tuple[List[Dict], List[Dict]]:
    """
    TODO 3 (depuis le Notebook 03) :
    Supprime les doublons et sépare en (sft_train, sft_test).
    """
    sft_train, sft_test = [], []
    # TODO 3 : Reporter ton code validé dans le Notebook 03
    return sft_train, sft_test


def build_hf_dataset_dict(
    cpt_texts: List[str],
    sft_train_list: List[Dict],
    sft_test_list: List[Dict]
) -> Dict[str, Dataset]:
    """
    TODO 4 (depuis le Notebook 03) :
    Construit le dictionnaire de Datasets Hugging Face ('cpt_train', 'sft_train', 'sft_test').
    """
    # TODO 4 : Reporter ton code validé dans le Notebook 03
    return {}
