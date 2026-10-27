"""
Module 1 : Collecte & Extraction Multi-Sources (WhatsApp & YouTube Data API v3)
===============================================================================
Ce fichier est le module Python natif correspondant au notebook :
`01_data_collection/01_Data_Collection_Scraping.ipynb`

Une fois que tes fonctions ont passé tous les `assert` dans le notebook de l'Étape 1,
copie tes implémentations des TODO 1 à 4 ici pour pouvoir les importer dans le reste
de la pipeline (`from src.data_pipeline.scraper import ...`).
"""

import re
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple


def clean_chat_text(chat_text: str) -> Optional[str]:
    """Nettoyage de base repris et amélioré depuis Cleansing_Dataset.py."""
    chat_text = str(chat_text)
    ignoring_tags = [
        "<Media omitted>", "You deleted this message", "This message was deleted",
        "Messages and calls are end-to-end encrypted", "<This message was edited>",
        "created group", "added you", "removed you", "audio omitted",
        "sticker omitted", "image omitted", "GIF omitted", "document omitted",
        "video omitted", "//", "voice call", "Voice call"
    ]
    for ignoring_tag in ignoring_tags:
        if ignoring_tag in chat_text:
            return None

    chat_text = re.sub(r'http\S+|www\.\S+', '', chat_text)
    chat_text = re.sub(r'@\s*[^\s@]+', '', chat_text)
    chat_text = re.sub(r'(.)\1{2,}', r'\1\1', chat_text)
    chat_text = re.sub(r'\s+', ' ', chat_text).strip()
    return chat_text if chat_text else None


def parse_whatsapp_to_records(lines: List[str], max_reply_delay_minutes: float = 30.0) -> List[Dict]:
    """
    TODO 1 (depuis le Notebook 01) :
    Transforme les lignes d'un export WhatsApp iOS (_chat.txt) en enregistrements
    avec 'comment_id' et 'parent_id' basés sur la proximité temporelle (<= max_reply_delay_minutes).
    """
    regex_ios = r'^\[(\d{2}/\d{2}/\d{2,4},\s+\d{2}:\d{2}:\d{2})\] ([^:]+): (.+)$'
    grouped_turns = []
    current_speaker = None
    current_time = None
    current_buffer = []

    for line in lines:
        line = line.replace('\u200e', '').replace('\u2069', '').strip()
        match = re.match(regex_ios, line)
        if match:
            date_str, speaker, message = match.groups()
            clean_msg = clean_chat_text(message)
            if not clean_msg:
                continue
            date_clean = re.sub(r'\s+', ' ', date_str)
            fmt = "%d/%m/%Y, %H:%M:%S" if len(date_clean.split(',')[0].split('/')[-1]) == 4 else "%d/%m/%y, %H:%M:%S"
            msg_time = datetime.strptime(date_clean, fmt)

            if speaker == current_speaker:
                current_buffer.append(clean_msg)
                current_time = msg_time
            else:
                if current_speaker and current_buffer:
                    full_text = " ".join(current_buffer)
                    if len(full_text.split()) >= 2:
                        grouped_turns.append({
                            "speaker": current_speaker,
                            "timestamp": current_time,
                            "text": full_text
                        })
                current_speaker = speaker
                current_time = msg_time
                current_buffer = [clean_msg]
        else:
            if current_speaker:
                clean_line = clean_chat_text(line)
                if clean_line:
                    current_buffer.append(clean_line)

    if current_speaker and current_buffer:
        full_text = " ".join(current_buffer)
        if len(full_text.split()) >= 2:
            grouped_turns.append({
                "speaker": current_speaker,
                "timestamp": current_time,
                "text": full_text
            })

    records = []
    for i, turn in enumerate(grouped_turns):
        comment_id = f"wa_{i}"
        parent_id = None
        # TODO 1 : Reporter ici ton calcul de delta_minutes depuis le Notebook 01
        records.append({
            "comment_id": comment_id,
            "parent_id": parent_id,
            "text": turn["text"],
            "like_count": 0,
            "source_video_title": "WhatsApp Chat",
            "source_platform": "whatsapp"
        })
    return records


def parse_youtube_thread_item(item: Dict, video_title: str) -> List[Dict]:
    """
    TODO 2 (depuis le Notebook 01) :
    Extrait le commentaire parent (parent_id=None) et ses réponses (parent_id=top_id)
    depuis un item commentThreads.list de YouTube Data API v3.
    """
    extracted = []
    # TODO 2 : Reporter ton code validé dans le Notebook 01
    return extracted


def fetch_youtube_video_comments(
    youtube_client,
    video_id: str,
    video_title: str,
    output_jsonl_path: Path,
    max_pages: int = 5
) -> int:
    """
    TODO 3 (depuis le Notebook 01) :
    Récupère les pages de commentaires d'une vidéo YouTube et les sauvegarde en mode append dans output_jsonl_path.
    """
    saved_count = 0
    # TODO 3 : Reporter ton code validé dans le Notebook 01
    return saved_count


def split_trees_and_isolated(records: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    TODO 4 (depuis le Notebook 01) :
    Sépare les enregistrements bruts en :
      - `sft_pairs` (dialogues Parent -> Enfant)
      - `isolated_comments` (commentaires isolés pour le CPT)
    """
    sft_pairs = []
    isolated_comments = []
    # TODO 4 : Reporter ton code validé dans le Notebook 01
    return sft_pairs, isolated_comments
