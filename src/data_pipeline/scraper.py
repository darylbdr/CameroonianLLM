"""
Module 1 : Collecte & Extraction Multi-Sources (WhatsApp & YouTube Data API v3)
===============================================================================
Module Python natif complet correspondant au notebook :
`01_data_collection/01_Data_Collection_Scraping.ipynb`
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
    Transforme les lignes d'un export WhatsApp iOS (_chat.txt) en enregistrements
    avec 'comment_id' et 'parent_id' basés sur la proximité temporelle (<= max_reply_delay_minutes).
    """
    regex_ios = r'^\[(\d{2}/\d{2}/\d{2,4},\s+\d{2}:\d{2}:\d{2})\] ([^:]+): (.+)$'
    grouped_turns = []
    current_speaker = None
    current_time = None
    current_buffer = []

    for line in lines:
        line = line.replace("document omitted", "").strip()
        line = line.replace('\u200e', '').replace('\u2069', '').strip()
        match = re.match(regex_ios, line)
        if match:
            date_str, speaker, message = match.groups()
            clean_msg = clean_chat_text(message)
            if not clean_msg:
                continue
            date_clean = re.sub(r'\s+', ' ', date_str)
            fmt = "%d/%m/%Y, %H:%M:%S" if len(date_clean.split(',')[0].split('/')[-1]) == 4 else "%d/%m/%y, %H:%M:%S"
            try:
                msg_time = datetime.strptime(date_clean, fmt)
            except ValueError:
                continue

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
        if i > 0 and turn["timestamp"] and grouped_turns[i - 1]["timestamp"]:
            delta_minutes = (turn["timestamp"] - grouped_turns[i - 1]["timestamp"]).total_seconds() / 60.0
            if 0.0 <= delta_minutes <= max_reply_delay_minutes:
                parent_id = f"wa_{i - 1}"

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
    Extrait le commentaire parent (parent_id=None) et ses réponses (parent_id=top_id)
    depuis un item commentThreads.list de YouTube Data API v3.
    """
    extracted = []

    top_comment = item["snippet"]["topLevelComment"]
    parent_id = top_comment["id"]
    top_snippet = top_comment["snippet"]

    extracted.append({
        "comment_id": parent_id,
        "parent_id": None,
        "text": top_snippet.get("textOriginal", ""),
        "like_count": int(top_snippet.get("likeCount", 0)),
        "source_video_title": video_title,
        "source_platform": "youtube"
    })

    replies = item.get("replies", {}).get("comments", [])
    for reply in replies:
        reply_snippet = reply.get("snippet", {})
        extracted.append({
            "comment_id": reply["id"],
            "parent_id": parent_id,
            "text": reply_snippet.get("textOriginal", ""),
            "like_count": int(reply_snippet.get("likeCount", 0)),
            "source_video_title": video_title,
            "source_platform": "youtube"
        })

    return extracted


def fetch_youtube_video_comments(
    youtube_client,
    video_id: str,
    video_title: str,
    output_jsonl_path: Path,
    max_pages: int = 5
) -> int:
    """
    Récupère les pages de commentaires d'une vidéo YouTube et les sauvegarde en mode append dans output_jsonl_path.
    """
    saved_count = 0
    next_page_token = None
    output_path = Path(output_jsonl_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for _ in range(max_pages):
        request = youtube_client.commentThreads().list(
            part="snippet,replies",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token,
            textFormat="plainText"
        )
        response = request.execute()

        with open(output_path, "a", encoding="utf-8") as f:
            for item in response.get("items", []):
                comments = parse_youtube_thread_item(item, video_title)
                for comment in comments:
                    f.write(json.dumps(comment, ensure_ascii=False) + "\n")
                    saved_count += 1

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return saved_count


def split_trees_and_isolated(records: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Sépare les enregistrements bruts en :
      - `sft_pairs` (dialogues Parent -> Enfant)
      - `isolated_comments` (commentaires isolés pour le CPT)
    """
    by_id = {r["comment_id"]: r for r in records if "comment_id" in r}
    parents_with_replies = set()
    sft_pairs = []

    for r in records:
        pid = r.get("parent_id")
        if pid is not None and pid in by_id:
            parent_rec = by_id[pid]
            parents_with_replies.add(pid)
            sft_pairs.append({
                "parent_id": pid,
                "child_id": r["comment_id"],
                "user": parent_rec["text"],
                "assistant": r["text"],
                "source": r.get("source_platform", parent_rec.get("source_video_title", "tree"))
            })

    isolated_comments = [
        r for r in records
        if r.get("parent_id") is None and r.get("comment_id") not in parents_with_replies
    ]

    return sft_pairs, isolated_comments
