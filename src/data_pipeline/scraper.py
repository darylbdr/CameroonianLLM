"""
Module 1 : Collecte & Extraction en Arbres (YouTube Data API v3 & Réseaux Sociaux)
==================================================================================
Module Python natif complet correspondant au notebook :
`01_data_collection/01_Data_Collection_Scraping.ipynb`

Fonctionnalités :
  - Nettoyage de base du bruit des réseaux sociaux (`clean_social_text`).
  - Recherche de vidéos ciblées (`search_youtube_videos`) et suivi des vidéos déjà scrapées
    (`load_scraped_video_ids` / `mark_video_as_scraped`) pour la reprise sur erreur.
  - Extraction des commentaires en arbres `Parent -> Enfant` (`parse_youtube_thread_item`, `fetch_youtube_video_comments`).
  - Séparation en paires de dialogue SFT (`Parent -> Enfant`) et commentaires isolés CPT (`split_trees_and_isolated`).
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set


def clean_social_text(text: str) -> Optional[str]:
    if not text:
        return None
    cleaned = str(text)
    cleaned = re.sub(r'http\S+|www\.\S+', '', cleaned)
    cleaned = re.sub(r'@\s*[^\s@]+', '', cleaned)
    cleaned = re.sub(r'(.)\1{2,}', r'\1\1', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned if cleaned else None


def load_scraped_video_ids(checkpoint_path: Path) -> Set[str]:
    if not checkpoint_path.exists():
        return set()
    return {
        line.strip()
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def mark_video_as_scraped(video_id: str, checkpoint_path: Path) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "a", encoding="utf-8") as f:
        f.write(f"{video_id}\n")


def search_youtube_videos(youtube_client, query: str, max_results: int = 10) -> List[Dict[str, str]]:
    request = youtube_client.search().list(
        part="snippet",
        q=query,
        type="video",
        maxResults=max_results,
        relevanceLanguage="fr"
    )
    response = request.execute()
    videos = []
    for item in response.get("items", []):
        vid = item.get("id", {}).get("videoId")
        title = item.get("snippet", {}).get("title", "")
        if vid:
            videos.append({"video_id": vid, "title": title})
    return videos


def parse_youtube_thread_item(item: Dict, video_title: str) -> List[Dict]:
    extracted = []

    top_comment = item["snippet"]["topLevelComment"]
    parent_id = top_comment["id"]
    top_snippet = top_comment["snippet"]
    parent_text = clean_social_text(top_snippet.get("textOriginal", ""))

    if parent_text:
        extracted.append({
            "comment_id": parent_id,
            "parent_id": None,
            "text": parent_text,
            "like_count": int(top_snippet.get("likeCount", 0)),
            "source_video_title": video_title,
            "source_platform": "youtube"
        })

    replies = item.get("replies", {}).get("comments", [])
    for reply in replies:
        reply_snippet = reply.get("snippet", {})
        reply_text = clean_social_text(reply_snippet.get("textOriginal", ""))
        if reply_text:
            extracted.append({
                "comment_id": reply["id"],
                "parent_id": parent_id,
                "text": reply_text,
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
    checkpoint_path: Optional[Path] = None,
    max_pages: int = 5
) -> int:
    output_path = Path(output_jsonl_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint_path is not None:
        already_done = load_scraped_video_ids(Path(checkpoint_path))
        if video_id in already_done:
            return 0

    saved_count = 0
    next_page_token = None

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

    if checkpoint_path is not None:
        mark_video_as_scraped(video_id, Path(checkpoint_path))

    return saved_count


def split_trees_and_isolated(records: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
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
                "source": r.get("source_platform", parent_rec.get("source_video_title", "youtube"))
            })

    isolated_comments = [
        r for r in records
        if r.get("parent_id") is None and r.get("comment_id") not in parents_with_replies
    ]

    return sft_pairs, isolated_comments
