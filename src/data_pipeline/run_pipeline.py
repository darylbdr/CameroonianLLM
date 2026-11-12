"""Orchestrateur de toute la Pipeline Locale (YouTube + conv_whatsapp.json + conv.json)."""

import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from src.data_pipeline.scraper import (
    search_youtube_videos,
    fetch_youtube_video_comments,
    split_trees_and_isolated,
)
from src.data_pipeline.scorer import load_lexicon, normalize_and_tokenize, score_camfranglais
from src.data_pipeline.builder import (
    is_good_sft_dialogue_pair,
    create_sft_entry,
    deduplicate_and_split_sft,
    build_hf_dataset_dict,
)
from src.data_pipeline.whatsapp import load_whatsapp_dataset


def run_youtube_scraping_if_configured(project_root: Path, raw_yt_jsonl: Path) -> None:
    """Lance le scraping YouTube si YOUTUBE_API_KEY est configurée dans .env."""
    yt_key = os.getenv("YOUTUBE_API_KEY", "").strip()
    if not yt_key or yt_key == "cle_youtube_api_v3_ici":
        return

    seeds_path = project_root / "config" / "youtube_seeds.json"
    if not seeds_path.exists():
        return
    try:
        from googleapiclient.discovery import build
    except ImportError:
        print("google-api-python-client non installé : `pip install -r requirements.txt` pour activer le scraping YouTube.")
        return

    seeds = json.loads(seeds_path.read_text(encoding="utf-8"))
    max_videos_per_query = int(seeds.get("max_videos_per_query", 15))
    max_pages_per_video = int(seeds.get("max_pages_per_video", 10))
    youtube = build("youtube", "v3", developerKey=yt_key)
    checkpoint_path = project_root / "data" / "raw" / "scraped_videos.txt"
    videos_to_scrape = [{"video_id": vid, "title": f"Video {vid}"} for vid in seeds.get("video_ids", [])]
    for query in seeds.get("search_queries", []):
        try:
            videos_to_scrape.extend(search_youtube_videos(youtube, query=query, max_results=max_videos_per_query))
        except Exception as exc:
            print(f"Erreur lors de la recherche YouTube '{query}' : {exc}")

    new_comments = 0
    for video in videos_to_scrape:
        try:
            new_comments += fetch_youtube_video_comments(
                youtube_client=youtube,
                video_id=video["video_id"],
                video_title=video["title"],
                output_jsonl_path=raw_yt_jsonl,
                checkpoint_path=checkpoint_path,
                max_pages=max_pages_per_video,
            )
        except Exception as exc:
            print(f"Ignore vidéo {video['video_id']} (commentaires désactivés ou quota) : {exc}")
    if new_comments:
        print(f"{new_comments} nouveaux commentaires YouTube extraits dans {raw_yt_jsonl.name} !")


def _add_cpt_text(target: list, seen: set, text: str) -> None:
    key = text.strip().casefold()
    if key and key not in seen:
        seen.add(key)
        target.append(text.strip())


def main():
    project_root = Path(__file__).resolve().parents[2]
    if load_dotenv is not None and (project_root / ".env").exists():
        load_dotenv(project_root / ".env")

    print("Démarrage de la Data Pipeline CameroonianLLM...")

    # 1. Lexique
    lexicon_path = project_root / "config" / "lexicon.json"
    lexicon = load_lexicon(lexicon_path)
    print(
        f"[1/5] Lexique chargé ({len(lexicon['strong_markers'])} marqueurs forts, "
        f"{len(lexicon['medium_markers'])} moyens)."
    )

    # 2. YouTube : extraction et séparation des fils parent -> enfant.
    raw_yt_jsonl = project_root / "data" / "raw" / "raw_comments.jsonl"
    # run_youtube_scraping_if_configured(project_root, raw_yt_jsonl)
    all_raw_records = []
    if raw_yt_jsonl.exists():
        with raw_yt_jsonl.open("r", encoding="utf-8") as file:
            all_raw_records = [json.loads(line) for line in file if line.strip()]
        print(f"[2/5] {len(all_raw_records)} commentaires YouTube chargés.")
    else:
        print("[2/5] Aucun data/raw/raw_comments.jsonl trouvé.")

    sft_candidate_pairs, _isolated_records = split_trees_and_isolated(all_raw_records)
    print(f"   -> {len(sft_candidate_pairs)} paires YouTube parent -> enfant.")

    # 3. WhatsApp : conv_whatsapp.json est la seule source conversationnelle mélangée avec YouTube.
    whatsapp_path = project_root / "conv_whatsapp.json"
    whatsapp_cpt, whatsapp_sft_pairs, whatsapp_stats = load_whatsapp_dataset(whatsapp_path, lexicon)
    if whatsapp_path.exists():
        print(
            "[3/5] WhatsApp (conv_whatsapp.json) : "
            f"{whatsapp_stats['messages_loaded']} messages lus, "
            f"{whatsapp_stats['messages_cleaned']} après nettoyage, "
            f"{whatsapp_stats['cpt_kept']} CPT, "
            f"{whatsapp_stats['sft_candidates']} candidats SFT."
        )
    else:
        print("[3/5] Aucun conv_whatsapp.json trouvé : source WhatsApp ignorée.")

    # 4. CPT : commentaires YouTube existants + messages WhatsApp nettoyés.
    cpt_texts = []
    seen_cpt = set()
    for record in tqdm(all_raw_records, desc="Scoring CPT YouTube"):
        text = record.get("text", "")
        result = score_camfranglais(text, lexicon, high_threshold=0.06)
        if result["zone"] == "high":
            _add_cpt_text(cpt_texts, seen_cpt, result["cleaned_text"])
    for text in tqdm(whatsapp_cpt, desc="Fusion CPT WhatsApp"):
        _add_cpt_text(cpt_texts, seen_cpt, text)

    # SFT : paires YouTube + paires WhatsApp (conv_whatsapp.json) + conv.json.
    sft_entries = []

    conv_path = project_root / "conv.json"
    if conv_path.exists():
        manual_convs = json.loads(conv_path.read_text(encoding="utf-8"))
        added_manual = 0
        for entry in manual_convs:
            conv = entry.get("conversations", [])
            if (
                isinstance(conv, list)
                and len(conv) >= 2
                and conv[0].get("content", "").strip()
                and conv[1].get("content", "").strip()
            ):
                sft_entries.append(
                    {
                        "conversations": conv,
                        "source": entry.get("source", "manual"),
                    }
                )
                added_manual += 1
        print(f"   -> {added_manual} conversations chargées depuis {conv_path.name}.")

    for pair in tqdm(sft_candidate_pairs, desc="Filtrage SFT YouTube"):
        result = score_camfranglais(pair["assistant"], lexicon, high_threshold=0.06)
        if is_good_sft_dialogue_pair(pair["user"], result["cleaned_text"], result["score"], min_score=0.06):
            cleaned_user, _ = normalize_and_tokenize(pair["user"])
            if cleaned_user:
                sft_entries.append(
                    create_sft_entry(cleaned_user, result["cleaned_text"], source=pair.get("source", "youtube"))
                )

    for pair in tqdm(whatsapp_sft_pairs, desc="Filtrage SFT WhatsApp"):
        result = score_camfranglais(pair["assistant"], lexicon, high_threshold=0.06)
        if is_good_sft_dialogue_pair(pair["user"], result["cleaned_text"], result["score"], min_score=0.06):
            sft_entries.append(
                create_sft_entry(pair["user"], result["cleaned_text"], source=pair.get("source", "whatsapp_group"))
            )

    print(
        f"[4/5] Filtrage terminé : {len(cpt_texts)} textes CPT certifiés | "
        f"{len(sft_entries)} dialogues SFT avant déduplication."
    )

    # 5. Déduplication, split et sauvegarde locale.
    sft_train, sft_test = deduplicate_and_split_sft(sft_entries, test_ratio=0.1)
    final_dir = project_root / "data" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    (final_dir / "cpt_train.json").write_text(
        json.dumps(cpt_texts, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (final_dir / "sft_train.json").write_text(
        json.dumps(sft_train, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (final_dir / "sft_test.json").write_text(
        json.dumps(sft_test, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[5/5] Datasets générés dans {final_dir} :")
    print(f"   - cpt_train.json : {len(cpt_texts)} textes")
    print(f"   - sft_train.json : {len(sft_train)} conversations")
    print(f"   - sft_test.json  : {len(sft_test)} conversations")

    hf_token = os.getenv("HF_TOKEN")
    whatsapp_used = bool(whatsapp_stats["messages_loaded"] and (whatsapp_cpt or whatsapp_sft_pairs))
    allow_whatsapp_push = os.getenv("ALLOW_WHATSAPP_HF_PUSH", "false").strip().lower() in {"1", "true", "yes", "oui"}
    if hf_token and hf_token != "hf_token_ici":
        if whatsapp_used and not allow_whatsapp_push:
            print(
                "Push Hugging Face ignoré : conv_whatsapp.json a contribué aux datasets. "
                "Pour autoriser explicitement le partage, renseigne "
                "ALLOW_WHATSAPP_HF_PUSH=true dans .env uniquement après accord des participants."
            )
        else:
            from datasets import DatasetDict

            dataset_dict = build_hf_dataset_dict(cpt_texts, sft_train, sft_test)
            if cpt_texts:
                dataset_dict["cpt_train"].push_to_hub("Cmr237/camfranglais-corpus-cpt", token=hf_token)
            DatasetDict({"train": dataset_dict["sft_train"], "test": dataset_dict["sft_test"]}).push_to_hub(
                "Cmr237/camfranglais-corpus-sft", token=hf_token
            )
            print("Datasets poussés sur Hugging Face.")


if __name__ == "__main__":
    main()
