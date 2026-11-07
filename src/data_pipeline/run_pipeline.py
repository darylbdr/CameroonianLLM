"""
Orchestrateur de la Data Pipeline Locale (Jumelage des Étapes 1 -> 2 -> 3)
==========================================================================
Exécute toute la pipeline de préparation de données en une seule commande depuis la racine :
    python -m src.data_pipeline.run_pipeline
"""

import os
import sys
import re
import json
from pathlib import Path

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
from src.data_pipeline.scorer import (
    load_lexicon,
    score_camfranglais,
)
from src.data_pipeline.builder import (
    is_good_sft_dialogue_pair,
    create_sft_entry,
    deduplicate_and_split_sft,
    build_hf_dataset_dict,
)


def load_clean_conv_bot(conv_path: Path):
    """Charge conv_bot.json en retirant les commentaires éventuels."""
    if not conv_path.exists():
        return []
    txt = conv_path.read_text(encoding="utf-8")
    txt = re.sub(r"//.*", "", txt)
    txt = re.sub(r",\s*]", "]", txt)
    txt = re.sub(r",\s*}", "}", txt)
    data = json.loads(txt)
    return [data] if isinstance(data, dict) else data


def run_youtube_scraping_if_configured(project_root: Path, raw_yt_jsonl: Path) -> None:
    """Lance automatiquement le scraping YouTube si YOUTUBE_API_KEY est définie dans .env."""
    yt_key = os.getenv("YOUTUBE_API_KEY", "").strip()
    if not yt_key or yt_key == "ta_cle_youtube_api_v3_ici":
        return

    seeds_path = project_root / "config" / "youtube_seeds.json"
    if not seeds_path.exists():
        return

    try:
        from googleapiclient.discovery import build
    except ImportError:
        print("⚠️ google-api-python-client non installé : `pip install -r requirements.txt` pour activer le scraping YouTube.")
        return

    seeds = json.loads(seeds_path.read_text(encoding="utf-8"))
    max_videos_per_query = int(seeds.get("max_videos_per_query", 15))
    max_pages_per_video = int(seeds.get("max_pages_per_video", 10))

    youtube = build("youtube", "v3", developerKey=yt_key)
    checkpoint_path = project_root / "data" / "raw" / "scraped_videos.txt"

    videos_to_scrape = [{"video_id": vid, "title": f"Video {vid}"} for vid in seeds.get("video_ids", [])]
    for query in seeds.get("search_queries", []):
        try:
            found = search_youtube_videos(youtube, query=query, max_results=max_videos_per_query)
            videos_to_scrape.extend(found)
        except Exception as e:
            print(f"⚠️ Erreur lors de la recherche YouTube '{query}' : {e}")

    new_comments = 0
    for v in videos_to_scrape:
        try:
            c = fetch_youtube_video_comments(
                youtube_client=youtube,
                video_id=v["video_id"],
                video_title=v["title"],
                output_jsonl_path=raw_yt_jsonl,
                checkpoint_path=checkpoint_path,
                max_pages=max_pages_per_video
            )
            new_comments += c
        except Exception as e:
            print(f"⚠️ Ignore vidéo {v['video_id']} (commentaires désactivés ou quota) : {e}")

    if new_comments > 0:
        print(f"📥 {new_comments} nouveaux commentaires YouTube extraits dans {raw_yt_jsonl.name} !")


def main():
    project_root = Path(__file__).resolve().parents[2]
    if load_dotenv is not None and (project_root / ".env").exists():
        load_dotenv(project_root / ".env")

    print("🚀 Démarrage de la Data Pipeline CameroonianAI (Local CPU)...")

    # 1. Chargement du lexique
    lexicon_path = project_root / "config" / "lexicon.json"
    lexicon = load_lexicon(lexicon_path)
    print(f"✅ [1/4] Lexique chargé ({len(lexicon['strong_markers'])} marqueurs forts, {len(lexicon['medium_markers'])} moyens).")

    # 2. Collecte / Lecture des commentaires YouTube / TikTok (data/raw/raw_comments.jsonl)
    raw_yt_jsonl = project_root / "data" / "raw" / "raw_comments.jsonl"
    run_youtube_scraping_if_configured(project_root, raw_yt_jsonl)

    all_raw_records = []
    if raw_yt_jsonl.exists():
        with open(raw_yt_jsonl, "r", encoding="utf-8") as f:
            all_raw_records = [json.loads(line) for line in f if line.strip()]
        print(f"✅ [2/4] Fichier {raw_yt_jsonl.name} : {len(all_raw_records)} commentaires chargés.")
    else:
        print("ℹ️ [2/4] Aucun fichier data/raw/raw_comments.jsonl trouvé pour l'instant (configure YOUTUBE_API_KEY dans .env pour scraper YouTube).")

    # Séparation en arbres SFT (Parent -> Enfant) vs Commentaires Isolés (CPT)
    sft_candidate_pairs, isolated_records = split_trees_and_isolated(all_raw_records)
    print(f"   -> {len(sft_candidate_pairs)} paires Parent->Enfant | {len(isolated_records)} commentaires isolés.")

    # 3. Filtrage & Scoring via la formule S = I(N >= 6) * D_cam * P_matrice (seuil high >= 0.06)
    cpt_texts = []
    seen_cpt = set()

    for rec in all_raw_records:
        res = score_camfranglais(rec["text"], lexicon, high_threshold=0.06)
        if res["zone"] == "high":
            key = res["cleaned_text"].lower()
            if key not in seen_cpt:
                seen_cpt.add(key)
                cpt_texts.append(res["cleaned_text"])

    sft_entries = []
    # 3.a Ajout des paires de référence de conv_bot.json
    conv_bot_entries = load_clean_conv_bot(project_root / "conv_bot.json")
    sft_entries.extend(conv_bot_entries)

    # 3.b Filtrage des paires Parent -> Enfant (Règle d'Or SFT + Filtre Dialogique)
    for pair in sft_candidate_pairs:
        res_assistant = score_camfranglais(pair["assistant"], lexicon, high_threshold=0.06)
        if is_good_sft_dialogue_pair(pair["user"], res_assistant["cleaned_text"], res_assistant["score"], min_score=0.06):
            sft_entries.append(
                create_sft_entry(pair["user"], res_assistant["cleaned_text"], source=pair.get("source", "youtube"))
            )

    print(f"✅ [3/4] Filtrage terminé : {len(cpt_texts)} textes CPT certifiés | {len(sft_entries)} dialogues SFT retenus.")

    # 4. Déduplication, Split Train/Test & Sauvegarde dans data/final/
    sft_train, sft_test = deduplicate_and_split_sft(sft_entries, test_ratio=0.1)
    final_dir = project_root / "data" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    (final_dir / "cpt_train.json").write_text(json.dumps(cpt_texts, indent=2, ensure_ascii=False), encoding="utf-8")
    (final_dir / "sft_train.json").write_text(json.dumps(sft_train, indent=2, ensure_ascii=False), encoding="utf-8")
    (final_dir / "sft_test.json").write_text(json.dumps(sft_test, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"✅ [4/4] Datasets finaux générés dans {final_dir} :")
    print(f"   - cpt_train.json : {len(cpt_texts)} textes bruts Camfranglais")
    print(f"   - sft_train.json : {len(sft_train)} conversations d'entraînement")
    print(f"   - sft_test.json  : {len(sft_test)} conversations de test (pour l'Étape 5)")

    # Push automatique sur Hugging Face Hub si HF_TOKEN est configuré dans .env
    hf_token = os.getenv("HF_TOKEN")
    if hf_token and hf_token != "hf_ton_token_huggingface_ici":
        from datasets import DatasetDict

        ds_dict = build_hf_dataset_dict(cpt_texts, sft_train, sft_test)
        if len(cpt_texts) > 0:
            ds_dict["cpt_train"].push_to_hub("Cmr237/camfranglais-corpus-cpt", token=hf_token)
        DatasetDict({
            "train": ds_dict["sft_train"],
            "test": ds_dict["sft_test"]
        }).push_to_hub("Cmr237/camfranglais-corpus-sft", token=hf_token)
        print("☁️ Datasets poussés avec succès sur Hugging Face Hub (Cmr237/camfranglais-corpus-cpt & sft) !")


if __name__ == "__main__":
    main()
