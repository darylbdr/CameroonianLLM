"""
Évaluation Scientifique
"""

import sys
import json
import argparse
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
from src.data_pipeline.scorer import load_lexicon, score_camfranglais
from src.data_pipeline.evaluator import (
    compute_perplexity_from_loss,
    evaluate_camfranglais_adherence,
    compute_semantic_similarity,
    plot_adherence_distributions,
)


def resolve_predictions_path(project_root: Path, custom_path: str = None) -> Path:
    """Recherche test_predictions.json à la racine ou dans data/final/."""
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p
    candidates = [
        project_root / "test_predictions.json",
        project_root / "data" / "final" / "test_predictions.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Fichier test_predictions.json introuvable (attendu à la racine du projet ou dans data/final/)."
    )


def main():
    parser = argparse.ArgumentParser(description="Évaluation du modèle CameroonianAI sur test_predictions.json")
    parser.add_argument("--predictions", type=str, default=None, help="Chemin vers test_predictions.json")
    parser.add_argument("--no-plot", action="store_true", help="Ne pas ouvrir la fenêtre matplotlib interactive")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    lexicon_path = project_root / "config" / "lexicon.json"
    predictions_path = resolve_predictions_path(project_root, args.predictions)

    print("=" * 72)
    print("ÉVALUATION CAMEROONIAN-AI — TRIPTYQUE DE MÉTRIQUES")
    print("=" * 72)
    print(f"Fichier de prédictions : {predictions_path}")

    lexicon = load_lexicon(lexicon_path)
    data = json.loads(predictions_path.read_text(encoding="utf-8"))

    eval_loss = float(data.get("eval_loss", 0.0))
    predictions = data.get("predictions", [])
    if not predictions:
        print("Aucune prédiction trouvée dans le fichier.")
        return

    # 1. Métrique 1 : Perplexité
    perplexity = compute_perplexity_from_loss(eval_loss)

    # 2. Métrique 2 : Camfranglais Adherence Score
    model_scores = []
    human_scores = []
    user_prompts = []
    model_responses = []
    human_references = []
    scored_examples = []

    for item in predictions:
        user_q = item.get("user", "").strip()
        human_ref = item.get("human_reference", "").strip()
        model_ans = item.get("model_response", "").strip()

        # On utilise min_words=3 en évaluation car certaines répliques conversationnelles sont courtes
        res_model = score_camfranglais(model_ans, lexicon, min_words=3)
        res_human = score_camfranglais(human_ref, lexicon, min_words=3)

        model_scores.append(res_model["score"])
        human_scores.append(res_human["score"])
        user_prompts.append(user_q)
        model_responses.append(model_ans)
        human_references.append(human_ref)

        scored_examples.append({
            "user": user_q,
            "model_response": model_ans,
            "human_reference": human_ref,
            "model_score_S": res_model["score"],
            "human_score_S": res_human["score"],
        })

    adherence_metrics = evaluate_camfranglais_adherence(model_scores, human_scores)

    # 3. Métrique 3 : Pertinence Sémantique
    print("\nCalcul des embeddings multilingues (paraphrase-multilingual-MiniLM-L12-v2)...")
    sim_ref_vs_model = compute_semantic_similarity(model_responses, human_references)
    sim_user_vs_model = compute_semantic_similarity(model_responses, user_prompts)

    # Assemblage du rapport complet
    report = {
        "num_test_dialogues": len(predictions),
        "1_perplexity": {
            "eval_loss": round(eval_loss, 4),
            "perplexity": round(perplexity, 2),
        },
        "2_camfranglais_adherence": adherence_metrics,
        "3_semantic_relevance": {
            "model_vs_human_reference": sim_ref_vs_model,
            "model_vs_user_prompt": sim_user_vs_model,
        },
    }

    # Sauvegarde du rapport JSON dans data/final/evaluation_report.json
    final_dir = project_root / "data" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    report_path = final_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Affichage
    print("\n" + "-" * 72)
    print("FLUIDITÉ & PRÉDICTION")
    print("-" * 72)
    print(f"   • Eval Loss sur sft_test        : {eval_loss:.4f}")
    print(f"   • Perplexité (PPL = e^Loss)     : {perplexity:.2f}")

    print("\n" + "-" * 72)
    print("ADHÉRENCE AU CAMFRANGLAIS")
    print("-" * 72)
    print(f"   • Score S moyen (Modèle)        : {adherence_metrics['model_mean']:.4f} (± {adherence_metrics['model_std']:.4f})")
    print(f"   • Score S moyen (Humains)       : {adherence_metrics['human_mean']:.4f} (± {adherence_metrics['human_std']:.4f})")
    print(f"   • Écart absolu |Modèle - Humain|: {adherence_metrics['mean_abs_gap']:.4f}")
    print(f"   • Taux de réponses avec marqueur: {adherence_metrics['active_camfranglais_rate'] * 100:.1f}%")

    print("\n" + "-" * 72)
    print("PERTINENCE SÉMANTIQUE")
    print("-" * 72)
    print(f"   • Cosinus Moyen (Modèle ↔ Réf.) : {sim_ref_vs_model['mean_cosine_similarity']:.4f} "
          f"[min: {sim_ref_vs_model['min_similarity']:.4f} | max: {sim_ref_vs_model['max_similarity']:.4f}]")
    print(f"   • Cosinus Moyen (Modèle ↔ User) : {sim_user_vs_model['mean_cosine_similarity']:.4f} "
          f"[min: {sim_user_vs_model['min_similarity']:.4f} | max: {sim_user_vs_model['max_similarity']:.4f}]")

    # Top 5 des réponses du modèle ayant le plus fort score S Camfranglais
    top_cam = sorted(scored_examples, key=lambda x: x["model_score_S"], reverse=True)[:5]
    print("\n" + "-" * 72)
    print("TOP 5 DES RÉPONSES GÉNÉRÉES EN CAMFRANGLAIS")
    print("-" * 72)
    for idx, ex in enumerate(top_cam, 1):
        print(f"   [{idx}] Score S = {ex['model_score_S']:.4f}")
        print(f"       User   : {ex['user']}")
        print(f"       Modèle : {ex['model_response']}")

    print("\n" + "=" * 72)
    print(f"Rapport complet sauvegardé dans : {report_path}")
    print("=" * 72)

    # Sauvegarde systématique de l'histogramme en PNG + affichage si demandé
    plot_path = final_dir / "adherence_distribution.png"
    if args.no_plot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.hist(human_scores, bins=15, alpha=0.6, label="Humains (sft_test)", color="#2b8cbe")
    plt.hist(model_scores, bins=15, alpha=0.6, label="Modèle Fine-Tuné", color="#e34a33")
    if human_scores:
        plt.axvline(float(adherence_metrics["human_mean"]), color="#045a8d", linestyle="--", label="Moyenne Humaine")
    if model_scores:
        plt.axvline(float(adherence_metrics["model_mean"]), color="#b30000", linestyle="--", label="Moyenne Modèle")
    plt.xlabel("Score Camfranglais (S)")
    plt.ylabel("Fréquence")
    plt.title("Camfranglais Adherence Score : Humains vs Modèle")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"Graphique sauvegardé dans       : {plot_path}")

    if not args.no_plot:
        plt.show()


if __name__ == "__main__":
    main()
