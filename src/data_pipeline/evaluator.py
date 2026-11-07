"""
Module 5 : Évaluation Rigoureuse (Le Triptyque de Métriques) en Local (CPU)
===========================================================================
Module Python natif complet correspondant au notebook :
`05_evaluation_and_deployment/05_Evaluation_and_GGUF_Export.ipynb`

Il regroupe les 3 métriques complémentaires :
  1. Perplexité : PPL = exp(eval_loss)
  2. Camfranglais Adherence Score : comparaison des distributions (mu, sigma) du score S
  3. Pertinence Sémantique : Similarité Cosinus des embeddings multilingues
"""

import math
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt


def compute_perplexity_from_loss(eval_loss: float) -> float:
    """
    Calcule PPL = exp(eval_loss).
    """
    if eval_loss > 20.0:
        return float("inf")
    return float(math.exp(eval_loss))


def evaluate_camfranglais_adherence(
    model_scores: List[float],
    human_scores: List[float]
) -> Dict[str, float]:
    """
    Compare la moyenne et l'écart-type des scores S du modèle avec ceux des humains.
    """
    m_arr = np.array(model_scores, dtype=float)
    h_arr = np.array(human_scores, dtype=float)

    model_mean = float(np.mean(m_arr)) if len(m_arr) > 0 else 0.0
    model_std = float(np.std(m_arr)) if len(m_arr) > 0 else 0.0
    human_mean = float(np.mean(h_arr)) if len(h_arr) > 0 else 0.0
    human_std = float(np.std(h_arr)) if len(h_arr) > 0 else 0.0
    mean_abs_gap = abs(model_mean - human_mean)
    active_rate = float(np.mean(m_arr > 0.0)) if len(m_arr) > 0 else 0.0

    return {
        "model_mean": round(model_mean, 4),
        "model_std": round(model_std, 4),
        "human_mean": round(human_mean, 4),
        "human_std": round(human_std, 4),
        "mean_abs_gap": round(mean_abs_gap, 4),
        "active_camfranglais_rate": round(active_rate, 4)
    }


def compute_semantic_similarity(
    generated_responses: List[str],
    reference_responses: List[str],
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
) -> Dict[str, float]:
    """
    Calcule la similarité cosinus moyenne entre les réponses générées et les vraies réponses humaines.
    """
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(model_name, device="cpu")
    emb_gen = embedder.encode(generated_responses, normalize_embeddings=True)
    emb_ref = embedder.encode(reference_responses, normalize_embeddings=True)

    cosine_sims = np.sum(emb_gen * emb_ref, axis=1)
    return {
        "mean_cosine_similarity": round(float(np.mean(cosine_sims)), 4),
        "min_similarity": round(float(np.min(cosine_sims)), 4),
        "max_similarity": round(float(np.max(cosine_sims)), 4)
    }


def plot_adherence_distributions(model_scores: List[float], human_scores: List[float]):
    """
    Trace l'histogramme comparatif de la densité de Camfranglais (Score S)
    entre les vraies réponses humaines et les réponses générées par le modèle.
    """
    plt.figure(figsize=(8, 4))
    plt.hist(human_scores, bins=15, alpha=0.6, label="Humains (Référence)", color="#2b8cbe")
    plt.hist(model_scores, bins=15, alpha=0.6, label="Modèle Fine-Tuné", color="#e34a33")
    if human_scores:
        plt.axvline(float(np.mean(human_scores)), color="#045a8d", linestyle="--", label="Moyenne Humaine")
    if model_scores:
        plt.axvline(float(np.mean(model_scores)), color="#b30000", linestyle="--", label="Moyenne Modèle")
    plt.xlabel("Score Camfranglais (S)")
    plt.ylabel("Fréquence")
    plt.title("Camfranglais Adherence Score : Humains vs Modèle")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
