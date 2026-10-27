"""
Module 5 : Évaluation Rigoureuse (Le Triptyque de Métriques) en Local (CPU)
===========================================================================
Ce fichier est le module Python natif correspondant au notebook :
`05_evaluation_and_deployment/05_Evaluation_and_GGUF_Export.ipynb`

Il regroupe les 3 métriques complémentaires :
  1. Perplexité : PPL = exp(eval_loss)
  2. Camfranglais Adherence Score : comparaison des distributions (mu, sigma) du score S
  3. Pertinence Sémantique : Similarité Cosinus des embeddings multilingues
"""

import math
from typing import List, Dict
import numpy as np


def compute_perplexity_from_loss(eval_loss: float) -> float:
    """
    TODO 1 (depuis le Notebook 05) :
    Calcule PPL = exp(eval_loss).
    """
    # TODO 1 : Reporter ton code validé dans le Notebook 05
    return 0.0


def evaluate_camfranglais_adherence(
    model_scores: List[float],
    human_scores: List[float]
) -> Dict[str, float]:
    """
    TODO 2 (depuis le Notebook 05) :
    Compare la moyenne et l'écart-type des scores S du modèle avec ceux des humains.
    """
    # TODO 2 : Reporter ton code validé dans le Notebook 05
    return {}


def compute_semantic_similarity(
    generated_responses: List[str],
    reference_responses: List[str],
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
) -> Dict[str, float]:
    """
    TODO 3 (depuis le Notebook 05) :
    Calcule la similarité cosinus moyenne entre les réponses générées et les vraies réponses humaines.
    """
    # TODO 3 : Reporter ton code validé dans le Notebook 05
    return {}
