"""
Pipeline de données locale (CPU) pour le projet Camfranglais :
- scraper.py   : Extraction WhatsApp (_chat.txt) et YouTube Data API v3 en arbres Parent -> Enfant.
- scorer.py    : Filtrage linguistique et calcul du score S = I(N >= 6) * D_cam * P_matrice.
- builder.py   : Construction des splits CPT et SFT + Rétro-Instruction + Push vers Hugging Face Hub.
- evaluator.py : Triptyque d'évaluation (Perplexité, Camfranglais Adherence Score, Similarité Sémantique).
- run_pipeline.py : Orchestrateur CLI qui jumelle toute la pipeline locale de bout en bout.
"""
