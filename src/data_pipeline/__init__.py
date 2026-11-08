"""
Pipeline de données locale pour le projet Camfranglais :
- scraper.py   : Extraction des données YouTube Data API v3 en arbres Parent -> Enfant.
- scorer.py    : Filtrage linguistique et calcul du score S = I(N >= 6) * D_cam * P_matrice.
- builder.py   : Construction des splits CPT et SFT + Rétro-Instruction + Push vers Hugging Face Hub.
- evaluator.py : Evaluation : Perplexité, Camfranglais Adherence Score, Similarité Sémantique.
- run_pipeline.py : Orchestrateur CLI qui jumelle toute la pipeline locale de bout en bout.
"""
