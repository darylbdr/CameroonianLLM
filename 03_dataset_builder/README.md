# 🏗️ Étape 3 — Construction des Datasets (CPT vs SFT), Rétro-Instruction & Publication Hugging Face

> **Objectif de cette étape** : Transformer les commentaires filtrés et tes données existantes ([`conv_bot.json`](../conv_bot.json) et [`conversations_dataset.json`](../conversations_dataset.json)) en trois splits propres (`cpt_train`, `sft_train`, `sft_test`) et les envoyer sur **Hugging Face Hub** pour découpler la préparation des données (CPU) de l'entraînement QLoRA (GPU Colab).

---

## 1. Le Pattern MLOps : `Local/Colab CPU` $\longrightarrow$ `Hugging Face Hub` $\longrightarrow$ `Colab GPU`

Si tu places le scraping, le filtrage, la création du dataset et l'entraînement dans un seul gros notebook GPU sur Colab :
- Tu gaspilles ton quota précieux de GPU Tesla T4 pendant que le CPU traite du texte.
- Tu dois ré-uploader tes fichiers `.json` à chaque déconnexion de Colab (comme c'était le cas avec `500_instructions_dataset.json` dans [`CameroonianFineTuning.ipynb`](../CameroonianFineTuning.ipynb)).

Avec le pattern **Hugging Face Hub** :
1. Ce notebook `03_Dataset_Builder_and_HF_Push.ipynb` tourne gratuitement sur **CPU**.
2. Il assemble et pousse tes jeux de données vers ton dépôt `Cmr237/camfranglais-corpus` via `dataset.push_to_hub(...)`.
3. Ton notebook d'entraînement GPU chargera tout en **3 secondes** avec `load_dataset("Cmr237/camfranglais-corpus")` !

---

## 2. Les Deux Corpus à Construire

### A. Le Corpus CPT (`cpt_train.jsonl`) — *Apprendre la Langue*
- **Contenu** : Tous les commentaires isolés (et les messages WhatsApp longs sans question-réponse directe) qui ont obtenu un score $S \ge \tau_{\text{haut}}$.
- **Format** : Une simple colonne `{"text": "...", "score": 0.32}`.
- **Astuce pour l'entraînement (Packing)** : Comme un commentaire fait souvent $10$ à $30$ mots et que la fenêtre de contexte du modèle est de $2048$ tokens, nous activerons `packing=True` dans l'étape 4 pour concaténer plusieurs commentaires séparés par `<|eos|>` dans un même bloc de $2048$ tokens (zéro temps GPU perdu sur du *padding* !).

---

### B. Le Corpus SFT (`sft_train.jsonl` & `sft_test.jsonl`) — *Apprendre à Dialoguer*
Un bon dataset SFT pour un chatbot en Camfranglais vise **$1\,000$ à $2\,000$ paires de haute qualité**. Attention au piège des commentaires YouTube/TikTok :
> Si quelqu'un répond à un commentaire juste pour dire *"Trop vrai ce que tu dis mbom, je valide fort !"*, c'est du Camfranglais, mais c'est un mauvais comportement d'assistant (qui doit répondre à des questions ou rebondir avec du fond).

Nous combinons donc **3 sources complémentaires** pour le SFT :
1. **Ton noyau manuel de haute qualité ([`conv_bot.json`](../conv_bot.json)) + WhatsApp filtré ([`conversations_dataset.json`](../conversations_dataset.json))** :
   - [`conv_bot.json`](../conv_bot.json) contient déjà d'excellents dialogues naturels `user` / `assistant`.
2. **Les meilleures paires naturelles `Parent -> Enfant` (YouTube / TikTok)** :
   - Celles où le **Parent pose une question ou exprime un problème** (présence d'un `?` ou de mots interrogatifs : *pourquoi, comment, qui, quand, où, quel, wanda, est-ce que...*) ET où **l'Enfant apporte une réponse développée en Camfranglais** ($N \ge 10$ mots et $S \ge \tau_{\text{haut}}$).
3. **La Génération Synthétique par "Rétro-Instruction" (*Back-translation*)** :
   - Tu prends un long commentaire riche du réservoir CPT (ex: $N \ge 15$ mots où quelqu'un raconte une anecdote ou donne un avis en pur Camfranglais) qui devient la **Réponse cible (`assistant`)**.
   - Tu demandes à un LLM (via une API gratuite comme Gemini Flash ou un modèle ouvert) :
     > *"Voici une réponse écrite en camfranglais par un jeune Camerounais : `[COMMENTAIRE]`. Génère uniquement le message court d'un utilisateur (en français standard ou en camfranglais) qui aurait naturellement provoqué cette réponse dans une discussion entre amis."*

---

## 🛠️ Guide des Exercices du Notebook `03_Dataset_Builder_and_HF_Push.ipynb`

1. **`TODO 1` — Filtre de Qualité Dialogique pour les paires `Parent -> Enfant`** :
   - Vérifier que le Parent est interrogatif ou engageant et que l'Enfant est développé ($N \ge 10$, $S \ge \tau_{\text{haut}}$) sans être un simple acquiescement générique.
2. **`TODO 2` — Rétro-Instruction (Back-translation) pour convertir un long commentaire CPT en paire SFT** :
   - Construire le prompt de rétro-instruction et formater la paire `{"conversations": [{"role": "user", "content": generated_question}, {"role": "assistant", "content": cpt_comment}]}`.
3. **`TODO 3` — Assemblage, Déduplication & Split Train / Test (`sft_train` vs `sft_test`)** :
   - Mettre de côté $100$ paires (ou $10\%$) dans `sft_test` pour l'évaluation rigoureuse de l'Étape 5 (en veillant à ce qu'aucune paire de `sft_test` ne fuite dans `sft_train` !).
4. **`TODO 4` — Publication sur Hugging Face Hub (`push_to_hub`)**.


---

## 🔄 4. Du Notebook au Script Python Natif ([`src/data_pipeline/builder.py`](../src/data_pipeline/builder.py)) & Jumelage Complet

Une fois tes fonctions validées dans [`03_Dataset_Builder_and_HF_Push.ipynb`](03_Dataset_Builder_and_HF_Push.ipynb) :
1. Ouvre [`src/data_pipeline/builder.py`](../src/data_pipeline/builder.py) et reporte le code de tes `TODO 1` à `TODO 4` :
   - `is_good_sft_dialogue_pair()`
   - `build_backtranslation_prompt()` et `create_sft_entry()`
   - `deduplicate_and_split_sft()`
   - `build_hf_dataset_dict()`

### 🔗 Jumeler toute la Data Pipeline en une seule commande ([`src/data_pipeline/run_pipeline.py`](../src/data_pipeline/run_pipeline.py))
Maintenant que [`scraper.py`](../src/data_pipeline/scraper.py), [`scorer.py`](../src/data_pipeline/scorer.py) et [`builder.py`](../src/data_pipeline/builder.py) sont complétés, regarde le fichier **[`src/data_pipeline/run_pipeline.py`](../src/data_pipeline/run_pipeline.py)** :
- Il importe tes 3 modules Python natifs.
- Il lit automatiquement [`_chat.txt`](../_chat.txt), `data/raw/raw_comments.jsonl` et [`conv_bot.json`](../conv_bot.json).
- Il sépare les arbres `Parent -> Enfant` des commentaires isolés, filtre tout avec ta formule $S$, déduplique, génère `data/final/cpt_train.json`, `data/final/sft_train.json` et `data/final/sft_test.json`, et prépare le push vers Hugging Face Hub !

Tu peux lancer toute ta pipeline locale d'un seul coup dans ton terminal avec :
```bash
python -m src.data_pipeline.run_pipeline
```
