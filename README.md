# 🇨🇲 CameroonianAI — Chatbot Conversationnel en Camfranglais (QLoRA & Approche Hybride CPT $\rightarrow$ SFT)

Bienvenue dans le dépôt **CameroonianAI** ! Ce projet a un objectif double :
1. **Objectif technique** : Entraîner un grand modèle de langage (LLM, ici `Mistral-Nemo-Instruct-2407-bnb-4bit` via **Unsloth** et **QLoRA**) pour en faire un **chatbot naturel et fluide en Camfranglais**.
2. **Objectif pédagogique** : Maîtriser de bout en bout un pipeline MLOps complet selon le pattern **Local CPU $\rightarrow$ Hugging Face Hub $\rightarrow$ Colab GPU**.

---

## 🏗️ Architecture MLOps : Notebooks/Scripts Locaux (CPU) vs Notebook Colab (GPU)

Conformément à notre discussion (page 11 du PDF), **toute la préparation des données et l'analyse d'évaluation tournent en local sur le CPU de ton PC** (dans VS Code ou Jupyter Lab), et **seul l'entraînement QLoRA (Étape 4) tourne sur le GPU de Google Colab** :

```mermaid
flowchart LR
    subgraph LocalPC ["💻 Sur ton PC en Local (CPU — VS Code / Jupyter)"]
        A["01_data_collection<br/>(Local : YouTube API / TikTok)"] --> B["02_filtering_scoring<br/>(Local : Formule S & Lexique)"]
        B --> C["03_dataset_builder<br/>(Local : CPT + SFT + Rétro-Prompting)"]
        E["05_evaluation_and_deployment<br/>(Local : Triptyque Métriques & GGUF Local)"]
    end
    C -->|"dataset.push_to_hub()"| HF[("🤗 Hugging Face Hub<br/>Dataset & Modèle")]
    subgraph ColabGPU ["☁️ Sur Google Colab (GPU Tesla T4)"]
        HF -->|"load_dataset()"| D["04_hybrid_training_qlora<br/>(Colab GPU : Phase 1 CPT -> Phase 2 SFT + Export GGUF)"]
    end
    D -->|"push_to_hub() + réponses test"| HF
    HF -->|"Téléchargement résultats & GGUF"| E
```

---

## 🧠 Pourquoi une Stratégie d'Entraînement Hybride (CPT $\rightarrow$ SFT) ?

| Approche | Volume de données | Ce que le modèle apprend | Limite principale |
| :--- | :--- | :--- | :--- |
| **1. Continued Pre-training (CPT) seul** | Grand (Texte brut : commentaires isolés) | Assimile le vocabulaire (*wanda, mbom, tchop, ngata*), le "flow" et les références culturelles via $P(w_t \mid w_1, \dots, w_{t-1})$. | **Catastrophic Forgetting** : le modèle désapprend son comportement d'assistant et répond comme une section commentaires TikTok/YouTube. |
| **2. Supervised Fine-Tuning (SFT) seul** | Moyen (Paires `User` $\rightarrow$ `Assistant`) | Apprend la posture d'interlocuteur et quand s'arrêter de parler (`<|eot_id|>` / `</s>`). | Camfranglais "scolaire" ou limité s'il ne possède pas déjà un grand réservoir d'expressions idiomatiques. |
| **3. Stratégie Hybride (CPT $\rightarrow$ Petit SFT)** ✅ | **Gros volume brut (CPT) + 1 000 à 2 000 paires de qualité (SFT)** | **Phase 1 (CPT)** : Injecte la langue, l'argot et la syntaxe. <br>**Phase 2 (SFT)** : Réapprend au modèle à dialoguer naturellement grâce au masquage de la *loss* sur l'instruction. | **Le "Sweet Spot" optimal !** |

---

## ⚙️ Installation de l'Environnement Local (pour les Étapes 1, 2, 3 et 5)

1. Installe les dépendances Python légères sur ton PC :
   ```bash
   pip install -r requirements.txt
   ```
2. Renseigne tes clés (`YOUTUBE_API_KEY`, `HF_TOKEN`, `GEMINI_API_KEY`) dans le fichier `.env` à la racine du dépôt (voir [`.env.example`](.env.example)).

---

## 🗺️ Feuille de Route des 5 Étapes

### 📦 Configuration Globale
- **[`config/lexicon.json`](config/lexicon.json)** : Le dictionnaire statique du Camfranglais (marqueurs exclusifs à poids fort $w_i = 2.0$, marqueurs partagés à poids moyen $w_i = 1.0$, ancres syntaxiques françaises $C_{\text{fr}}$ et ancres Pidgin/Anglais $C_{\text{en/pid}}$).
- **[`config/youtube_seeds.json`](config/youtube_seeds.json)** : Liste de requêtes et d'IDs de vidéos YouTube camerounaises à scraper (clips Mbolé, rap 237, sketchs, web-séries, débats urbains).
- **[`conv_bot.json`](conv_bot.json)** : Noyau de conversations SFT manuelles de haute qualité en Camfranglais.

### 1️⃣ [Étape 1 — Collecte & Extraction en Arbres (`01_data_collection/`)](01_data_collection/README.md) 💻 *Exécution : EN LOCAL*
- **Objectif** : Collecter les commentaires YouTube Data API v3 (`commentThreads.list`) en préservant la hiérarchie **Commentaires Isolés (CPT)** vs **Fils `Parent -> Enfant` (SFT)** avec reprise sur erreur (`scraped_videos.txt`).
- **Supports** : [📖 Guide Étape 1](01_data_collection/README.md) | [📓 Notebook Local](01_data_collection/01_Data_Collection_Scraping.ipynb) | [🐍 Module `scraper.py`](src/data_pipeline/scraper.py)

### 2️⃣ [Étape 2 — Filtrage Linguistique & Formule de Scoring $S$ (`02_filtering_scoring/`)](02_filtering_scoring/README.md) 💻 *Exécution : EN LOCAL*
- **Objectif** : Coder le *Funnel Pipeline* en 3 étages et la formule mathématique purement linguistique qui distingue le vrai Camfranglais du Français standard et du Pidgin/Anglais pur :
  $$S = \mathbb{I}(N \ge 6) \times D_{\text{cam}} \times P_{\text{matrice}}$$
- **Supports** : [📖 Guide Étape 2](02_filtering_scoring/README.md) | [📓 Notebook Local](02_filtering_scoring/02_Filtering_and_Scoring.ipynb) | [🐍 Module `scorer.py`](src/data_pipeline/scorer.py)

### 3️⃣ [Étape 3 — Construction des Datasets (CPT & SFT) & Push HF (`03_dataset_builder/`)](03_dataset_builder/README.md) 💻 *Exécution : EN LOCAL*
- **Objectif** : Assembler `cpt_train.json`, `sft_train.json` et `sft_test.json` en appliquant la **Règle d'Or SFT** (seul l'Enfant doit passer le filtre $S \ge \tau_{\text{haut}}$), fusionner [`conv_bot.json`](conv_bot.json), générer des paires synthétiques par *Back-translation*, et pousser vers Hugging Face Hub depuis ton PC.
- **Supports** : [📖 Guide Étape 3](03_dataset_builder/README.md) | [📓 Notebook Local](03_dataset_builder/03_Dataset_Builder_and_HF_Push.ipynb) | [🐍 Module `builder.py`](src/data_pipeline/builder.py)

### 4️⃣ [Étape 4 — Entraînement Hybride QLoRA : CPT $\rightarrow$ SFT (`04_hybrid_training_qlora/`)](04_hybrid_training_qlora/README.md) ☁️ *Exécution : SUR GOOGLE COLAB (GPU)*
- **Objectif** : **Le seul notebook qui tourne sur Google Colab GPU !**
  1. Téléchargement instantané du dataset depuis Hugging Face Hub (`load_dataset`).
  2. **Phase 1 (CPT)** : Apprentissage de la langue brute avec *Packing* (`packing=True`).
  3. **Phase 2 (SFT Conversationnel)** : Alignement dialogue avec le *Chat Template* natif de Mistral (`[INST] ... [/INST]`) et masquage de la *loss* (`train_on_responses_only`).
  4. Génération des réponses sur `sft_test` + export **GGUF (`q4_k_m`)** vers Hugging Face Hub.
- **Supports** : [📖 Guide Étape 4](04_hybrid_training_qlora/README.md) | [📓 Notebook Colab GPU](04_hybrid_training_qlora/04_Hybrid_Training_CPT_and_SFT.ipynb)

### 5️⃣ [Étape 5 — Évaluation (Le Triptyque de Métriques) & Inférence Locale (`05_evaluation_and_deployment/`)](05_evaluation_and_deployment/README.md) 💻 *Exécution : EN LOCAL*
- **Objectif** : Évaluer scientifiquement en local sur ton PC les réponses générées par le modèle sur `sft_test` via nos 3 métriques complémentaires :
  1. **Perplexité** : $\text{PPL} = \exp(\text{Loss})$
  2. **Camfranglais Adherence Score** : Comparaison de la distribution $(\mu, \sigma)$ du score $S$ du modèle avec celle des humains.
  3. **Pertinence Sémantique** : Similarité Cosinus des embeddings multilingues & LLM-as-a-Judge.
- **Supports** : [📖 Guide Étape 5](05_evaluation_and_deployment/README.md) | [📓 Notebook Local](05_evaluation_and_deployment/05_Evaluation_and_GGUF_Export.ipynb) | [🐍 Module `evaluator.py`](src/data_pipeline/evaluator.py)

---

## 🚀 Exécuter toute la Data Pipeline en une seule commande

Le script orchestrateur **[`src/data_pipeline/run_pipeline.py`](src/data_pipeline/run_pipeline.py)** jumelle automatiquement `scraper.py`, `scorer.py` et `builder.py` :

```bash
python -m src.data_pipeline.run_pipeline
```
