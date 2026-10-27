# 📊 Étape 5 — Évaluation Rigoureuse (Le Triptyque de Métriques) & Inférence Locale GGUF 💻 *(Notebook Local)*

> **Environnement d'exécution** : **En local sur ton PC (CPU)** via VS Code ou Jupyter Lab.
> **Objectif de cette étape** : Évaluer scientifiquement ton chatbot en Camfranglais sur ton jeu de test (`../data/final/sft_test.json`) avec 3 métriques complémentaires sur CPU, et faire tourner ton modèle **GGUF** directement sur ton PC !

---

## 1. Pourquoi ne peut-on pas évaluer le modèle uniquement avec la formule $S$ ?

Ton idée de réutiliser la formule mathématique de filtrage $S = \mathbb{I}(N \ge 6) \times D_{\text{cam}} \times P_{\text{matrice}}$ pour évaluer le modèle est excellente, mais avec une nuance fondamentale :
- **$S$ mesure le *style linguistique*** (est-ce que le modèle parle en vrai Camfranglais ?).
- **$S$ ne mesure PAS la *cohérence sémantique*** (est-ce que le modèle répond intelligemment à la question posée ?).

> **Contre-exemple** : Un modèle qui aurait "vrillé" (*mode collapse*) et répondrait à toutes tes questions par *"Gars le mbom a ndem il a tchop le kolo au ngata"* obtiendrait un score $S$ exceptionnel ($S > 0.80$), alors que c'est un très mauvais chatbot !

Pour prouver que ton entraînement hybride (CPT $\rightarrow$ SFT) a réussi, nous calculons **en local sur ton PC** un triptyque de métriques complémentaires sur le jeu de test `sft_test`.

---

## 2. Le Triptyque de Métriques d'Évaluation (Calculé en Local sur CPU)

### Métrique 1 : La Perplexité ($\text{PPL} = \exp(\text{Loss})$) — *La Surprise Mathématique*
- **Définition** : À partir de l'`eval_loss` sur le jeu de test `sft_test`, on calcule :
  $$\text{PPL} = \exp\left( -\frac{1}{T} \sum_{t=1}^{T} \log P_{\theta}(w_t \mid w_{<t}) \right)$$
- **Interprétation** : Plus la perplexité baisse après ton CPT et ton SFT, moins le modèle est "surpris" par les tournures de phrases et l'argot du Camfranglais.

---

### Métrique 2 : Le *Camfranglais Adherence Score* ($S$ adapté) — *Le Dosage Naturel du Style*
- **Protocole** :
  1. On applique la fonction `score_camfranglais()` (utilisant ton fichier local [`../config/lexicon.json`](../config/lexicon.json)) sur les réponses générées par le modèle et sur les vraies réponses humaines de `../data/final/sft_test.json`.
  2. Au lieu de chercher à maximiser $S \to \infty$, on compare la **distribution (moyenne $\mu$ et écart-type $\sigma$)** du score $S$ du modèle avec celle des humains :
     $$\Delta_{\text{style}} = \left| \mu_{\text{modèle}} - \mu_{\text{humain}} \right|$$
  - Si les humains ont un $\mu_{\text{humain}} = 0.25$ et que ton modèle est à $0.23$, son dosage de Camfranglais est ultra naturel !

---

### Métrique 3 : Pertinence Sémantique (Similarité Cosinus & LLM-as-a-Judge) — *Le Sens*
Pour vérifier que le chatbot répond bien à la question posée :
1. **Similarité Cosinus d'Embeddings Multilingues (100% CPU)** :
   - On encode la réponse générée $\vec{u}$ et la vraie réponse humaine $\vec{v}$ avec `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (modèle très léger de ~400 Mo qui tourne en quelques secondes sur le CPU de ton PC) :
     $$\text{Sim}(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \, \|\vec{v}\|}$$
2. **LLM-as-a-Judge (Note sur 5)** :
   - Via l'API gratuite Gemini Flash (`GEMINI_API_KEY` dans ton fichier `../.env`).

---

## 🛠️ Guide des Exercices du Notebook Local `05_Evaluation_and_GGUF_Export.ipynb`

1. **`TODO 1` — Calcul de la Perplexité ($\text{PPL} = \exp(\text{Loss})$)**.
2. **`TODO 2` — Calcul du *Camfranglais Adherence Score* ($\mu_{\text{modèle}}$ vs $\mu_{\text{humain}}$) avec [`../config/lexicon.json`](../config/lexicon.json)**.
3. **`TODO 3` — Calcul de la Similarité Sémantique Cosinus sur CPU avec `SentenceTransformer`**.
4. **`TODO 4` — Inférence Locale du Modèle GGUF sur ton PC**.


---

## 🔄 4. Du Notebook au Script Python Natif ([`src/data_pipeline/evaluator.py`](../src/data_pipeline/evaluator.py))

Une fois tes fonctions d'évaluation validées dans [`05_Evaluation_and_GGUF_Export.ipynb`](05_Evaluation_and_GGUF_Export.ipynb) :
1. Ouvre [`src/data_pipeline/evaluator.py`](../src/data_pipeline/evaluator.py).
2. Reporte le code de tes `TODO 1`, `TODO 2` et `TODO 3` :
   - `compute_perplexity_from_loss()`
   - `evaluate_camfranglais_adherence()`
   - `compute_semantic_similarity()`
3. Tu disposes ainsi d'un module d'évaluation réutilisable à chaque nouvelle version de ton modèle Camfranglais !
