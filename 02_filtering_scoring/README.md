# 🧮 Étape 2 — Filtrage Linguistique & Formule Mathématique de Scoring ($S$)

> **Objectif de cette étape** : Construire le filtre algorithmique (*Funnel Pipeline*) capable de détecter automatiquement si une phrase est réellement en **Camfranglais**, tout en rejetant le bruit des réseaux sociaux, le **Français standard** et les faux positifs en **Pidgin/Anglais pur**.

---

## 1. Pourquoi les détecteurs de langue classiques échouent-ils ?

Le Camfranglais repose sur du *code-switching* : c'est **un lexique argotique et anglophone/pidgin inséré dans une matrice syntaxique française**.
- Si tu passes *"Je wanda comment le gars va tchop tout l'argent"* dans `langdetect` ou `FastText`, ils vont le classer en Français ou en Anglais avec un score de confiance médiocre.
- Pire : si tu filtres uniquement par présence de mots-clés comme `wanda` ou `tchop`, tu vas laisser passer des phrases en **Pidgin pur** (*"I di wanda how that masa go tchop all the money"*), ce qui polluerait ton chatbot !

Pour éviter de payer une API de LLM sur $500\,000$ commentaires bruts, nous construisons un **pipeline en entonnoir (Funnel Pipeline) en 3 étages** sur CPU.

---

## 2. Les 3 Étages de l'Entonnoir & La Formule Mathématique $S$

### Étage 1 : Nettoyage Structurel & Filtre de Longueur ($\mathbb{I}(N \ge 6)$)
On enrichit ta fonction `clean_chat_text()` de [`Cleansing_Dataset.py`](../Cleansing_Dataset.py) :
1. **Suppression** des URLs, des mentions `@user`, et des hashtags `#...` en fin de phrase.
2. **Dédoublonnage des émojis** : On ne supprime pas tous les émojis (un chatbot en Camfranglais sans émojis paraîtrait froid), mais on réduit les répétitions (`"🤣🤣🤣😭😭"` $\rightarrow$ `"🤣😭"`).
3. **Comptage des mots alphabétiques ($N$)** : On extrait uniquement les mots alphabétiques (`re.findall(r"[a-zA-ZÀ-ÿ']+", text.lower())`). Si $N < 6$, la fonction indicatrice $\mathbb{I}(N \ge 6)$ vaut $0$ (élimine instantanément les *"Mdr le gars"*, *"First"*, *"Force à toi boss"*).

---

### Étage 2 : La Densité Lexicale Pondérée ($D_{\text{cam}}$) & Tolérance Phonétique
À partir du dictionnaire statique [`config/lexicon.json`](../config/lexicon.json), chaque mot du commentaire reçoit un poids $w_i$ :
- **Poids fort ($w_i = 2.0$) — Marqueurs exclusifs 237** : Mots qui n'existent ni en français, ni en anglais standard, ni en pidgin classique (*mbom, ndem, ngata, feyman, mbindi, kolo, mouf, boboh, nga, djoss, pala, chab, guif, grap...*).
- **Poids moyen ($w_i = 1.0$) — Marqueurs partagés avec le Pidgin/Anglais** : Mots essentiels en Camfranglais mais ambigus s'ils sont utilisés seuls (*tchop, wanda, fala, go, know, lap, massa, ashia, buy, ways, send, look...*).
- **Autres mots ($w_i = 0.0$)**.

Pour un commentaire de $N$ mots alphabétiques, la densité brute est :
$$D_{\text{cam}} = \frac{\sum_{i=1}^{N} w_i}{N}$$

#### 💡 Gérer l'orthographe phonétique (*wanda, awanda, wanday*)
Comme l'argot camerounais a un vocabulaire stable mais que chacun l'écrit au son sur les réseaux sociaux, notre fonction de matching d'un mot `word` procède en 3 niveaux :
1. **Match exact** dans `strong_markers` ou `medium_markers`.
2. **Match de racine (sous-chaîne $\ge 4$ lettres)** : ex: `wanday` ou `awanda` contient la racine `wanda` ; `tchopper` contient `tchop`.
3. **Distance d'édition légère** (ex: `difflib.SequenceMatcher` avec ratio $\ge 0.84$) pour capter les fautes de frappe (`mboutmann`, `feymann`).

---

### Étage 3 : Le *Syntax Anchor* Anti-Pidgin / Anti-Anglais ($P_{\text{matrice}}$)
C'est l'ingrédient secret pour éliminer les faux positifs en Pidgin ou Anglais pur ! On compte dans la phrase :
- **$C_{\text{fr}}$ (Ancres françaises)** : Pronoms, prépositions, articles et auxiliaires français (*je, tu, il, on, le, la, les, un, une, des, dans, avec, pour, mais, c'est, qu'il, va, est...*).
- **$C_{\text{en/pid}}$ (Ancres Pidgin / Anglais)** : Mots-outils anglais ou marqueurs grammaticaux du pidgin (*the, that, how, with, di, dey, wetin, pikin, una, dem, na, wey...*).

On calcule alors le facteur de pénalité de matrice $P_{\text{matrice}} \in [0, 1]$ :
$$P_{\text{matrice}} = \frac{C_{\text{fr}}}{C_{\text{fr}} + C_{\text{en/pid}} + \epsilon} \quad (\text{avec } \epsilon = 10^{-5})$$

- **Cas 1 (Pidgin pur)** : *"I di wanda how that masa go tchop all the money"* $\rightarrow C_{\text{fr}} = 0, C_{\text{en/pid}} = 5 \implies P_{\text{matrice}} = 0 \implies \mathbf{S = 0}$ (Rejeté !).
- **Cas 2 (Camfranglais)** : *"Je wanda comment le gars va tchop tout l'argent"* $\rightarrow C_{\text{fr}} = 5, C_{\text{en/pid}} = 0 \implies P_{\text{matrice}} \approx 1 \implies \mathbf{S > 0}$ (Validé !).

---

## 3. Formule Finale & Les 3 Zones de Routage

Sans le bonus d'engagement social (qui biaisait le score envers les vieux commentaires populaires), notre score linguistique pur s'écrit :

$$\boxed{S = \mathbb{I}(N \ge 6) \times D_{\text{cam}} \times P_{\text{matrice}}}$$

Au lieu d'un seuil binaire brutal, on route chaque commentaire dans **3 zones** (avec par exemple $\tau_{\text{haut}} = 0.15$ et $\tau_{\text{bas}} = 0.05$) :
1. **Zone Haute ($S \ge \tau_{\text{haut}}$)** : Camfranglais certifié $\rightarrow$ retenu directement pour le dataset !
2. **Zone Grise ($\tau_{\text{bas}} \le S < \tau_{\text{haut}}$)** : Camfranglais léger ou orthographe atypique $\rightarrow$ mis de côté pour inspection.
3. **Zone Basse ($S < \tau_{\text{bas}}$)** : Français standard ($D_{\text{cam}} = 0$) ou Pidgin/Anglais ($P_{\text{matrice}} \approx 0$) $\rightarrow$ rejeté.

> ⚠️ **Règle d'Or pour les paires SFT (`Parent -> Enfant`)** :
> Dans une paire de dialogue, **seul l'Enfant (la réponse de l'assistant) DOIT être en Zone Haute ($S_{\text{enfant}} \ge \tau_{\text{haut}}$)**. Le Parent peut être en Français standard ($S_{\text{parent}} = 0$) ou en Camfranglais ! Cela permet à ton chatbot de répondre en Camfranglais même quand on lui pose une question en français classique.

---

## 🛠️ Guide des Exercices du Notebook `02_Filtering_and_Scoring.ipynb`

1. **`TODO 1` — Tokenisation & Filtre de Longueur ($\mathbb{I}(N \ge 6)$)**.
2. **`TODO 2` — Matching Lexical avec Tolérance Phonétique & Calcul de $D_{\text{cam}}$**.
3. **`TODO 3` — Calcul du *Syntax Anchor* $P_{\text{matrice}}$ et du Score Global $S$**.
4. **`TODO 4` — Application de la Règle d'Or sur ton fichier existant [`conversations_dataset.json`](../conversations_dataset.json)** pour mesurer combien de paires WhatsApp passent réellement le filtre Camfranglais !


---

## 🔄 4. Du Notebook au Script Python Natif ([`src/data_pipeline/scorer.py`](../src/data_pipeline/scorer.py))

Ton module de scoring est **la brique centrale** du projet : il est utilisé à la fois à l'Étape 3 (pour construire les datasets CPT et SFT) et à l'Étape 5 (pour calculer le *Camfranglais Adherence Score* lors de l'évaluation du modèle).

Une fois que tous tes `assert` sont validés dans [`02_Filtering_and_Scoring.ipynb`](02_Filtering_and_Scoring.ipynb) :
1. Ouvre [`src/data_pipeline/scorer.py`](../src/data_pipeline/scorer.py).
2. Reporte tes implémentations validées dans :
   - `normalize_and_tokenize()` (`TODO 1`)
   - `compute_d_cam()` (`TODO 2`)
   - `compute_p_matrice()` et `score_camfranglais()` (`TODO 3`)
   - `filter_sft_conversations()` (`TODO 4`)
3. Tu pourras ainsi l'importer proprement depuis n'importe quel autre script ou notebook :
   ```python
   from src.data_pipeline.scorer import load_lexicon, score_camfranglais, filter_sft_conversations
   ```
