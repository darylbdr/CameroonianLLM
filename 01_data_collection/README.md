# 📥 Étape 1 — Collecte & Extraction Multi-Sources en Arbres (`Parent -> Enfant`) 💻 *(Notebook Local)*

> **Environnement d'exécution** : **En local sur ton PC (CPU)** via VS Code ou Jupyter Lab.
> **Objectif de cette étape** : Construire le pipeline d'acquisition de données brutes pour le Camfranglais en combinant tes données locales existantes (WhatsApp [`../_chat.txt`](../_chat.txt) et [`../conv_bot.json`](../conv_bot.json)) avec une collecte automatisée sur **YouTube Data API v3**, tout en préservant la structure en **arbres de discussion**.

---

## 1. Pourquoi scraper des *arbres* plutôt que du texte plat ?

Comme ton objectif final est de créer un **chatbot conversationnel** (via la stratégie hybride **CPT $\rightarrow$ SFT**), la manière dont tu collectes et stockes les données dès la première étape change tout :

1. **Le Réservoir CPT (Texte brut isolé)** :
   - Sur YouTube ou TikTok, beaucoup de commentaires de premier niveau (*top-level comments*) n'ont pas de réponses.
   - S'ils sont longs et riches en Camfranglais, ils serviront au **Continued Pre-training (CPT)** pour enseigner le vocabulaire, la grammaire et les expressions au modèle.

2. **Le Réservoir SFT (Fils de discussion `Parent -> Enfant`)** :
   - Lorsqu'un commentaire principal $A$ (`parent`) reçoit une ou plusieurs réponses $B, C$ (`replies`), tu obtiens naturellement des paires de dialogue écrites par de vrais humains :
     $$\text{Message Utilisateur (Parent } A\text{)} \longrightarrow \text{Réponse du Chatbot (Enfant } B\text{)}$$
   - En sauvegardant dès le départ `comment_id` et `parent_id`, tu n'as plus besoin d'inventer des dizaines de milliers d'instructions à la main !

---

## 2. Analyse de ton Code Existant ([`Cleansing_Dataset.py`](../Cleansing_Dataset.py))

Dans ton script actuel [`Cleansing_Dataset.py`](../Cleansing_Dataset.py), tu as déjà codé deux briques très utiles :
- **`parse_ios_whatsapp(file_path)`** : Extrait les messages d'un export WhatsApp iOS (`[DD/MM/YYYY, HH:MM:SS] Auteur: Message`), regroupe les messages consécutifs d'un même interlocuteur, et crée des paires `Message i (user) -> Message i+1 (assistant)`.
- **`load_conv_bot(conv_path)`** : Charge [`conv_bot.json`](../conv_bot.json) en nettoyant les éventuels commentaires `//` ou virgules flottantes.

### 🔍 Ce qu'il faut améliorer sur la partie WhatsApp :
Dans [`Cleansing_Dataset.py`](../Cleansing_Dataset.py) (lignes 124-135), **chaque message est utilisé deux fois** dans une fenêtre glissante (`msg_0 -> msg_1`, puis `msg_1 -> msg_2`), et **même si `msg_{i+1}` a été envoyé 3 jours plus tard** sur un tout autre sujet !
- **Amélioration guidée dans le Notebook 01** : Ajouter un filtre de **fenêtre temporelle** (ex: si l'écart entre le message $A$ et le message $B$ dépasse $30$ minutes, ce n'est pas une réponse directe, mais un nouveau sujet isolé qui ira dans le réservoir CPT !).

---

## 3. Collecte YouTube Data API v3 en Local (La Fondation Propre et Rapide)

L'API officielle **YouTube Data API v3** est gratuite (quota de $10\,000$ unités/jour, sachant qu'un appel `commentThreads.list` ne coûte qu'**$1$ unité** et récupère jusqu'à $100$ fils de commentaires d'un coup !).
- Ajoute ta clé dans ton fichier local `.env` à la racine du projet : `YOUTUBE_API_KEY="ta_cle_ici"`.

### Schéma JSONL brut à sauvegarder (`../data/raw/raw_comments.jsonl`)
Chaque ligne de ton fichier local `../data/raw/raw_comments.jsonl` doit respecter cette structure :

```json
{
  "comment_id": "UgzX9...",
  "parent_id": null,
  "text": "Gars le son là est trop djoss, le mbom a tout cassé sur le beat !",
  "like_count": 142,
  "source_video_title": "Clip Officiel Mbolé 237",
  "source_platform": "youtube"
}
```

---

## 🛠️ Guide des Exercices du Notebook Local `01_Data_Collection_Scraping.ipynb`

Ouvre [`01_Data_Collection_Scraping.ipynb`](01_Data_Collection_Scraping.ipynb) dans VS Code ou Jupyter Lab sur ton PC et complète les **4 blocs `# TODO`** :

1. **`TODO 1` — Amélioration du Parser WhatsApp sur `../_chat.txt` avec Fenêtre Temporelle ($\Delta t \le 30\text{ min}$)**.
2. **`TODO 2` — Extraction d'une vidéo YouTube en Arbres (`parse_youtube_thread_item`)**.
3. **`TODO 3` — Système de Sauvegarde Incrémentale dans `../data/raw/raw_comments.jsonl`**.
4. **`TODO 4` — Reconstruction des Paires `Parent -> Enfant` vs Commentaires Isolés (`split_trees_and_isolated`)**.


---

## 🔄 4. Du Notebook au Script Python Natif ([`src/data_pipeline/scraper.py`](../src/data_pipeline/scraper.py))

En ingénierie MLOps, **un notebook sert à explorer, prototyper et valider visuellement ses fonctions**, tandis qu'**un fichier `.py` sert à industrialiser et réutiliser le code** dans les étapes suivantes sans copier-coller de cellules.

Une fois que les 4 cellules de test (`assert`) de ton notebook [`01_Data_Collection_Scraping.ipynb`](01_Data_Collection_Scraping.ipynb) passent au vert :
1. Ouvre le fichier [`src/data_pipeline/scraper.py`](../src/data_pipeline/scraper.py).
2. Reporte le code de tes `TODO 1`, `TODO 2`, `TODO 3` et `TODO 4` dans les fonctions correspondantes :
   - `parse_whatsapp_to_records()`
   - `parse_youtube_thread_item()`
   - `fetch_youtube_video_comments()`
   - `split_trees_and_isolated()`
3. Tes fonctions deviennent alors directement importables par l'orchestrateur [`src/data_pipeline/run_pipeline.py`](../src/data_pipeline/run_pipeline.py) :
   ```python
   from src.data_pipeline.scraper import parse_whatsapp_to_records, split_trees_and_isolated
   ```
