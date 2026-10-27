# ⚡ Étape 4 — Entraînement Hybride QLoRA avec Unsloth (Phase 1 : CPT $\rightarrow$ Phase 2 : SFT)

> **Objectif de cette étape** : Reprendre ton notebook existant [`CameroonianFineTuning.ipynb`](../CameroonianFineTuning.ipynb) basé sur **Unsloth** et `Mistral-Nemo-Instruct-2407-bnb-4bit`, corriger ses subtilités d'entraînement, et implémenter le **pipeline hybride en deux phases** :
> 1. **Phase 1 — Continued Pre-training (CPT)** sur les commentaires isolés avec *Packing*.
> 2. **Phase 2 — Supervised Fine-Tuning (SFT)** sur les dialogues avec le vrai *Chat Template* et **masquage de la *loss* sur l'instruction** (`train_on_responses_only`).

---

## 1. Retour d'Expérience sur ton Notebook Existant ([`CameroonianFineTuning.ipynb`](../CameroonianFineTuning.ipynb))

Dans [`CameroonianFineTuning.ipynb`](../CameroonianFineTuning.ipynb), tu avais déjà mis en place d'excellentes fondations :
- ✅ **Quantification 4-bit (QLoRA)** avec `FastLanguageModel.from_pretrained("unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit", load_in_4bit=True)`.
- ✅ **Ciblage complet des couches linéaires d'attention et du MLP** (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) avec `r = 16` et `use_gradient_checkpointing = "unsloth"`.
- ✅ **Optimiseur 8-bit (`adamw_8bit`)** pour économiser la VRAM de la Tesla T4 (15 Go).

### 🔍 Les 3 Points Critiques que nous Corrigeons dans cette Étape :

1. **Incohérence de Format entre Entraînement et Inférence (*Template Mismatch*)** :
   - **Dans ton ancien notebook** : L'entraînement formattait les données avec `alpaca_prompt` (`Below is an instruction... ### Instruction: ... ### Response:`), **MAIS** la cellule d'inférence utilisait `tokenizer.apply_chat_template(messages)` (qui génère `[INST] ... [/INST]`).
   - **Conséquence** : En inférence, le modèle ne voyait pas les balises `### Response:` qu'il avait apprises pendant le fine-tuning !
   - **Correction** : Utiliser `tokenizer.apply_chat_template(conversations, tokenize=False)` **à la fois pendant le SFT et pendant l'inférence**.

2. **Le Masquage de la Loss en SFT (`train_on_responses_only`)** :
   - En CPT, la loss est calculée sur **tous les tokens** de la séquence :
     $$\mathcal{L}_{\text{CPT}} = -\sum_{t=1}^{T} \log P(w_t \mid w_1, \dots, w_{t-1})$$
   - En SFT, si tu ne masques pas la question de l'utilisateur (`user`), le modèle gaspille sa capacité à apprendre à générer des questions en français standard ! Avec `train_on_responses_only` d'Unsloth, les tokens de la question reçoivent le label `-100` (ignoré par PyTorch `CrossEntropyLoss`), et le modèle n'est pénalisé **que sur la réponse de l'assistant**.

3. **Le Warning `per_device_train_batch_size = 1` et le Learning Rate** :
   - Dans [`CameroonianFineTuning.ipynb`](../CameroonianFineTuning.ipynb) (ligne 413), TRL affichait : *"Using a batch size of 1 annihilates the benefits of padding-free training. Please consider increasing the batch size to at least 2."*
   - Nous passons à `per_device_train_batch_size = 2` et `gradient_accumulation_steps = 4` (soit un *batch effectif* de $8$), avec un *learning rate* de $1\times 10^{-4}$ en CPT puis $5\times 10^{-5}$ en SFT pour ne pas écraser les connaissances acquises.

> 🔐 **Note de Sécurité Importante** : Dans la dernière cellule de [`CameroonianFineTuning.ipynb`](../CameroonianFineTuning.ipynb) (lignes 1026-1027), ton token Hugging Face `hf_...` était écrit en clair au lieu d'utiliser la variable `hf_token = userdata.get('hf_token')`. Pense à révoquer/régénérer ce token sur Hugging Face et à toujours utiliser `userdata.get('hf_token')` comme dans le notebook de cette étape !

---

## 2. Faut-il dégeler `embed_tokens` et `lm_head` pour le Camfranglais ?

C'est une question classique quand on adapte un LLM à une nouvelle langue :
- **Quand faut-il ajouter `embed_tokens` et `lm_head` aux `target_modules` ?** Quand la langue utilise un nouvel alphabet ou des milliers de racines complètement absentes du tokenizer. Mais cela ajoute $\sim 200\text{ Mo}$ à $1\text{ Go}$ de paramètres entraînables en VRAM.
- **Pourquoi les 7 couches `q, k, v, o, gate, up, down` avec `r=16` ou `r=32` suffisent sur T4 ?** Parce que le Camfranglais s'écrit en alphabet latin et que le tokenizer de Mistral-Nemo découpe déjà très bien les sous-mots (`mbom`, `ngata`, `tchop`, `wanda`) en 1 ou 2 sub-tokens. Adapter les couches MLP (`gate_proj`, `up_proj`, `down_proj`) et d'attention suffit à réassocier ces sub-tokens au bon sens sémantique tout en tenant largement dans les 15 Go d'une Tesla T4 gratuite !

---

## 🛠️ Guide des Exercices du Notebook `04_Hybrid_Training_CPT_and_SFT.ipynb`

1. **`TODO 1` — Configuration QLoRA de Mistral-Nemo avec Unsloth**.
2. **`TODO 2` — Phase 1 : Préparation du Texte Brut & Entraînement CPT (`packing=True`)**.
3. **`TODO 3` — Phase 2 : Formatage Chat Template & Masquage de la Loss (`train_on_responses_only`) pour le SFT**.
4. **`TODO 4` — Sauvegarde Sécurisée des Adaptateurs LoRA sur Hugging Face Hub**.
