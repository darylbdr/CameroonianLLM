import re
import json

def nettoyer_message(texte):
    # Nettoyage de base
    texte = str(texte)
    if "<Media omitted>" in texte or "image omitted" in texte or "video omitted" in texte:
        return None
    # Suppression des liens et mentions
    texte = re.sub(r'http\S+|www\.\S+', '', texte) #retire les URLs
    texte = re.sub(r'@[^\s@]+', '', texte) #retire les mentions (@ suivi de caractères non-espace)
    texte = re.sub(r'(.)\1{2,}', r'\1\1', texte) # Réduction des caractères répétés (mdrrr -> mdrr)
    texte = re.sub(r'\s+', ' ', texte).strip() # Nettoyage des espaces
    return texte

def parse_ios_whatsapp(chemin_fichier):
    # --- ETAPE 1 : Regex Spécial iOS ---
    # Format typique iOS : [25/12/2023, 14:00:15] Nom: Message
    # On cherche : Crochet ouvrant + Date/Heure + Crochet fermant + Espace + Nom + : + Espace + Message
    regex_ios = r'^\[(\d{2}/\d{2}/\d{2,4},\s+\d{2}:\d{2}:\d{2})\] ([^:]+): (.+)$'
    
    # Liste pour stocker les messages "fusionnés" (Un bloc par personne)
    messages_groupes = []
    current_speaker = None
    current_text_buffer = []

    with open(chemin_fichier, 'r', encoding='utf-8') as f:
        # On lit ligne par ligne
        lignes = f.readlines()

    # --- ETAPE 2 : Extraction et Fusion ---
    for ligne in lignes:
        # On enlève les caractères invisibles bizarres des exports iOS (LTR marks)
        ligne = ligne.replace('\u200e', '').strip()
        
        match = re.match(regex_ios, ligne)
        
        if match:
            # C'est une nouvelle ligne de message
            date, speaker, message = match.groups()
            clean_msg = nettoyer_message(message)
            
            if not clean_msg: # Si le message est vide ou "media omitted", on ignore
                continue

            # LOGIQUE DE FUSION :
            # Si c'est le même parleur qu'avant, on ajoute à son texte
            if speaker == current_speaker:
                current_text_buffer.append(clean_msg)
            else:
                # Si le parleur change, on sauvegarde le bloc précédent
                if current_speaker and current_text_buffer:
                    full_text = " ".join(current_text_buffer)
                    # On ne garde que si assez long (min 2 mots)
                    if len(full_text.split()) >= 2:
                        messages_groupes.append({"role": current_speaker, "content": full_text})
                
                # On commence le nouveau bloc
                current_speaker = speaker
                current_text_buffer = [clean_msg]
        else:
            # Si pas de date, c'est un retour à la ligne du message précédent
            if current_speaker:
                clean_line = nettoyer_message(ligne)
                if clean_line:
                    current_text_buffer.append(clean_line)

    # N'oublie pas d'ajouter le tout dernier message à la fin de la boucle
    if current_speaker and current_text_buffer:
        full_text = " ".join(current_text_buffer)
        if len(full_text.split()) >= 2:
            messages_groupes.append({"role": current_speaker, "content": full_text})

    # --- ETAPE 3 : Création des Paires (Sliding Window) ---
    dataset_final = []
    
    # On prend les messages 2 par 2
    # Message i devient "User", Message i+1 devient "Assistant"
    for i in range(len(messages_groupes) - 1):
        msg_A = messages_groupes[i]
        msg_B = messages_groupes[i+1]
        
        # On crée une conversation. 
        # Note : On ignore le nom réel (Junior, Tonton) et on met les rôles génériques
        dataset_final.append({
            "conversations": [
                {"role": "user", "content": msg_A['content']},
                {"role": "assistant", "content": msg_B['content']}
            ]
        })

    return dataset_final

# --- EXECUTION ---
# Remplace par ton vrai fichier (dézippé de l'archive WhatsApp)
nom_fichier = "_chat.txt" 

try:
    data = parse_ios_whatsapp(nom_fichier)
    print(f"Extraction réussie ! {len(data)} conversations générées.")
    
    # Sauvegarde
    with open('dataset_whatsapp_ios.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
except FileNotFoundError:
    print("Erreur : Fichier non trouvé. Assure-toi d'avoir uploadé le fichier .txt")
except Exception as e:
    print(f"Une erreur s'est produite : {e}")