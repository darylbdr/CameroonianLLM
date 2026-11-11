import os
import modal
from threading import Thread
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, StreamingResponse

# 1. Nom de l'application et volume de cache pour les poids
app = modal.App("camfranglais-llm-chat")
hf_cache_volume = modal.Volume.from_name("hf-model-cache", create_if_missing=True)

CACHE_DIR = "/root/hf_cache"
MODEL_ID = "Cmr237/Camfranglais:Q8_0"  

# Dépendances système et Python
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "peft",
        "fastapi[standard]",
        "pydantic"
    )
    .env({"HF_HOME": CACHE_DIR})
)

# 2. Classe d'inférence GPU
@app.cls(
    gpu="A10G",               
    scaledown_window=120,   
    image=image,
    volumes={CACHE_DIR: hf_cache_volume},
    # secrets=[modal.Secret.from_name("huggingface-secret")] # Utiser uniquement lorsque le dépot sera mis en privée
)
class ModelInference:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"Chargement du tokenizer et du modèle depuis {MODEL_ID}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            cache_dir=CACHE_DIR,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.model.eval()
        
        # Valide et sauvegarde les fichiers téléchargés dans le volume persistant
        hf_cache_volume.commit()
        print("Modèle chargé et cache persisté.")

    @modal.method()
    def generate_stream(self, message: str, system_prompt: str, temperature: float, max_tokens: int):
        import torch
        from transformers import TextIteratorStreamer

        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Streamer asynchrone
        streamer = TextIteratorStreamer(
            self.tokenizer,
            timeout=20.0,
            skip_prompt=True,
            skip_special_tokens=True
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # La génération tourne en arrière-plan pendant que les tokens sont renvoyés
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text


# 3. Interface HTML statique avec gestion du streaming SSE
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Camfranglais LLM Demo</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-950 text-slate-100 flex flex-col h-screen">
    <header class="p-4 border-b border-slate-800 bg-slate-900/60 flex items-center justify-between">
        <h1 class="text-lg font-bold text-emerald-400">Camfranglais LLM Chat (Streaming)</h1>
        <span class="text-xs text-slate-400">Modal Serverless + Persistance</span>
    </header>

    <div class="flex-1 flex overflow-hidden">
        <!-- Paramètres -->
        <aside class="w-80 border-r border-slate-800 p-4 space-y-4 hidden md:block bg-slate-900/30 text-sm">
            <div>
                <label class="block font-medium mb-1 text-slate-300">Prompt Système</label>
                <textarea id="sysPrompt" rows="3" class="w-full bg-slate-950 border border-slate-700 rounded p-2 text-xs">Tu es un assistant utile qui répond naturellement en Camfranglais.</textarea>
            </div>
            <div>
                <label class="block font-medium mb-1 text-slate-300">Température (<span id="tempVal">0.7</span>)</label>
                <input id="temp" type="range" min="0.1" max="1.5" step="0.05" value="0.7" class="w-full" oninput="tempVal.innerText=this.value">
            </div>
            <div>
                <label class="block font-medium mb-1 text-slate-300">Tokens Max</label>
                <input id="maxTokens" type="number" value="256" step="64" min="64" max="1024" class="w-full bg-slate-950 border border-slate-700 rounded p-1">
            </div>
        </aside>

        <!-- Fenêtre de chat -->
        <main class="flex-1 flex flex-col">
            <div id="chatBox" class="flex-1 overflow-y-auto p-4 space-y-3">
                <div class="bg-slate-800/60 p-3 rounded-lg max-w-xl text-slate-300 text-sm">
                    Posez vos questions ou testez votre vocabulaire en camfranglais !
                </div>
            </div>

            <form id="chatForm" class="p-4 border-t border-slate-800 bg-slate-900/40 flex gap-2">
                <input id="userInput" type="text" placeholder="Écrivez un message (ex: Comment tu vas le mola ?)..." required
                       class="flex-1 bg-slate-950 border border-slate-700 rounded-lg px-4 py-2 text-sm focus:outline-none focus:border-emerald-500">
                <button type="submit" id="sendBtn" class="bg-emerald-600 hover:bg-emerald-500 font-semibold px-4 py-2 rounded-lg text-sm transition">
                    Envoyer
                </button>
            </form>
        </main>
    </div>

    <script>
        const chatBox = document.getElementById("chatBox");
        const chatForm = document.getElementById("chatForm");
        const userInput = document.getElementById("userInput");
        const sendBtn = document.getElementById("sendBtn");

        function addMessage(text, isUser = false) {
            const div = document.createElement("div");
            div.className = isUser 
                ? "bg-emerald-700/60 ml-auto p-3 rounded-lg max-w-xl text-sm" 
                : "bg-slate-800/80 mr-auto p-3 rounded-lg max-w-xl text-sm whitespace-pre-wrap";
            div.textContent = text;
            chatBox.appendChild(div);
            chatBox.scrollTop = chatBox.scrollHeight;
            return div;
        }

        chatForm.onsubmit = async (e) => {
            e.preventDefault();
            const text = userInput.value.trim();
            if (!text) return;

            addMessage(text, true);
            userInput.value = "";
            sendBtn.disabled = true;
            sendBtn.innerText = "Génération...";

            // Bulle vide pour accueillir les tokens au fur et à mesure
            const botBubble = addMessage("");

            try {
                const response = await fetch("/api/stream", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        message: text,
                        system_prompt: document.getElementById("sysPrompt").value,
                        temperature: parseFloat(document.getElementById("temp").value),
                        max_tokens: parseInt(document.getElementById("maxTokens").value)
                    })
                });

                // Lecture du flux SSE (Server-Sent Events)
                const reader = response.body.getReader();
                const decoder = new TextDecoder("utf-8");

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    const chunk = decoder.decode(value, { stream: true });
                    const lines = chunk.split("\\n");
                    
                    for (const line of lines) {
                        if (line.startsWith("data: ")) {
                            const token = line.replace("data: ", "");
                            botBubble.textContent += token;
                            chatBox.scrollTop = chatBox.scrollHeight;
                        }
                    }
                }
            } catch (err) {
                botBubble.textContent = "Erreur de connexion lors de la génération.";
            } finally {
                sendBtn.disabled = false;
                sendBtn.innerText = "Envoyer";
            }
        };
    </script>
</body>
</html>
"""

# 4. Point d'entrée Web
class ChatRequest(BaseModel):
    message: str
    system_prompt: str = "Tu es un assistant utile qui répond en Camfranglais."
    temperature: float = 0.7
    max_tokens: int = 256

@app.function(image=image)
@modal.asgi_app()
def ui():
    from fastapi import FastAPI
    web_app = FastAPI()

    @web_app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse(content=HTML_CONTENT)

    @web_app.post("/api/stream")
    def api_stream(req: ChatRequest):
        def event_generator():
            # Appelle le générateur sur le conteneur GPU
            for chunk in ModelInference().generate_stream.remote_gen(
                message=req.message,
                system_prompt=req.system_prompt,
                temperature=req.temperature,
                max_tokens=req.max_tokens
            ):
                # Formatage SSE standard
                yield f"data: {chunk}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    return web_app