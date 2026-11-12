import os
import json
import modal
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, StreamingResponse

# Configuration Modal & Stockage Persistant
app = modal.App("camfranglais-llm-chat")
hf_cache_volume = modal.Volume.from_name("hf-model-cache", create_if_missing=True)

CACHE_DIR = "/root/hf_cache"
MODEL_REPO = "Cmr237/Camfranglais"
MODEL_FILENAME = "Camfranglais-Q8_0.gguf"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "nvidia-cuda-runtime-cu12",
        "nvidia-cublas-cu12",
        "huggingface_hub[hf_transfer]",
        "hf-transfer",
        "fastapi[standard]",
        "pydantic",
        "llama-cpp-python",
        extra_index_url="https://abetlen.github.io/llama-cpp-python/whl/cu124"
    )
    .env({
        "LD_LIBRARY_PATH": "/usr/local/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib",
        "HF_HOME": CACHE_DIR,
        "HF_HUB_ENABLE_HF_TRANSFER": "1"
    })
)

# Fonction de préchargement du modèle dans le volume persistant
@app.function(
    image=image,
    volumes={CACHE_DIR: hf_cache_volume},
    timeout=1800,
    secrets=[modal.Secret.from_dotenv()]
)
def preload():
    from huggingface_hub import hf_hub_download

    hf_cache_volume.reload()
    print(f"--> Vérification / Téléchargement de {MODEL_FILENAME} ({MODEL_REPO})...")
    path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        cache_dir=CACHE_DIR,
        token=os.environ.get("HF_TOKEN")
    )
    hf_cache_volume.commit()
    print(f"--> Fichier en cache à jour : {path}")
    return path


# Inférence GPU A10G 
@app.cls(
    gpu=["A10G"],
    scaledown_window=120,
    timeout=600,
    startup_timeout=600,
    image=image,
    volumes={CACHE_DIR: hf_cache_volume},
    secrets=[modal.Secret.from_dotenv()]
)
class ModelInference:
    @modal.enter()
    def load_model(self):
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama

        hf_cache_volume.reload()
        try:
            print(f"--> Vérification du modèle {MODEL_FILENAME} ({MODEL_REPO}) sur Hugging Face...")
            model_path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename=MODEL_FILENAME,
                cache_dir=CACHE_DIR,
                token=os.environ.get("HF_TOKEN")
            )
            hf_cache_volume.commit()
        except Exception as e:
            print(f"--> Vérification HF impossible ({e}), repli sur le cache local...")
            model_path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename=MODEL_FILENAME,
                cache_dir=CACHE_DIR,
                local_files_only=True
            )

        print(f"--> Chargement du modèle {model_path} dans llama-cpp-python (GPU)...")
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=8192,
            verbose=True
        )
        print("--> Modèle chargé avec succès et prêt !")

    @staticmethod
    def _format_mistral_prompt(messages: list, system_prompt: str) -> str:
        """Formate les messages exactement comme le template Unsloth 'mistral' utilisé en SFT."""
        sys_text = system_prompt
        convo = []
        for msg in messages:
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if role == "system" and content:
                sys_text = content
            elif role in ("user", "assistant"):
                convo.append({"role": role, "content": content})

        parts = []
        first_user = True
        for msg in convo:
            if msg["role"] == "user":
                user_content = f"{sys_text}\n\n{msg['content']}" if (first_user and sys_text) else msg["content"]
                first_user = False
                parts.append(f"[INST] {user_content} [/INST]")
            else:
                parts.append(f" {msg['content']}</s>")

        return "".join(parts)

    @modal.method()
    def generate_stream(
        self,
        messages: list,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 50,
        min_p: float = 0.0,
        repeat_penalty: float = 1.1,
    ):
        SYSTEM_PROMPT = "Tu parles comme un camerounais"
        STOP_TOKENS = ["[INST]", "[/INST]", "</s>"]

        prompt = self._format_mistral_prompt(messages, SYSTEM_PROMPT)

        stream = self.llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repeat_penalty=repeat_penalty,
            stop=STOP_TOKENS,
            stream=True
        )

        for chunk in stream:
            text = chunk["choices"][0].get("text", "")
            if text:
                yield text


# Interface Web 
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Camfranglais</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body {
            background-color: #131314;
            color: #e3e3e3;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }

        
        /* Halo Vert-Rouge-Jaune Cameroun d'accueil */
        .ambient-glow {
            position: absolute;
            top: 28%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 760px;
            height: 460px;
            display: flex;
            align-items: center;
            justify-content: center;
            pointer-events: none;
            z-index: 0;
            transition: opacity 0.4s ease;
        }

        .ambient-glow::before {
            content: "";
            position: absolute;
            inset: 0;
            background: 
                radial-gradient(circle at 20% 50%, rgba(0, 122, 61, 0.22) 0%, transparent 60%),
                radial-gradient(circle at 50% 50%, rgba(206, 17, 38, 0.18) 0%, transparent 55%),
                radial-gradient(circle at 80% 50%, rgba(252, 209, 22, 0.16) 0%, transparent 60%);
            filter: blur(55px);
            z-index: -1;
        }

        .ambient-star {
            color: #fcd116;
            filter: drop-shadow(0 0 8px rgba(252, 209, 22, 0.45));
            opacity: 0.7;
            animation: pulse-star 3s ease-in-out infinite;
        }

        @keyframes pulse-star {
            0%, 100% { transform: scale(1); opacity: 0.65; }
            50% { transform: scale(1.08); opacity: 0.85; }
        }

        #sidebar {
            transition: width 0.25s cubic-bezier(0.4, 0, 0.2, 1),
                        opacity 0.2s ease,
                        padding 0.25s ease;
        }

        #sidebar.collapsed {
            width: 0 !important;
            min-width: 0 !important;
            padding: 0 !important;
            border-right-width: 0 !important;
            opacity: 0 !important;
            pointer-events: none;
            overflow: hidden;
        }

        .chat-scroll::-webkit-scrollbar {
            width: 5px;
        }
        .chat-scroll::-webkit-scrollbar-thumb {
            background-color: #282a2c;
            border-radius: 4px;
        }
    </style>
</head>
<body class="flex h-screen overflow-hidden bg-[#131314]">

    <!-- Barre latérale -->
    <aside id="sidebar" class="w-64 bg-[#1e1f20]/50 flex flex-col border-r border-[#282a2c] z-20 shrink-0">
        <div class="p-3">
            <button onclick="createNewChat()" class="flex items-center gap-3 bg-[#1e1f20] hover:bg-[#282a2c] text-sm text-neutral-200 px-4 py-2.5 rounded-full transition w-full shadow-sm">
                <i class="fa-solid fa-plus text-xs"></i>
                <span class="font-medium">Nouvelle discussion</span>
            </button>
        </div>

        <div class="px-4 py-2 text-xs font-semibold text-neutral-500 uppercase tracking-wider">Récents</div>
        <div id="historyList" class="flex-1 overflow-y-auto px-2 space-y-1 chat-scroll text-sm"></div>

        <div class="p-3 border-t border-[#282a2c] text-xs text-neutral-400 flex items-center justify-between">
            <span class="flex items-center gap-2">
                <span class="w-2 h-2 rounded-full bg-emerald-500"></span>
                Camfranglais LLM 🇨🇲
            </span>
            <button onclick="clearAllHistory()" title="Effacer l'historique" class="hover:text-red-400 transition p-1">
                <i class="fa-regular fa-trash-can"></i>
            </button>
        </div>
    </aside>

    <!-- Zone Principale -->
    <main class="flex-1 flex flex-col relative bg-[#131314] overflow-hidden">
        <!-- Halo Vert-Rouge-Jaune avec étoile d'or Cameroun -->
        <div id="ambientGlow" class="ambient-glow">
            <div class="ambient-star" title="Cameroun">
                <svg class="w-7 h-7" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
                </svg>
            </div>
        </div>

        <!-- En-tête : Menu Hamburger à gauche + Options à droite -->
        <div class="p-3 flex items-center justify-between z-20">
            <button onclick="toggleSidebar()" class="text-neutral-400 hover:text-white p-2 rounded-lg hover:bg-[#282a2c] transition" title="Menu">
                <i class="fa-solid fa-bars text-base"></i>
            </button>
            <!-- <div class="text-neutral-500 hover:text-neutral-300 transition p-2 cursor-pointer">
                <i class="fa-solid fa-ellipsis-vertical text-base"></i>
            </div> -->
        </div>

        <!-- Zone des messages -->
        <div id="chatContainer" class="flex-1 overflow-y-auto px-4 md:px-32 chat-scroll z-10 flex flex-col">
            <!-- Écran d'accueil centré -->
            <div id="welcomeScreen" class="m-auto text-center space-y-4 max-w-lg">
                <h1 class="text-3xl font-normal text-[#e3e3e3] tracking-tight"> Tu veux un way? </h1>
            </div>

            <!-- Liste des bulles de messages -->
            <div id="messagesList" class="space-y-7 my-4 hidden max-w-3xl w-full mx-auto"></div>
        </div>

        <!-- Barre de saisie en bas (Style Gemini) -->
        <div class="p-4 md:px-32 z-10">
            <form id="chatForm" class="max-w-3xl mx-auto bg-[#1e1f20] hover:bg-[#232426] focus-within:bg-[#232426] rounded-full px-5 py-2.5 flex items-center gap-3 transition border border-[#282a2c]/60">
                <input id="userInput" type="text" placeholder="Waye tu veux quoi bro?..." autocomplete="off" required
                       class="flex-1 bg-transparent text-sm text-neutral-100 placeholder-neutral-400 focus:outline-none py-1">
                
                <button type="submit" id="submitBtn" class="w-8 h-8 rounded-full bg-neutral-200 hover:bg-white text-black flex items-center justify-center transition disabled:opacity-30 disabled:cursor-not-allowed shadow-sm shrink-0" title="Envoyer">
                    <i class="fa-solid fa-arrow-up text-xs font-bold"></i>
                </button>
            </form>
        </div>
    </main>

    <script>
        let sessions = JSON.parse(localStorage.getItem("camfranglais_sessions") || "[]");
        let currentSessionId = localStorage.getItem("current_session_id") || null;

        const sidebar = document.getElementById("sidebar");
        const historyList = document.getElementById("historyList");
        const welcomeScreen = document.getElementById("welcomeScreen");
        const ambientGlow = document.getElementById("ambientGlow");
        const messagesList = document.getElementById("messagesList");
        const chatForm = document.getElementById("chatForm");
        const userInput = document.getElementById("userInput");
        const submitBtn = document.getElementById("submitBtn");
        const chatContainer = document.getElementById("chatContainer");

        function init() {
            if (sessions.length === 0) {
                createNewChat();
            } else {
                if (!currentSessionId || !sessions.find(s => s.id === currentSessionId)) {
                    currentSessionId = sessions[0].id;
                }
                renderHistory();
                loadCurrentSession();
            }
        }

        // Bascule de la barre latérale via le bouton à 3 barres
        function toggleSidebar() {
            sidebar.classList.toggle("collapsed");
        }

        function createNewChat() {
            const newId = "session_" + Date.now();
            sessions.unshift({ id: newId, title: "Nouvelle discussion", messages: [] });
            currentSessionId = newId;
            saveSessions();
            renderHistory();
            loadCurrentSession();
        }

        function saveSessions() {
            localStorage.setItem("camfranglais_sessions", JSON.stringify(sessions));
            localStorage.setItem("current_session_id", currentSessionId);
        }

        function renderHistory() {
            historyList.innerHTML = "";
            sessions.forEach(sess => {
                const btn = document.createElement("button");
                btn.className = `w-full text-left px-3 py-2.5 rounded-lg truncate text-xs flex items-center gap-2.5 transition ${sess.id === currentSessionId ? 'bg-[#282a2c] text-white font-medium' : 'text-neutral-400 hover:bg-[#1e1f20] hover:text-neutral-200'}`;
                btn.innerHTML = `<i class="fa-regular fa-message text-[10px]"></i> <span class="truncate">${sess.title}</span>`;
                btn.onclick = () => {
                    currentSessionId = sess.id;
                    saveSessions();
                    renderHistory();
                    loadCurrentSession();
                };
                historyList.appendChild(btn);
            });
        }

        function loadCurrentSession() {
            const sess = sessions.find(s => s.id === currentSessionId);
            messagesList.innerHTML = "";
            
            // Si pas de message : écran d'accueil + halo bleu actif
            if (!sess || sess.messages.length === 0) {
                welcomeScreen.classList.remove("hidden");
                messagesList.classList.add("hidden");
                ambientGlow.classList.remove("hidden");
            } else {
                // Dès qu'il y a des messages : écran noir pur, halo masqué
                welcomeScreen.classList.add("hidden");
                messagesList.classList.remove("hidden");
                ambientGlow.classList.add("hidden");
                sess.messages.forEach(m => appendBubble(m.role, m.content));
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        }

        // Création des bulles 
        function appendBubble(role, text) {
            const row = document.createElement("div");
            if (role === "user") {
                row.className = "flex justify-end";
                row.innerHTML = `<div class="bg-[#282a2c] text-neutral-100 px-4 py-2 rounded-2xl max-w-xl text-sm leading-relaxed">${text}</div>`;
            } else {
                row.className = "flex flex-col items-start gap-2 w-full text-left";
                row.innerHTML = `
                    <div class="text-[#e3e3e3] text-sm leading-relaxed whitespace-pre-wrap content-bubble">${text}</div>
                    <div class="flex items-center gap-4 text-neutral-500 text-xs mt-1">
                        <button class="hover:text-neutral-300 transition" title="Copier" onclick="copyContent(this)"><i class="fa-regular fa-copy"></i></button>
                        <button class="hover:text-neutral-300 transition"><i class="fa-solid fa-ellipsis"></i></button>
                    </div>
                `;
            }
            messagesList.appendChild(row);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            return row.querySelector(".content-bubble");
        }

        function copyContent(btn) {
            const text = btn.closest(".flex-col").querySelector(".content-bubble").innerText;
            navigator.clipboard.writeText(text);
            btn.innerHTML = `<i class="fa-solid fa-check text-emerald-400"></i>`;
            setTimeout(() => { btn.innerHTML = `<i class="fa-regular fa-copy"></i>`; }, 1500);
        }

        chatForm.onsubmit = async (e) => {
            e.preventDefault();
            const text = userInput.value.trim();
            if (!text) return;

            const sess = sessions.find(s => s.id === currentSessionId);
            if (sess.messages.length === 0) {
                sess.title = text.slice(0, 26) + (text.length > 26 ? "..." : "");
                welcomeScreen.classList.add("hidden");
                messagesList.classList.remove("hidden");
                // Bascule immédiate sur fond noir pur
                ambientGlow.classList.add("hidden");
                renderHistory();
            }

            sess.messages.push({ role: "user", content: text });
            appendBubble("user", text);
            userInput.value = "";
            if (submitBtn) submitBtn.disabled = true;

            const botBubble = appendBubble("assistant", "");
            botBubble.innerHTML = '<span class="italic text-neutral-400">Génération en cours...</span>';
            let fullResponse = "";
            let receivedFirstChunk = false;

            try {
                const response = await fetch("/api/stream", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ messages: sess.messages })
                });

                if (!response.ok) {
                    const errorText = await response.text();
                    botBubble.textContent = "Erreur (" + response.status + "): " + (errorText || "Connexion impossible.");
                    if (submitBtn) submitBtn.disabled = false;
                    return;
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder("utf-8");
                let buffer = "";

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const parts = buffer.split("\\n\\n");
                    buffer = parts.pop();

                    for (const part of parts) {
                        for (const line of part.split("\\n")) {
                            if (line.startsWith("data: ")) {
                                try {
                                    const parsed = JSON.parse(line.slice(6));
                                    if (!receivedFirstChunk) {
                                        fullResponse = "";
                                        receivedFirstChunk = true;
                                    }
                                    fullResponse += parsed.text;
                                    botBubble.textContent = fullResponse;
                                    chatContainer.scrollTop = chatContainer.scrollHeight;
                                } catch (_) {}
                            }
                        }
                    }
                }

                if (buffer.trim()) {
                    for (const line of buffer.split("\\n")) {
                        if (line.startsWith("data: ")) {
                            try {
                                const parsed = JSON.parse(line.slice(6));
                                if (!receivedFirstChunk) {
                                    fullResponse = "";
                                    receivedFirstChunk = true;
                                }
                                fullResponse += parsed.text;
                                botBubble.textContent = fullResponse;
                            } catch (_) {}
                        }
                    }
                }

                if (!fullResponse && !receivedFirstChunk) {
                    botBubble.textContent = "Aucune réponse reçue du modèle.";
                }

                sess.messages.push({ role: "assistant", content: fullResponse });
                saveSessions();
            } catch (err) {
                botBubble.textContent = "Erreur de connexion : " + err.message;
            } finally {
                if (submitBtn) submitBtn.disabled = false;
            }
        };

        function clearAllHistory() {
            if (confirm("Voulez-vous supprimer tout l'historique des discussions ?")) {
                localStorage.removeItem("camfranglais_sessions");
                sessions = [];
                createNewChat();
            }
        }

        init();
    </script>
</body>
</html>
"""

# 4. Routeur Web FastAPI
class ChatRequest(BaseModel):
    messages: list

@app.function(image=image, timeout=600)
@modal.asgi_app()
def ui():
    import asyncio
    from fastapi import FastAPI
    web_app = FastAPI()

    @web_app.get("/", response_class=HTMLResponse)
    async def index():
        return HTMLResponse(content=HTML_CONTENT)

    @web_app.post("/api/stream")
    async def api_stream(req: ChatRequest):
        async def event_generator():
            queue = asyncio.Queue()
            stop_event = asyncio.Event()

            async def consume_gpu():
                try:
                    model = ModelInference()
                    async for chunk in model.generate_stream.remote_gen.aio(
                        messages=req.messages
                    ):
                        await queue.put(f"data: {json.dumps({'text': chunk})}\n\n")
                except Exception as e:
                    await queue.put(f"data: {json.dumps({'text': f'Erreur: {str(e)}'})}\n\n")
                finally:
                    stop_event.set()

            task = asyncio.create_task(consume_gpu())

            # Premier ping immédiat pour acquitter immédiatement la connexion HTTP
            yield ": ping\n\n"

            while not stop_event.is_set() or not queue.empty():
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=4.0)
                    yield item
                except asyncio.TimeoutError:
                    if not stop_event.is_set():
                        yield ": ping\n\n"

            await task

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )

    return web_app