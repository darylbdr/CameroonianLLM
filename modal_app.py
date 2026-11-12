import os
import json
import uuid
import modal
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, StreamingResponse

# Configuration Modal & Stockage Persistant
app = modal.App("camfranglais-llm-chat")
hf_cache_volume = modal.Volume.from_name("hf-model-cache", create_if_missing=True)

CACHE_DIR = "/root/hf_cache"
BASE_MODEL_REPO = "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit"
LORA_REPO = "Cmr237/Camfranglais"

# Image légère CPU uniquement pour l'interface Web FastAPI
web_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi[standard]",
    "pydantic",
)

# Image GPU dédiée à vLLM + BitsAndBytes + LoRA
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface_hub",
        "transformers",
        "bitsandbytes>=0.45.0",
        "vllm>=0.6.0",
    )
    .env({
        "HF_HOME": CACHE_DIR,
        "HF_XET_HIGH_PERFORMANCE": "1",
    })
)

# Fonction de préchargement du modèle de base + adaptateur LoRA dans le volume persistant
@app.function(
    image=vllm_image,
    volumes={CACHE_DIR: hf_cache_volume},
    timeout=1800,
    secrets=[modal.Secret.from_dotenv()]
)
def preload():
    from huggingface_hub import snapshot_download

    hf_cache_volume.reload()
    token = os.environ.get("HF_TOKEN")
    print(f"--> Préchargement du modèle de base {BASE_MODEL_REPO}...")
    snapshot_download(repo_id=BASE_MODEL_REPO, cache_dir=CACHE_DIR, token=token)
    print(f"--> Préchargement de l'adaptateur LoRA {LORA_REPO}...")
    lora_path = snapshot_download(
        repo_id=LORA_REPO,
        cache_dir=CACHE_DIR,
        ignore_patterns=["*.gguf"],
        token=token,
    )
    hf_cache_volume.commit()
    print(f"--> Fichiers en cache à jour : {lora_path}")
    return lora_path


# Inférence GPU A10G avec vLLM + LoRA pour réduire les coûts
@app.cls(
    gpu=["A10G"],
    scaledown_window=60,
    max_containers=3,
    timeout=600,
    startup_timeout=600,
    image=vllm_image,
    volumes={CACHE_DIR: hf_cache_volume},
    secrets=[modal.Secret.from_dotenv()]
)
@modal.concurrent(max_inputs=32)
class ModelInference:
    @modal.enter()
    def load_model(self):
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        hf_cache_volume.reload()
        token = os.environ.get("HF_TOKEN")
        try:
            print(f"--> Vérification du modèle {BASE_MODEL_REPO} et LoRA {LORA_REPO}...")
            base_path = snapshot_download(
                repo_id=BASE_MODEL_REPO,
                cache_dir=CACHE_DIR,
                token=token,
            )
            self.lora_path = snapshot_download(
                repo_id=LORA_REPO,
                cache_dir=CACHE_DIR,
                ignore_patterns=["*.gguf"],
                token=token,
            )
            hf_cache_volume.commit()
        except Exception as e:
            print(f"--> Vérification HF impossible ({e}), repli sur le cache local...")
            base_path = snapshot_download(
                repo_id=BASE_MODEL_REPO,
                cache_dir=CACHE_DIR,
                local_files_only=True,
            )
            self.lora_path = snapshot_download(
                repo_id=LORA_REPO,
                cache_dir=CACHE_DIR,
                ignore_patterns=["*.gguf"],
                local_files_only=True,
            )

        print(f"--> Chargement de vLLM (Base 4-bit + LoRA {self.lora_path})...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.lora_path)
        engine_args = AsyncEngineArgs(
            model=base_path,
            tokenizer=self.lora_path,
            quantization="bitsandbytes",
            load_format="bitsandbytes",
            enable_lora=True,
            max_loras=1,
            max_lora_rank=32,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            max_model_len=4096,
            enforce_eager=True,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        print("--> Moteur vLLM + LoRA chargé avec succès et prêt !")

    @modal.method()
    async def generate_stream(
        self,
        messages: list,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 50,
        min_p: float = 0.0,
        repetition_penalty: float = 1.1,
    ):
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        SYSTEM_PROMPT = "Tu parles comme un camerounais"

        formatted_messages = list(messages)
        if not formatted_messages or formatted_messages[0].get("role") != "system":
            formatted_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        prompt = self.tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            stop=["[INST]", "[/INST]", "</s>"],
        )

        lora_request = LoRARequest("camfranglais", 1, self.lora_path)
        request_id = str(uuid.uuid4())
        results_generator = self.engine.generate(
            prompt,
            sampling_params,
            request_id,
            lora_request=lora_request,
        )

        previous_text = ""
        async for request_output in results_generator:
            current_text = request_output.outputs[0].text
            delta = current_text[len(previous_text):]
            previous_text = current_text
            if delta:
                yield delta


# Interface Web 
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
    <title>Camfranglais LLM</title>
    <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 900 600'%3E%3Crect width='300' height='600' fill='%23007a5e'/%3E%3Crect x='300' width='300' height='600' fill='%23ce1126'/%3E%3Crect x='600' width='300' height='600' fill='%23fcd116'/%3E%3Cpolygon points='450,230 468,285 526,285 479,319 497,374 450,340 403,374 421,319 374,285 432,285' fill='%23fcd116'/%3E%3C/svg%3E">
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body {
            background-color: #131314;
            color: #e3e3e3;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            height: 100vh;
            height: 100dvh;
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
<body class="flex overflow-hidden bg-[#131314]">

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
                <span>Camfranglais LLM</span>
                <svg class="w-4 h-3 rounded-[2px] shadow-sm inline-block" viewBox="0 0 900 600" aria-label="Cameroun">
                    <rect width="300" height="600" fill="#007a5e"/>
                    <rect x="300" width="300" height="600" fill="#ce1126"/>
                    <rect x="600" width="300" height="600" fill="#fcd116"/>
                    <polygon points="450,230 468,285 526,285 479,319 497,374 450,340 403,374 421,319 374,285 432,285" fill="#fcd116"/>
                </svg>
            </span>
            <button onclick="clearAllHistory()" title="Effacer l'historique" class="hover:text-red-400 transition p-1">
                <i class="fa-regular fa-trash-can"></i>
            </button>
        </div>
    </aside>

    <!-- Zone Principale -->
    <main class="flex-1 min-w-0 flex flex-col relative bg-[#131314] overflow-hidden">
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
        <div class="p-4 md:px-32 z-10 shrink-0" style="padding-bottom: max(1rem, env(safe-area-inset-bottom));">
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
            if (window.innerWidth < 768) {
                sidebar.classList.add("collapsed");
            }
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

@app.function(image=web_image, timeout=600)
@modal.concurrent(max_inputs=100)
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