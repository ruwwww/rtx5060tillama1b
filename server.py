import argparse
import asyncio
import json
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

from config import EngineConfig, ModelConfig
from engine import LlamaEngine


class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 40


def create_app(engine: LlamaEngine) -> FastAPI:
    app = FastAPI(title="RTX 5060 Ti Llama 1B Server")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>RTX 5060 Ti - Llama 1B</title>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
            <style>
                :root {
                    --bg-dark: #09090b;
                    --bg-card: rgba(20, 20, 25, 0.6);
                    --accent: #10b981;
                    --accent-glow: rgba(16, 185, 129, 0.4);
                    --text-primary: #f4f4f5;
                    --text-secondary: #a1a1aa;
                    --border: rgba(255, 255, 255, 0.08);
                }

                * {
                    box-sizing: border-box;
                    margin: 0;
                    padding: 0;
                    font-family: 'Inter', sans-serif;
                }

                body {
                    background-color: var(--bg-dark);
                    color: var(--text-primary);
                    overflow: hidden;
                    height: 100vh;
                    display: flex;
                }

                /* Layout */
                .container {
                    display: flex;
                    width: 100%;
                    height: 100%;
                }

                /* Sidebar */
                .sidebar {
                    width: 320px;
                    border-right: 1px solid var(--border);
                    background: rgba(15, 15, 20, 0.9);
                    padding: 2rem;
                    display: flex;
                    flex-direction: column;
                    gap: 1.5rem;
                }

                .logo-section {
                    display: flex;
                    align-items: center;
                    gap: 0.75rem;
                    margin-bottom: 1rem;
                }

                .logo-icon {
                    width: 12px;
                    height: 12px;
                    background: var(--accent);
                    border-radius: 50%;
                    box-shadow: 0 0 12px var(--accent);
                }

                .logo-text {
                    font-size: 1.1rem;
                    font-weight: 700;
                    letter-spacing: -0.025em;
                }

                .setting-group {
                    display: flex;
                    flex-direction: column;
                    gap: 0.5rem;
                }

                label {
                    font-size: 0.8rem;
                    color: var(--text-secondary);
                    font-weight: 600;
                }

                input[type="range"] {
                    width: 100%;
                    accent-color: var(--accent);
                    background: #27272a;
                    border-radius: 4px;
                    height: 6px;
                    outline: none;
                }

                .value-display {
                    display: flex;
                    justify-content: space-between;
                    font-size: 0.85rem;
                    font-family: 'JetBrains Mono', monospace;
                    color: var(--accent);
                }

                /* Main Chat Area */
                .chat-area {
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                    background: radial-gradient(circle at top right, rgba(16, 185, 129, 0.05), transparent 60%);
                }

                .chat-header {
                    padding: 1.5rem 2rem;
                    border-bottom: 1px solid var(--border);
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                }

                .status-badge {
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                    font-size: 0.8rem;
                    color: var(--text-secondary);
                }

                .status-dot {
                    width: 8px;
                    height: 8px;
                    background: #10b981;
                    border-radius: 50%;
                    animation: pulse 2s infinite;
                }

                /* Messages */
                .messages-container {
                    flex: 1;
                    padding: 2rem;
                    overflow-y: auto;
                    display: flex;
                    flex-direction: column;
                    gap: 1.5rem;
                }

                .message {
                    display: flex;
                    flex-direction: column;
                    gap: 0.5rem;
                    max-width: 80%;
                    animation: fadeIn 0.3s ease-out;
                }

                .message.user {
                    align-self: flex-end;
                }

                .message.assistant {
                    align-self: flex-start;
                }

                .message-bubble {
                    padding: 1rem 1.25rem;
                    border-radius: 12px;
                    font-size: 0.95rem;
                    line-height: 1.5;
                }

                .message.user .message-bubble {
                    background: var(--accent);
                    color: #fff;
                    border-bottom-right-radius: 2px;
                }

                .message.assistant .message-bubble {
                    background: var(--bg-card);
                    border: 1px solid var(--border);
                    border-bottom-left-radius: 2px;
                    backdrop-filter: blur(8px);
                }

                .message-meta {
                    font-size: 0.75rem;
                    color: var(--text-secondary);
                }

                .message.user .message-meta {
                    align-self: flex-end;
                }

                /* Input Section */
                .input-section {
                    padding: 2rem;
                    display: flex;
                    gap: 1rem;
                    border-top: 1px solid var(--border);
                }

                .input-wrapper {
                    flex: 1;
                    position: relative;
                }

                textarea {
                    width: 100%;
                    height: 50px;
                    background: var(--bg-card);
                    border: 1px solid var(--border);
                    border-radius: 8px;
                    padding: 0.8rem 1rem;
                    color: var(--text-primary);
                    outline: none;
                    resize: none;
                    font-size: 0.95rem;
                    transition: border-color 0.2s;
                }

                textarea:focus {
                    border-color: var(--accent);
                    box-shadow: 0 0 10px var(--accent-glow);
                }

                .send-btn {
                    padding: 0 1.5rem;
                    background: var(--accent);
                    color: #fff;
                    border: none;
                    border-radius: 8px;
                    font-weight: 600;
                    cursor: pointer;
                    transition: transform 0.1s, opacity 0.2s;
                }

                .send-btn:hover {
                    opacity: 0.9;
                }

                .send-btn:active {
                    transform: scale(0.98);
                }

                @keyframes pulse {
                    0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
                    70% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
                    100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
                }

                @keyframes fadeIn {
                    from { opacity: 0; transform: translateY(8px); }
                    to { opacity: 1; transform: translateY(0); }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="sidebar">
                    <div class="logo-section">
                        <div class="logo-icon"></div>
                        <div class="logo-text">RTX 5060 Ti Llama 1B</div>
                    </div>
                    
                    <div class="setting-group">
                        <label for="temp">Temperature</label>
                        <input type="range" id="temp" min="0" max="2.0" step="0.1" value="0.8">
                        <div class="value-display"><span id="temp-val">0.8</span></div>
                    </div>

                    <div class="setting-group">
                        <label for="top_p">Top P</label>
                        <input type="range" id="top_p" min="0" max="1.0" step="0.05" value="0.95">
                        <div class="value-display"><span id="top_p-val">0.95</span></div>
                    </div>

                    <div class="setting-group">
                        <label for="max_tokens">Max Tokens</label>
                        <input type="range" id="max_tokens" min="16" max="512" step="16" value="128">
                        <div class="value-display"><span id="max_tokens-val">128</span></div>
                    </div>
                </div>

                <div class="chat-area">
                    <div class="chat-header">
                        <div class="logo-text">Active Session</div>
                        <div class="status-badge">
                            <div class="status-dot"></div>
                            Online
                        </div>
                    </div>

                    <div class="messages-container" id="chat-messages">
                        <div class="message assistant">
                            <div class="message-bubble">Hello! I am Llama-3.2-1B-Instruct loaded from HuggingFace safetensors. Ask me anything!</div>
                            <div class="message-meta">System</div>
                        </div>
                    </div>

                    <div class="input-section">
                        <div class="input-wrapper">
                            <textarea id="prompt-input" placeholder="Type your prompt here..."></textarea>
                        </div>
                        <button class="send-btn" id="send-btn">Send</button>
                    </div>
                </div>
            </div>

            <script>
                // Slider value updates
                const sliders = ['temp', 'top_p', 'max_tokens'];
                sliders.forEach(id => {
                    const el = document.getElementById(id);
                    const val = document.getElementById(id + '-val');
                    el.addEventListener('input', () => {
                        val.textContent = el.value;
                    });
                });

                const messagesContainer = document.getElementById('chat-messages');
                const promptInput = document.getElementById('prompt-input');
                const sendBtn = document.getElementById('send-btn');

                async function sendMessage() {
                    const prompt = promptInput.value.trim();
                    if (!prompt) return;

                    promptInput.value = '';

                    // Add user message
                    const userMsg = document.createElement('div');
                    userMsg.className = 'message user';
                    userMsg.innerHTML = `<div class="message-bubble">${prompt}</div><div class="message-meta">User</div>`;
                    messagesContainer.appendChild(userMsg);
                    messagesContainer.scrollTop = messagesContainer.scrollHeight;

                    // Add assistant temporary message
                    const assistantMsg = document.createElement('div');
                    assistantMsg.className = 'message assistant';
                    const bubble = document.createElement('div');
                    bubble.className = 'message-bubble';
                    bubble.textContent = '...';
                    assistantMsg.appendChild(bubble);
                    assistantMsg.innerHTML += '<div class="message-meta">Assistant</div>';
                    messagesContainer.appendChild(assistantMsg);
                    const bubbleRef = assistantMsg.querySelector('.message-bubble');

                    const response = await fetch('/v1/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            prompt: prompt,
                            temperature: parseFloat(document.getElementById('temp').value),
                            top_p: parseFloat(document.getElementById('top_p').value),
                            max_tokens: parseInt(document.getElementById('max_tokens').value)
                        })
                    });

                    bubbleRef.textContent = '';
                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();

                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        const chunk = decoder.decode(value);
                        
                        // Parse events
                        const lines = chunk.split('\\n');
                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                try {
                                    const data = JSON.parse(line.slice(6));
                                    if (data.text) {
                                        bubbleRef.textContent += data.text;
                                        messagesContainer.scrollTop = messagesContainer.scrollHeight;
                                    }
                                } catch (e) {}
                            }
                        }
                    }
                }

                sendBtn.addEventListener('click', sendMessage);
                promptInput.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        sendMessage();
                    }
                });
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)

    @app.post("/v1/chat")
    async def chat(req: ChatRequest):
        async def event_generator():
            async for chunk in engine.generate(
                prompt=req.prompt,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k
            ):
                yield f"data: {json.dumps({'text': chunk.text, 'finished': chunk.finished})}\n\n"
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    return app


def main():
    parser = argparse.ArgumentParser(description="Llama 1B Safetensors Inference Server")
    parser.add_argument("--model-path", type=str, default="/home/kuroko/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6", help="HF model local path")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    args = parser.parse_args()

    model_cfg = ModelConfig(hf_path=args.model_path)
    engine_cfg = EngineConfig()

    print("Initializing RTX 5060 Ti Llama 1B Engine...")
    engine = LlamaEngine(model_cfg, engine_cfg)
    app = create_app(engine)

    print(f"Server starting on http://localhost:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
