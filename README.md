<div align="center">

<h1> Alpha AI</h1>

<p>A production-grade conversational AI assistant built with <strong>Streamlit</strong>, <strong>LangChain</strong>, and <strong>Groq</strong>.<br/>
Multi-model · Multi-persona · Real-time analytics · Session persistence · Rate-limited</p>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-1C3C3C?style=flat-square&logo=chainlink&logoColor=white)](https://langchain.com)
[![Groq](https://img.shields.io/badge/Groq-API-F55036?style=flat-square&logo=groq&logoColor=white)](https://groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-8B5CF6?style=flat-square)](LICENSE)

![Alpha AI Demo](https://placehold.co/900x480/0d0d1a/8b5cf6?text=Alpha+AI+%E2%9C%A6&font=inter)

</div>

---

## Overview

**Alpha** is a full-stack AI chat application that lets users interact with multiple large language models through a polished, dark-themed UI. It ships with a built-in sliding-window rate limiter, LangChain-powered conversation memory that survives model/persona switches, persistent session storage, live analytics, and one-click export — all in a single, well-structured Python file.

> **Live demo:** *(deploy to [Streamlit Cloud](https://streamlit.io/cloud) — free in 2 minutes)*

---

## Features

| Category | Details |
|---|---|
| **13 LLMs via Groq** | LLaMA 3.3 70B · LLaMA 3.1 70B/8B · Mixtral 8x7B · Gemma 2 9B · DeepSeek R1 70B · QwQ 32B · Mistral Saba 24B · and more |
| **5 AI Personas** | Coding Assistant · FAANG Interview Coach · System Design Expert · Code Reviewer · General Assistant |
| **Sliding-Window Rate Limiter** | `deque`-backed O(1) amortized limiter — 20 req / 60 s per session |
| **Context-Preserving Chain Rebuild** | Switching model or persona replays history into the new `ConversationBufferMemory` — no context lost |
| **Session Persistence** | Save, load, and delete named sessions backed by a local JSON store |
| **Conversation Export** | Download full history as `.json` or `.txt` |
| **Live Session Analytics** | Avg / min / max latency, message counts, real-time rate-limit gauge |
| **Keyword Search** | Filter conversation history with inline bold highlighting |
| **Configurable LLM Params** | Temperature slider + model selector; chain rebuilds transparently |
| **Modern Dark UI** | Glassmorphic header, animated chat bubbles, Inter font, custom scrollbar |
| **Graceful Error Handling** | API failures are caught, displayed, and stored — never silently dropped |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **UI Framework** | [Streamlit](https://streamlit.io/) — reactive, component-driven Python UI |
| **LLM Orchestration** | [LangChain](https://langchain.com/) — prompt templates, memory, chain abstraction |
| **Inference API** | [Groq](https://groq.com/) — ultra-low-latency LLaMA 3 / Mixtral serving |
| **Environment Config** | [python-dotenv](https://pypi.org/project/python-dotenv/) |
| **Language** | Python 3.10+ (type-annotated throughout) |

---

## Architecture

```
chatbot_ui.py
│
├── RateLimiter                  Sliding-window deque — O(1) amortized per call
│   ├── allow() → bool
│   ├── remaining → int
│   └── reset_in → float
│
├── build_chain()                Constructs LangChain ConversationChain
│   ├── Loads persona system prompt
│   ├── Attaches ChatGroq LLM (model + temperature)
│   └── Replays message history into ConversationBufferMemory
│
├── Persistence layer
│   ├── save_session()           Upsert session into chat_history.json
│   ├── load_session()           Restore state + rebuild chain
│   └── delete_session()        Remove entry from JSON store
│
├── Sidebar                      Settings · Analytics · Search · Sessions · Export
│
└── Chat loop
    ├── Rate-gate (allow or block with countdown)
    ├── chain.invoke({input})
    ├── Latency measurement (perf_counter)
    └── Message append + render
```

**Memory continuity:** `ConversationBufferMemory` is chain-bound. When settings change, `build_chain()` constructs a fresh chain and replays all prior `(user, assistant)` pairs into the new memory — preserving full context without a separate state store.

**Rate limiting:** A sliding-window rather than a fixed-window counter prevents the 2× burst problem at window boundaries. Each call is timestamped in a `deque`; expired entries are evicted lazily in O(1).

---

## Getting Started

### Prerequisites

- Python **3.10+**
- A free [Groq API key](https://console.groq.com/) (no credit card required)

### 1 · Clone & install

```bash
git clone https://github.com/yourusername/alpha-ai.git
cd alpha-ai

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2 · Configure

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3 · Run

```bash
streamlit run chatbot_ui.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage Guide

1. **Pick a persona** from the sidebar — the system prompt updates immediately.
2. **Choose a model** — 70B for quality, 8B/3B for speed.
3. **Tune temperature** — the sidebar shows the recommended range per persona.
4. **Chat** — responses display latency; the analytics panel updates live.
5. **Save your session** — timestamped, loadable across restarts.
6. **Search** — keyword filter highlights matches inline.
7. **Export** — download as JSON (structured) or TXT (readable).

---

## Project Structure

```
alpha-ai/
├── chatbot_ui.py        # Full application (~380 lines, type-annotated)
├── requirements.txt     # Pinned dependencies
├── .env                 # API keys — git-ignored
├── chat_history.json    # Auto-generated session store — git-ignored
└── README.md
```

---

## Design Decisions

**Single-file architecture**  
For a portfolio/prototype, one well-structured file is easier to read, fork, and deploy than a premature package split. The natural next steps — splitting `RateLimiter`, persistence, and UI into modules — are tracked in the roadmap.

**Sliding-window rate limiter over fixed-window**  
A fixed-window counter allows 2× burst at window edges. The `deque`-based sliding window eliminates this, with O(1) amortized cost and no background thread.

**Chain rebuild on settings change**  
Rather than coupling model/persona state to the chain object, `build_chain()` is a pure factory. Any settings change triggers a clean rebuild, with history replayed automatically. This makes the settings logic trivially testable and free of hidden side effects.

---

## Roadmap

- [ ] Streaming token output (SSE via Streamlit's `write_stream`)
- [ ] RAG pipeline — upload PDFs and chat with your documents
- [ ] PostgreSQL / SQLite session backend (replace JSON store)
- [ ] User authentication with per-user session isolation
- [ ] Docker image + GitHub Actions CI
- [ ] Deployment guide (Streamlit Cloud / Hugging Face Spaces / Railway)

---

## Local `.gitignore` recommendations

```gitignore
.env
chat_history.json
__pycache__/
venv/
*.pyc
```

---

## License

[MIT](LICENSE) — free to use, modify, and distribute.

---

## Author

Built by **Labibul Ahsan Alif**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Labibul%20Ahsan%20Alif-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/labibul-ahsan-alif-b70974291/)
[![Email](https://img.shields.io/badge/Email-labibalif2001%40gmail.com-EA4335?style=flat-square&logo=gmail)](mailto:labibalif2001@gmail.com)

---

<div align="center">
<sub>If you found this useful, consider giving it a ⭐ — it helps others discover the project.</sub>
</div>
