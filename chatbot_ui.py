import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
import os
import re
from datetime import datetime
from collections import deque

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Alpha AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS  –  Modern dark theme
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ──────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background: linear-gradient(160deg, #0d0d1a 0%, #111128 60%, #0d1117 100%);
    min-height: 100vh;
}

/* ── Sidebar ─────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: rgba(13, 13, 26, 0.97) !important;
    border-right: 1px solid rgba(139, 92, 246, 0.18) !important;
}
[data-testid="stSidebar"] .stMarkdown h2 {
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #8b5cf6 !important;
    margin-top: 8px !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stTextInput label {
    color: #cbd5e1 !important;
    font-size: 12px !important;
}

/* ── Header card ─────────────────────────────────────── */
.alpha-header {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #a855f7 100%);
    border-radius: 18px;
    padding: 22px 28px 18px;
    margin-bottom: 18px;
    box-shadow: 0 8px 40px rgba(124, 58, 237, 0.35);
    display: flex;
    align-items: center;
    gap: 16px;
}
.alpha-logo {
    width: 48px; height: 48px;
    background: rgba(255,255,255,0.15);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 26px;
    backdrop-filter: blur(10px);
    flex-shrink: 0;
}
.alpha-title { color: #fff; font-size: 26px; font-weight: 700; letter-spacing: -0.5px; margin: 0; }
.alpha-sub   { color: rgba(255,255,255,0.65); font-size: 13px; margin: 2px 0 0; }

/* ── Persona badge ───────────────────────────────────── */
.persona-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    color: white;
    margin-top: 10px;
    letter-spacing: 0.02em;
}
.model-chip {
    display: inline-flex;
    align-items: center;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 11px;
    color: #94a3b8;
    margin-left: 8px;
    vertical-align: middle;
}

/* ── Chat bubbles ────────────────────────────────────── */
[data-testid="stChatMessage"] {
    border-radius: 14px !important;
    padding: 14px 18px !important;
    margin-bottom: 10px !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    animation: fadeUp 0.25s ease both;
}
[data-testid="stChatMessage"][data-role="user"] {
    background: linear-gradient(135deg,
        rgba(79, 70, 229, 0.14) 0%,
        rgba(99, 102, 241, 0.08) 100%) !important;
    border-color: rgba(99, 102, 241, 0.25) !important;
}
[data-testid="stChatMessage"][data-role="assistant"] {
    background: rgba(255,255,255,0.025) !important;
    border-color: rgba(139, 92, 246, 0.15) !important;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Chat input ──────────────────────────────────────── */
[data-testid="stChatInputContainer"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(139, 92, 246, 0.35) !important;
    border-radius: 14px !important;
    box-shadow: 0 0 0 0 rgba(139,92,246,0);
    transition: box-shadow 0.2s;
}
[data-testid="stChatInputContainer"]:focus-within {
    box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.18) !important;
    border-color: rgba(139, 92, 246, 0.6) !important;
}

/* ── Metrics ─────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 8px 10px;
}
[data-testid="stMetricValue"] { font-size: 18px !important; color: #e2e8f0 !important; }
[data-testid="stMetricLabel"] { font-size: 10px !important; color: #64748b !important; }

/* ── Buttons ─────────────────────────────────────────── */
.stButton > button {
    background: rgba(139, 92, 246, 0.12) !important;
    border: 1px solid rgba(139, 92, 246, 0.35) !important;
    color: #c4b5fd !important;
    border-radius: 9px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: all 0.18s !important;
}
.stButton > button:hover {
    background: rgba(139, 92, 246, 0.25) !important;
    border-color: rgba(139, 92, 246, 0.6) !important;
    color: #fff !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(139,92,246,0.2) !important;
}

/* ── Progress bar ────────────────────────────────────── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #7c3aed, #a855f7) !important;
    border-radius: 9999px !important;
}

/* ── Welcome screen ──────────────────────────────────── */
.welcome-card {
    text-align: center;
    padding: 60px 40px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(139, 92, 246, 0.12);
    border-radius: 20px;
    margin: 20px auto;
    max-width: 580px;
}
.welcome-icon { font-size: 52px; margin-bottom: 16px; }
.welcome-title { font-size: 24px; font-weight: 700; color: #e2e8f0; margin-bottom: 8px; }
.welcome-sub   { font-size: 14px; color: #64748b; line-height: 1.6; }
.suggestion-row { display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; margin-top: 24px; }
.suggestion-chip {
    padding: 8px 16px;
    background: rgba(139, 92, 246, 0.1);
    border: 1px solid rgba(139, 92, 246, 0.25);
    border-radius: 20px;
    font-size: 13px;
    color: #c4b5fd;
    cursor: pointer;
}

/* ── Response time stamp ─────────────────────────────── */
.response-time {
    font-size: 10px;
    color: #475569;
    text-align: right;
    margin-top: 6px;
    letter-spacing: 0.03em;
}

/* ── Divider ─────────────────────────────────────────── */
hr { border-color: rgba(139,92,246,0.12) !important; }

/* ── Scrollbar ───────────────────────────────────────── */
::-webkit-scrollbar       { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(139,92,246,0.35); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: rgba(139,92,246,0.6); }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
BOT_NAME = "Alpha"

MODELS: dict[str, str] = {
    "LLaMA 3.3 70B   (Best quality)":    "llama-3.3-70b-versatile",
    "LLaMA 3.1 70B   (High quality)":    "llama-3.1-70b-versatile",
    "LLaMA 3.1 8B    (Fastest)":         "llama-3.1-8b-instant",
    "LLaMA 3 70B     (Reliable)":        "llama3-70b-8192",
    "LLaMA 3 8B      (Compact)":         "llama3-8b-8192",
    "LLaMA 3.2 3B    (Ultra-fast)":      "llama-3.2-3b-preview",
    "LLaMA 3.2 1B    (Lightest)":        "llama-3.2-1b-preview",
    "Mixtral 8x7B    (Balanced)":        "mixtral-8x7b-32768",
    "Gemma 2 9B      (Google)":          "gemma2-9b-it",
    "Gemma 7B        (Google Compact)":  "gemma-7b-it",
    "DeepSeek R1 70B (Reasoning)":       "deepseek-r1-distill-llama-70b",
    "QwQ 32B         (Qwen Reasoning)":  "qwen-qwq-32b",
    "Mistral Saba 24B":                  "mistral-saba-24b",
}

PERSONAS: dict[str, dict] = {
    "💻 Coding Assistant": {
        "prompt": (
            "You are an expert software engineer and coding assistant. "
            "Write clean, efficient, and well-commented code. "
            "Always include time and space complexity for algorithms. "
            "Prefer idiomatic solutions and mention trade-offs when relevant."
        ),
        "color": "#22c55e",
        "temp_hint": ("0.0 – 0.3", "precise, deterministic code output"),
    },
    "🎯 FAANG Interview Coach": {
        "prompt": (
            "You are a FAANG senior engineer conducting mock technical interviews. "
            "Guide candidates through DSA problems step by step. Ask clarifying questions. "
            "Evaluate solutions on correctness, efficiency, and code quality. "
            "Give structured, constructive feedback after each problem."
        ),
        "color": "#3b82f6",
        "temp_hint": ("0.2 – 0.5", "structured, consistent interview feedback"),
    },
    "🏗️ System Design Expert": {
        "prompt": (
            "You are a principal engineer specializing in large-scale distributed systems. "
            "Help design scalable, fault-tolerant architectures. "
            "Always discuss trade-offs, CAP theorem, sharding strategies, caching layers, "
            "load balancing, and real-world constraints like cost and latency."
        ),
        "color": "#f59e0b",
        "temp_hint": ("0.3 – 0.6", "thorough yet focused architecture analysis"),
    },
    "📝 Code Reviewer": {
        "prompt": (
            "You are a meticulous senior code reviewer. "
            "Analyse submitted code for bugs, security vulnerabilities (OWASP Top 10), "
            "performance bottlenecks, and style issues. "
            "Suggest improvements using SOLID principles and relevant design patterns. "
            "Always explain WHY a change is needed, not just what to change."
        ),
        "color": "#ef4444",
        "temp_hint": ("0.0 – 0.3", "precise, bug-focused review"),
    },
    "🧠 General Assistant": {
        "prompt": (
            "You are a knowledgeable and helpful AI assistant named Alpha. "
            "Answer questions clearly and concisely with supporting examples when useful."
        ),
        "color": "#8b5cf6",
        "temp_hint": ("0.5 – 0.8", "balanced and conversational"),
    },
}

HISTORY_FILE = "chat_history.json"
RATE_LIMIT_CALLS = 20
RATE_LIMIT_WINDOW = 60


# ──────────────────────────────────────────────────────────────────────────────
# Rate Limiter
# ──────────────────────────────────────────────────────────────────────────────
class RateLimiter:
    def __init__(self, max_calls: int, window_seconds: int) -> None:
        self.max_calls = max_calls
        self.window = window_seconds
        self._calls: deque[float] = deque()

    def allow(self) -> bool:
        now = time.time()
        while self._calls and self._calls[0] < now - self.window:
            self._calls.popleft()
        if len(self._calls) < self.max_calls:
            self._calls.append(now)
            return True
        return False

    @property
    def remaining(self) -> int:
        now = time.time()
        while self._calls and self._calls[0] < now - self.window:
            self._calls.popleft()
        return self.max_calls - len(self._calls)

    @property
    def reset_in(self) -> float:
        if not self._calls:
            return 0.0
        return max(0.0, self.window - (time.time() - self._calls[0]))


# ──────────────────────────────────────────────────────────────────────────────
# Session State Initialisation
# ──────────────────────────────────────────────────────────────────────────────
def _init_session() -> None:
    defaults: dict = {
        "messages":       [],
        "session_id":     datetime.now().strftime("%Y%m%d_%H%M%S"),
        "response_times": [],
        "rate_limiter":   RateLimiter(RATE_LIMIT_CALLS, RATE_LIMIT_WINDOW),
        "model":          list(MODELS.keys())[0],
        "persona":        list(PERSONAS.keys())[0],
        "temperature":    0.7,
        "max_tokens":     1024,
        "conversation":   None,
        "search_query":   "",
        "show_metrics":   True,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session()


# ──────────────────────────────────────────────────────────────────────────────
# Conversation Chain Builder
# ──────────────────────────────────────────────────────────────────────────────
def build_chain(model_key: str, persona_key: str, temperature: float) -> ConversationChain:
    persona_prompt = PERSONAS[persona_key]["prompt"]
    prompt = ChatPromptTemplate.from_messages([
        ("system", persona_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    llm = ChatGroq(model_name=MODELS[model_key], temperature=temperature)
    memory = ConversationBufferMemory(return_messages=True)

    msgs = st.session_state.messages
    for i in range(0, len(msgs) - 1, 2):
        if i + 1 < len(msgs):
            if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
                memory.chat_memory.add_user_message(msgs[i]["content"])
                memory.chat_memory.add_ai_message(msgs[i + 1]["content"])

    return ConversationChain(llm=llm, memory=memory, prompt=prompt)


if st.session_state.conversation is None:
    st.session_state.conversation = build_chain(
        st.session_state.model,
        st.session_state.persona,
        st.session_state.temperature,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Persistence Helpers
# ──────────────────────────────────────────────────────────────────────────────
def save_session() -> None:
    data: dict = {}
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    data[st.session_state.session_id] = {
        "messages": st.session_state.messages,
        "persona":  st.session_state.persona,
        "model":    st.session_state.model,
        "saved_at": datetime.now().isoformat(),
    }
    with open(HISTORY_FILE, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def load_all_sessions() -> dict:
    if not os.path.exists(HISTORY_FILE):
        return {}
    with open(HISTORY_FILE, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_session(session_id: str) -> None:
    sessions = load_all_sessions()
    if session_id not in sessions:
        return
    s = sessions[session_id]
    st.session_state.messages       = s["messages"]
    st.session_state.persona        = s.get("persona", list(PERSONAS.keys())[0])
    st.session_state.model          = s.get("model",   list(MODELS.keys())[0])
    st.session_state.session_id     = session_id
    st.session_state.response_times = []
    st.session_state.conversation   = build_chain(
        st.session_state.model,
        st.session_state.persona,
        st.session_state.temperature,
    )


def delete_session(session_id: str) -> None:
    sessions = load_all_sessions()
    sessions.pop(session_id, None)
    with open(HISTORY_FILE, "w", encoding="utf-8") as fh:
        json.dump(sessions, fh, indent=2, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand mark in sidebar
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;padding:4px 0 16px'>"
        f"<div style='width:32px;height:32px;background:linear-gradient(135deg,#4f46e5,#a855f7);"
        f"border-radius:9px;display:flex;align-items:center;justify-content:center;"
        f"font-size:16px;'>✦</div>"
        f"<span style='font-size:17px;font-weight:700;color:#e2e8f0;letter-spacing:-0.3px'>{BOT_NAME}</span>"
        f"<span style='font-size:10px;color:#6366f1;background:rgba(99,102,241,0.15);"
        f"padding:2px 8px;border-radius:99px;font-weight:600;'>AI</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("## ⚙️ Configuration")

    new_model = st.selectbox(
        "Model",
        list(MODELS.keys()),
        index=list(MODELS.keys()).index(st.session_state.model)
              if st.session_state.model in MODELS else 0,
    )
    new_persona = st.selectbox(
        "Persona",
        list(PERSONAS.keys()),
        index=list(PERSONAS.keys()).index(st.session_state.persona)
              if st.session_state.persona in PERSONAS else 0,
    )
    new_temp = st.slider(
        "Temperature", 0.0, 1.0, st.session_state.temperature, 0.05,
        help="Lower = more deterministic  ·  Higher = more creative",
    )
    hint_range, hint_desc = PERSONAS[new_persona]["temp_hint"]
    st.caption(f"Recommended for **{new_persona}**: `{hint_range}` — {hint_desc}")

    if (new_model   != st.session_state.model    or
        new_persona != st.session_state.persona  or
        new_temp    != st.session_state.temperature):
        st.session_state.model        = new_model
        st.session_state.persona      = new_persona
        st.session_state.temperature  = new_temp
        st.session_state.conversation = build_chain(new_model, new_persona, new_temp)
        st.success("Settings applied — context preserved.")

    st.divider()

    # ── Analytics ──────────────────────────────────────────────────────────
    st.markdown("## 📊 Analytics")
    total_msgs = len(st.session_state.messages)
    user_msgs  = sum(1 for m in st.session_state.messages if m["role"] == "user")
    rts        = st.session_state.response_times
    avg_rt     = sum(rts) / len(rts) if rts else 0.0
    fastest    = min(rts) if rts else 0.0

    c1, c2 = st.columns(2)
    c1.metric("Messages",    total_msgs)
    c2.metric("Your turns",  user_msgs)
    c1.metric("Avg latency", f"{avg_rt:.2f}s")
    c2.metric("Fastest",     f"{fastest:.2f}s")

    rl_remaining = st.session_state.rate_limiter.remaining
    st.progress(
        rl_remaining / RATE_LIMIT_CALLS,
        text=f"Rate limit · {rl_remaining}/{RATE_LIMIT_CALLS}",
    )

    st.divider()

    # ── Search ─────────────────────────────────────────────────────────────
    st.markdown("## 🔍 Search")
    st.session_state.search_query = st.text_input(
        "Filter", placeholder="Search messages…", label_visibility="collapsed"
    )

    st.divider()

    # ── Session Management ─────────────────────────────────────────────────
    st.markdown("## 💾 Sessions")

    col_save, col_clear = st.columns(2)
    if col_save.button("💾 Save", use_container_width=True):
        save_session()
        st.success("Saved!")

    if col_clear.button("🗑 Clear", use_container_width=True):
        st.session_state.messages       = []
        st.session_state.response_times = []
        st.session_state.conversation   = build_chain(
            st.session_state.model, st.session_state.persona, st.session_state.temperature
        )
        st.rerun()

    sessions = load_all_sessions()
    if sessions:
        selected = st.selectbox(
            "Saved sessions",
            ["— select —"] + list(sessions.keys()),
            format_func=lambda s: s if s == "— select —"
                else f"{s}  ({sessions[s].get('persona','?')[:12]}…)",
        )
        col_load, col_del = st.columns(2)
        if col_load.button("📂 Load", use_container_width=True) and selected != "— select —":
            load_session(selected)
            st.rerun()
        if col_del.button("❌ Delete", use_container_width=True) and selected != "— select —":
            delete_session(selected)
            st.rerun()

    st.divider()

    # ── Export ─────────────────────────────────────────────────────────────
    st.markdown("## 📤 Export")
    if st.session_state.messages:
        json_blob = json.dumps(st.session_state.messages, indent=2, ensure_ascii=False)
        st.download_button(
            "⬇ JSON", json_blob,
            file_name=f"alpha_{st.session_state.session_id}.json",
            mime="application/json", use_container_width=True,
        )
        txt_lines = [f"[{m['role'].upper()}]\n{m['content']}\n" for m in st.session_state.messages]
        st.download_button(
            "⬇ TXT", "\n".join(txt_lines),
            file_name=f"alpha_{st.session_state.session_id}.txt",
            mime="text/plain", use_container_width=True,
        )
    else:
        st.caption("No messages to export yet.")

    st.divider()
    st.caption(f"Session · `{st.session_state.session_id}`")


# ──────────────────────────────────────────────────────────────────────────────
# Main Header
# ──────────────────────────────────────────────────────────────────────────────
persona_color = PERSONAS[st.session_state.persona]["color"]
model_short   = MODELS[st.session_state.model].split("-")[0].capitalize()

st.markdown(
    f"""
    <div class="alpha-header">
        <div class="alpha-logo">✦</div>
        <div>
            <div class="alpha-title">{BOT_NAME}</div>
            <div class="alpha-sub">Created By Alif· Always thinking, always ready</div>
        </div>
    </div>
    <div style="margin-bottom:18px">
        <span class="persona-badge" style="background:{persona_color}22;
              border:1px solid {persona_color}55;color:{persona_color}">
            {st.session_state.persona}
        </span>
        <span class="model-chip">⚡ {MODELS[st.session_state.model]}</span>
        <span class="model-chip">🌡 {st.session_state.temperature}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Welcome screen (empty state)
# ──────────────────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown(
        f"""
        <div class="welcome-card">
            <div class="welcome-icon">✦</div>
            <div class="welcome-title">Hi, I'm {BOT_NAME}</div>
            <div class="welcome-sub">
                Your intelligent AI assistant powered by state-of-the-art language models.<br>
                Ask me anything — code, design, analysis, or just a conversation.
            </div>
            <div class="suggestion-row">
                <span class="suggestion-chip">🧑‍💻 Write a sorting algorithm</span>
                <span class="suggestion-chip">🏗️ Design a URL shortener</span>
                <span class="suggestion-chip">📝 Review my code</span>
                <span class="suggestion-chip">🎯 Mock interview me</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Chat History Display
# ──────────────────────────────────────────────────────────────────────────────
search_term   = st.session_state.search_query.strip().lower()
visible_count = 0

for message in st.session_state.messages:
    content = message["content"]

    if search_term and search_term not in content.lower():
        continue
    visible_count += 1

    with st.chat_message(message["role"]):
        if search_term:
            highlighted = re.sub(
                f"(?i)({re.escape(search_term)})",
                r"**\1**",
                content,
            )
            st.markdown(highlighted)
        else:
            st.markdown(content)

        if message["role"] == "assistant" and "response_time" in message:
            st.markdown(
                f"<div class='response-time'>⏱ {message['response_time']:.2f}s</div>",
                unsafe_allow_html=True,
            )

if search_term and visible_count == 0:
    st.info(f"No messages match **{search_term}**.")

# ──────────────────────────────────────────────────────────────────────────────
# Rate-limit warning
# ──────────────────────────────────────────────────────────────────────────────
rl = st.session_state.rate_limiter
if rl.remaining <= 5:
    st.warning(
        f"⚠️ **{rl.remaining}** request(s) remaining · resets in ~{rl.reset_in:.0f}s"
    )

# ──────────────────────────────────────────────────────────────────────────────
# Chat Input
# ──────────────────────────────────────────────────────────────────────────────
user_input = st.chat_input(f"Message {BOT_NAME}…")

if user_input:
    rl = st.session_state.rate_limiter
    if not rl.allow():
        st.error(f"⛔ Rate limit reached. Wait **{rl.reset_in:.0f}s** before sending again.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner(f"{BOT_NAME} is thinking…"):
            t_start = time.perf_counter()
            try:
                response: str = st.session_state.conversation.invoke(
                    {"input": user_input}
                )["response"]
                elapsed = time.perf_counter() - t_start
                st.session_state.response_times.append(elapsed)
                st.markdown(response)
                st.markdown(
                    f"<div class='response-time'>⏱ {elapsed:.2f}s</div>",
                    unsafe_allow_html=True,
                )
                st.session_state.messages.append({
                    "role":          "assistant",
                    "content":       response,
                    "response_time": elapsed,
                })
            except Exception as exc:
                err = f"❌ Error: {exc}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
