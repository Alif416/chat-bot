import base64
import json
import os
import re
import time
from datetime import datetime
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from rate_limiter import RateLimiter

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Alpha AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

.stApp {
    background: #FAF9F6;
}
.stApp > header { background: transparent !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #1A1A1A !important;
    border-right: 1px solid #2A2A2A !important;
}
[data-testid="stSidebar"] .stMarkdown h2 {
    font-size: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #6E6E73 !important;
    margin-top: 4px !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stTextInput label {
    color: #AEAEB2 !important;
    font-size: 12px !important;
}
[data-testid="stSidebar"] .stCaption p,
[data-testid="stSidebar"] .stMarkdown p {
    color: #6E6E73 !important;
    font-size: 11px !important;
}
[data-testid="stSidebar"] [data-testid="stMetric"] {
    background: #242424 !important;
    border: 1px solid #333333 !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: #E5E5E7 !important;
    font-size: 15px !important;
}
[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
    color: #6E6E73 !important;
    font-size: 10px !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: #242424 !important;
    border: 1px solid #333333 !important;
    color: #C7C7CC !important;
    border-radius: 7px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    transition: all 0.15s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #2C2C2C !important;
    border-color: #D97757 !important;
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] [data-testid="stProgress"] > div > div {
    background: #D97757 !important;
    border-radius: 99px !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    padding: 14px 18px !important;
    margin-bottom: 4px !important;
    border-radius: 10px !important;
}
[data-testid="stChatMessage"][data-role="user"] {
    background: #EFEEE9 !important;
    border: 1px solid #E2E1DC !important;
}
[data-testid="stChatMessage"][data-role="assistant"] {
    background: transparent !important;
    border: none !important;
}

/* ── Chat input ── */
[data-testid="stChatInputContainer"] {
    background: #FFFFFF !important;
    border: 1px solid #DDDCDA !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
}
[data-testid="stChatInputContainer"]:focus-within {
    border-color: #D97757 !important;
    box-shadow: 0 0 0 3px rgba(217,119,87,0.12) !important;
}

/* ── Buttons (main area) ── */
.stButton > button {
    background: #FFFFFF !important;
    border: 1px solid #DDDCDA !important;
    color: #3A3935 !important;
    border-radius: 7px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    background: #F5F4F0 !important;
    border-color: #B5B3AE !important;
}

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div {
    background: #D97757 !important;
    border-radius: 99px !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #F5F4F0;
    border: 1px solid #E4E3DF;
    border-radius: 8px;
    padding: 8px 10px;
}
[data-testid="stMetricValue"] { font-size: 18px !important; color: #1C1C1C !important; }
[data-testid="stMetricLabel"] { font-size: 10px !important; color: #8A8A88 !important; }

/* ── Divider ── */
hr { border-color: #E4E3DF !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #CFCEC9; border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: #B5B3AE; }

/* ── Header ── */
.alpha-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding-bottom: 16px;
    border-bottom: 1px solid #E4E3DF;
    margin-bottom: 20px;
}
.alpha-logo {
    width: 36px;
    height: 36px;
    background: #D97757;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 17px;
    flex-shrink: 0;
}
.alpha-title { font-size: 20px; font-weight: 700; color: #1C1C1C; letter-spacing: -0.3px; }
.alpha-sub   { font-size: 12px; color: #8A8A88; margin-top: 1px; }

.chip {
    display: inline-flex;
    align-items: center;
    background: #F0EFEB;
    border: 1px solid #E2E1DC;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 11px;
    color: #6A6963;
    margin-right: 6px;
    vertical-align: middle;
}
.persona-tag {
    display: inline-flex;
    align-items: center;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    margin-right: 6px;
    vertical-align: middle;
}

/* ── Welcome ── */
.welcome-wrap {
    text-align: center;
    padding: 64px 40px;
    max-width: 560px;
    margin: 0 auto;
}
.welcome-icon {
    width: 52px;
    height: 52px;
    background: #D97757;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    margin: 0 auto 18px;
}
.welcome-title { font-size: 22px; font-weight: 600; color: #1C1C1C; margin-bottom: 8px; }
.welcome-sub   { font-size: 14px; color: #7A7973; line-height: 1.7; }
.chip-row { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 24px; }
.prompt-chip {
    padding: 7px 14px;
    background: #F5F4F0;
    border: 1px solid #E4E3DF;
    border-radius: 20px;
    font-size: 13px;
    color: #3A3935;
}

/* ── Chat text colour ── */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3,
[data-testid="stChatMessage"] td,
[data-testid="stChatMessage"] th {
    color: #1C1C1C !important;
}

/* ── Code blocks (dark bg → white text) ── */
[data-testid="stChatMessage"] pre {
    background: #1E1E1E !important;
    border-radius: 8px !important;
    border: 1px solid #333 !important;
}
[data-testid="stChatMessage"] pre code {
    color: #E8E8E8 !important;
    background: transparent !important;
}

/* ── Inline code (light bg → dark text) ── */
[data-testid="stChatMessage"] code {
    color: #C7383A !important;
    background: #F5F4F0 !important;
    border-radius: 4px !important;
    padding: 1px 5px !important;
    font-size: 0.88em !important;
}

/* ── Response timestamp ── */
.response-time {
    font-size: 10px;
    color: #B5B3AE;
    text-align: right;
    margin-top: 4px;
}
"""

st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
BOT_NAME = "Alpha"
HISTORY_FILE = "chat_history.json"
RATE_LIMIT_CALLS = 20
RATE_LIMIT_WINDOW = 60

MODELS: dict[str, str] = {
    # ── OpenAI OSS (via Groq) ─────────────────────────────────────────────
    "GPT OSS 120B    (Most powerful)":   "openai/gpt-oss-120b",
    "GPT OSS 20B     (Balanced)":        "openai/gpt-oss-20b",
    # ── Meta LLaMA ────────────────────────────────────────────────────────
    "LLaMA 3.3 70B   (High quality)":    "llama-3.3-70b-versatile",
    "LLaMA 3.1 8B    (Fast)":            "llama-3.1-8b-instant",
    # ── Reasoning ─────────────────────────────────────────────────────────
    "DeepSeek R1 70B (Reasoning)":       "deepseek-r1-distill-llama-70b",
    "Qwen 3 32B      (Reasoning)":       "qwen/qwen3-32b",
}

# Vision-capable models (image input supported)
VISION_MODELS: set[str] = set()  # no confirmed vision model in current Groq lineup

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
            "You are a principal engineer specialising in large-scale distributed systems. "
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
        "color": "#D97757",
        "temp_hint": ("0.5 – 0.8", "balanced and conversational"),
    },
}


# ── Session init ──────────────────────────────────────────────────────────────
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
        "pending_image":  None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session()


# ── Chain builder ─────────────────────────────────────────────────────────────
def build_chain(
    model_key: str, persona_key: str, temperature: float, max_tokens: int
) -> ConversationChain:
    persona_prompt = PERSONAS[persona_key]["prompt"]
    prompt = ChatPromptTemplate.from_messages([
        ("system", persona_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    llm = ChatGroq(
        model_name=MODELS[model_key],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    memory = ConversationBufferMemory(return_messages=True)

    msgs = st.session_state.messages
    for i in range(0, len(msgs) - 1, 2):
        if i + 1 < len(msgs) and msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
            memory.chat_memory.add_user_message(msgs[i]["content"])
            memory.chat_memory.add_ai_message(msgs[i + 1]["content"])

    return ConversationChain(llm=llm, memory=memory, prompt=prompt)


if st.session_state.conversation is None:
    st.session_state.conversation = build_chain(
        st.session_state.model,
        st.session_state.persona,
        st.session_state.temperature,
        st.session_state.max_tokens,
    )


# ── Audio transcription ───────────────────────────────────────────────────────
def transcribe_audio(audio_bytes: bytes, mime_type: str) -> str:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    ext = mime_type.split("/")[-1].split(";")[0]
    transcription = client.audio.transcriptions.create(
        file=(f"audio.{ext}", audio_bytes),
        model="whisper-large-v3-turbo",
        response_format="text",
    )
    return str(transcription).strip()


# ── Vision inference (bypasses ConversationChain for multimodal input) ────────
def invoke_with_image(text: str, image_bytes: bytes, mime_type: str) -> str:
    llm = ChatGroq(
        model_name=MODELS[st.session_state.model],
        temperature=st.session_state.temperature,
        max_tokens=st.session_state.max_tokens,
    )
    msgs: list = [SystemMessage(content=PERSONAS[st.session_state.persona]["prompt"])]
    for m in st.session_state.messages:
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        else:
            msgs.append(AIMessage(content=m["content"]))

    b64 = base64.b64encode(image_bytes).decode()
    msgs.append(HumanMessage(content=[
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
    ]))

    response = llm.invoke(msgs)

    # Keep the text chain's memory in sync for follow-up messages
    conv = st.session_state.conversation
    conv.memory.chat_memory.add_user_message(text)
    conv.memory.chat_memory.add_ai_message(response.content)

    return response.content


# ── Persistence helpers ───────────────────────────────────────────────────────
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
    st.session_state.messages = s["messages"]

    saved_persona = s.get("persona", "")
    st.session_state.persona = saved_persona if saved_persona in PERSONAS else list(PERSONAS.keys())[0]

    saved_model = s.get("model", "")
    st.session_state.model = saved_model if saved_model in MODELS else list(MODELS.keys())[0]
    st.session_state.session_id     = session_id
    st.session_state.response_times = []
    st.session_state.conversation   = build_chain(
        st.session_state.model,
        st.session_state.persona,
        st.session_state.temperature,
        st.session_state.max_tokens,
    )


def delete_session(session_id: str) -> None:
    sessions = load_all_sessions()
    sessions.pop(session_id, None)
    with open(HISTORY_FILE, "w", encoding="utf-8") as fh:
        json.dump(sessions, fh, indent=2, ensure_ascii=False)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;padding:4px 0 16px'>"
        f"<div style='width:28px;height:28px;background:#D97757;border-radius:7px;"
        f"display:flex;align-items:center;justify-content:center;font-size:14px;'>✦</div>"
        f"<span style='font-size:16px;font-weight:700;color:#E5E5E7;letter-spacing:-0.3px'>{BOT_NAME}</span>"
        f"<span style='font-size:10px;color:#D97757;background:rgba(217,119,87,0.15);"
        f"padding:2px 8px;border-radius:99px;font-weight:600;'>AI</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("## Configuration")

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

    new_max_tokens = st.slider(
        "Max tokens", 256, 4096, st.session_state.max_tokens, 256,
        help="Maximum tokens in the model's response",
    )

    settings_changed = (
        new_model      != st.session_state.model       or
        new_persona    != st.session_state.persona     or
        new_temp       != st.session_state.temperature or
        new_max_tokens != st.session_state.max_tokens
    )
    if settings_changed:
        st.session_state.model        = new_model
        st.session_state.persona      = new_persona
        st.session_state.temperature  = new_temp
        st.session_state.max_tokens   = new_max_tokens
        st.session_state.conversation = build_chain(new_model, new_persona, new_temp, new_max_tokens)
        st.success("Settings applied — context preserved.")

    st.divider()

    # Analytics
    st.markdown("## Analytics")
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

    # Search
    st.markdown("## Search")
    st.session_state.search_query = st.text_input(
        "Filter", placeholder="Search messages…", label_visibility="collapsed"
    )

    st.divider()

    # Session management
    st.markdown("## Sessions")

    col_save, col_clear = st.columns(2)
    if col_save.button("Save", use_container_width=True):
        save_session()
        st.success("Saved!")

    if col_clear.button("Clear", use_container_width=True):
        st.session_state.messages       = []
        st.session_state.response_times = []
        st.session_state.conversation   = build_chain(
            st.session_state.model, st.session_state.persona,
            st.session_state.temperature, st.session_state.max_tokens,
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
        if col_load.button("Load", use_container_width=True) and selected != "— select —":
            load_session(selected)
            st.rerun()
        if col_del.button("Delete", use_container_width=True) and selected != "— select —":
            delete_session(selected)
            st.rerun()

    st.divider()

    # Export
    st.markdown("## Export")
    if st.session_state.messages:
        json_blob = json.dumps(st.session_state.messages, indent=2, ensure_ascii=False)
        st.download_button(
            "Download JSON", json_blob,
            file_name=f"alpha_{st.session_state.session_id}.json",
            mime="application/json", use_container_width=True,
        )
        txt_lines = [f"[{m['role'].upper()}]\n{m['content']}\n" for m in st.session_state.messages]
        st.download_button(
            "Download TXT", "\n".join(txt_lines),
            file_name=f"alpha_{st.session_state.session_id}.txt",
            mime="text/plain", use_container_width=True,
        )
    else:
        st.caption("No messages to export yet.")

    st.divider()
    st.caption(f"Session · `{st.session_state.session_id}`")


# ── Header ────────────────────────────────────────────────────────────────────
persona_color = PERSONAS[st.session_state.persona]["color"]

st.markdown(
    f"""
    <div class="alpha-header">
        <div class="alpha-logo">✦</div>
        <div>
            <div class="alpha-title">{BOT_NAME}</div>
            <div class="alpha-sub">Created by Alif · Always thinking, always ready</div>
        </div>
    </div>
    <div style="margin-bottom:20px">
        <span class="persona-tag"
              style="background:{persona_color}18;border:1px solid {persona_color}40;color:{persona_color}">
            {st.session_state.persona}
        </span>
        <span class="chip">⚡ {MODELS[st.session_state.model]}</span>
        <span class="chip">🌡 {st.session_state.temperature}</span>
        <span class="chip">↕ {st.session_state.max_tokens} tok</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Welcome screen ────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown(
        f"""
        <div class="welcome-wrap">
            <div class="welcome-icon">✦</div>
            <div class="welcome-title">Hi, I'm {BOT_NAME}</div>
            <div class="welcome-sub">
                Your intelligent AI assistant powered by state-of-the-art language models.<br>
                Ask me anything — code, design, analysis, or just a conversation.
            </div>
            <div class="chip-row">
                <span class="prompt-chip">Write a sorting algorithm</span>
                <span class="prompt-chip">Design a URL shortener</span>
                <span class="prompt-chip">Review my code</span>
                <span class="prompt-chip">Mock interview me</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Chat history ──────────────────────────────────────────────────────────────
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

# ── Rate-limit warning ────────────────────────────────────────────────────────
rl = st.session_state.rate_limiter
if rl.remaining <= 5:
    st.warning(f"⚠️ **{rl.remaining}** request(s) remaining · resets in ~{rl.reset_in:.0f}s")

# ── Attachments ───────────────────────────────────────────────────────────────
is_vision_model = MODELS[st.session_state.model] in VISION_MODELS

with st.expander("📎 Attach  ·  🎙 Voice", expanded=bool(st.session_state.pending_image)):
    att_col, aud_col = st.columns(2)

    with att_col:
        if is_vision_model:
            img_file = st.file_uploader(
                "Image", type=["jpg", "jpeg", "png", "webp"],
                key="img_upload", label_visibility="collapsed",
            )
            if img_file:
                img_bytes = img_file.read()
                st.session_state.pending_image = {"bytes": img_bytes, "type": img_file.type}
                st.image(BytesIO(img_bytes), width=180)
            elif st.session_state.pending_image:
                st.image(BytesIO(st.session_state.pending_image["bytes"]), width=180)
                if st.button("Remove image", use_container_width=True):
                    st.session_state.pending_image = None
                    st.rerun()
        else:
            st.caption("Switch to **Llama 4 Scout** or **Maverick** to attach images.")

    with aud_col:
        audio_file = st.file_uploader(
            "Audio (auto-transcribe)", type=["wav", "mp3", "m4a", "webm", "ogg"],
            key="aud_upload", label_visibility="collapsed",
        )
        if audio_file:
            if st.button("Transcribe & Send", use_container_width=True):
                with st.spinner("Transcribing with Whisper…"):
                    try:
                        transcript = transcribe_audio(audio_file.read(), audio_file.type)
                        st.session_state["_voice_input"] = transcript
                        st.rerun()
                    except Exception as exc:
                        st.error(f"❌ Transcription failed: {exc}")

# ── Chat input ────────────────────────────────────────────────────────────────
voice_prefill = st.session_state.pop("_voice_input", None)
user_input = st.chat_input(f"Message {BOT_NAME}…") or voice_prefill

if user_input:
    rl = st.session_state.rate_limiter
    if not rl.allow():
        st.error(f"⛔ Rate limit reached. Wait **{rl.reset_in:.0f}s** before sending again.")
        st.stop()

    pending = st.session_state.pending_image
    st.session_state.pending_image = None

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        if pending:
            st.image(BytesIO(pending["bytes"]), width=220)

    with st.chat_message("assistant"):
        with st.spinner(f"{BOT_NAME} is thinking…"):
            t_start = time.perf_counter()
            try:
                if pending and is_vision_model:
                    response = invoke_with_image(user_input, pending["bytes"], pending["type"])
                else:
                    response = st.session_state.conversation.invoke(
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
                st.error(f"❌ {exc}")
