"""
Frontier Model Personal Assistant — Google Gemini 1.5 Flash (Free Tier)
Falls back to HuggingFace Inference API (Mistral-7B) if no Gemini key.
Features: Multi-turn memory, tool use, safety guardrails, observability
Interface: Streamlit (beautiful dark UI)
Author: Krishna Murthi
"""

import os
import sys
import time
import json
import re

import streamlit as st
from huggingface_hub import InferenceClient

# Add parent directory to path for shared modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.memory import ConversationMemory
from shared.tools import get_tool_descriptions, calculator, get_datetime, web_search, unit_converter
from shared.guardrails import check_input_safety, check_output_safety, get_refusal_message

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

GEMINI_KEY = os.environ.get("GOOGLE_API_KEY", None)
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# Use Gemini if available, otherwise fall back to Mistral via HF
if GEMINI_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_KEY)
        MODEL_NAME = "gemini-1.5-flash"
        USE_GEMINI = True
    except ImportError:
        USE_GEMINI = False
        MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
else:
    USE_GEMINI = False
    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

SYSTEM_PROMPT = f"""You are an advanced AI assistant — powerful, precise, and principled.

Your core traits:
• Accuracy: Always strive for factual correctness; say "I'm not certain" when unsure
• Helpfulness: Give complete, actionable responses — not vague generalities  
• Safety: Firmly but politely decline harmful requests
• Honesty: Never fabricate information or pretend to have capabilities you don't

{get_tool_descriptions()}

When you need to use a tool, format it as: [TOOL: tool_name(arguments)]
After tool results are shown, continue your response naturally and helpfully."""

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit Page Config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Frontier AI Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Frontier AI Assistant — Ollive AI Assignment by Krishna Murthi"
    }
)

# ──────────────────────────────────────────────────────────────────────────────
# CSS Styling
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root ── */
:root {
    --bg-primary: #0a0a0f;
    --bg-secondary: #111118;
    --bg-card: #16161f;
    --accent: #7c3aed;
    --accent-light: #a78bfa;
    --accent-glow: rgba(124,58,237,0.3);
    --text-primary: #f1f1f5;
    --text-secondary: #9ca3af;
    --border: rgba(255,255,255,0.08);
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
}

/* ── App Background ── */
.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0d0a1a 50%, #0a0f0a 100%);
    font-family: 'Inter', system-ui, sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0a1a 0%, #0a0a0f 100%) !important;
    border-right: 1px solid var(--border) !important;
}

/* ── Chat Messages ── */
[data-testid="stChatMessage"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 16px !important;
    margin-bottom: 12px !important;
    backdrop-filter: blur(10px);
    transition: all 0.2s ease;
}

[data-testid="stChatMessage"]:hover {
    border-color: var(--accent-glow) !important;
    box-shadow: 0 0 20px var(--accent-glow) !important;
}

/* ── User Messages ── */
[data-testid="stChatMessage"][data-testid*="user"] {
    background: linear-gradient(135deg, rgba(124,58,237,0.1), rgba(167,139,250,0.05)) !important;
    border-color: rgba(124,58,237,0.2) !important;
}

/* ── Input Box ── */
[data-testid="stChatInput"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
}

/* ── Metrics ── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    transition: transform 0.2s ease;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-value {
    font-size: 24px;
    font-weight: 700;
    color: var(--accent-light);
    font-family: 'JetBrains Mono', monospace;
}
.metric-label {
    font-size: 12px;
    color: var(--text-secondary);
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Safety Badge ── */
.safe-badge {
    display: inline-block;
    background: rgba(16,185,129,0.15);
    border: 1px solid rgba(16,185,129,0.3);
    color: #10b981;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
}
.unsafe-badge {
    background: rgba(239,68,68,0.15);
    border-color: rgba(239,68,68,0.3);
    color: #ef4444;
}

/* ── Code blocks ── */
code {
    background: rgba(124,58,237,0.1) !important;
    border: 1px solid rgba(124,58,237,0.2) !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--accent-light) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 3px; }

/* ── Animations ── */
@keyframes glow-pulse {
    0%, 100% { box-shadow: 0 0 10px var(--accent-glow); }
    50% { box-shadow: 0 0 25px var(--accent-glow), 0 0 50px rgba(124,58,237,0.1); }
}

.header-glow {
    animation: glow-pulse 3s ease-in-out infinite;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────────────────────────────────────

if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory(max_turns=12, system_prompt=SYSTEM_PROMPT)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "total_latency" not in st.session_state:
    st.session_state.total_latency = []

if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0

if "safety_events" not in st.session_state:
    st.session_state.safety_events = []

# ──────────────────────────────────────────────────────────────────────────────
# Tool Execution
# ──────────────────────────────────────────────────────────────────────────────

def extract_and_run_tools(text: str) -> str:
    tool_pattern = re.compile(r'\[TOOL:\s*(\w+)\(([^)]*)\)\]')

    def execute_tool(match):
        tool_name = match.group(1).strip()
        args_str = match.group(2).strip()
        try:
            if tool_name == "calculator":
                result = calculator(args_str)
            elif tool_name == "datetime":
                result = get_datetime()
            elif tool_name == "search":
                result = web_search(args_str)
            elif tool_name == "convert":
                parts = [p.strip() for p in args_str.split(',')]
                result = unit_converter(float(parts[0]), parts[1], parts[2]) if len(parts) == 3 else "Usage: convert(value, from_unit, to_unit)"
            else:
                result = f"Unknown tool: {tool_name}"
        except Exception as e:
            result = f"Tool error: {str(e)}"
        return f"\n\n> 🔧 **Tool: `{tool_name}`** → {result}\n\n"

    return tool_pattern.sub(execute_tool, text)


# ──────────────────────────────────────────────────────────────────────────────
# LLM Inference
# ──────────────────────────────────────────────────────────────────────────────

def call_gemini(messages: list) -> str:
    import google.generativeai as genai
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=SYSTEM_PROMPT,
    )
    # Convert messages to Gemini format
    history = []
    for msg in messages[1:]:  # Skip system message
        role = "user" if msg["role"] == "user" else "model"
        history.append({"role": role, "parts": [msg["content"]]})

    chat_session = model.start_chat(history=history[:-1] if len(history) > 1 else [])
    response = chat_session.send_message(history[-1]["parts"][0] if history else "Hello")
    return response.text


def call_hf_inference(messages: list) -> str:
    client = InferenceClient(token=HF_TOKEN)
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=768,
        temperature=0.7,
        top_p=0.9,
    )
    return completion.choices[0].message.content


def generate_response(messages: list) -> tuple[str, float]:
    start = time.time()
    try:
        if USE_GEMINI:
            response = call_gemini(messages)
        else:
            response = call_hf_inference(messages)
    except Exception as e:
        response = f"⚠️ I encountered an error: {str(e)}\n\nPlease try again or check your connection."
    latency = (time.time() - start) * 1000
    return response, latency


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px;">
        <div style="font-size:48px; margin-bottom:8px;">⚡</div>
        <h2 style="color:#a78bfa; margin:0; font-weight:700;">Frontier AI</h2>
        <p style="color:#6b7280; font-size:13px; margin-top:4px;">Personal Assistant</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Model info
    st.markdown("**🤖 Model**")
    model_badge = "🌟 Gemini 1.5 Flash" if USE_GEMINI else "🔷 Mistral-7B-Instruct"
    st.info(f"{model_badge}")
    st.caption(f"ID: `{MODEL_NAME}`")

    st.divider()

    # Live Metrics
    st.markdown("**📊 Session Metrics**")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Turns",
            st.session_state.memory.turn_count,
            help="Total conversation turns"
        )
    with col2:
        avg_latency = (
            sum(st.session_state.total_latency) / len(st.session_state.total_latency)
            if st.session_state.total_latency else 0
        )
        st.metric(
            "Avg Latency",
            f"{avg_latency:.0f}ms",
            help="Average response time"
        )

    col3, col4 = st.columns(2)
    with col3:
        st.metric("Tokens (est.)", st.session_state.total_tokens)
    with col4:
        st.metric("Safety Events", len(st.session_state.safety_events))

    st.divider()

    # Controls
    st.markdown("**⚙️ Controls**")
    show_metrics = st.toggle("Show response metrics", value=True)
    show_safety = st.toggle("Show safety analysis", value=False)

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.memory.clear()
        st.session_state.messages = []
        st.session_state.total_latency = []
        st.session_state.total_tokens = 0
        st.session_state.safety_events = []
        st.rerun()

    if st.button("📥 Export Conversation", use_container_width=True):
        data = st.session_state.memory.to_json()
        st.download_button(
            "Download JSON",
            data=data,
            file_name="conversation_export.json",
            mime="application/json",
            use_container_width=True,
        )

    st.divider()
    st.caption("Built by **Krishna Murthi**\nOllive AI Assignment")


# ──────────────────────────────────────────────────────────────────────────────
# Main UI
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="header-glow" style="
    background: linear-gradient(135deg, rgba(124,58,237,0.15), rgba(167,139,250,0.05));
    border: 1px solid rgba(124,58,237,0.3);
    border-radius: 20px;
    padding: 24px 32px;
    margin-bottom: 24px;
    text-align: center;
">
    <h1 style="
        margin: 0;
        font-size: 32px;
        font-weight: 800;
        background: linear-gradient(135deg, #7c3aed, #a78bfa, #c4b5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    ">⚡ Frontier AI Assistant</h1>
    <p style="color: #9ca3af; margin: 8px 0 0; font-size: 15px;">
        Advanced AI with Memory · Tools · Safety · Observability
    </p>
</div>
""", unsafe_allow_html=True)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "⚡"):
        st.markdown(msg["content"])
        if show_metrics and msg.get("metrics"):
            m = msg["metrics"]
            st.caption(
                f"⏱️ {m.get('latency_ms', 0):.0f}ms · "
                f"📝 ~{m.get('tokens', 0)} tokens · "
                f"🛡️ {m.get('safety_category', 'safe')}"
            )

# ──────────────────────────────────────────────────────────────────────────────
# Chat Input
# ──────────────────────────────────────────────────────────────────────────────

if user_input := st.chat_input("Ask me anything... (try math, questions, or code!)"):

    # ── Safety Check ──────────────────────────────────────────────────────
    safety = check_input_safety(user_input)

    if show_safety:
        badge_class = "safe-badge" if safety.is_safe else "safe-badge unsafe-badge"
        badge_text = f"✅ {safety.reason}" if safety.is_safe else f"🚨 {safety.reason}"
        st.markdown(f'<div class="{badge_class}">{badge_text}</div>', unsafe_allow_html=True)

    if not safety.is_safe:
        st.session_state.safety_events.append(safety)
        refusal = get_refusal_message(safety.category)

        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="⚡"):
            st.markdown(f"🛡️ {refusal}")

        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": f"🛡️ {refusal}"})
        st.rerun()

    else:
        # ── Display user message ───────────────────────────────────────────
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_input)

        # ── Update memory ──────────────────────────────────────────────────
        st.session_state.memory.add_user_message(user_input)
        messages = st.session_state.memory.get_full_prompt()

        # ── Generate response with streaming indicator ─────────────────────
        with st.chat_message("assistant", avatar="⚡"):
            with st.spinner("Thinking..."):
                raw_response, latency_ms = generate_response(messages)

            # ── Tool execution ─────────────────────────────────────────────
            response_with_tools = extract_and_run_tools(raw_response)

            # ── Output safety check ────────────────────────────────────────
            is_safe, final_response = check_output_safety(response_with_tools)
            if not is_safe:
                final_response = "🛡️ " + final_response

            st.markdown(final_response)

            # ── Metrics display ────────────────────────────────────────────
            est_tokens = (len(user_input) + len(final_response)) // 4
            st.session_state.total_tokens += est_tokens
            st.session_state.total_latency.append(latency_ms)

            metrics = {
                "latency_ms": latency_ms,
                "tokens": est_tokens,
                "safety_category": safety.category,
            }

            if show_metrics:
                st.caption(
                    f"⏱️ {latency_ms:.0f}ms · 📝 ~{est_tokens} tokens · "
                    f"💰 {'Free (OSS)' if not USE_GEMINI else 'Gemini Free'} · "
                    f"🛡️ {safety.category}"
                )

        # ── Update state ───────────────────────────────────────────────────
        st.session_state.memory.add_assistant_message(final_response)
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_response,
            "metrics": metrics
        })
