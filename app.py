# streamlit_app.py
"""
Streamlit Agent Chat — Input fixed at bottom, chat scrolls up (ChatGPT-like)
Requires: agent.py with create_agent() and chat(user_input, agent_executor) -> str
"""

import streamlit as st
import uuid
import time
import traceback
from typing import List, Dict
import streamlit.components.v1 as components

# === Import agent utilities ===
try:
    from agent import create_agent, chat as agent_chat
except Exception as imp_err:
    create_agent = None
    agent_chat = None
    IMPORT_ERROR = imp_err
else:
    IMPORT_ERROR = None

# === Page config and CSS ===
st.set_page_config(page_title="Agentic AI — Chat", page_icon="🤖", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    /* Layout */
    .app-container { max-width: 1100px; margin: 10px auto; }
    .header { display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }
    .title { font-size:22px; font-weight:700; }
    .subtitle { color:#7b8794; font-size:12px; }

    /* Chatbox: fixed height and scrollable */
    #chatbox {
        border-radius: 12px;
        padding: 18px;
        background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.02));
        box-shadow: 0 8px 28px rgba(2,6,23,0.06);
        max-height: 68vh;          /* controls visible height */
        overflow-y: auto;
    }

    /* message bubbles */
    .msg-user { margin-left:auto; margin-bottom:12px; padding:12px 14px; border-radius:12px; background: linear-gradient(180deg,#dff3ff,#e9fbff); color:#052a3a; max-width:78%; box-shadow: 0 6px 18px rgba(2,6,23,0.06); word-wrap:break-word; }
    .msg-assistant { margin-right:auto; margin-bottom:12px; padding:12px 14px; border-radius:12px; background: linear-gradient(180deg,#0f1724,#111827); color:#e6eef8; max-width:78%; box-shadow: 0 6px 18px rgba(2,6,23,0.12); word-wrap:break-word; }
    .meta { font-size:11px; color:#98a0a6; margin-top:6px; }

    /* Input area fixed visually at bottom of the page content */
    .input-row { margin-top:12px; display:flex; gap:8px; align-items:center; }
    .send-btn { padding:10px 14px; border-radius:12px; background:#0f62fe; color:white; border:none; cursor:pointer; }
    .small-muted { color:#98a0a6; font-size:12px; }

    /* Responsive */
    @media (max-width: 800px) {
        #chatbox { max-height: 60vh; padding:12px; }
        .msg-user, .msg-assistant { max-width:92%; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# === Cached agent initializer ===
@st.cache_resource(show_spinner=False)
def get_agent_executor_cached():
    if create_agent is None:
        raise RuntimeError(f"agent.create_agent import failed: {IMPORT_ERROR}")
    return create_agent()

def ensure_agent_ready():
    if st.session_state.get("agent_ready") and st.session_state.get("agent_executor"):
        return st.session_state["agent_executor"]
    agent_exec = get_agent_executor_cached()
    st.session_state["agent_executor"] = agent_exec
    st.session_state["agent_ready"] = True
    return agent_exec

# === Session state init ===
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role","content","ts"}

if "agent_ready" not in st.session_state:
    st.session_state.agent_ready = False

if "last_error" not in st.session_state:
    st.session_state.last_error = None

# === Header ===
st.markdown('<div class="app-container">', unsafe_allow_html=True)
col_left, col_right = st.columns([4,1])
with col_left:
    st.markdown('<div class="header"><div><div class="title">🤖 Agentic AI Chat</div><div class="subtitle">Integrated LangChain agent — Tools: datetime, weather, web search (tavily)</div></div></div>', unsafe_allow_html=True)
with col_right:
    st.markdown(f"<div class='small-muted'>Session: {st.session_state.session_id.split('-')[0]}</div>", unsafe_allow_html=True)

st.divider()

# === Minimal status area ===
status_col, _ = st.columns([3,1])
with status_col:
    if st.session_state.agent_ready:
        st.success("Agent initialized")
    else:
        st.info("Agent not initialized (initializing automatically)")

# === Main chat container ===
chatbox_placeholder = st.empty()

# Initialize agent lazily (cold start)
if not st.session_state.agent_ready:
    try:
        with st.spinner("Initializing agent (cold start may take a bit)..."):
            ensure_agent_ready()
            st.session_state.agent_ready = True
    except Exception:
        st.session_state.last_error = traceback.format_exc()
        st.error("Agent initialization failed. Open logs in the sidebar.")
        # Render minimal chatbox and stop
        chat_html = "<div id='chatbox'><div style='color:#e53e3e'>Agent init failed. Check logs.</div></div>"
        chatbox_placeholder.markdown(chat_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

# Function to render chat HTML (inside chatbox div)
def render_chat_html(messages):
    html_parts = ["<div id='chatbox'>"]
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        ts = msg.get("ts", time.time())
        ts_str = time.strftime("%H:%M", time.localtime(ts))
        safe_content = content.replace("\n", "<br>")  # simple newline -> <br>
        if role == "user":
            html_parts.append(
                f"<div style='display:flex; justify-content:flex-end;'><div class='msg-user'>{safe_content}<div class='meta'>You • {ts_str}</div></div></div>"
            )
        else:
            html_parts.append(
                f"<div style='display:flex; justify-content:flex-start;'><div class='msg-assistant'>{safe_content}<div class='meta'>Agent • {ts_str}</div></div></div>"
            )
    html_parts.append("</div>")
    return "\n".join(html_parts)

# Render current messages initially
chatbox_placeholder.markdown(render_chat_html(st.session_state.messages), unsafe_allow_html=True)

# === Input area: placed below the chatbox so it visually remains at bottom ===
# We use a form to submit the user message (clears automatically)
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area("Type your message:", key="user_input", height=120, placeholder="Ask me anything...")
    submit = st.form_submit_button("Send")

if submit and user_input and user_input.strip():
    text = user_input.strip()
    # Append user message immediately to session messages
    st.session_state.messages.append({"role": "user", "content": text, "ts": time.time()})

    # Re-render chatbox with updated messages (user message shown)
    chatbox_placeholder.markdown(render_chat_html(st.session_state.messages), unsafe_allow_html=True)

    # Create assistant placeholder in the chat history (so UI shows a bubble while thinking)
    st.session_state.messages.append({"role": "assistant", "content": "Thinking...", "ts": time.time()})
    chatbox_placeholder.markdown(render_chat_html(st.session_state.messages), unsafe_allow_html=True)

    # Ensure agent ready
    try:
        agent_exec = st.session_state.get("agent_executor") or ensure_agent_ready()
        start = time.time()
        # Call your agent.chat synchronously
        try:
            response = agent_chat(text, agent_exec)
            if response is None:
                response = "I didn't receive a response. Please try again."
            elif not isinstance(response, str):
                try:
                    response = str(response)
                except Exception as e:
                    response = f"Received an unexpected response format: {type(response).__name__}"
        except Exception as e:
            response = f"I encountered an error while processing your request: {str(e)}"

        # Update assistant message in session_state with final text
        # Find last assistant message index (should be the last element)
        for i in range(len(st.session_state.messages) - 1, -1, -1):
            if st.session_state.messages[i]["role"] == "assistant":
                st.session_state.messages[i]["content"] = response
                st.session_state.messages[i]["ts"] = time.time()
                break
        # Re-render chat with final assistant response
        chatbox_placeholder.markdown(render_chat_html(st.session_state.messages), unsafe_allow_html=True)
        elapsed = time.time() - start
        # show small response time line below chatbox (optional)
        st.markdown(f"<div style='text-align:right; color:#98a0a6; font-size:12px'>Response time: {elapsed:.2f}s</div>", unsafe_allow_html=True)

    except Exception:
        tb = traceback.format_exc()
        st.session_state.last_error = tb
        # Replace assistant placeholder with friendly error
        for i in range(len(st.session_state.messages) - 1, -1, -1):
            if st.session_state.messages[i]["role"] == "assistant":
                st.session_state.messages[i]["content"] = "⚠️ Agent failed to respond. Check logs."
                st.session_state.messages[i]["ts"] = time.time()
                break
        chatbox_placeholder.markdown(render_chat_html(st.session_state.messages), unsafe_allow_html=True)
        # Show error in sidebar
        with st.sidebar.expander("Last Error (traceback)"):
            st.code(tb)

    # After updating chatbox, use a tiny JS snippet to scroll the chatbox to bottom
    scroll_js = """
    <script>
    const cb = document.getElementById('chatbox');
    if (cb) { cb.scrollTop = cb.scrollHeight; }
    </script>
    """
    components.html(scroll_js, height=0)

# Ensure chat is scrolled to bottom on initial page load as well
components.html(
    """
    <script>
    const cb = document.getElementById('chatbox');
    if (cb) { cb.scrollTop = cb.scrollHeight; }
    </script>
    """,
    height=0,
)

st.markdown("</div>", unsafe_allow_html=True)
# Minimal footer
st.markdown("<div style='color:#98a0a6; margin-top:8px; font-size:12px; text-align:center;'>Keep API keys in Streamlit Secrets • Built with LangChain</div>", unsafe_allow_html=True)