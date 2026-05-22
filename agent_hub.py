"""
AI Agent Hub — Multi-Agent Research Assistant (LangChain Edition)
===================================================================
A comprehensive multi-agent system with:
  1. User Authentication & API Configuration
  2. LangChain Agent with Tools (Autonomous Routing)
  3. PDF Document Retriever & Q&A with Guardrails
  4. Web Search Agent (Tavily Tool)
  5. Blog Analyzer with Scope Detection
  6. Sentiment Analysis Tool
  7. Guardrails: Out-of-scope detection + Web Search fallback

Tech Stack:
  - Streamlit, LangChain Tools & Agents
  - Azure OpenAI GPT-4o with Tool Calling
  - Tavily for web search
  - PyMuPDF for PDF processing
  - BeautifulSoup for blog scraping
  - SQLite for persistence
"""

import streamlit as st
import os
import re
import hashlib
import sqlite3
import uuid
import json
import base64
import html as html_lib
import io
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import pymupdf
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from warnings import filterwarnings
from typing import List, Dict, Any, Optional

# LangChain imports
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.tavily_search import TavilySearchResults

filterwarnings("ignore")

st.set_page_config(
    page_title="AI Agent Hub",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS - Modern Agent Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --primary: #7C3AED;
    --primary-dim: rgba(124,58,237,0.12);
    --primary-border: rgba(124,58,237,0.25);
    --primary-glow: rgba(124,58,237,0.35);
    --accent: #10B981;
    --accent-dim: rgba(16,185,129,0.12);
    --warning: #F59E0B;
    --danger: #EF4444;
    --info: #3B82F6;
    --bg-root: #0F0F1A;
    --bg-surface: #151521;
    --bg-card: #1C1C2E;
    --bg-elevated: #252538;
    --text-100: #F8FAFC;
    --text-200: #E2E8F0;
    --text-300: #94A3B8;
    --text-400: #64748B;
    --border: rgba(255,255,255,0.08);
    --border-focus: rgba(124,58,237,0.5);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
}

html, body, .stApp {
    background: var(--bg-root) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-100) !important;
}
#MainMenu, footer { visibility: hidden; }
.stDeployButton { display: none; }

/* Keep sidebar toggle button visible and styled */
button[data-testid="stSidebarCollapseButton"],
button[data-testid="collapsedControl"] {
    visibility: visible !important;
    display: flex !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-200) !important;
    padding: 0.5rem !important;
    margin: 0.5rem !important;
    cursor: pointer !important;
    z-index: 999 !important;
}
button[data-testid="stSidebarCollapseButton"]:hover,
button[data-testid="collapsedControl"]:hover {
    background: var(--bg-elevated) !important;
    border-color: var(--primary-border) !important;
}

section[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] .stMarkdown span {
    color: var(--text-200) !important; font-size: 0.85rem !important;
}

.brand-box {
    padding: 1.5rem 1rem 1.2rem 1rem; text-align: center;
    border-bottom: 1px solid var(--border); margin-bottom: 1rem;
}
.brand-box .logo {
    font-size: 1.2rem; font-weight: 700; color: var(--primary);
    letter-spacing: 0.06em;
    display: flex; align-items: center; justify-content: center; gap: 0.5rem;
}
.brand-box .sub {
    font-size: 0.65rem; color: var(--text-400);
    letter-spacing: 0.12em; text-transform: uppercase; margin-top: 4px;
}

.user-pill {
    display: flex; align-items: center; gap: 0.6rem;
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius-sm); padding: 0.6rem 0.8rem; margin: 0.6rem 0;
}
.user-pill .av {
    width: 32px; height: 32px; border-radius: 50%;
    background: linear-gradient(135deg, var(--primary), var(--accent));
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.8rem; color: #fff;
}
.user-pill .nm { font-weight: 600; font-size: 0.85rem; color: var(--text-100); }
.user-pill .rl { font-size: 0.7rem; color: var(--text-400); }

.sd { border: none; border-top: 1px solid var(--border); margin: 0.9rem 0; }

/* Agent Mode Buttons */
.agent-btn {
    display: flex; align-items: center; gap: 0.6rem;
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius-sm); padding: 0.7rem 0.9rem; margin: 0.35rem 0;
    cursor: pointer; transition: all 0.2s; width: 100%;
}
.agent-btn:hover { border-color: var(--primary-border); background: var(--bg-elevated); }
.agent-btn.active { border-color: var(--primary); background: var(--primary-dim); }
.agent-btn .a-icon { font-size: 1.15rem; }
.agent-btn .a-name { font-size: 0.82rem; font-weight: 500; color: var(--text-100); }
.agent-btn .a-desc { font-size: 0.68rem; color: var(--text-400); }

/* Login */
.login-brand { text-align: center; margin-bottom: 2rem; }
.login-brand .lb-icon {
    width: 72px; height: 72px; border-radius: 22px;
    background: linear-gradient(135deg, var(--primary), var(--accent));
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 2rem; margin-bottom: 1rem; box-shadow: 0 8px 32px rgba(124,58,237,0.35);
}
.login-brand h1 { font-size: 1.7rem; font-weight: 700; color: var(--text-100); margin: 0; }
.login-brand h1 span { color: var(--accent); }
.login-brand .lb-tag {
    font-size: 0.7rem; color: var(--text-400); letter-spacing: 0.15em;
    text-transform: uppercase; margin-top: 0.3rem;
}
.login-brand .lb-desc {
    font-size: 0.85rem; color: var(--text-300); margin-top: 1rem;
    line-height: 1.6; max-width: 400px; margin-left: auto; margin-right: auto;
}
.login-features {
    display: flex; justify-content: center; gap: 0.8rem; margin-top: 1.2rem; flex-wrap: wrap;
}
.login-feat {
    display: flex; align-items: center; gap: 0.3rem;
    font-size: 0.72rem; color: var(--text-300);
    background: var(--bg-elevated); border: 1px solid var(--border);
    border-radius: 20px; padding: 0.35rem 0.75rem;
}
.login-feat .lf-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); }

/* Chat */
[data-testid="stChatMessage"] {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important; padding: 0.9rem 1.1rem !important;
    margin-bottom: 0.6rem !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(135deg, rgba(124,58,237,0.12), rgba(124,58,237,0.05)) !important;
    border: 1px solid var(--primary-border) !important;
}
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] td, [data-testid="stChatMessage"] span {
    color: var(--text-100) !important;
}
[data-testid="stChatMessage"] code {
    background: rgba(255,255,255,0.08) !important; color: #A78BFA !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stChatMessage"] a { color: var(--info) !important; }

/* Sentiment Badges */
.sentiment-positive { background: rgba(16,185,129,0.15); color: #10B981; border: 1px solid rgba(16,185,129,0.3); }
.sentiment-negative { background: rgba(239,68,68,0.15); color: #EF4444; border: 1px solid rgba(239,68,68,0.3); }
.sentiment-neutral { background: rgba(245,158,11,0.15); color: #F59E0B; border: 1px solid rgba(245,158,11,0.3); }
.sentiment-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.5rem 1rem; border-radius: 20px; font-weight: 600; font-size: 0.85rem;
}

/* Agent Badge */
.agent-badge {
    display: inline-flex; align-items: center; gap: 0.3rem;
    background: var(--primary-dim); border: 1px solid var(--primary-border);
    border-radius: 15px; padding: 0.25rem 0.6rem;
    font-size: 0.7rem; font-weight: 500; color: var(--primary);
}

/* Source chips */
.src-box { margin-top: 0.6rem; padding-top: 0.5rem; border-top: 1px solid var(--border); }
.src-label { font-size: 0.65rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-400); margin-bottom: 0.4rem; }
.src-chips { display: flex; flex-wrap: wrap; gap: 0.35rem; }
.src-chip {
    display: inline-flex; align-items: center; gap: 0.25rem;
    background: var(--bg-elevated); border: 1px solid var(--border);
    border-radius: 20px; padding: 0.25rem 0.65rem;
    font-size: 0.7rem; color: var(--text-300); text-decoration: none !important; transition: all 0.15s;
}
.src-chip:hover { border-color: var(--primary-border); color: var(--primary); }
.src-chip .sd2 { width: 5px; height: 5px; border-radius: 50%; background: var(--info); }

/* Welcome */
.welcome-area { text-align: center; padding: 8vh 2rem 3rem 2rem; }
.welcome-area .w-icon {
    width: 80px; height: 80px; border-radius: 24px;
    background: linear-gradient(135deg, var(--primary), var(--accent));
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 2.2rem; margin-bottom: 1.2rem; box-shadow: 0 12px 40px rgba(124,58,237,0.35);
}
.welcome-area h2 { font-size: 1.5rem; font-weight: 600; color: var(--text-100); margin-bottom: 0.5rem; }
.welcome-area p { font-size: 0.9rem; color: var(--text-400); max-width: 500px; margin: 0 auto; line-height: 1.65; }

/* Agent Cards */
.agent-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem; margin-top: 1.5rem; max-width: 850px; margin-left: auto; margin-right: auto;
}
.agent-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius-md); padding: 1.1rem; text-align: center; transition: all 0.2s;
}
.agent-card:hover { border-color: var(--primary-border); transform: translateY(-3px); }
.agent-card .ac-icon { font-size: 1.8rem; margin-bottom: 0.5rem; }
.agent-card .ac-title { font-size: 0.9rem; font-weight: 600; color: var(--text-100); margin-bottom: 0.25rem; }
.agent-card .ac-desc { font-size: 0.72rem; color: var(--text-400); line-height: 1.4; }

/* Doc item */
.doc-item {
    display: flex; align-items: center; gap: 0.4rem;
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 6px; padding: 0.4rem 0.65rem; margin: 0.25rem 0;
    font-size: 0.75rem; color: var(--text-200);
}
.doc-item .di { color: var(--primary); }

/* Inputs */
.stTextInput > div > div > input, .stTextArea > div > div > textarea {
    background: var(--bg-elevated) !important; border: 1px solid var(--border) !important;
    color: var(--text-100) !important; border-radius: var(--radius-sm) !important;
}
.stTextInput > div > div > input:focus { border-color: var(--border-focus) !important; }
.stButton > button {
    font-family: 'Inter', sans-serif !important; font-weight: 600 !important;
    border-radius: var(--radius-sm) !important; transition: all 0.2s !important;
}
form .stButton > button {
    background: linear-gradient(135deg, var(--primary), #9333EA) !important;
    color: #fff !important; border: none !important;
}
form .stButton > button:hover { box-shadow: 0 4px 20px var(--primary-glow) !important; }
section[data-testid="stSidebar"] .stButton > button {
    background: var(--bg-card) !important; color: var(--text-200) !important; border: 1px solid var(--border) !important;
}
.stChatInput > div { background: var(--bg-card) !important; border: 1px solid var(--border) !important; }
.stChatInput textarea { color: var(--text-100) !important; }
.stSpinner > div > div { border-top-color: var(--accent) !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: var(--text-300) !important; font-weight: 600 !important; }
.stTabs [aria-selected="true"] { color: var(--primary) !important; border-bottom-color: var(--primary) !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-thumb { background: var(--bg-elevated); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════

DB_PATH = "ai_agent_hub.db"

def init_database():
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY, username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL, display_name TEXT,
        role TEXT DEFAULT 'user', created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS conversations (
        conv_id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
        agent_type TEXT, title TEXT DEFAULT 'New Chat',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS messages (
        msg_id TEXT PRIMARY KEY, conv_id TEXT NOT NULL,
        role TEXT NOT NULL, content TEXT NOT NULL,
        agent_used TEXT, sources TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS documents (
        doc_id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
        filename TEXT NOT NULL, doc_type TEXT, content TEXT,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS blogs (
        blog_id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
        url TEXT NOT NULL, title TEXT, content TEXT,
        added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()

def _hash(pw): return hashlib.sha256(f"ai_agent_hub_2026_{pw}".encode()).hexdigest()

def register_user(username, password):
    uid = str(uuid.uuid4()); display = username.strip().title()
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("INSERT INTO users VALUES (?,?,?,?,?,CURRENT_TIMESTAMP)",
                     (uid, username.lower().strip(), _hash(password), display, "user"))
        conn.commit(); conn.close()
        return {"user_id": uid, "username": username.lower().strip(), "display_name": display, "role": "user"}
    except sqlite3.IntegrityError:
        conn.close(); return None

def authenticate(username, password):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT user_id,username,display_name,role FROM users WHERE username=? AND password_hash=?",
                       (username.lower().strip(), _hash(password))).fetchone()
    conn.close()
    return {"user_id": row[0], "username": row[1], "display_name": row[2], "role": row[3]} if row else None

def seed_defaults():
    conn = sqlite3.connect(DB_PATH)
    if conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
        conn.execute("INSERT OR IGNORE INTO users VALUES (?,?,?,?,?,CURRENT_TIMESTAMP)",
                     (str(uuid.uuid4()), "admin", _hash("admin123"), "Admin", "admin"))
    conn.commit(); conn.close()

def save_document(uid, filename, doc_type, content):
    did = str(uuid.uuid4()); conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO documents VALUES (?,?,?,?,?,CURRENT_TIMESTAMP)", (did, uid, filename, doc_type, content[:50000]))
    conn.commit(); conn.close(); return did

def get_user_documents(uid):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT doc_id,filename,doc_type,uploaded_at FROM documents WHERE user_id=? ORDER BY uploaded_at DESC", (uid,)).fetchall()
    conn.close()
    return [{"id": r[0], "filename": r[1], "type": r[2], "uploaded_at": r[3]} for r in rows]

def get_document_content(doc_id):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT content FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
    conn.close()
    return row[0] if row else None

def delete_document(doc_id, uid):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM documents WHERE doc_id=? AND user_id=?", (doc_id, uid))
    conn.commit(); conn.close()

def save_blog(uid, url, title, content):
    bid = str(uuid.uuid4()); conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO blogs VALUES (?,?,?,?,?,CURRENT_TIMESTAMP)", (bid, uid, url, title, content[:50000]))
    conn.commit(); conn.close(); return bid

def get_user_blogs(uid):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT blog_id,url,title,added_at FROM blogs WHERE user_id=? ORDER BY added_at DESC", (uid,)).fetchall()
    conn.close()
    return [{"id": r[0], "url": r[1], "title": r[2], "added_at": r[3]} for r in rows]

def get_blog_content(blog_id):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT content,title,url FROM blogs WHERE blog_id=?", (blog_id,)).fetchone()
    conn.close()
    return {"content": row[0], "title": row[1], "url": row[2]} if row else None

def delete_blog(blog_id, uid):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM blogs WHERE blog_id=? AND user_id=?", (blog_id, uid))
    conn.commit(); conn.close()


# ═══════════════════════════════════════════════
# FILE & CONTENT PROCESSING
# ═══════════════════════════════════════════════

def extract_pdf(f):
    try:
        f.seek(0)
        doc = pymupdf.open(stream=f.read(), filetype="pdf")
        txt = "\n".join(p.get_text() for p in doc); doc.close()
        return txt if txt.strip() else "[No extractable text in PDF]"
    except Exception as e: return f"[PDF error: {e}]"

def extract_txt(f):
    try:
        f.seek(0); d = f.read()
        try: return d.decode("utf-8")
        except: return d.decode("latin-1")
    except Exception as e: return f"[TXT error: {e}]"

def load_blog_content(url):
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Get title
        title = soup.find("title")
        title_text = title.get_text().strip() if title else "Untitled"
        
        # Clean content
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]): tag.decompose()
        
        # Try to get article content
        article = soup.find("article") or soup.find("main") or soup.find("body")
        if article:
            text = article.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)
        
        lines = [l.strip() for l in text.splitlines() if l.strip() and len(l.strip()) > 20]
        content = "\n".join(lines)
        return title_text, content[:30000] if content else "[No content extracted]"
    except Exception as e:
        return "Error", f"[Blog load error: {e}]"


# ═══════════════════════════════════════════════
# ENV FILE PARSING
# ═══════════════════════════════════════════════

def parse_env_file(content: str) -> dict:
    """Parse an .env file content and return a dictionary of key-value pairs."""
    config = {}
    for line in content.splitlines():
        line = line.strip()
        # Skip comments and empty lines
        if not line or line.startswith('#'):
            continue
        # Parse key=value pairs
        if '=' in line:
            key, _, value = line.partition('=')
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            config[key] = value
    return config

def load_env_to_session(config: dict):
    """Load parsed env config into session state with proper key mapping."""
    # Map env file keys to session state keys
    key_mapping = {
        'AZURE_ENDPOINT': 'azure_endpoint',
        'AZURE_OPENAI_API_KEY': 'azure_api_key',
        'MODEL_DEPLOYMENT': 'model_deployment',
        'CHAT_MODEL_NAME': 'model_deployment',  # Alternative key
        'API_VERSION': 'api_version',
        'api_version': 'api_version',  # Alternative format
        'TAVILY_API_KEY': 'tavily_key',
        'WEATHER_API_KEY': 'weather_key',
    }
    
    for env_key, session_key in key_mapping.items():
        if env_key in config and config[env_key]:
            st.session_state[session_key] = config[env_key]
    
    # Set default model if not specified
    if 'model_deployment' not in st.session_state or not st.session_state['model_deployment']:
        st.session_state['model_deployment'] = 'gpt-4o'
    
    # Set default API version if not specified
    if 'api_version' not in st.session_state or not st.session_state['api_version']:
        st.session_state['api_version'] = '2024-12-01-preview'

def try_load_default_env():
    """Try to load DENTSU_AZURE.env from the current directory on startup."""
    if st.session_state.get("env_loaded"):
        return
    
    env_path = "DENTSU_AZURE.env"
    if os.path.exists(env_path):
        try:
            with open(env_path, 'r') as f:
                content = f.read()
            config = parse_env_file(content)
            load_env_to_session(config)
            st.session_state["env_loaded"] = True
            st.session_state["env_file_name"] = env_path
        except Exception:
            pass  # Silently fail if file can't be read


# ═══════════════════════════════════════════════
# LLM HELPER
# ═══════════════════════════════════════════════

def get_llm(max_tokens=2000, temperature=0.7):
    return AzureChatOpenAI(
        azure_deployment=st.session_state.get("model_deployment", "gpt-4o"),
        api_version=st.session_state.get("api_version", "2024-12-01-preview"),
        azure_endpoint=st.session_state.get("azure_endpoint", ""),
        api_key=st.session_state.get("azure_api_key", ""),
        temperature=temperature, max_tokens=max_tokens
    )


# ═══════════════════════════════════════════════
# PAGE & AGENT DEFINITIONS
# ═══════════════════════════════════════════════

# Pages for navigation
PAGES = {
    "home": {
        "name": "🏠 Multi-Agent Hub",
        "short": "Home",
        "icon": "🏠",
        "desc": "AI auto-routes to the best agent",
        "agent": "orchestrator"
    },
    "pdf": {
        "name": "📄 PDF Chat",
        "short": "PDFs",
        "icon": "📄",
        "desc": "Dedicated PDF document Q&A",
        "agent": "pdf_qa"
    },
    "web": {
        "name": "🔍 Web Search",
        "short": "Web",
        "icon": "🔍",
        "desc": "Search the internet",
        "agent": "web_search"
    },
    "blog": {
        "name": "📝 Blog Chat",
        "short": "Blogs",
        "icon": "📝",
        "desc": "Analyze blog articles",
        "agent": "blog_qa"
    },
    "sentiment": {
        "name": "💬 Sentiment",
        "short": "Sentiment",
        "icon": "💬",
        "desc": "Analyze text emotions",
        "agent": "sentiment"
    },
    "weather": {
        "name": "🌤️ Weather",
        "short": "Weather",
        "icon": "🌤️",
        "desc": "Current weather & forecasts",
        "agent": "weather"
    },
    "image": {
        "name": "🖼️ Image Chat",
        "short": "Image",
        "icon": "🖼️",
        "desc": "Analyze & ask about images",
        "agent": "image_qa"
    },
    "data": {
        "name": "📊 Data Analysis",
        "short": "Data",
        "icon": "📊",
        "desc": "Analyze CSV data & create charts",
        "agent": "data_analysis"
    }
}

# Agent configs (used internally)
AGENTS = {
    "orchestrator": {"name": "Multi-Agent Hub", "icon": "🤖", "color": "#7C3AED"},
    "pdf_qa": {"name": "PDF Retriever", "icon": "📄", "color": "#3B82F6"},
    "web_search": {"name": "Web Search", "icon": "🔍", "color": "#10B981"},
    "blog_qa": {"name": "Blog Analyzer", "icon": "📝", "color": "#F59E0B"},
    "sentiment": {"name": "Sentiment Analyzer", "icon": "💬", "color": "#EC4899"},
    "weather": {"name": "Weather Agent", "icon": "🌤️", "color": "#06B6D4"},
    "image_qa": {"name": "Image Analyzer", "icon": "🖼️", "color": "#8B5CF6"},
    "data_analysis": {"name": "Data Analyst", "icon": "📊", "color": "#F97316"}
}


# ═══════════════════════════════════════════════
# LANGCHAIN TOOLS
# ═══════════════════════════════════════════════

from langchain_core.tools import tool

@tool
def search_pdf_documents(query: str) -> str:
    """Search and retrieve information from uploaded PDF documents. Use this when user asks about their uploaded documents."""
    user = st.session_state.get("user", {})
    docs = get_user_documents(user.get("user_id", ""))
    if not docs:
        return "NO_DOCUMENTS: No PDF documents uploaded yet."
    
    doc_contexts = []
    for doc in docs[:5]:
        content = get_document_content(doc["id"])
        if content:
            doc_contexts.append(f"[Document: {doc['filename']}]\n{content[:8000]}")
    
    return "\n\n---\n\n".join(doc_contexts) if doc_contexts else "NO_CONTENT: Could not extract content from documents."


@tool
def search_web(query: str) -> str:
    """Search the internet for current information, news, facts, or real-time data. Use this for questions requiring up-to-date information."""
    tavily_key = st.session_state.get("tavily_key", "")
    if not tavily_key:
        return "NO_API_KEY: Tavily API key not configured."
    
    try:
        os.environ["TAVILY_API_KEY"] = tavily_key
        tavily = TavilySearchResults(max_results=5, search_depth="advanced")
        results = tavily.invoke(query)
        
        if not results:
            return "NO_RESULTS: No search results found."
        
        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get('title', 'No Title')
            content = r.get('content', '')[:400]
            url = r.get('url', '')
            formatted.append(f"[{i}] {title}\n{content}\nSource: {url}")
        
        return "\n\n".join(formatted)
    except Exception as e:
        return f"SEARCH_ERROR: {str(e)}"


@tool
def search_blog_articles(query: str) -> str:
    """Search and analyze added blog articles. Use this when user asks about blog content they've added."""
    user = st.session_state.get("user", {})
    blogs = get_user_blogs(user.get("user_id", ""))
    if not blogs:
        return "NO_BLOGS: No blog articles added yet."
    
    blog_contexts = []
    for blog in blogs[:5]:
        data = get_blog_content(blog["id"])
        if data:
            blog_contexts.append(f"[Blog: {data['title']}]\nURL: {data['url']}\n\n{data['content'][:6000]}")
    
    return "\n\n---\n\n".join(blog_contexts) if blog_contexts else "NO_CONTENT: Could not extract blog content."


@tool
def analyze_sentiment(text: str) -> str:
    """Analyze the sentiment and emotions in text. Use this for reviews, feedback, or any text requiring emotional analysis."""
    return f"ANALYZE_SENTIMENT: {text}"


@tool
def get_weather(location: str) -> str:
    """Get current weather and forecast for a location. Use this when user asks about weather, temperature, forecast, or climate conditions."""
    weather_key = st.session_state.get("weather_key", "")
    if not weather_key:
        return "NO_WEATHER_KEY: Weather API key not configured."
    
    try:
        # WeatherAPI.com endpoint
        url = f"http://api.weatherapi.com/v1/forecast.json?key={weather_key}&q={location}&days=3&aqi=yes"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        current = data.get("current", {})
        location_info = data.get("location", {})
        forecast = data.get("forecast", {}).get("forecastday", [])
        
        result = f"""📍 **Location**: {location_info.get('name', location)}, {location_info.get('country', '')}
🕐 **Local Time**: {location_info.get('localtime', 'N/A')}

## Current Weather
- 🌡️ **Temperature**: {current.get('temp_c', 'N/A')}°C ({current.get('temp_f', 'N/A')}°F)
- 🤔 **Feels Like**: {current.get('feelslike_c', 'N/A')}°C
- ☁️ **Condition**: {current.get('condition', {}).get('text', 'N/A')}
- 💧 **Humidity**: {current.get('humidity', 'N/A')}%
- 💨 **Wind**: {current.get('wind_kph', 'N/A')} km/h {current.get('wind_dir', '')}
- 👁️ **Visibility**: {current.get('vis_km', 'N/A')} km
- 🌡️ **UV Index**: {current.get('uv', 'N/A')}
"""
        
        if forecast:
            result += "\n## 3-Day Forecast\n"
            for day in forecast:
                date = day.get("date", "")
                day_data = day.get("day", {})
                result += f"\n**{date}**\n"
                result += f"- High: {day_data.get('maxtemp_c', 'N/A')}°C | Low: {day_data.get('mintemp_c', 'N/A')}°C\n"
                result += f"- {day_data.get('condition', {}).get('text', 'N/A')}\n"
                result += f"- Rain chance: {day_data.get('daily_chance_of_rain', 0)}%\n"
        
        return result
    except requests.exceptions.HTTPError as e:
        return f"WEATHER_ERROR: Could not find weather for '{location}'. Please check the city name."
    except Exception as e:
        return f"WEATHER_ERROR: {str(e)}"


# ═══════════════════════════════════════════════
# GUARDRAILS
# ═══════════════════════════════════════════════

def check_scope(query: str, context: str, agent_type: str) -> dict:
    """Check if the query is within scope of the provided context."""
    guardrail_prompt = f"""You are a scope detection system. Analyze if the user's question can be answered using the provided context.

CONTEXT TYPE: {agent_type}
CONTEXT PREVIEW: {context[:2000]}...

USER QUESTION: {query}

Respond with ONLY a JSON object (no markdown):
{{"in_scope": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}}

Rules:
- in_scope=true if the context contains relevant information to answer the question
- in_scope=false if the question is about something not covered in the context
- Be strict: if there's no clear match, return false"""

    try:
        llm = get_llm(max_tokens=200, temperature=0)
        response = llm.invoke([SystemMessage(content="You are a JSON-only scope detector."), HumanMessage(content=guardrail_prompt)])
        
        # Clean response
        text = response.content.strip()
        if text.startswith("```"): text = text.split("```")[1].replace("json", "").strip()
        
        return json.loads(text)
    except:
        return {"in_scope": True, "confidence": 0.5, "reason": "Could not determine scope"}


def run_with_guardrails(prompt: str, context: str, agent_type: str, system_prompt: str) -> tuple:
    """Run agent with guardrails - detect out-of-scope and offer web search."""
    # Check scope
    scope_check = check_scope(prompt, context, agent_type)
    
    if not scope_check.get("in_scope", True) and scope_check.get("confidence", 0) > 0.7:
        return {
            "response": f"⚠️ **Question appears to be outside the scope of your {agent_type}.**\n\n" +
                        f"*Reason: {scope_check.get('reason', 'No relevant information found')}*\n\n" +
                        f"Your question: \"{prompt}\"\n\n" +
                        "The uploaded content doesn't seem to contain information about this topic.",
            "out_of_scope": True,
            "reason": scope_check.get("reason", "")
        }
    
    # In scope - proceed with answer
    try:
        llm = get_llm(max_tokens=2000, temperature=0.3)
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Context:\n{context}\n\n---\n\nQuestion: {prompt}")
        ])
        return {"response": response.content, "out_of_scope": False}
    except Exception as e:
        return {"response": f"Error: {str(e)}", "out_of_scope": False}


# ═══════════════════════════════════════════════
# AGENT IMPLEMENTATIONS (with LangChain)
# ═══════════════════════════════════════════════

def run_orchestrator(prompt: str, user: dict, chat_history: list = None) -> tuple:
    """Smart multi-agent orchestrator that autonomously decides which agent to use."""
    today_str = datetime.now().strftime("%B %d, %Y")
    
    # Check available resources
    has_pdfs = bool(get_user_documents(user["user_id"]))
    has_blogs = bool(get_user_blogs(user["user_id"]))
    has_tavily = bool(st.session_state.get("tavily_key"))
    has_weather = bool(st.session_state.get("weather_key"))
    has_image = bool(st.session_state.get("uploaded_image_data"))
    has_csv = bool(st.session_state.get("uploaded_csv_data"))
    
    # Build resource summary for LLM
    resources = []
    if has_pdfs:
        docs = get_user_documents(user["user_id"])
        doc_names = ", ".join([d["filename"][:20] for d in docs[:3]])
        resources.append(f"PDF_DOCUMENTS: User has uploaded PDFs: {doc_names}")
    if has_blogs:
        resources.append("BLOG_ARTICLES: User has added blog articles")
    if has_image:
        img_name = st.session_state.get("uploaded_image_name", "image")
        resources.append(f"IMAGE: User has uploaded an image: {img_name}")
    if has_csv:
        csv_name = st.session_state.get("uploaded_csv_name", "data.csv")
        resources.append(f"CSV_DATA: User has uploaded data file: {csv_name}")
    
    resources_str = "\n".join(resources) if resources else "No user resources uploaded."
    
    # Build routing prompt
    routing_prompt = f"""You are a routing system. Analyze the user's question and decide which agent to use.

TODAY: {today_str}

USER'S RESOURCES:
{resources_str}

AVAILABLE AGENTS:
- PDF_AGENT: For questions about uploaded PDF documents
- WEB_AGENT: For web search, current events, news, facts, real-time info
- BLOG_AGENT: For questions about added blog articles  
- SENTIMENT_AGENT: For sentiment/emotion analysis of text
- WEATHER_AGENT: For weather information (requires location)
- IMAGE_AGENT: For analyzing uploaded images
- DATA_AGENT: For analyzing CSV data, creating charts
- DIRECT: For simple greetings or questions you can answer directly

ROUTING RULES:
1. If user mentions "document", "PDF", "uploaded", "file", "report" AND has PDFs → PDF_AGENT
2. If user mentions "blog", "article I added" AND has blogs → BLOG_AGENT
3. If user asks about "weather", "temperature", "forecast" → WEATHER_AGENT
4. If user asks to "analyze sentiment", "emotions", "tone" of text → SENTIMENT_AGENT
5. If user mentions "image", "picture", "photo" AND has image → IMAGE_AGENT
6. If user mentions "data", "chart", "CSV", "visualization" AND has CSV → DATA_AGENT
7. For general questions, facts, news, "what is", "who is" → WEB_AGENT
8. For simple greetings ("hi", "hello") → DIRECT

USER QUESTION: {prompt}

Respond with ONLY ONE of: PDF_AGENT, WEB_AGENT, BLOG_AGENT, SENTIMENT_AGENT, WEATHER_AGENT, IMAGE_AGENT, DATA_AGENT, or DIRECT
Your answer:"""

    try:
        # Step 1: Determine which agent to route to
        llm = get_llm(max_tokens=50, temperature=0)
        route_response = llm.invoke([HumanMessage(content=routing_prompt)])
        route = route_response.content.strip().upper()
        
        # Step 2: Execute the appropriate agent function directly (with user context)
        if "PDF" in route and has_pdfs:
            response, agent_used, sources, out_of_scope = run_pdf_retriever(prompt, user, chat_history)
            return response, agent_used, sources
            
        elif "BLOG" in route and has_blogs:
            response, agent_used, sources, out_of_scope = run_blog_analyzer(prompt, user, chat_history)
            return response, agent_used, sources
            
        elif "WEATHER" in route and has_weather:
            response, agent_used, sources, out_of_scope = run_weather_agent(prompt, user, chat_history)
            return response, agent_used, sources
            
        elif "SENTIMENT" in route:
            response, agent_used, sources, out_of_scope = run_sentiment_analyzer(prompt, user, chat_history)
            return response, agent_used, sources
            
        elif "IMAGE" in route and has_image:
            response, agent_used, sources, out_of_scope = run_image_analyzer(prompt, user, chat_history)
            return response, agent_used, sources
            
        elif "DATA" in route and has_csv:
            response, agent_used, sources, out_of_scope = run_data_analyzer(prompt, user, chat_history)
            return response, agent_used, sources
            
        elif "WEB" in route and has_tavily:
            response, agent_used, sources, out_of_scope = run_web_search(prompt, user, chat_history)
            return response, agent_used, sources
            
        elif "DIRECT" in route:
            # Handle simple responses directly
            llm = get_llm(max_tokens=500, temperature=0.7)
            response = llm.invoke([
                SystemMessage(content=f"You are the AI Agent Hub assistant. Today is {today_str}. Respond helpfully and concisely."),
                HumanMessage(content=prompt)
            ])
            return response.content, "orchestrator", []
            
        else:
            # Default: try web search if available, otherwise direct response
            if has_tavily:
                response, agent_used, sources, out_of_scope = run_web_search(prompt, user, chat_history)
                return response, agent_used, sources
            else:
                llm = get_llm(max_tokens=1500, temperature=0.5)
                response = llm.invoke([
                    SystemMessage(content=f"You are the AI Agent Hub. Today is {today_str}. Answer the user's question to the best of your knowledge."),
                    HumanMessage(content=prompt)
                ])
                return response.content, "orchestrator", []
                
    except Exception as e:
        return f"Error: {str(e)}", "orchestrator", []


def detect_agent_used(result: dict) -> str:
    """Detect which agent/tool was used from the agent result."""
    try:
        # Check intermediate steps for tool usage
        steps = result.get("intermediate_steps", [])
        if steps:
            for step in steps:
                if hasattr(step[0], 'tool'):
                    tool_name = step[0].tool
                    if "pdf" in tool_name.lower():
                        return "pdf_qa"
                    elif "web" in tool_name.lower() or "search" in tool_name.lower():
                        return "web_search"
                    elif "blog" in tool_name.lower():
                        return "blog_qa"
                    elif "sentiment" in tool_name.lower():
                        return "sentiment"
    except:
        pass
    return "orchestrator"


def run_pdf_retriever(prompt: str, user: dict, chat_history: list = None) -> tuple:
    """Answer questions from uploaded PDF documents with guardrails and memory."""
    # Get all user documents
    docs = get_user_documents(user["user_id"])
    if not docs:
        return "📄 **No PDFs uploaded yet.**\n\nPlease upload PDF documents in the sidebar to use the PDF Retriever agent.", "pdf_qa", [], False
    
    # Collect document contents
    doc_contexts = []
    for doc in docs[:5]:
        content = get_document_content(doc["id"])
        if content:
            doc_contexts.append(f"[Document: {doc['filename']}]\n{content[:8000]}")
    
    full_context = "\n\n---\n\n".join(doc_contexts)
    
    # Build conversation context for follow-ups
    conv_context = ""
    if chat_history:
        recent = chat_history[-4:]  # Last 4 messages
        conv_parts = []
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            conv_parts.append(f"{role}: {msg['content'][:300]}")
        conv_context = f"\n\nPrevious conversation:\n" + "\n".join(conv_parts)
    
    system_prompt = f"""You are the PDF Retriever Agent. Your role is to answer questions based ONLY on the provided document context.

Rules:
1. Answer using information from the documents
2. Quote relevant passages when helpful
3. If the answer isn't in the documents, say so clearly
4. Cite which document the information comes from
5. Use markdown formatting for clarity
6. You have conversation history - handle follow-up questions naturally{conv_context}"""

    # Run with guardrails
    result = run_with_guardrails(prompt, full_context, "PDF Documents", system_prompt)
    sources = [{"name": d["filename"], "type": "PDF"} for d in docs[:5]]
    
    return result["response"], "pdf_qa", sources, result.get("out_of_scope", False)


def run_web_search(prompt: str, user: dict, chat_history: list = None) -> tuple:
    """Search the web for current information with conversation memory."""
    tavily_key = st.session_state.get("tavily_key", "")
    if not tavily_key:
        return "🔍 **Tavily API Key not configured.**\n\nPlease add your Tavily API key in the sidebar under API Configuration.", "web_search", [], False
    
    # Build search query with context from history if it's a follow-up
    search_query = prompt
    if chat_history and len(prompt.split()) < 5:  # Short query likely a follow-up
        # Get context from last exchange
        for msg in reversed(chat_history[-4:]):
            if msg["role"] == "user" and msg["content"] != prompt:
                search_query = f"{msg['content']} {prompt}"
                break
    
    try:
        # Use LangChain Tavily tool
        search_results = search_web.invoke(search_query)
        
        if search_results.startswith("NO_") or search_results.startswith("SEARCH_ERROR"):
            return f"🔍 {search_results}", "web_search", [], False
        
        # Extract sources from results
        sources = []
        urls = re.findall(r'Source: (https?://[^\s]+)', search_results)
        for url in urls[:5]:
            domain = re.sub(r'^https?://(www\.)?', '', url).split('/')[0]
            sources.append({"url": url, "title": domain})
        
        # Generate synthesized answer with conversation context
        llm = get_llm(max_tokens=1500, temperature=0.5)
        today_str = datetime.now().strftime("%B %d, %Y")
        
        # Build conversation context
        conv_context = ""
        if chat_history:
            recent = chat_history[-4:]
            conv_parts = [f"{'User' if m['role']=='user' else 'Assistant'}: {m['content'][:200]}" for m in recent]
            conv_context = f"\n\nPrevious conversation:\n" + "\n".join(conv_parts)
        
        response = llm.invoke([
            SystemMessage(content=f"You are the Web Search Agent. Today is {today_str}. Synthesize the search results to answer the user's question. Cite sources with URLs. Use markdown formatting. Handle follow-up questions naturally.{conv_context}"),
            HumanMessage(content=f"Search Results:\n{search_results}\n\n---\n\nQuestion: {prompt}")
        ])
        
        return response.content, "web_search", sources, False
    except Exception as e:
        return f"Search error: {str(e)}", "web_search", [], False


def run_blog_analyzer(prompt: str, user: dict, chat_history: list = None) -> tuple:
    """Answer questions about added blog articles with guardrails and memory."""
    
    blogs = get_user_blogs(user["user_id"])
    if not blogs:
        return "📝 **No blogs added yet.**\n\nPaste a blog URL in the sidebar to add articles for analysis.", "blog_qa", [], False
    
    # Collect blog contents
    blog_contexts = []
    sources = []
    for blog in blogs[:5]:
        data = get_blog_content(blog["id"])
        if data:
            blog_contexts.append(f"[Blog: {data['title']}]\nURL: {data['url']}\n\n{data['content'][:6000]}")
            sources.append({"title": data['title'], "url": data['url']})
    
    full_context = "\n\n---\n\n".join(blog_contexts)
    
    # Build conversation context
    conv_context = ""
    if chat_history:
        recent = chat_history[-4:]
        conv_parts = [f"{'User' if m['role']=='user' else 'Assistant'}: {m['content'][:300]}" for m in recent]
        conv_context = f"\n\nPrevious conversation:\n" + "\n".join(conv_parts)
    
    system_prompt = f"""You are the Blog Analyzer Agent. Your role is to analyze and answer questions about the provided blog articles.

Capabilities:
1. Summarize blog content
2. Answer specific questions about the articles
3. Compare information across multiple blogs
4. Extract key points and insights
5. Identify the author's tone and perspective

Always cite which blog article the information comes from. Use markdown formatting.
Handle follow-up questions naturally using conversation history.{conv_context}"""

    # Run with guardrails
    result = run_with_guardrails(prompt, full_context, "Blog Articles", system_prompt)
    
    return result["response"], "blog_qa", sources, result.get("out_of_scope", False)


def run_sentiment_analyzer(prompt: str, user: dict, chat_history: list = None) -> tuple:
    """Analyze sentiment of reviews and text with conversation memory."""
    # Guardrail: Check if text is too short (unless it's a follow-up)
    is_followup = chat_history and len(chat_history) > 0
    if len(prompt.strip()) < 10 and not is_followup:
        return "⚠️ **Text too short for sentiment analysis.**\n\nPlease provide a longer text (review, feedback, comment) to analyze.", "sentiment", [], False
    
    # Build conversation context
    conv_context = ""
    if chat_history:
        recent = chat_history[-4:]
        conv_parts = [f"{'User' if m['role']=='user' else 'Assistant'}: {m['content'][:300]}" for m in recent]
        conv_context = f"\n\nPrevious conversation:\n" + "\n".join(conv_parts)
    
    system_prompt = f"""You are the Sentiment Analyzer Agent. Analyze the sentiment of the provided text.

For each analysis, provide:
1. **Overall Sentiment**: POSITIVE, NEGATIVE, or NEUTRAL (with confidence percentage)
2. **Sentiment Score**: -1.0 (most negative) to +1.0 (most positive)
3. **Key Emotions**: List detected emotions (joy, anger, sadness, fear, surprise, etc.)
4. **Tone Analysis**: Formal/informal, sincere/sarcastic, etc.
5. **Key Phrases**: Important positive and negative phrases
6. **Summary**: Brief explanation of the sentiment

Format your response clearly with markdown. Be objective and thorough.
Handle follow-up questions naturally using conversation history.{conv_context}"""

    try:
        llm = get_llm(max_tokens=1500, temperature=0.3)
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Analyze the sentiment of this text:\n\n{prompt}")
        ])
        
        # Determine sentiment for badge
        content_lower = response.content.lower()
        if "positive" in content_lower[:200]:
            sentiment = "positive"
        elif "negative" in content_lower[:200]:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return response.content, "sentiment", [{"sentiment": sentiment}], False
    except Exception as e:
        return f"Error: {str(e)}", "sentiment", [], False


def run_weather_agent(prompt: str, user: dict, chat_history: list = None) -> tuple:
    """Get weather information for a location with conversation memory."""
    weather_key = st.session_state.get("weather_key", "")
    if not weather_key:
        return "🌤️ **Weather API Key not configured.**\n\nPlease add your Weather API key in the sidebar under API Configuration.", "weather", [], False
    
    # Build conversation context for follow-ups
    conv_context = ""
    if chat_history:
        recent = chat_history[-4:]
        conv_parts = [f"{'User' if m['role']=='user' else 'Assistant'}: {m['content'][:200]}" for m in recent]
        conv_context = f"\n\nPrevious conversation:\n" + "\n".join(conv_parts)
    
    # Extract location from prompt
    try:
        llm = get_llm(max_tokens=100, temperature=0)
        extract_prompt = f"""Extract the city/location name from this weather question. Return ONLY the city name, nothing else.
If user says "there" or similar, use conversation history to find the last mentioned location.{conv_context}

Question: {prompt}
City:"""
        location_response = llm.invoke([HumanMessage(content=extract_prompt)])
        location = location_response.content.strip().strip('"').strip("'")
        
        if not location or len(location) < 2:
            return "🌤️ **Please specify a location.**\n\nExample: 'What's the weather in Tokyo?' or 'Weather forecast for New York'", "weather", [], False
        
        # Get weather data using the tool
        weather_data = get_weather.invoke(location)
        
        if weather_data.startswith("WEATHER_ERROR") or weather_data.startswith("NO_"):
            return f"⚠️ **Could not get weather for '{location}'**\n\nPlease check the city name and try again. Examples:\n- 'Weather in London'\n- 'Tokyo forecast'\n- 'Temperature in Paris'", "weather", [], False
        
        # Enhance with LLM response
        response_llm = get_llm(max_tokens=800, temperature=0.5)
        enhanced = response_llm.invoke([
            SystemMessage(content=f"You are the Weather Agent. Present this weather data in a friendly, informative way. Add relevant advice (bring umbrella, wear sunscreen, etc.). Use emojis appropriately. Handle follow-up questions naturally.{conv_context}"),
            HumanMessage(content=f"Weather data:\n{weather_data}\n\nUser question: {prompt}")
        ])
        
        return enhanced.content, "weather", [{"location": location, "type": "weather"}], False
    except Exception as e:
        return f"Error getting weather: {str(e)}", "weather", [], False


def run_image_analyzer(prompt: str, user: dict, chat_history: list = None) -> tuple:
    """Analyze images and answer questions about them using GPT-4o vision."""
    # Check if image is uploaded
    image_data = st.session_state.get("uploaded_image_data")
    image_name = st.session_state.get("uploaded_image_name", "image")
    
    if not image_data:
        return "🖼️ **No image uploaded yet.**\n\nPlease upload an image in the sidebar to use the Image Analyzer.", "image_qa", [], False
    
    # Build conversation context for follow-ups
    conv_context = ""
    if chat_history:
        recent = chat_history[-4:]
        conv_parts = [f"{'User' if m['role']=='user' else 'Assistant'}: {m['content'][:200]}" for m in recent]
        conv_context = f"\n\nPrevious conversation about this image:\n" + "\n".join(conv_parts)
    
    system_prompt = f"""You are the Image Analyzer Agent. You have excellent vision capabilities and can analyze images in detail.

Your capabilities:
1. Describe what you see in the image
2. Identify objects, people, text, colors, and patterns
3. Answer specific questions about the image content
4. Read and extract text from images (OCR)
5. Analyze charts, diagrams, and visual data
6. Identify emotions, settings, and contexts
7. Compare elements within the image

Guidelines:
- Be detailed and accurate in your observations
- If you're uncertain about something, say so
- Use markdown formatting for clarity
- Handle follow-up questions naturally using conversation history{conv_context}"""

    try:
        # Create vision-capable LLM
        llm = AzureChatOpenAI(
            azure_deployment=st.session_state.get("model_deployment", "gpt-4o"),
            api_version=st.session_state.get("api_version", "2024-12-01-preview"),
            azure_endpoint=st.session_state.get("azure_endpoint", ""),
            api_key=st.session_state.get("azure_api_key", ""),
            temperature=0.5,
            max_tokens=2000
        )
        
        # Create message with image (base64)
        message_content = [
            {"type": "text", "text": f"{system_prompt}\n\nUser question: {prompt}"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_data}",
                    "detail": "high"
                }
            }
        ]
        
        response = llm.invoke([HumanMessage(content=message_content)])
        
        return response.content, "image_qa", [{"name": image_name, "type": "image"}], False
    except Exception as e:
        return f"Error analyzing image: {str(e)}", "image_qa", [], False


def run_data_analyzer(prompt: str, user: dict, chat_history: list = None) -> tuple:
    """Analyze CSV data, answer questions, and create visualizations using pandas."""
    # Check if data is uploaded
    csv_data = st.session_state.get("uploaded_csv_data")
    csv_name = st.session_state.get("uploaded_csv_name", "data.csv")
    
    if csv_data is None:
        return "📊 **No CSV file uploaded yet.**\n\nPlease upload a CSV file in the sidebar to use the Data Analyst.", "data_analysis", [], False
    
    # Load the dataframe
    try:
        df = pd.read_csv(io.StringIO(csv_data))
    except Exception as e:
        return f"Error reading CSV: {str(e)}", "data_analysis", [], False
    
    # Build data context
    data_info = f"""**Dataset: {csv_name}**
- Rows: {len(df):,}
- Columns: {len(df.columns)}
- Column names: {', '.join(df.columns.tolist())}

**Column Types:**
{df.dtypes.to_string()}

**Sample Data (first 5 rows):**
{df.head().to_markdown()}

**Basic Statistics:**
{df.describe().to_markdown()}"""

    # Build conversation context
    conv_context = ""
    if chat_history:
        recent = chat_history[-4:]
        conv_parts = [f"{'User' if m['role']=='user' else 'Assistant'}: {m['content'][:300]}" for m in recent]
        conv_context = f"\n\nPrevious conversation:\n" + "\n".join(conv_parts)
    
    # Check if user wants a visualization
    viz_keywords = ["chart", "graph", "plot", "visualize", "visualization", "histogram", "bar", "pie", "scatter", "line", "show me", "draw", "create a"]
    wants_viz = any(kw in prompt.lower() for kw in viz_keywords)
    
    if wants_viz:
        # Generate visualization code
        viz_prompt = f"""You are a Python data visualization expert. Given this dataset info and user request, generate ONLY executable Python code to create the visualization.

{data_info}

User request: {prompt}

Requirements:
1. Use matplotlib.pyplot as plt (already imported)
2. The dataframe is available as 'df'
3. Use a clean, modern style with plt.style.use('seaborn-v0_8-whitegrid') or similar
4. Set figure size appropriately: plt.figure(figsize=(10, 6))
5. Add title, labels, and legend as needed
6. Use colors that look good: consider using a color palette
7. For categorical data with many values, limit to top 10-15
8. End with plt.tight_layout()
9. DO NOT call plt.show() or plt.savefig()
10. Return ONLY the Python code, no explanations, no markdown code blocks

Generate the code:"""

        try:
            llm = get_llm(max_tokens=1000, temperature=0.2)
            code_response = llm.invoke([
                SystemMessage(content="You are a Python code generator. Output ONLY executable Python code, nothing else."),
                HumanMessage(content=viz_prompt)
            ])
            
            code = code_response.content.strip()
            # Clean up code if wrapped in markdown
            if code.startswith("```"):
                code = code.split("```")[1]
                if code.startswith("python"):
                    code = code[6:]
                code = code.strip()
            
            # Execute the visualization code with safe style
            try:
                plt.style.use('seaborn-v0_8-whitegrid')
            except:
                try:
                    plt.style.use('ggplot')
                except:
                    pass  # Use default style
            
            plt.figure(figsize=(10, 6))
            
            # Create a safe execution environment
            exec_globals = {"df": df, "plt": plt, "pd": pd}
            exec(code, exec_globals)
            
            plt.tight_layout()
            
            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            buf.seek(0)
            plt.close('all')
            
            # Store chart in session for display
            chart_b64 = base64.b64encode(buf.read()).decode('utf-8')
            st.session_state["last_chart"] = chart_b64
            
            # Generate explanation
            explain_llm = get_llm(max_tokens=500, temperature=0.5)
            explanation = explain_llm.invoke([
                SystemMessage(content="You are a data analyst. Briefly explain what the chart shows based on the data."),
                HumanMessage(content=f"Dataset info:\n{data_info}\n\nUser asked for: {prompt}\n\nProvide a brief, insightful explanation of what the visualization reveals.")
            ])
            
            response = f"📊 **Chart Generated!**\n\n{explanation.content}\n\n*(Chart displayed below)*"
            return response, "data_analysis", [{"name": csv_name, "type": "chart", "chart": chart_b64}], False
            
        except Exception as e:
            # Fallback to text analysis if viz fails
            plt.close('all')
            error_msg = str(e)
            # Continue to text analysis with note about viz failure
    
    # Text-based analysis
    system_prompt = f"""You are the Data Analyst Agent. You analyze CSV data and provide insights.

{data_info}

Your capabilities:
1. Answer questions about the data
2. Calculate statistics (mean, median, mode, std, etc.)
3. Identify trends and patterns
4. Find correlations between columns
5. Summarize key insights
6. Suggest further analyses

Guidelines:
- Be precise with numbers
- Use markdown tables for structured data
- Highlight key findings
- If a question can't be answered with the data, say so
- Handle follow-up questions using conversation history{conv_context}"""

    try:
        llm = get_llm(max_tokens=2000, temperature=0.3)
        
        # For specific calculations, try to compute them
        analysis_prompt = f"""Based on the dataset information provided, answer this question:

{prompt}

If the question requires specific calculations, compute them from the data summary provided.
If it requires data not available in the summary, explain what additional data would be needed."""

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=analysis_prompt)
        ])
        
        return response.content, "data_analysis", [{"name": csv_name, "type": "data"}], False
    except Exception as e:
        return f"Error analyzing data: {str(e)}", "data_analysis", [], False


# ═══════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════

def extract_sources(text):
    urls = re.findall(r'https?://[^\s\)\]>"\'`,]+', text)
    seen = set(); unique = []
    for u in urls:
        u = u.rstrip('.,;:!?)')
        domain = re.sub(r'^https?://(www\.)?', '', u).split('/')[0]
        if domain not in seen and len(domain) > 2:
            seen.add(domain); unique.append({"url": u, "domain": domain})
    return unique[:6]

def render_sources(sources, agent_type):
    if not sources:
        return
    
    if agent_type == "sentiment":
        # Render sentiment badge
        sentiment = sources[0].get("sentiment", "neutral")
        emoji = "😊" if sentiment == "positive" else "😞" if sentiment == "negative" else "😐"
        st.markdown(f'<div class="sentiment-badge sentiment-{sentiment}">{emoji} {sentiment.upper()}</div>', unsafe_allow_html=True)
    elif agent_type == "web_search":
        chips = ""
        for s in sources:
            url = html_lib.escape(s.get("url", "#"))
            title = s.get("title", "Source")[:30]
            chips += f'<a href="{url}" target="_blank" class="src-chip"><span class="sd2"></span>{title}</a>'
        st.markdown(f'<div class="src-box"><div class="src-label">🔗 Sources</div><div class="src-chips">{chips}</div></div>', unsafe_allow_html=True)
    elif agent_type in ["pdf_qa", "blog_qa"]:
        names = ", ".join([s.get("name", s.get("title", "Document"))[:20] for s in sources])
        st.markdown(f'<div class="agent-badge">📎 {names}</div>', unsafe_allow_html=True)
    elif agent_type == "weather":
        location = sources[0].get("location", "Location") if sources else "Weather"
        st.markdown(f'<div class="agent-badge">🌍 {location}</div>', unsafe_allow_html=True)
    elif agent_type == "image_qa":
        img_name = sources[0].get("name", "Image")[:25] if sources else "Image"
        st.markdown(f'<div class="agent-badge">🖼️ {img_name}</div>', unsafe_allow_html=True)
    elif agent_type == "data_analysis":
        csv_name = sources[0].get("name", "data.csv")[:25] if sources else "Data"
        data_type = sources[0].get("type", "data") if sources else "data"
        icon = "📊" if data_type == "chart" else "📈"
        st.markdown(f'<div class="agent-badge">{icon} {csv_name}</div>', unsafe_allow_html=True)

def generate_title(content):
    words = content.split()[:6]
    return " ".join(words) + ("..." if len(content.split()) > 6 else "")


# ═══════════════════════════════════════════════
# LOGIN
# ═══════════════════════════════════════════════

def render_login():
    st.markdown("<style>section[data-testid='stSidebar']{display:none;}</style>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.0, 1.8, 1.0])
    with c2:
        st.markdown("""
        <div class="login-brand">
            <div class="lb-icon">🤖</div>
            <h1>AI Agent <span>Hub</span></h1>
            <div class="lb-tag">Multi-Agent Research System</div>
            <div class="lb-desc">
                Your intelligent multi-agent assistant — PDF analysis, web search, 
                blog insights, sentiment, weather, image analysis, data analytics, and more.
            </div>
            <div class="login-features">
                <div class="login-feat"><span class="lf-dot"></span> PDF Q&A</div>
                <div class="login-feat"><span class="lf-dot"></span> Web Search</div>
                <div class="login-feat"><span class="lf-dot"></span> Blog Analysis</div>
                <div class="login-feat"><span class="lf-dot"></span> Sentiment</div>
                <div class="login-feat"><span class="lf-dot"></span> Weather</div>
                <div class="login-feat"><span class="lf-dot"></span> Image AI</div>
                <div class="login-feat"><span class="lf-dot"></span> Data Analytics</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        t1, t2 = st.tabs(["Sign In", "Create Account"])
        with t1:
            with st.form("login_form", clear_on_submit=False):
                u = st.text_input("Username", placeholder="Enter your username", key="lu")
                p = st.text_input("Password", type="password", placeholder="Enter your password", key="lp")
                if st.form_submit_button("Sign In", use_container_width=True):
                    if u and p:
                        user = authenticate(u, p)
                        if user:
                            st.session_state.update(authenticated=True, user=user, current_page="home", messages_home=[])
                            st.rerun()
                        else: st.error("Invalid credentials.")
                    else: st.warning("Please fill in both fields.")
        with t2:
            with st.form("signup_form", clear_on_submit=True):
                nu = st.text_input("Choose a username", placeholder="e.g. john.doe", key="su")
                p1 = st.text_input("Create password", type="password", placeholder="Min 4 characters", key="sp1")
                p2 = st.text_input("Confirm password", type="password", placeholder="Re-enter password", key="sp2")
                if st.form_submit_button("Create Account", use_container_width=True):
                    nc = (nu or "").strip()
                    if not nc or not p1: st.warning("Both fields required.")
                    elif len(nc) < 3: st.warning("Username must be 3+ characters.")
                    elif len(p1) < 4: st.warning("Password must be 4+ characters.")
                    elif p1 != p2: st.error("Passwords don't match.")
                    else:
                        r = register_user(nc, p1)
                        if r: st.success(f"Account created! Sign in as **{nc}**.")
                        else: st.error("Username already taken.")
        
        st.markdown("<p style='text-align:center;color:var(--text-400);font-size:0.7rem;margin-top:1.5rem;'>Powered by AI Agent Hub · GPT-4o</p>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════

def render_sidebar():
    user = st.session_state["user"]
    current_page = st.session_state.get("current_page", "home")
    
    # Try to auto-load env file on first render
    try_load_default_env()
    
    with st.sidebar:
        st.markdown('<div class="brand-box"><div class="logo">🤖 AI Agent Hub</div><div class="sub">Multi-Agent System</div></div>', unsafe_allow_html=True)
        ini = user["display_name"][0].upper()
        st.markdown(f'<div class="user-pill"><div class="av">{ini}</div><div><div class="nm">{user["display_name"]}</div><div class="rl">{user["role"].title()}</div></div></div>', unsafe_allow_html=True)
        
        st.markdown('<hr class="sd">', unsafe_allow_html=True)
        
        # API Configuration via ENV File
        is_configured = st.session_state.get("azure_endpoint") and st.session_state.get("azure_api_key")
        with st.expander("🔑 API Configuration", expanded=not is_configured):
            # Show current status
            if is_configured:
                env_file = st.session_state.get("env_file_name", "Manual")
                st.markdown(f'<div class="doc-item"><span class="di">✅</span>Configured ({env_file})</div>', unsafe_allow_html=True)
                
                # Show loaded keys (masked)
                endpoint = st.session_state.get("azure_endpoint", "")
                if endpoint:
                    st.markdown(f"<p style='font-size:0.7rem;color:var(--text-400);margin:0.2rem 0;'>Endpoint: {endpoint[:30]}...</p>", unsafe_allow_html=True)
                
                has_tavily = "✓" if st.session_state.get("tavily_key") else "✗"
                has_weather = "✓" if st.session_state.get("weather_key") else "✗"
                st.markdown(f"<p style='font-size:0.7rem;color:var(--text-400);margin:0.2rem 0;'>Tavily: {has_tavily} | Weather: {has_weather}</p>", unsafe_allow_html=True)
                
                if st.button("🔄 Reload / Change Config", use_container_width=True, key="reload_config"):
                    # Clear current config
                    for key in ["azure_endpoint", "azure_api_key", "model_deployment", "api_version", "tavily_key", "weather_key", "env_loaded", "env_file_name"]:
                        st.session_state.pop(key, None)
                    st.rerun()
            else:
                st.markdown("<p style='font-size:0.75rem;color:var(--text-300);margin-bottom:0.5rem;'>Upload your DENTSU_AZURE.env file to configure all API keys at once.</p>", unsafe_allow_html=True)
                
                # ENV File Upload
                uploaded_env = st.file_uploader("Upload .env file", type=["env", "txt"], key="env_uploader", label_visibility="collapsed")
                if uploaded_env:
                    try:
                        content = uploaded_env.read().decode("utf-8")
                        config = parse_env_file(content)
                        
                        # Validate required keys
                        if config.get("AZURE_ENDPOINT") and config.get("AZURE_OPENAI_API_KEY"):
                            load_env_to_session(config)
                            st.session_state["env_loaded"] = True
                            st.session_state["env_file_name"] = uploaded_env.name
                            st.success(f"✓ Loaded {len(config)} settings")
                            st.rerun()
                        else:
                            st.error("Missing AZURE_ENDPOINT or AZURE_OPENAI_API_KEY")
                    except Exception as e:
                        st.error(f"Error reading file: {e}")
                
                st.markdown("<p style='font-size:0.68rem;color:var(--text-400);margin-top:0.5rem;'>— or —</p>", unsafe_allow_html=True)
                
                # Manual fallback button
                if st.button("Enter Manually", use_container_width=True, key="manual_config"):
                    st.session_state["show_manual_config"] = True
                    st.rerun()
                
                # Manual entry (hidden by default)
                if st.session_state.get("show_manual_config"):
                    st.markdown('<hr class="sd">', unsafe_allow_html=True)
                    ep = st.text_input("Azure Endpoint", placeholder="https://your-resource.openai.azure.com/", key="ep_input")
                    ak = st.text_input("Azure API Key", placeholder="Enter API key", key="ak_input", type="password")
                    tavily = st.text_input("Tavily API Key", placeholder="tvly-...", key="tavily_input", type="password")
                    weather = st.text_input("Weather API Key", placeholder="weatherapi.com key", key="weather_input", type="password")
                    
                    if st.button("Save", use_container_width=True, key="save_manual_config"):
                        if ep.strip() and ak.strip():
                            st.session_state["azure_endpoint"] = ep.strip()
                            st.session_state["azure_api_key"] = ak.strip()
                            st.session_state["model_deployment"] = "gpt-4o"
                            st.session_state["api_version"] = "2024-12-01-preview"
                            st.session_state["tavily_key"] = tavily.strip()
                            st.session_state["weather_key"] = weather.strip()
                            st.session_state["env_file_name"] = "Manual"
                            st.session_state.pop("show_manual_config", None)
                            st.success("✓ Saved")
                            st.rerun()
                        else:
                            st.warning("Endpoint and API Key required")
        
        st.markdown('<hr class="sd">', unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.72rem;font-weight:700;color:var(--text-400);letter-spacing:0.1em;text-transform:uppercase;'>📂 Pages</p>", unsafe_allow_html=True)
        
        # Page Navigation (replaces agent selection)
        for page_id, page_info in PAGES.items():
            is_active = page_id == current_page
            btn_type = "primary" if is_active else "secondary"
            if st.button(f"{page_info['icon']} {page_info['short']}", key=f"page_{page_id}",
                        use_container_width=True, type=btn_type):
                st.session_state["current_page"] = page_id
                # Each page has its own message history
                if f"messages_{page_id}" not in st.session_state:
                    st.session_state[f"messages_{page_id}"] = []
                st.rerun()
        
        st.markdown('<hr class="sd">', unsafe_allow_html=True)
        
        # ═══════════════════════════════════════════════
        # UPLOAD WIDGETS IN SIDEBAR
        # ═══════════════════════════════════════════════
        
        # PDF Upload
        st.markdown("<p style='font-size:0.72rem;font-weight:700;color:var(--text-400);letter-spacing:0.1em;text-transform:uppercase;'>📄 PDF Documents</p>", unsafe_allow_html=True)
        uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_uploader", label_visibility="collapsed")
        if uploaded_pdf:
            # Check if this PDF was already processed to prevent rerun loop
            existing_docs = [d["filename"] for d in get_user_documents(user["user_id"])]
            last_processed = st.session_state.get("last_processed_pdf", "")
            if uploaded_pdf.name not in existing_docs and uploaded_pdf.name != last_processed:
                with st.spinner("Processing PDF..."):
                    content = extract_pdf(uploaded_pdf)
                    if not content.startswith("["):
                        save_document(user["user_id"], uploaded_pdf.name, "pdf", content)
                        st.session_state["last_processed_pdf"] = uploaded_pdf.name
                        st.success(f"✓ {uploaded_pdf.name}")
                    else:
                        st.session_state["last_processed_pdf"] = uploaded_pdf.name
                        st.error(content)
        
        # Show uploaded docs
        docs = get_user_documents(user["user_id"])
        for doc in docs[:5]:
            cols = st.columns([5, 1])
            with cols[0]:
                st.markdown(f'<div class="doc-item"><span class="di">📄</span>{doc["filename"][:22]}</div>', unsafe_allow_html=True)
            with cols[1]:
                if st.button("×", key=f"del_doc_{doc['id']}"):
                    delete_document(doc["id"], user["user_id"])
                    st.rerun()
        
        st.markdown('<hr class="sd">', unsafe_allow_html=True)
        
        # Blog URL Input
        st.markdown("<p style='font-size:0.72rem;font-weight:700;color:var(--text-400);letter-spacing:0.1em;text-transform:uppercase;'>📝 Blog Articles</p>", unsafe_allow_html=True)
        blog_url = st.text_input("Add blog URL", placeholder="https://example.com/blog", key="blog_url", label_visibility="collapsed")
        if st.button("Add Blog", key="add_blog_btn", use_container_width=True):
            if blog_url and blog_url.startswith("http"):
                existing = [b["url"] for b in get_user_blogs(user["user_id"])]
                if blog_url not in existing:
                    with st.spinner("Loading blog..."):
                        title, content = load_blog_content(blog_url)
                        if not content.startswith("["):
                            save_blog(user["user_id"], blog_url, title, content)
                            st.success(f"✓ Added")
                            st.rerun()
                        else:
                            st.error(content)
        
        # Show added blogs
        blogs = get_user_blogs(user["user_id"])
        for blog in blogs[:5]:
            cols = st.columns([5, 1])
            with cols[0]:
                st.markdown(f'<div class="doc-item"><span class="di">📝</span>{(blog["title"] or "Blog")[:22]}</div>', unsafe_allow_html=True)
            with cols[1]:
                if st.button("×", key=f"del_blog_{blog['id']}"):
                    delete_blog(blog["id"], user["user_id"])
                    st.rerun()
        
        st.markdown('<hr class="sd">', unsafe_allow_html=True)
        
        # Image Upload
        st.markdown("<p style='font-size:0.72rem;font-weight:700;color:var(--text-400);letter-spacing:0.1em;text-transform:uppercase;'>🖼️ Image Analysis</p>", unsafe_allow_html=True)
        uploaded_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "gif", "webp"], key="image_uploader", label_visibility="collapsed")
        if uploaded_image:
            current_image_name = st.session_state.get("uploaded_image_name", "")
            if uploaded_image.name != current_image_name:
                uploaded_image.seek(0)
                image_bytes = uploaded_image.read()
                image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                st.session_state["uploaded_image_data"] = image_b64
                st.session_state["uploaded_image_name"] = uploaded_image.name
                st.success(f"✓ {uploaded_image.name}")
        
        # Show current image
        if st.session_state.get("uploaded_image_data"):
            img_name = st.session_state.get("uploaded_image_name", "image")
            cols = st.columns([5, 1])
            with cols[0]:
                st.markdown(f'<div class="doc-item"><span class="di">🖼️</span>{img_name[:22]}</div>', unsafe_allow_html=True)
            with cols[1]:
                if st.button("×", key="del_image"):
                    st.session_state.pop("uploaded_image_data", None)
                    st.session_state.pop("uploaded_image_name", None)
                    st.rerun()
        
        st.markdown('<hr class="sd">', unsafe_allow_html=True)
        
        # CSV Upload
        st.markdown("<p style='font-size:0.72rem;font-weight:700;color:var(--text-400);letter-spacing:0.1em;text-transform:uppercase;'>📊 Data Analysis</p>", unsafe_allow_html=True)
        uploaded_csv = st.file_uploader("Upload CSV", type=["csv"], key="csv_uploader", label_visibility="collapsed")
        if uploaded_csv:
            current_csv_name = st.session_state.get("uploaded_csv_name", "")
            if uploaded_csv.name != current_csv_name:
                try:
                    uploaded_csv.seek(0)
                    csv_content = uploaded_csv.read().decode("utf-8")
                    st.session_state["uploaded_csv_data"] = csv_content
                    st.session_state["uploaded_csv_name"] = uploaded_csv.name
                    df_preview = pd.read_csv(io.StringIO(csv_content))
                    st.success(f"✓ {len(df_preview):,} rows")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # Show current CSV
        if st.session_state.get("uploaded_csv_data"):
            csv_name = st.session_state.get("uploaded_csv_name", "data.csv")
            cols = st.columns([5, 1])
            with cols[0]:
                st.markdown(f'<div class="doc-item"><span class="di">📊</span>{csv_name[:22]}</div>', unsafe_allow_html=True)
            with cols[1]:
                if st.button("×", key="del_csv"):
                    st.session_state.pop("uploaded_csv_data", None)
                    st.session_state.pop("uploaded_csv_name", None)
                    st.session_state.pop("last_chart", None)
                    st.rerun()
        
        st.markdown('<hr class="sd">', unsafe_allow_html=True)
        
        # New Chat (clears current page's messages)
        if st.button("＋ New Chat", use_container_width=True):
            current_page = st.session_state.get("current_page", "home")
            st.session_state[f"messages_{current_page}"] = []
            st.rerun()
        
        st.markdown('<hr class="sd">', unsafe_allow_html=True)
        
        # Sign Out
        if st.button("Sign out", use_container_width=True, key="logout"):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()
        
        st.markdown("<p style='color:var(--text-400);font-size:0.62rem;text-align:center;padding-top:0.8rem;'>AI Agent Hub v1.0</p>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# MAIN CHAT
# ═══════════════════════════════════════════════

def render_chat():
    if not st.session_state.get("azure_endpoint") or not st.session_state.get("azure_api_key"):
        st.markdown("""
        <div class="welcome-area">
            <div class="w-icon">🔑</div>
            <h2>Configure API Credentials</h2>
            <p>Open <b>🔑 API Configuration</b> in the sidebar to add your Azure OpenAI endpoint and API key.</p>
        </div>""", unsafe_allow_html=True)
        return
    
    # Get current page and agent
    current_page = st.session_state.get("current_page", "home")
    page_info = PAGES.get(current_page, PAGES["home"])
    current_agent = page_info["agent"]
    agent_info = AGENTS.get(current_agent, AGENTS["orchestrator"])
    
    # Get page-specific messages
    messages_key = f"messages_{current_page}"
    if messages_key not in st.session_state:
        st.session_state[messages_key] = []
    messages = st.session_state[messages_key]
    user = st.session_state["user"]
    
    # Page header
    st.markdown(f"""<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:1rem;">
        <span style="font-size:1.5rem;">{page_info['icon']}</span>
        <span style="font-size:1.1rem;font-weight:600;color:var(--text-100);">{page_info['name']}</span>
        <span style="font-size:0.75rem;color:var(--text-400);margin-left:0.5rem;">{page_info['desc']}</span>
    </div>""", unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════
    # CHECK IF UPLOAD REQUIRED FOR THIS PAGE
    # ═══════════════════════════════════════════════
    
    needs_upload = False
    upload_message = ""
    
    if current_page == "pdf" and not get_user_documents(user["user_id"]):
        needs_upload = True
        upload_message = """
        <div class="welcome-area" style="padding-top:6vh;">
            <div class="w-icon">📄</div>
            <h2>Upload PDF Documents First</h2>
            <p>To use the PDF Chat agent, please upload one or more PDF documents using the <b>📄 PDF Documents</b> section in the sidebar.</p>
            <p style="margin-top:1rem;font-size:0.85rem;color:var(--text-300);">Once uploaded, you can ask questions about the content of your documents.</p>
        </div>
        """
    elif current_page == "blog" and not get_user_blogs(user["user_id"]):
        needs_upload = True
        upload_message = """
        <div class="welcome-area" style="padding-top:6vh;">
            <div class="w-icon">📝</div>
            <h2>Add Blog Articles First</h2>
            <p>To use the Blog Chat agent, please add blog URLs using the <b>📝 Blog Articles</b> section in the sidebar.</p>
            <p style="margin-top:1rem;font-size:0.85rem;color:var(--text-300);">Paste a blog URL and click "Add Blog" to load the article content.</p>
        </div>
        """
    elif current_page == "image" and not st.session_state.get("uploaded_image_data"):
        needs_upload = True
        upload_message = """
        <div class="welcome-area" style="padding-top:6vh;">
            <div class="w-icon">🖼️</div>
            <h2>Upload an Image First</h2>
            <p>To use the Image Analyzer agent, please upload an image using the <b>🖼️ Image Analysis</b> section in the sidebar.</p>
            <p style="margin-top:1rem;font-size:0.85rem;color:var(--text-300);">Supported formats: PNG, JPG, JPEG, GIF, WebP</p>
        </div>
        """
    elif current_page == "data" and not st.session_state.get("uploaded_csv_data"):
        needs_upload = True
        upload_message = """
        <div class="welcome-area" style="padding-top:6vh;">
            <div class="w-icon">📊</div>
            <h2>Upload CSV Data First</h2>
            <p>To use the Data Analyst agent, please upload a CSV file using the <b>📊 Data Analysis</b> section in the sidebar.</p>
            <p style="margin-top:1rem;font-size:0.85rem;color:var(--text-300);">You can ask questions about your data and create visualizations.</p>
        </div>
        """
    
    # If upload is required, show the message and return early
    if needs_upload:
        st.markdown(upload_message, unsafe_allow_html=True)
        return
    
    # Welcome screen for Home page
    if not messages and current_page == "home":
        st.markdown(f"""
        <div class="welcome-area" style="padding-top:4vh;">
            <div class="w-icon">🤖</div>
            <h2>Multi-Agent Hub</h2>
            <p>Ask anything! I'll automatically route your question to the best specialist agent.</p>
        </div>
        
        <div class="agent-grid">
            <div class="agent-card">
                <div class="ac-icon">📄</div>
                <div class="ac-title">PDF Questions</div>
                <div class="ac-desc">Upload & query PDFs</div>
            </div>
            <div class="agent-card">
                <div class="ac-icon">🔍</div>
                <div class="ac-title">Web Search</div>
                <div class="ac-desc">Search the internet</div>
            </div>
            <div class="agent-card">
                <div class="ac-icon">📝</div>
                <div class="ac-title">Blog Analysis</div>
                <div class="ac-desc">Add & analyze blogs</div>
            </div>
            <div class="agent-card">
                <div class="ac-icon">💬</div>
                <div class="ac-title">Sentiment</div>
                <div class="ac-desc">Analyze emotions in text</div>
            </div>
            <div class="agent-card">
                <div class="ac-icon">🌤️</div>
                <div class="ac-title">Weather</div>
                <div class="ac-desc">Get weather forecasts</div>
            </div>
            <div class="agent-card">
                <div class="ac-icon">🖼️</div>
                <div class="ac-title">Image Analysis</div>
                <div class="ac-desc">Upload & analyze images</div>
            </div>
            <div class="agent-card">
                <div class="ac-icon">📊</div>
                <div class="ac-title">Data Analytics</div>
                <div class="ac-desc">Upload CSV & visualize</div>
            </div>
        </div>
        
        <p style="text-align:center;color:var(--text-400);font-size:0.8rem;margin-top:1.5rem;">
            💡 Select an agent page from the sidebar to get started
        </p>
        """, unsafe_allow_html=True)
    elif not messages and current_page not in ["pdf", "blog", "image", "data"]:
        # Welcome for agent pages without uploads (web, sentiment, weather)
        st.markdown(f"""
        <div class="welcome-area" style="padding-top:4vh;">
            <div class="w-icon">{agent_info['icon']}</div>
            <h2>{agent_info['name']}</h2>
            <p>{get_agent_welcome(current_agent)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display messages
    for msg in messages:
        # For home page, show which agent was used
        msg_agent = msg.get("agent_type", current_agent)
        msg_agent_info = AGENTS.get(msg_agent, agent_info)
        avatar = "👤" if msg["role"] == "user" else msg_agent_info["icon"]
        
        with st.chat_message(msg["role"], avatar=avatar):
            # Show agent badge for home page responses (always show which agent)
            if msg["role"] == "assistant" and current_page == "home":
                badge_style = "background:linear-gradient(135deg,var(--primary-dim),var(--accent-dim));border:1px solid var(--primary-border);padding:0.4rem 0.8rem;border-radius:20px;font-size:0.75rem;font-weight:600;display:inline-flex;align-items:center;gap:0.4rem;margin-bottom:0.6rem;"
                agent_label = msg_agent_info["name"] if msg_agent != "orchestrator" else "Direct Response"
                st.markdown(f'<div style="{badge_style}">{msg_agent_info["icon"]} {agent_label}</div>', unsafe_allow_html=True)
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_sources(msg["sources"], msg_agent)
                # Display chart if present in sources
                if any(s.get("type") == "chart" for s in msg.get("sources", [])):
                    chart_source = next((s for s in msg["sources"] if s.get("type") == "chart"), None)
                    if chart_source and chart_source.get("chart"):
                        st.image(f"data:image/png;base64,{chart_source['chart']}", use_container_width=True)
    
    # Check for pending web search redirect
    if st.session_state.get("redirect_to_web_search"):
        pending_query = st.session_state.pop("redirect_to_web_search")
        st.session_state["current_page"] = "web"
        st.session_state["pending_query"] = pending_query
        st.rerun()
    
    # Auto-execute pending query after redirect
    pending_query = st.session_state.pop("pending_query", None)
    
    # Chat input
    placeholder = get_page_placeholder(current_page)
    prompt = pending_query or st.chat_input(placeholder, key=f"chat_input_{current_page}")
    
    if prompt:
        # Add user message to page-specific history
        st.session_state[messages_key].append({
            "role": "user", "content": prompt, "agent_type": current_agent
        })
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        # Run appropriate agent based on page
        with st.chat_message("assistant", avatar=agent_info["icon"]):
            spinner_text = "Routing to best agent..." if current_page == "home" else f"{agent_info['name']} is thinking..."
            with st.spinner(spinner_text):
                out_of_scope = False
                
                if current_page == "home":  # Multi-agent orchestrator
                    response, agent_used, sources = run_orchestrator(prompt, user, messages)
                elif current_agent == "pdf_qa":
                    response, agent_used, sources, out_of_scope = run_pdf_retriever(prompt, user, messages)
                elif current_agent == "web_search":
                    response, agent_used, sources, out_of_scope = run_web_search(prompt, user, messages)
                elif current_agent == "blog_qa":
                    response, agent_used, sources, out_of_scope = run_blog_analyzer(prompt, user, messages)
                elif current_agent == "sentiment":
                    response, agent_used, sources, out_of_scope = run_sentiment_analyzer(prompt, user, messages)
                elif current_agent == "weather":
                    response, agent_used, sources, out_of_scope = run_weather_agent(prompt, user, messages)
                elif current_agent == "image_qa":
                    response, agent_used, sources, out_of_scope = run_image_analyzer(prompt, user, messages)
                elif current_agent == "data_analysis":
                    response, agent_used, sources, out_of_scope = run_data_analyzer(prompt, user, messages)
                else:
                    response, agent_used, sources = run_orchestrator(prompt, user)
            
            # Show agent badge for home page (always show which agent was used)
            if current_page == "home":
                used_agent_info = AGENTS.get(agent_used, agent_info)
                badge_style = "background:linear-gradient(135deg,var(--primary-dim),var(--accent-dim));border:1px solid var(--primary-border);padding:0.4rem 0.8rem;border-radius:20px;font-size:0.75rem;font-weight:600;display:inline-flex;align-items:center;gap:0.4rem;margin-bottom:0.6rem;"
                agent_label = used_agent_info["name"] if agent_used != "orchestrator" else "Direct Response"
                st.markdown(f'<div style="{badge_style}">{used_agent_info["icon"]} {agent_label}</div>', unsafe_allow_html=True)
            
            st.markdown(response)
            render_sources(sources, agent_used)
            
            # Display chart if one was generated
            if sources and any(s.get("type") == "chart" for s in sources):
                chart_source = next((s for s in sources if s.get("type") == "chart"), None)
                if chart_source and chart_source.get("chart"):
                    st.image(f"data:image/png;base64,{chart_source['chart']}", use_container_width=True)
            
            # Show "Search Web" button if out of scope (only on dedicated pages)
            if out_of_scope and st.session_state.get("tavily_key") and current_page != "home":
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🔍 Search the Web Instead", key=f"web_search_{current_page}_{len(messages)}", use_container_width=True, type="primary"):
                        st.session_state["redirect_to_web_search"] = prompt
                        st.rerun()
        
        st.session_state[messages_key].append({
            "role": "assistant", "content": response,
            "agent_type": agent_used, "sources": sources,
            "out_of_scope": out_of_scope
        })


def get_agent_welcome(agent_type):
    welcomes = {
        "orchestrator": "Ask me anything — I'll automatically route to the best specialist agent!",
        "pdf_qa": "This is your dedicated PDF chat. Upload PDFs in the sidebar, then ask questions about their content.",
        "web_search": "This is your dedicated web search chat. Ask about news, events, facts, or anything you want to look up online.",
        "blog_qa": "This is your dedicated blog chat. Add blog URLs in the sidebar, and ask questions about their content.",
        "sentiment": "This is your dedicated sentiment analysis chat. Paste reviews, feedback, or any text to analyze emotions.",
        "weather": "This is your dedicated weather chat. Ask about current weather, forecasts, or conditions in any city worldwide.",
        "image_qa": "This is your dedicated image chat. Upload an image in the sidebar, then ask questions about what you see!",
        "data_analysis": "This is your dedicated data analysis chat. Upload a CSV file in the sidebar, then ask questions, request statistics, or create visualizations!"
    }
    return welcomes.get(agent_type, welcomes["orchestrator"])


def get_page_placeholder(page_id):
    placeholders = {
        "home": "Ask anything — I'll route to the best agent...",
        "pdf": "Ask about your uploaded documents...",
        "web": "Search for current information...",
        "blog": "Ask about the blog articles...",
        "sentiment": "Paste text to analyze sentiment...",
        "weather": "Ask about weather in any city...",
        "image": "Ask a question about the uploaded image...",
        "data": "Ask about your data, request stats or charts..."
    }
    return placeholders.get(page_id, placeholders["home"])


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════

def main():
    init_database()
    seed_defaults()
    
    if not st.session_state.get("authenticated"):
        render_login()
    else:
        render_sidebar()
        render_chat()


if __name__ == "__main__":
    main()
