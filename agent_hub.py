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
from datetime import datetime
from warnings import filterwarnings
from typing import List, Dict, Any, Optional

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
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

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
        import pymupdf; f.seek(0)
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
        import requests
        from bs4 import BeautifulSoup
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
# LLM HELPER
# ═══════════════════════════════════════════════

def get_llm(max_tokens=2000, temperature=0.7):
    from langchain_openai import AzureChatOpenAI
    return AzureChatOpenAI(
        azure_deployment=st.session_state.get("model_deployment", "gpt-4o"),
        api_version=st.session_state.get("api_version", "2024-12-01-preview"),
        azure_endpoint=st.session_state.get("azure_endpoint", ""),
        api_key=st.session_state.get("azure_api_key", ""),
        temperature=temperature, max_tokens=max_tokens
    )


# ═══════════════════════════════════════════════
# AGENT DEFINITIONS
# ═══════════════════════════════════════════════

AGENTS = {
    "orchestrator": {
        "name": "Smart Assistant",
        "icon": "🤖",
        "desc": "General AI assistant that can route to specialists",
        "color": "#7C3AED"
    },
    "pdf_qa": {
        "name": "PDF Retriever",
        "icon": "📄",
        "desc": "Answer questions from uploaded PDF documents",
        "color": "#3B82F6"
    },
    "web_search": {
        "name": "Web Search",
        "icon": "🔍",
        "desc": "Search the internet for current information",
        "color": "#10B981"
    },
    "blog_qa": {
        "name": "Blog Analyzer",
        "icon": "📝",
        "desc": "Analyze and answer questions about blogs",
        "color": "#F59E0B"
    },
    "sentiment": {
        "name": "Sentiment Analyzer",
        "icon": "💬",
        "desc": "Analyze sentiment of reviews and text",
        "color": "#EC4899"
    }
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
        from langchain_community.tools.tavily_search import TavilySearchResults
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


# ═══════════════════════════════════════════════
# GUARDRAILS
# ═══════════════════════════════════════════════

def check_scope(query: str, context: str, agent_type: str) -> dict:
    """Check if the query is within scope of the provided context."""
    from langchain_core.messages import HumanMessage, SystemMessage
    
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
        
        import json
        # Clean response
        text = response.content.strip()
        if text.startswith("```"): text = text.split("```")[1].replace("json", "").strip()
        
        return json.loads(text)
    except:
        return {"in_scope": True, "confidence": 0.5, "reason": "Could not determine scope"}


def run_with_guardrails(prompt: str, context: str, agent_type: str, system_prompt: str) -> tuple:
    """Run agent with guardrails - detect out-of-scope and offer web search."""
    from langchain_core.messages import HumanMessage, SystemMessage
    
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

def run_orchestrator(prompt: str, user: dict) -> tuple:
    """Smart assistant with LangChain tools for autonomous routing."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    
    today_str = datetime.now().strftime("%B %d, %Y")
    
    # Check available resources
    has_pdfs = bool(get_user_documents(user["user_id"]))
    has_blogs = bool(get_user_blogs(user["user_id"]))
    has_tavily = bool(st.session_state.get("tavily_key"))
    
    # Build available tools list
    available_tools = []
    tool_info = []
    
    if has_pdfs:
        available_tools.append(search_pdf_documents)
        tool_info.append("- search_pdf_documents: Query uploaded PDF documents")
    if has_blogs:
        available_tools.append(search_blog_articles)
        tool_info.append("- search_blog_articles: Query added blog articles")
    if has_tavily:
        available_tools.append(search_web)
        tool_info.append("- search_web: Search the internet for current information")
    available_tools.append(analyze_sentiment)
    tool_info.append("- analyze_sentiment: Analyze emotions in text")
    
    tools_str = "\n".join(tool_info) if tool_info else "No specialized tools available."
    
    system_prompt = f"""You are AI Agent Hub's Smart Orchestrator with access to specialized tools. Today is {today_str}.

You have access to these tools:
{tools_str}

Guidelines:
1. For questions about uploaded documents, use search_pdf_documents
2. For current events, news, or real-time info, use search_web
3. For questions about blog articles, use search_blog_articles
4. For sentiment/emotion analysis, use analyze_sentiment
5. For general knowledge questions, answer directly without tools

Always be helpful, accurate, and use markdown formatting."""

    try:
        llm = get_llm(max_tokens=2000, temperature=0.7)
        
        # If tools available, use agent
        if available_tools:
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
            
            try:
                agent = create_tool_calling_agent(llm, available_tools, prompt_template)
                executor = AgentExecutor(agent=agent, tools=available_tools, verbose=False, max_iterations=3)
                result = executor.invoke({"input": prompt, "chat_history": []})
                return result.get("output", "No response generated."), "orchestrator", [{"tools_used": True}]
            except Exception as agent_error:
                # Fallback to direct LLM if agent fails
                response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=prompt)])
                return response.content, "orchestrator", []
        else:
            # No tools - direct response
            response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=prompt)])
            return response.content, "orchestrator", []
    except Exception as e:
        return f"Error: {str(e)}", "orchestrator", []


def run_pdf_retriever(prompt: str, user: dict) -> tuple:
    """Answer questions from uploaded PDF documents with guardrails."""
    
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
    
    system_prompt = """You are the PDF Retriever Agent. Your role is to answer questions based ONLY on the provided document context.

Rules:
1. Answer using information from the documents
2. Quote relevant passages when helpful
3. If the answer isn't in the documents, say so clearly
4. Cite which document the information comes from
5. Use markdown formatting for clarity"""

    # Run with guardrails
    result = run_with_guardrails(prompt, full_context, "PDF Documents", system_prompt)
    sources = [{"name": d["filename"], "type": "PDF"} for d in docs[:5]]
    
    return result["response"], "pdf_qa", sources, result.get("out_of_scope", False)


def run_web_search(prompt: str, user: dict) -> tuple:
    """Search the web for current information using LangChain Tavily tool."""
    from langchain_core.messages import HumanMessage, SystemMessage
    
    tavily_key = st.session_state.get("tavily_key", "")
    if not tavily_key:
        return "🔍 **Tavily API Key not configured.**\n\nPlease add your Tavily API key in the sidebar under API Configuration.", "web_search", [], False
    
    try:
        # Use LangChain Tavily tool
        search_results = search_web.invoke(prompt)
        
        if search_results.startswith("NO_") or search_results.startswith("SEARCH_ERROR"):
            return f"🔍 {search_results}", "web_search", [], False
        
        # Extract sources from results
        sources = []
        import re
        urls = re.findall(r'Source: (https?://[^\s]+)', search_results)
        for url in urls[:5]:
            domain = re.sub(r'^https?://(www\.)?', '', url).split('/')[0]
            sources.append({"url": url, "title": domain})
        
        # Generate synthesized answer
        llm = get_llm(max_tokens=1500, temperature=0.5)
        today_str = datetime.now().strftime("%B %d, %Y")
        
        response = llm.invoke([
            SystemMessage(content=f"You are the Web Search Agent. Today is {today_str}. Synthesize the search results to answer the user's question. Cite sources with URLs. Use markdown formatting."),
            HumanMessage(content=f"Search Results:\n{search_results}\n\n---\n\nQuestion: {prompt}")
        ])
        
        return response.content, "web_search", sources, False
    except Exception as e:
        return f"Search error: {str(e)}", "web_search", [], False


def run_blog_analyzer(prompt: str, user: dict) -> tuple:
    """Answer questions about added blog articles with guardrails."""
    
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
    
    system_prompt = """You are the Blog Analyzer Agent. Your role is to analyze and answer questions about the provided blog articles.

Capabilities:
1. Summarize blog content
2. Answer specific questions about the articles
3. Compare information across multiple blogs
4. Extract key points and insights
5. Identify the author's tone and perspective

Always cite which blog article the information comes from. Use markdown formatting."""

    # Run with guardrails
    result = run_with_guardrails(prompt, full_context, "Blog Articles", system_prompt)
    
    return result["response"], "blog_qa", sources, result.get("out_of_scope", False)


def run_sentiment_analyzer(prompt: str, user: dict) -> tuple:
    """Analyze sentiment of reviews and text using LangChain."""
    from langchain_core.messages import HumanMessage, SystemMessage
    
    # Guardrail: Check if text is too short
    if len(prompt.strip()) < 10:
        return "⚠️ **Text too short for sentiment analysis.**\n\nPlease provide a longer text (review, feedback, comment) to analyze.", "sentiment", [], False
    
    system_prompt = """You are the Sentiment Analyzer Agent. Analyze the sentiment of the provided text.

For each analysis, provide:
1. **Overall Sentiment**: POSITIVE, NEGATIVE, or NEUTRAL (with confidence percentage)
2. **Sentiment Score**: -1.0 (most negative) to +1.0 (most positive)
3. **Key Emotions**: List detected emotions (joy, anger, sadness, fear, surprise, etc.)
4. **Tone Analysis**: Formal/informal, sincere/sarcastic, etc.
5. **Key Phrases**: Important positive and negative phrases
6. **Summary**: Brief explanation of the sentiment

Format your response clearly with markdown. Be objective and thorough."""

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
                blog insights, sentiment analysis, and more. All in one place.
            </div>
            <div class="login-features">
                <div class="login-feat"><span class="lf-dot"></span> PDF Q&A</div>
                <div class="login-feat"><span class="lf-dot"></span> Web Search</div>
                <div class="login-feat"><span class="lf-dot"></span> Blog Analysis</div>
                <div class="login-feat"><span class="lf-dot"></span> Sentiment</div>
                <div class="login-feat"><span class="lf-dot"></span> Multi-Agent</div>
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
                            st.session_state.update(authenticated=True, user=user, current_agent="orchestrator", messages=[])
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
    with st.sidebar:
        st.markdown('<div class="brand-box"><div class="logo">🤖 AI Agent Hub</div><div class="sub">Multi-Agent System</div></div>', unsafe_allow_html=True)
        ini = user["display_name"][0].upper()
        st.markdown(f'<div class="user-pill"><div class="av">{ini}</div><div><div class="nm">{user["display_name"]}</div><div class="rl">{user["role"].title()}</div></div></div>', unsafe_allow_html=True)
        
        st.markdown('<hr class="sd">', unsafe_allow_html=True)
        
        # API Configuration
        with st.expander("🔑 API Configuration", expanded=not st.session_state.get("azure_endpoint")):
            ep = st.text_input("Azure Endpoint", value=st.session_state.get("azure_endpoint", ""),
                              placeholder="https://your-resource.openai.azure.com/", key="ep_input")
            ak = st.text_input("Azure API Key", value=st.session_state.get("azure_api_key", ""),
                              placeholder="Enter API key", key="ak_input", type="password")
            model = st.text_input("Model Deployment", value=st.session_state.get("model_deployment", "gpt-4o"),
                                 placeholder="gpt-4o", key="model_input")
            tavily = st.text_input("Tavily API Key (for web search)", value=st.session_state.get("tavily_key", ""),
                                  placeholder="tvly-...", key="tavily_input", type="password")
            
            if st.button("Save Configuration", use_container_width=True, key="save_config"):
                if ep.strip() and ak.strip():
                    st.session_state["azure_endpoint"] = ep.strip()
                    st.session_state["azure_api_key"] = ak.strip()
                    st.session_state["model_deployment"] = model.strip() or "gpt-4o"
                    st.session_state["tavily_key"] = tavily.strip()
                    st.success("Configuration saved!"); st.rerun()
                else: st.warning("Endpoint and API Key required.")
        
        st.markdown('<hr class="sd">', unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.72rem;font-weight:700;color:var(--text-400);letter-spacing:0.1em;text-transform:uppercase;'>Select Agent</p>", unsafe_allow_html=True)
        
        # Agent Selection
        current_agent = st.session_state.get("current_agent", "orchestrator")
        for agent_id, agent_info in AGENTS.items():
            is_active = agent_id == current_agent
            btn_type = "primary" if is_active else "secondary"
            if st.button(f"{agent_info['icon']} {agent_info['name']}", key=f"agent_{agent_id}",
                        use_container_width=True, type=btn_type):
                st.session_state["current_agent"] = agent_id
                st.session_state["messages"] = []
                st.rerun()
        
        st.markdown('<hr class="sd">', unsafe_allow_html=True)
        
        # PDF Upload
        st.markdown("<p style='font-size:0.72rem;font-weight:700;color:var(--text-400);letter-spacing:0.1em;text-transform:uppercase;'>📄 PDF Documents</p>", unsafe_allow_html=True)
        
        uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_uploader", label_visibility="collapsed")
        if uploaded_pdf:
            with st.spinner("Processing PDF..."):
                content = extract_pdf(uploaded_pdf)
                if not content.startswith("["):
                    save_document(user["user_id"], uploaded_pdf.name, "pdf", content)
                    st.success(f"Uploaded: {uploaded_pdf.name}")
                    st.rerun()
                else:
                    st.error(content)
        
        # Show uploaded docs
        docs = get_user_documents(user["user_id"])
        for doc in docs[:5]:
            cols = st.columns([5, 1])
            with cols[0]:
                st.markdown(f'<div class="doc-item"><span class="di">📄</span>{doc["filename"][:25]}</div>', unsafe_allow_html=True)
            with cols[1]:
                if st.button("×", key=f"del_doc_{doc['id']}"):
                    delete_document(doc["id"], user["user_id"])
                    st.rerun()
        
        st.markdown('<hr class="sd">', unsafe_allow_html=True)
        
        # Blog URL Input
        st.markdown("<p style='font-size:0.72rem;font-weight:700;color:var(--text-400);letter-spacing:0.1em;text-transform:uppercase;'>📝 Blog Articles</p>", unsafe_allow_html=True)
        
        blog_url = st.text_input("Paste blog URL", placeholder="https://example.com/blog-post", key="blog_url", label_visibility="collapsed")
        if blog_url and blog_url.startswith("http"):
            existing = [b["url"] for b in get_user_blogs(user["user_id"])]
            if blog_url not in existing:
                with st.spinner("Loading blog..."):
                    title, content = load_blog_content(blog_url)
                    if not content.startswith("["):
                        save_blog(user["user_id"], blog_url, title, content)
                        st.success(f"Added: {title[:30]}...")
                        st.rerun()
                    else:
                        st.error(content)
        
        # Show added blogs
        blogs = get_user_blogs(user["user_id"])
        for blog in blogs[:5]:
            cols = st.columns([5, 1])
            with cols[0]:
                st.markdown(f'<div class="doc-item"><span class="di">📝</span>{(blog["title"] or "Blog")[:25]}</div>', unsafe_allow_html=True)
            with cols[1]:
                if st.button("×", key=f"del_blog_{blog['id']}"):
                    delete_blog(blog["id"], user["user_id"])
                    st.rerun()
        
        st.markdown('<hr class="sd">', unsafe_allow_html=True)
        
        # New Chat
        if st.button("＋ New Chat", use_container_width=True):
            st.session_state["messages"] = []
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
    
    current_agent = st.session_state.get("current_agent", "orchestrator")
    agent_info = AGENTS.get(current_agent, AGENTS["orchestrator"])
    messages = st.session_state.get("messages", [])
    user = st.session_state["user"]
    
    # Welcome screen
    if not messages:
        st.markdown(f"""
        <div class="welcome-area">
            <div class="w-icon">{agent_info['icon']}</div>
            <h2>{agent_info['name']}</h2>
            <p>{get_agent_welcome(current_agent)}</p>
        </div>
        
        <div class="agent-grid">
            <div class="agent-card">
                <div class="ac-icon">🤖</div>
                <div class="ac-title">Smart Assistant</div>
                <div class="ac-desc">General AI help & routing</div>
            </div>
            <div class="agent-card">
                <div class="ac-icon">📄</div>
                <div class="ac-title">PDF Retriever</div>
                <div class="ac-desc">Q&A from your documents</div>
            </div>
            <div class="agent-card">
                <div class="ac-icon">🔍</div>
                <div class="ac-title">Web Search</div>
                <div class="ac-desc">Real-time web information</div>
            </div>
            <div class="agent-card">
                <div class="ac-icon">📝</div>
                <div class="ac-title">Blog Analyzer</div>
                <div class="ac-desc">Insights from articles</div>
            </div>
            <div class="agent-card">
                <div class="ac-icon">💬</div>
                <div class="ac-title">Sentiment</div>
                <div class="ac-desc">Analyze review emotions</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Display messages
    for msg in messages:
        avatar = "👤" if msg["role"] == "user" else agent_info["icon"]
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_sources(msg["sources"], msg.get("agent_type", current_agent))
    
    # Check for pending web search redirect
    if st.session_state.get("redirect_to_web_search"):
        pending_query = st.session_state.pop("redirect_to_web_search")
        st.session_state["current_agent"] = "web_search"
        st.session_state["pending_query"] = pending_query
        st.rerun()
    
    # Auto-execute pending query after redirect
    pending_query = st.session_state.pop("pending_query", None)
    
    # Chat input
    placeholder = get_agent_placeholder(current_agent)
    prompt = pending_query or st.chat_input(placeholder, key="chat_input")
    
    if prompt:
        # Add user message
        st.session_state.setdefault("messages", []).append({
            "role": "user", "content": prompt, "agent_type": current_agent
        })
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        # Run appropriate agent
        with st.chat_message("assistant", avatar=agent_info["icon"]):
            with st.spinner(f"{agent_info['name']} is thinking..."):
                out_of_scope = False
                
                if current_agent == "orchestrator":
                    response, agent_used, sources = run_orchestrator(prompt, user)
                elif current_agent == "pdf_qa":
                    response, agent_used, sources, out_of_scope = run_pdf_retriever(prompt, user)
                elif current_agent == "web_search":
                    response, agent_used, sources, out_of_scope = run_web_search(prompt, user)
                elif current_agent == "blog_qa":
                    response, agent_used, sources, out_of_scope = run_blog_analyzer(prompt, user)
                elif current_agent == "sentiment":
                    response, agent_used, sources, out_of_scope = run_sentiment_analyzer(prompt, user)
                else:
                    response, agent_used, sources = run_orchestrator(prompt, user)
            
            st.markdown(response)
            render_sources(sources, agent_used)
            
            # Show "Search Web" button if out of scope
            if out_of_scope and st.session_state.get("tavily_key"):
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🔍 Search the Web Instead", key=f"web_search_{len(messages)}", use_container_width=True, type="primary"):
                        st.session_state["redirect_to_web_search"] = prompt
                        st.rerun()
        
        st.session_state["messages"].append({
            "role": "assistant", "content": response,
            "agent_type": agent_used, "sources": sources,
            "out_of_scope": out_of_scope
        })


def get_agent_welcome(agent_type):
    welcomes = {
        "orchestrator": "I'm your intelligent assistant. Ask me anything — I can help with questions, writing, analysis, or guide you to specialized agents.",
        "pdf_qa": "I analyze your uploaded PDF documents. Upload PDFs in the sidebar, then ask me questions about their content.",
        "web_search": "I search the web for current information. Ask about news, events, facts, or anything you want to look up online.",
        "blog_qa": "I analyze blog articles. Add blog URLs in the sidebar, and I'll answer questions about their content.",
        "sentiment": "I analyze sentiment in text. Paste a review, feedback, or any text, and I'll identify the emotions and sentiment."
    }
    return welcomes.get(agent_type, welcomes["orchestrator"])


def get_agent_placeholder(agent_type):
    placeholders = {
        "orchestrator": "Ask me anything...",
        "pdf_qa": "Ask about your uploaded documents...",
        "web_search": "Search for current information...",
        "blog_qa": "Ask about the blog articles...",
        "sentiment": "Paste text to analyze sentiment..."
    }
    return placeholders.get(agent_type, placeholders["orchestrator"])


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
