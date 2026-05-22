"""
Multi-Agent Dentsu Buddy - v2.0 (Enhanced)
===========================================
Combines Multi-Agent RAG Architecture with Smart Buddy Features:

Multi-Agent Architecture:
  - Orchestrator Agent: Routes queries to appropriate specialists
  - Campaign Agent: Searches campaign database
  - Research Agent: Searches marketing research PDFs
  - Compliance Agent: Searches legal/compliance articles
  - Web Agent: Searches live web for current events
  - Validator Agent: Validates all responses before delivery

Smart Features (from v5.1):
  1. User authentication & session management
  2. SQLite persistence (conversations, documents)
  3. CSV/Excel analysis with chart generation
  4. Multi-document interaction (images + docs + data)
  5. Smart keyword-based routing
  6. Professional dark theme UI
  7. Weather API integration
"""

import os
import re
import json
import warnings
import tempfile
import base64
import hashlib
import sqlite3
import uuid
import io
from io import BytesIO
from datetime import datetime
from typing import Annotated, Sequence, TypedDict, Literal, List, Dict, Any, Optional

warnings.filterwarnings("ignore")

import streamlit as st
from dotenv import load_dotenv

# Document loading and splitting
from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Embeddings and vector store
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_chroma import Chroma

# Prompt templates and output parsers
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool, create_retriever_tool

# LangGraph components
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Structured output
from pydantic import BaseModel, Field

# Web search
from langchain_community.tools.tavily_search import TavilySearchResults


# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="Multi-Agent Dentsu Buddy",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Professional Dark Theme CSS (from Smart Buddy v5.1)
# ============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --primary: #E30613;
    --primary-dim: rgba(227,6,19,0.10);
    --primary-border: rgba(227,6,19,0.18);
    --primary-glow: rgba(227,6,19,0.25);
    --accent: #00C4B3;
    --accent-dim: rgba(0,196,179,0.10);
    --bg-root: #0A0D12;
    --bg-surface: #10141B;
    --bg-card: #161B25;
    --bg-elevated: #1D2330;
    --text-100: #F0F2F5;
    --text-200: #C4CAD4;
    --text-300: #8B95A5;
    --text-400: #5E6878;
    --border: rgba(255,255,255,0.06);
    --border-focus: rgba(227,6,19,0.45);
    --blue: #6B8AFF;
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
    color: var(--text-200) !important; font-family: 'Inter', sans-serif !important; font-size: 0.85rem !important;
}

.brand-box {
    padding: 1.5rem 1rem 1.2rem 1rem; text-align: center;
    border-bottom: 1px solid var(--border); margin-bottom: 1rem;
}
.brand-box .logo {
    font-size: 1.15rem; font-weight: 700; color: var(--primary);
    letter-spacing: 0.08em;
    display: flex; align-items: center; justify-content: center; gap: 0.45rem;
}
.brand-box .sub {
    font-size: 0.62rem; color: var(--text-400);
    letter-spacing: 0.15em; text-transform: uppercase; margin-top: 4px;
}

.user-pill {
    display: flex; align-items: center; gap: 0.6rem;
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius-sm); padding: 0.55rem 0.75rem; margin: 0.6rem 0;
}
.user-pill .av {
    width: 30px; height: 30px; border-radius: 50%;
    background: linear-gradient(135deg, var(--primary), var(--accent));
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.75rem; color: #fff; flex-shrink: 0;
}
.user-pill .nm { font-weight: 600; font-size: 0.82rem; color: var(--text-100); }
.user-pill .rl { font-size: 0.68rem; color: var(--text-400); }

.sd { border: none; border-top: 1px solid var(--border); margin: 0.9rem 0; }

/* Login styling */
.login-brand {
    text-align: center; margin-bottom: 2rem;
}
.login-brand .lb-icon {
    width: 64px; height: 64px; border-radius: 20px;
    background: linear-gradient(135deg, var(--primary), var(--accent));
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 1.8rem; margin-bottom: 1rem; box-shadow: 0 8px 24px rgba(227,6,19,0.3);
}
.login-brand h1 {
    font-size: 1.55rem; font-weight: 700; color: var(--text-100);
    letter-spacing: 0.04em; margin: 0 0 0.15rem 0;
}
.login-brand h1 span { color: var(--accent); }
.login-brand .lb-tag {
    font-size: 0.68rem; color: var(--text-400); letter-spacing: 0.2em;
    text-transform: uppercase; margin-top: 0.2rem;
}
.login-brand .lb-desc {
    font-size: 0.82rem; color: var(--text-300); margin-top: 0.8rem;
    line-height: 1.55; max-width: 400px; margin-left: auto; margin-right: auto;
}
.login-features {
    display: flex; justify-content: center; gap: 1.2rem; margin-top: 1.2rem; flex-wrap: wrap;
}
.login-feat {
    display: flex; align-items: center; gap: 0.35rem;
    font-size: 0.7rem; color: var(--text-300);
    background: var(--bg-elevated); border: 1px solid var(--border);
    border-radius: 20px; padding: 0.3rem 0.7rem;
}
.login-feat .lf-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); flex-shrink: 0; }

.src-box { margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px solid rgba(255,255,255,0.06); }
.src-label { font-size: 0.62rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #5E6878; margin-bottom: 0.35rem; }
.src-chips { display: flex; flex-wrap: wrap; gap: 0.3rem; }
.src-chip {
    display: inline-flex; align-items: center; gap: 0.25rem;
    background: #1E2430; border: 1px solid rgba(255,255,255,0.06);
    border-radius: 20px; padding: 0.2rem 0.6rem;
    font-size: 0.68rem; color: #8B95A5; text-decoration: none !important; transition: all 0.15s;
}
.src-chip:hover { border-color: rgba(227,6,19,0.3); color: #E30613; background: rgba(227,6,19,0.08); }
.src-chip .sd2 { width: 5px; height: 5px; border-radius: 50%; background: #6B8AFF; flex-shrink: 0; }

.tbadge {
    display: inline-block; margin-top: 0.3rem;
    background: rgba(227,6,19,0.08); border: 1px solid rgba(227,6,19,0.18);
    border-radius: 20px; padding: 0.12rem 0.55rem;
    font-size: 0.62rem; font-weight: 500; color: #E30613; font-family: 'JetBrains Mono', monospace;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important; padding: 0.8rem 1rem !important;
    margin-bottom: 0.5rem !important; color: var(--text-100) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(135deg, rgba(227,6,19,0.10), rgba(227,6,19,0.04)) !important;
    border: 1px solid var(--primary-border) !important;
}
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] td, [data-testid="stChatMessage"] span {
    color: var(--text-100) !important; font-family: 'Inter', sans-serif !important;
}
[data-testid="stChatMessage"] code {
    background: rgba(255,255,255,0.06) !important; color: #e0c285 !important;
    font-family: 'JetBrains Mono', monospace !important; padding: 0.15em 0.4em !important;
}

/* Agent badges */
.agent-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 16px;
    font-size: 0.75em;
    font-weight: 600;
    margin: 2px;
}
.agent-campaign { background-color: rgba(30,136,229,0.2); color: #42A5F5; }
.agent-research { background-color: rgba(156,39,176,0.2); color: #BA68C8; }
.agent-compliance { background-color: rgba(255,152,0,0.2); color: #FFB74D; }
.agent-web { background-color: rgba(76,175,80,0.2); color: #81C784; }
.agent-validator { background-color: rgba(244,67,54,0.2); color: #EF5350; }
.agent-orchestrator { background-color: rgba(0,150,136,0.2); color: #4DB6AC; }
.agent-memory { background-color: rgba(255,193,7,0.2); color: #FFD54F; }
.agent-guardrail { background-color: rgba(121,85,72,0.2); color: #A1887F; }
.agent-consolidator { background-color: rgba(63,81,181,0.2); color: #7986CB; }
.agent-upload { background-color: rgba(3,169,244,0.2); color: #4FC3F7; }
.agent-data { background-color: rgba(0,188,212,0.2); color: #4DD0E1; }

.agent-step {
    background-color: var(--bg-elevated);
    padding: 12px;
    border-radius: 8px;
    margin: 8px 0;
    border-left: 4px solid var(--primary);
}

.doc-item {
    display: flex; align-items: center; gap: 0.4rem;
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 6px; padding: 0.35rem 0.6rem; margin: 0.25rem 0;
    font-size: 0.75rem; color: var(--text-200);
}
.doc-item .di { color: var(--primary); flex-shrink: 0; }

.welcome-area { text-align: center; padding: 10vh 2rem 4rem 2rem; }
.welcome-area .w-icon {
    width: 64px; height: 64px; border-radius: 20px;
    background: linear-gradient(135deg, var(--primary), var(--accent));
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 1.6rem; margin-bottom: 1.2rem; box-shadow: 0 8px 24px rgba(227,6,19,0.25);
}
.welcome-area h2 { font-size: 1.4rem; font-weight: 600; color: var(--text-100); margin-bottom: 0.5rem; }
.welcome-area p { font-size: 0.88rem; color: var(--text-400); max-width: 550px; margin: 0 auto; line-height: 1.6; }

.stTextInput > div > div > input {
    background: var(--bg-elevated) !important; border: 1px solid var(--border) !important;
    color: var(--text-100) !important; border-radius: var(--radius-sm) !important;
}
.stTextInput > div > div > input:focus { border-color: var(--border-focus) !important; }
.stButton > button {
    font-family: 'Inter', sans-serif !important; font-weight: 600 !important;
    border-radius: var(--radius-sm) !important; transition: all 0.2s !important;
}
form .stButton > button { background: linear-gradient(135deg, var(--primary), #FF4444) !important; color: #fff !important; border: none !important; }
form .stButton > button:hover { box-shadow: 0 4px 16px var(--primary-glow) !important; }
section[data-testid="stSidebar"] .stButton > button {
    background: var(--bg-card) !important; color: var(--text-200) !important; border: 1px solid var(--border) !important;
}
.stFileUploader > div { background: var(--bg-card) !important; border: 1px dashed rgba(255,255,255,0.1) !important; }
.stChatInput > div { background: var(--bg-card) !important; border: 1px solid var(--border) !important; }
.stChatInput textarea { color: var(--text-100) !important; font-family: 'Inter', sans-serif !important; }
.stSpinner > div > div { border-top-color: var(--accent) !important; }
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: var(--text-300) !important;
    font-family: 'Inter', sans-serif !important; font-weight: 600 !important;
}
.stTabs [aria-selected="true"] { color: var(--primary) !important; border-bottom-color: var(--primary) !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--bg-elevated); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Database Functions (from Smart Buddy v5.1)
# ============================================================================

DB_PATH = "multi_agent_buddy.db"

def init_database():
    """Initialize SQLite database for users, conversations, and documents."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY, username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL, display_name TEXT,
        role TEXT DEFAULT 'user', created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS conversations (
        conversation_id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
        title TEXT DEFAULT 'New Chat',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS messages (
        message_id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL,
        role TEXT NOT NULL, content TEXT NOT NULL,
        agents_used TEXT, trace TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS documents (
        doc_id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
        filename TEXT NOT NULL, file_type TEXT, content_preview TEXT,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit()
    conn.close()

def _hash(pw):
    return hashlib.sha256(f"dentsu_multi_agent_{pw}".encode()).hexdigest()

def register_user(username, password):
    uid = str(uuid.uuid4())
    display = username.strip().title()
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("INSERT INTO users VALUES (?,?,?,?,?,CURRENT_TIMESTAMP)",
                     (uid, username.lower().strip(), _hash(password), display, "user"))
        conn.commit()
        conn.close()
        return {"user_id": uid, "username": username.lower().strip(), "display_name": display, "role": "user"}
    except sqlite3.IntegrityError:
        conn.close()
        return None

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
    conn.commit()
    conn.close()

def create_conversation(uid, title="New Chat"):
    cid = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO conversations VALUES (?,?,?,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)", (cid, uid, title))
    conn.commit()
    conn.close()
    return cid

def get_conversations(uid):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT conversation_id,title,created_at,updated_at FROM conversations WHERE user_id=? ORDER BY updated_at DESC", (uid,)).fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1], "created_at": r[2], "updated_at": r[3]} for r in rows]

def delete_conversation(cid):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM messages WHERE conversation_id=?", (cid,))
    conn.execute("DELETE FROM conversations WHERE conversation_id=?", (cid,))
    conn.commit()
    conn.close()

def save_message(cid, role, content, agents_used=None, trace=None):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO messages VALUES (?,?,?,?,?,?,CURRENT_TIMESTAMP)",
                 (str(uuid.uuid4()), cid, role, content,
                  json.dumps(agents_used) if agents_used else None,
                  json.dumps(trace) if trace else None))
    conn.execute("UPDATE conversations SET updated_at=CURRENT_TIMESTAMP WHERE conversation_id=?", (cid,))
    conn.commit()
    conn.close()

def get_messages(cid):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT role,content,agents_used,trace,timestamp FROM messages WHERE conversation_id=? ORDER BY timestamp ASC", (cid,)).fetchall()
    conn.close()
    out = []
    for r in rows:
        agents = None
        trace = None
        try:
            if r[2]: agents = json.loads(r[2])
            if r[3]: trace = json.loads(r[3])
        except: pass
        out.append({"role": r[0], "content": r[1], "agents_used": agents, "trace": trace, "timestamp": r[4]})
    return out

def save_doc_record(uid, fname, ftype, preview):
    conn = sqlite3.connect(DB_PATH)
    existing = conn.execute("SELECT doc_id FROM documents WHERE user_id=? AND filename=?", (uid, fname)).fetchone()
    if existing:
        conn.close()
        return
    conn.execute("INSERT INTO documents VALUES (?,?,?,?,?,CURRENT_TIMESTAMP)",
                 (str(uuid.uuid4()), uid, fname, ftype, preview[:500]))
    conn.commit()
    conn.close()

def get_user_docs(uid):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT doc_id,filename,file_type,uploaded_at FROM documents WHERE user_id=? ORDER BY uploaded_at DESC", (uid,)).fetchall()
    conn.close()
    return [{"id": r[0], "filename": r[1], "type": r[2], "uploaded_at": r[3]} for r in rows]

def delete_document(doc_id, uid):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT filename FROM documents WHERE doc_id=? AND user_id=?", (doc_id, uid)).fetchone()
    fname = row[0] if row else None
    conn.execute("DELETE FROM documents WHERE doc_id=? AND user_id=?", (doc_id, uid))
    conn.commit()
    conn.close()
    return fname

def generate_title(content):
    words = content.split()[:7]
    return " ".join(words) + ("..." if len(content.split()) > 7 else "")


# ============================================================================
# LLM Helper
# ============================================================================

def get_llm(max_tokens=2000):
    """Get LLM from session state or environment."""
    if st.session_state.get("llm"):
        return st.session_state.llm
    
    # Fallback to dynamic credentials
    from langchain_openai import AzureChatOpenAI
    return AzureChatOpenAI(
        azure_deployment=st.session_state.get("model_name", "gpt-4o"),
        api_version=st.session_state.get("api_version", "2024-12-01-preview"),
        azure_endpoint=st.session_state.get("azure_endpoint", ""),
        api_key=st.session_state.get("azure_api_key", ""),
        temperature=0,
        max_tokens=max_tokens
    )


# ============================================================================
# File Processing (Enhanced from Smart Buddy v5.1)
# ============================================================================

def extract_pdf(f):
    try:
        import pymupdf
        f.seek(0)
        doc = pymupdf.open(stream=f.read(), filetype="pdf")
        txt = "\n".join(p.get_text() for p in doc)
        doc.close()
        return txt if txt.strip() else "[No extractable text in PDF]"
    except Exception as e:
        return f"[PDF error: {e}]"

def extract_docx(f):
    try:
        from docx import Document
        f.seek(0)
        return "\n".join(p.text for p in Document(io.BytesIO(f.read())).paragraphs if p.text.strip())
    except Exception as e:
        return f"[DOCX error: {e}]"

def extract_txt(f):
    try:
        f.seek(0)
        d = f.read()
        try:
            return d.decode("utf-8")
        except:
            return d.decode("latin-1")
    except Exception as e:
        return f"[TXT error: {e}]"

def extract_image(f):
    try:
        f.seek(0)
        data = f.read()
        if not data:
            return "[Image error: empty file]"
        b64 = base64.b64encode(data).decode("utf-8")
        ext = f.name.rsplit(".", 1)[-1].lower()
        mime_map = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif", "webp": "image/webp"}
        return f"__IMAGE_B64__{json.dumps({'mime': mime_map.get(ext, 'image/png'), 'b64': b64})}"
    except Exception as e:
        return f"[Image error: {e}]"

def extract_csv_excel(f):
    """Read CSV/Excel into pandas, store as JSON-safe marker."""
    try:
        import pandas as pd
        f.seek(0)
        name = f.name.lower()
        if name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(f.read()))
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(f.read()))
        else:
            return "[Unsupported tabular format]"

        payload = {
            "shape": f"{df.shape[0]} rows × {df.shape[1]} columns",
            "columns": {col: str(df[col].dtype) for col in df.columns},
            "preview_md": df.head(20).to_markdown(index=False),
            "data_json": df.to_json(orient="records", date_format="iso")
        }
        return f"__DATAFRAME__{json.dumps(payload)}"
    except Exception as e:
        return f"[CSV/Excel error: {e}]"

def is_image_data(text):
    return text and text.startswith("__IMAGE_B64__")

def is_dataframe_data(text):
    return text and text.startswith("__DATAFRAME__")

def parse_image_data(text):
    try:
        payload = json.loads(text[len("__IMAGE_B64__"):])
        return payload["mime"], payload["b64"]
    except:
        return None, None

def parse_dataframe(text):
    try:
        import pandas as pd
        payload = json.loads(text[len("__DATAFRAME__"):])
        df = pd.DataFrame(json.loads(payload["data_json"]))
        return df, payload
    except:
        return None, None

def process_upload(f):
    n = f.name.lower()
    if n.endswith(".pdf"):
        return extract_pdf(f)
    if n.endswith(".docx"):
        return extract_docx(f)
    if n.endswith(".txt"):
        return extract_txt(f)
    if n.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
        return extract_image(f)
    if n.endswith((".csv", ".xlsx", ".xls")):
        return extract_csv_excel(f)
    return f"[Unsupported: {f.name}]"

def classify_doc_types():
    """Classify all uploaded docs into categories."""
    image_entries = {}
    text_entries = {}
    df_entries = {}
    for fn, txt in st.session_state.get("doc_texts", {}).items():
        if is_image_data(txt):
            image_entries[fn] = txt
        elif is_dataframe_data(txt):
            df_entries[fn] = txt
        else:
            text_entries[fn] = txt
    return image_entries, text_entries, df_entries


# ============================================================================
# Smart Routing — keyword-based, no LLM hallucination
# ============================================================================

def build_doc_topic_keywords():
    """Extract actual keywords from document content for matching."""
    keywords = set()
    for fn, txt in st.session_state.get("doc_texts", {}).items():
        if is_image_data(txt):
            keywords.update(["image", "picture", "photo", "diagram", "screenshot", fn.lower()])
        elif is_dataframe_data(txt):
            _, payload = parse_dataframe(txt)
            if payload:
                for col in payload.get("columns", {}).keys():
                    keywords.update(col.lower().split("_"))
                    keywords.update(col.lower().split(" "))
            keywords.update(["data", "csv", "excel", "table", "column", "row", fn.lower()])
        else:
            # Extract top words from text documents
            words = re.findall(r'\b[a-zA-Z]{3,}\b', txt[:3000].lower())
            # Get most frequent meaningful words
            from collections import Counter
            common = Counter(words).most_common(50)
            stop = {"the","and","for","are","but","not","you","all","can","had","her","was","one",
                    "our","out","has","have","been","from","this","that","with","they","will","each",
                    "which","their","said","what","its","about","than","into","them","some","could",
                    "other","more","very","when","come","make","like","over","such","also","most"}
            for w, _ in common:
                if w not in stop: keywords.add(w)
            keywords.add(fn.lower().replace(".pdf","").replace(".docx","").replace(".txt",""))
    return keywords

def is_query_about_docs(query):
    """Keyword overlap check — fast, no hallucination."""
    doc_keywords = build_doc_topic_keywords()
    if not doc_keywords:
        return False

    query_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()))

    # Direct reference to document or file
    direct_refs = {"document", "file", "uploaded", "pdf", "report", "article", "csv",
                   "excel", "spreadsheet", "image", "picture", "photo", "data", "table",
                   "chart", "plot", "graph", "column", "analyze", "summarize", "summary",
                   "extract", "describe", "explain", "what", "list", "content"}
    if query_words & direct_refs:
        return True

    # Check overlap with actual doc content keywords
    overlap = query_words & doc_keywords
    # If at least 2 content words match, or 1 match + short query
    if len(overlap) >= 2:
        return True
    if len(overlap) >= 1 and len(query_words) <= 5:
        return True

    return False


# ============================================================================
# Helper Functions
# ============================================================================

def extract_sources(text):
    """Extract source URLs from text."""
    import html as html_lib
    urls = re.findall(r'https?://[^\s\)\]>"\'`,]+', text)
    seen = set(); unique = []
    for u in urls:
        u = u.rstrip('.,;:!?)')
        domain = re.sub(r'^https?://(www\.)?', '', u).split('/')[0]
        if domain not in seen and len(domain) > 2:
            seen.add(domain); unique.append({"url": u, "domain": domain})
    return unique[:8]

def clean_answer_urls(text, sources):
    """Clean raw URLs from text while preserving markdown links."""
    if not sources: return text
    protected = text; phs = {}
    for i, m in enumerate(re.finditer(r'\[([^\]]+)\]\(https?://[^\)]+\)', text)):
        ph = f"MDLNK{i}X"; phs[ph] = m.group(0); protected = protected.replace(m.group(0), ph, 1)
    cleaned = re.sub(r'https?://[^\s\)\]>"\'`,]+', '', protected)
    for ph, orig in phs.items(): cleaned = cleaned.replace(ph, orig)
    cleaned = re.sub(r'Source:\s*$', '', cleaned, flags=re.M)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()

def get_domain_label(domain):
    """Get display label from domain."""
    parts = domain.split('.'); return parts[-2].capitalize() if len(parts) >= 2 else domain.capitalize()

def render_sources_and_tools(sources, tools):
    """Render source chips and tool badges."""
    import html as html_lib
    parts = []
    if sources:
        chips = ""
        for s in sources:
            d = s.get("domain", ""); u = html_lib.escape(s.get("url", "#"))
            chips += f'<a href="{u}" target="_blank" rel="noopener" class="src-chip"><span class="sd2"></span>{get_domain_label(d)} · {d}</a>'
        parts.append(f'<div class="src-box"><div class="src-label">📎 Sources</div><div class="src-chips">{chips}</div></div>')
    if tools:
        t = str(tools)
        t = t.replace("search_web_extract_info", "Web Search").replace("get_weather", "Weather")
        parts.append(f'<div class="tbadge">⚡ {t}</div>')
    if parts:
        st.markdown("\n".join(parts), unsafe_allow_html=True)


# ============================================================================
# URL Processing
# ============================================================================

def detect_url(text):
    match = re.search(r'https?://[^\s<>"\']+', text)
    return match.group(0) if match else None

def load_url_content(url):
    try:
        import requests
        from bs4 import BeautifulSoup
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        content = "\n".join(lines)
        return content[:15000] if content else "[No content extracted from URL]"
    except Exception as e:
        return f"[URL load error: {e}]"

def store_url_as_doc(url, content):
    from urllib.parse import urlparse
    domain = urlparse(url).netloc
    doc_name = f"🔗 {domain}"
    st.session_state.setdefault("doc_texts", {})[doc_name] = content
    if doc_name not in st.session_state.get("processed_files", []):
        st.session_state.setdefault("processed_files", []).append(doc_name)
    st.session_state.setdefault("url_docs", {})[url] = doc_name
    return doc_name


# ============================================================================
# Load Environment Variables
# ============================================================================
@st.cache_resource
def load_credentials():
    """Load Azure OpenAI credentials from .env file."""
    env_files = ['DENTSU_AZURE.env', 'Dentsu_AZURE.env', '.env']
    
    for env_file in env_files:
        if os.path.exists(env_file):
            load_dotenv(env_file)
            break
    
    credentials = {
        "endpoint": os.environ.get("MODEL_ENDPOINT"),
        "model_name": os.environ.get("CHAT_MODEL_NAME"),
        "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
        "api_version": os.environ.get("api_version"),
        "embeddings_model": os.environ.get("EMBEDDING_MODEL_NAME"),
        "embeddings_endpoint": os.environ.get("MODEL_ENDPOINT_EMBEDDING"),
        "api_version_embedding": os.environ.get("api_version_embedding"),
        "tavily_key": os.environ.get("TAVILY_API_KEY")
    }
    
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    
    return credentials


# ============================================================================
# Initialize LLM and Embeddings
# ============================================================================
@st.cache_resource
def initialize_models(_credentials):
    """Initialize LLM and Embeddings models."""
    llm = AzureChatOpenAI(
        azure_deployment=_credentials["model_name"],
        api_version=_credentials["api_version"],
        azure_endpoint=_credentials["endpoint"],
        api_key=_credentials["api_key"],
        temperature=0
    )
    
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=_credentials["embeddings_endpoint"],
        azure_deployment=_credentials["embeddings_model"],
        openai_api_version=_credentials["api_version_embedding"],
        api_key=_credentials["api_key"]
    )
    
    return llm, embeddings


# ============================================================================
# Knowledge Base Setup (Multi-Agent RAG)
# ============================================================================
@st.cache_resource
def setup_campaign_db(_embeddings):
    """Load and index campaign database."""
    PERSIST_DIR = "campaigns_db_vectorstore"
    
    if os.path.exists(PERSIST_DIR):
        vectordb = Chroma(
            collection_name="campaigns-db",
            persist_directory=PERSIST_DIR,
            embedding_function=_embeddings,
        )
    else:
        if not os.path.exists("campaigns_db.json"):
            return None
        with open("campaigns_db.json", "r") as f:
            campaigns_data = json.load(f)
        
        campaign_documents = []
        for idx, campaign in enumerate(campaigns_data):
            fields = '\n'.join(f"{key}: {value}" for key, value in campaign.items())
            campaign_documents.append(
                Document(
                    page_content=fields,
                    metadata={
                        "source": "campaigns_db.json",
                        "index": idx,
                        "industry": campaign.get("industry", ""),
                        "client": campaign.get("client", ""),
                        "campaign_name": campaign.get("campaign_name", "")
                    }
                )
            )
        
        vectordb = Chroma.from_documents(
            documents=campaign_documents,
            collection_name="campaigns-db",
            embedding=_embeddings,
            persist_directory=PERSIST_DIR,
        )
    
    retriever = vectordb.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.5, "k": 5}
    )
    
    tool = create_retriever_tool(
        retriever=retriever,
        name='search_campaigns_db',
        description='Search and return information about marketing campaigns, advertising strategies, and media plans.',
    )
    
    return tool


@st.cache_resource
def setup_pdf_db(_embeddings):
    """Load and index research PDFs."""
    PERSIST_DIR = "marketing_pdf_db"
    
    if os.path.exists(PERSIST_DIR):
        vectordb = Chroma(
            collection_name="marketing-pdf-docs",
            persist_directory=PERSIST_DIR,
            embedding_function=_embeddings,
        )
    else:
        pdf_files = [
            'Content_Effects_Advertising_Marketing.pdf',
            'Digital_Transformation_in_Marketing.pdf'
        ]
        
        all_pdf_docs = []
        for pdf_file in pdf_files:
            if os.path.exists(pdf_file):
                docs = PyMuPDFLoader(pdf_file).load_and_split()
                all_pdf_docs.extend(docs)
        
        if not all_pdf_docs:
            return None
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        pdf_texts = text_splitter.split_documents(all_pdf_docs)
        
        vectordb = Chroma.from_documents(
            documents=pdf_texts,
            collection_name="marketing-pdf-docs",
            embedding=_embeddings,
            persist_directory=PERSIST_DIR,
        )
    
    retriever = vectordb.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.5, "k": 5}
    )
    
    tool = create_retriever_tool(
        retriever=retriever,
        name='search_marketing_research',
        description='Search marketing and advertising research papers about content effectiveness, digital transformation, MarTech.',
    )
    
    return tool


@st.cache_resource
def setup_web_articles_db(_embeddings):
    """Load and index marketing law web articles."""
    PERSIST_DIR = "marketing_law_articles_db"
    
    if os.path.exists(PERSIST_DIR):
        vectordb = Chroma(
            collection_name="marketing-web-docs",
            persist_directory=PERSIST_DIR,
            embedding_function=_embeddings,
        )
    else:
        marketing_urls = [
            "https://tenthings.blog/2023/06/30/ten-things-marketing-law-basics-for-in-house-counsel/",
            "https://blog.ipleaders.in/marketing-media-consumer-protection-law-india/"
        ]
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        
        try:
            docs = [WebBaseLoader(url).load() for url in marketing_urls]
            docs_list = [item for sublist in docs for item in sublist]
            web_texts = text_splitter.split_documents(docs_list)
            
            vectordb = Chroma.from_documents(
                documents=web_texts,
                collection_name="marketing-web-docs",
                embedding=_embeddings,
                persist_directory=PERSIST_DIR,
            )
        except Exception:
            return None
    
    retriever = vectordb.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.5, "k": 5}
    )
    
    tool = create_retriever_tool(
        retriever=retriever,
        name='search_marketing_law_articles',
        description='Search legal and compliance insights on marketing, advertising, consumer protection.',
    )
    
    return tool


def create_web_search_tool():
    """Create live web search tool using Tavily."""
    tavily_search = TavilySearchResults(
        max_results=5,
        search_depth='advanced',
        include_raw_content=True
    )
    
    @tool
    def search_web(query: str) -> str:
        """Search the web for current information, news, recent events."""
        results = tavily_search.invoke(query)
        
        formatted_results = []
        for r in results:
            title = r.get('title', 'No Title')
            content = r.get('content', 'No Content')
            url = r.get('url', '')
            formatted_results.append(f"Title: {title}\nContent: {content}\nSource: {url}")
        
        return "\n\n---\n\n".join(formatted_results) if formatted_results else "No results found."
    
    return search_web


# ============================================================================
# Data Analysis with Charts (from Smart Buddy v5.1)
# ============================================================================

def run_dataframe_analysis(query, df_entries, llm):
    """Analyze CSV/Excel data using pandas. Returns (answer, chart_fig_or_None)."""
    try:
        import pandas as pd

        all_dfs = {}
        for fn, raw in df_entries.items():
            df, payload = parse_dataframe(raw)
            if df is not None:
                all_dfs[fn] = (df, payload)

        if not all_dfs:
            return "No tabular data could be loaded.", None

        primary_name = list(all_dfs.keys())[0]
        df, payload = all_dfs[primary_name]

        schema_info = f"DataFrame '{primary_name}': {payload['shape']}\nColumns:\n"
        for col in df.columns:
            sample = df[col].dropna().head(3).tolist()
            schema_info += f"  - {col} ({df[col].dtype}): sample = {sample}\n"

        wants_chart = any(kw in query.lower() for kw in [
            "chart", "plot", "graph", "visuali", "bar", "pie", "line",
            "histogram", "scatter", "trend", "distribution", "show me"
        ])

        if wants_chart:
            code_prompt = f"""Given this DataFrame schema:
{schema_info}

User wants: {query}

Write Python code that:
1. Uses variable `df` (pandas DataFrame, already loaded)
2. Uses matplotlib to create the chart
3. Creates figure: fig, ax = plt.subplots(figsize=(10, 6))
4. Style: dark background '#161B25', white text, colors ['#E30613','#00C4B3','#6B8AFF','#FF6B6B','#FFB347']
5. Add title, labels. DO NOT call plt.show()

Output ONLY Python code. No explanation. No markdown fences."""

            resp = llm.invoke([
                SystemMessage(content="Output ONLY valid Python code. Nothing else."),
                HumanMessage(content=code_prompt)
            ])
            code = resp.content.strip()
            for fence in ["```python", "```py", "```"]:
                code = code.replace(fence, "")
            code = code.strip()

            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import numpy as np

                exec_globals = {"pd": pd, "plt": plt, "np": np, "df": df}
                exec(code, exec_globals)
                fig = exec_globals.get("fig", plt.gcf())

                resp2 = llm.invoke([
                    SystemMessage(content="You are a data analyst. Be concise — 2-3 sentences with specific numbers."),
                    HumanMessage(content=f"Schema:\n{schema_info}\n\nUser asked: {query}\nChart was generated. Write a brief insight.")
                ])
                return resp2.content.strip(), fig

            except Exception as e:
                return f"**Chart error:** {e}\n\nFalling back to basic stats:\n\n{df.describe().to_markdown()}", None

        else:
            # Text-based analysis
            resp = llm.invoke([
                SystemMessage(content=f"You are a data analyst. Analyze this data:\n{schema_info}\n\nFirst rows:\n{df.head().to_markdown(index=False)}"),
                HumanMessage(content=query)
            ])
            return resp.content.strip(), None

    except Exception as e:
        return f"Analysis error: {str(e)}", None


# ============================================================================
# Guardrail System
# ============================================================================
class MarketingDecision(BaseModel):
    decision: Literal["YES", "NO"] = Field(
        description="YES if the query is marketing/advertising related, NO otherwise"
    )


GUARDRAIL_PROMPT = """
Classify the following user query according to whether it is related to marketing, advertising,
media planning, brand strategy, campaign management, digital marketing, content strategy,
programmatic advertising, media buying, creative strategy, audience targeting, marketing analytics,
ROI measurement, social media marketing, SEO/SEM, influencer marketing, MarTech, CRM,
advertising technology, marketing operations, or legal and compliance topics related to marketing.

Return either YES or NO as per the mentioned schema.
"""


def check_marketing_guardrail(llm, user_query: str) -> dict:
    """Validates whether a query is marketing/advertising-related."""
    messages = [
        SystemMessage(content=GUARDRAIL_PROMPT),
        HumanMessage(content=user_query),
    ]
    
    llm_classifier = llm.with_structured_output(MarketingDecision)
    classifier_response = llm_classifier.invoke(messages)
    
    if classifier_response.decision != "YES":
        return {
            "approved": False,
            "message": (
                "**Out of Scope**\n\n"
                "Sorry, I can only assist with marketing, advertising, and related industry queries. "
                "Please try again with a marketing or advertising-related question."
            )
        }
    
    return {"approved": True, "message": None}


# ============================================================================
# Multi-Agent System
# ============================================================================

class MultiAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_query: str
    selected_agents: List[str]
    agent_responses: Dict[str, str]
    validation_result: Optional[Dict[str, Any]]
    final_answer: str
    revision_count: int


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


class AgentSelection(BaseModel):
    agents: List[Literal["campaign", "research", "compliance", "web"]] = Field(
        description="List of specialist agents to invoke for this query"
    )
    reasoning: str = Field(
        description="Brief explanation of why these agents were selected"
    )


class ValidationResult(BaseModel):
    accuracy_score: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        description="Assessment of factual accuracy based on source documents"
    )
    compliance_check: bool = Field(
        description="Whether the response follows marketing guidelines"
    )
    completeness_score: Literal["COMPLETE", "PARTIAL", "INCOMPLETE"] = Field(
        description="How thoroughly the response addresses the user's question"
    )
    issues: List[str] = Field(default_factory=list)
    recommendation: Literal["APPROVE", "REVISE", "REJECT"] = Field(
        description="Final recommendation for the response"
    )
    revision_guidance: str = Field(default="")


ORCHESTRATOR_PROMPT = """You are an intelligent query router for a marketing research system.
Analyze the user's question and select which specialist agents should handle it.

Available agents:
- **campaign**: For questions about specific campaigns, case studies, channel recommendations, performance benchmarks
- **research**: For questions about marketing research, content effectiveness, digital transformation, academic insights
- **compliance**: For questions about marketing law, legal compliance, consumer protection, endorsements, disclosures
- **web**: For questions about current events, recent news, platform updates, real-time information

Rules:
1. Select 1-3 agents most relevant to the question
2. For general marketing knowledge, prefer "campaign" or "research"
3. For legal/regulatory questions, always include "compliance"
4. For questions about recent events or "latest" anything, include "web"

Return the list of agents to invoke and your reasoning."""


VALIDATOR_PROMPT = """You are a quality assurance validator for a marketing research assistant.
Validate responses before they are shown to users.

Validation Criteria:
1. **Accuracy**: Does the response match source documents? No hallucinated metrics.
2. **Compliance**: Does it include appropriate disclaimers?
3. **Completeness**: Does it fully address the user's question?

Rules:
- APPROVE: Response is accurate, complete, and follows guidelines
- REVISE: Response has minor issues (missing disclaimer, incomplete)
- REJECT: Response has major issues (factual errors, off-topic)"""


def create_agent_node(llm, tools):
    def agent(state: AgentState):
        model_with_tools = llm.bind_tools(tools)
        response = model_with_tools.invoke(state['messages'])
        return {'messages': [response]}
    return agent


def create_generate_node(llm, agent_type: str = "general"):
    agent_prompts = {
        "campaign": "You are a campaign research specialist for Dentsu. Focus on campaign details, channel strategies, KPIs.",
        "research": "You are a marketing research analyst for Dentsu. Focus on research findings and evidence-based recommendations.",
        "compliance": "You are a marketing compliance advisor for Dentsu. Focus on legal requirements and regulatory compliance.",
        "web": "You are a marketing intelligence specialist for Dentsu. Focus on current events and industry news.",
        "general": "You are a helpful marketing and advertising research assistant for Dentsu."
    }
    
    def generate(state: AgentState):
        messages = state['messages']
        question = messages[0].content
        docs = messages[-1].content
        
        system_prompt = agent_prompts.get(agent_type, agent_prompts["general"])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             f"{system_prompt}\n\n"
             "Use the following documents to answer the question.\n"
             "Include disclaimer: 'This information is for research purposes only and should be validated.'"),
            ("human", "Documents:\n{context}\n\nQuestion: {question}\n\nAnswer:")
        ])
        
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({'context': docs, 'question': question})
        
        return {'messages': [answer]}
    
    return generate


def create_rewrite_node(llm):
    def rewrite(state: AgentState):
        original_question = state['messages'][0].content
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Refine this search query to be clearer and more specific."),
            ("human", "Original question: {question}\n\nRewrite as a clearer, more searchable question:")
        ])
        
        chain = prompt | llm
        response = chain.invoke({'question': original_question})
        
        return {'messages': [HumanMessage(content=response.content)]}
    
    return rewrite


def create_grade_documents(llm):
    class Grade(BaseModel):
        score: str = Field(description="'yes' if the document is relevant, 'no' if not")
    
    def grade_documents(state: AgentState) -> Literal['generate', 'rewrite']:
        grader = llm.with_structured_output(Grade)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Determine if the retrieved document helps answer the user's question. Reply 'yes' or 'no'."),
            ("human", "Document:\n{context}\n\nQuestion: {question}\n\nRelevant? (yes/no)")
        ])
        
        chain = prompt | grader
        
        question = state['messages'][0].content
        docs = state['messages'][-1].content
        
        result = chain.invoke({'question': question, 'context': docs})
        
        return 'generate' if result.score == 'yes' else 'rewrite'
    
    return grade_documents


def build_specialist_agent(llm, tools, agent_type: str):
    """Build a specialist agent sub-graph."""
    workflow = StateGraph(AgentState)
    
    workflow.add_node('agent', create_agent_node(llm, tools))
    workflow.add_node('retrieve', ToolNode(tools))
    workflow.add_node('generate', create_generate_node(llm, agent_type))
    workflow.add_node('rewrite', create_rewrite_node(llm))
    
    workflow.add_edge(START, 'agent')
    workflow.add_conditional_edges('agent', tools_condition, {
        'tools': 'retrieve',
        END: END,
    })
    workflow.add_conditional_edges('retrieve', create_grade_documents(llm))
    workflow.add_edge('generate', END)
    workflow.add_edge('rewrite', 'agent')
    
    return workflow.compile()


@st.cache_resource
def build_specialist_agents(_llm, _tools_dict):
    """Build all specialist agents."""
    agents = {}
    
    if "campaign" in _tools_dict and _tools_dict["campaign"]:
        agents["campaign"] = build_specialist_agent(_llm, [_tools_dict["campaign"]], "campaign")
    
    if "research" in _tools_dict and _tools_dict["research"]:
        agents["research"] = build_specialist_agent(_llm, [_tools_dict["research"]], "research")
    
    if "compliance" in _tools_dict and _tools_dict["compliance"]:
        agents["compliance"] = build_specialist_agent(_llm, [_tools_dict["compliance"]], "compliance")
    
    if "web" in _tools_dict and _tools_dict["web"]:
        agents["web"] = build_specialist_agent(_llm, [_tools_dict["web"]], "web")
    
    return agents


def format_conversation_history(messages: List[Dict], max_turns: int = 5) -> str:
    if not messages:
        return ""
    
    recent_messages = messages[-(max_turns * 2):]
    
    formatted = []
    for msg in recent_messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")[:500]
        formatted.append(f"{role}: {content}")
    
    return "\n\n".join(formatted)


def run_orchestrator(llm, user_query: str, conversation_history: str = "") -> Dict[str, Any]:
    context_section = ""
    if conversation_history:
        context_section = f"\n\nConversation History:\n{conversation_history}\n\n"
    
    messages = [
        SystemMessage(content=ORCHESTRATOR_PROMPT),
        HumanMessage(content=f"{context_section}Current user question: {user_query}")
    ]
    
    orchestrator = llm.with_structured_output(AgentSelection)
    result = orchestrator.invoke(messages)
    
    return {
        "selected_agents": result.agents,
        "reasoning": result.reasoning
    }


def run_validator(llm, user_query: str, agent_responses: Dict[str, str]) -> ValidationResult:
    formatted_responses = "\n\n".join([
        f"**{agent.upper()} AGENT:**\n{response}"
        for agent, response in agent_responses.items()
    ])
    
    messages = [
        SystemMessage(content=VALIDATOR_PROMPT),
        HumanMessage(content=f"User Question: {user_query}\n\nAgent Responses:\n{formatted_responses}\n\nValidate:")
    ]
    
    validator = llm.with_structured_output(ValidationResult)
    return validator.invoke(messages)


def consolidate_responses(llm, user_query: str, agent_responses: Dict[str, str], conversation_history: str = "") -> str:
    if len(agent_responses) == 1:
        return list(agent_responses.values())[0]
    
    formatted_responses = "\n\n---\n\n".join([
        f"**From {agent.upper()} Agent:**\n{response}"
        for agent, response in agent_responses.items()
    ])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a senior marketing research editor. Consolidate multiple agent responses into a single coherent answer."),
        ("human", "User Question: {question}\n\nAgent Responses:\n{responses}\n\nConsolidate:")
    ])
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({'question': user_query, 'responses': formatted_responses})


def run_multi_agent_system(
    llm, 
    specialist_agents: Dict, 
    question: str, 
    conversation_history: str = "",
    uploaded_file_context: str = "",
    enable_validation: bool = True
) -> Dict[str, Any]:
    """Run the full multi-agent system with orchestration and validation."""
    trace = []
    
    if conversation_history:
        trace.append({
            "step": "Context",
            "agent": "memory",
            "action": "📝 Using conversation history",
            "status": "pass"
        })
    
    if uploaded_file_context:
        trace.append({
            "step": "Uploaded Files",
            "agent": "upload",
            "action": "📄 Using uploaded file content",
            "status": "pass"
        })
    
    # Step 1: Guardrail Check
    trace.append({
        "step": "Guardrail",
        "agent": "guardrail",
        "action": "Checking if query is marketing-related...",
        "status": "running"
    })
    
    guardrail_result = check_marketing_guardrail(llm, question)
    
    if not guardrail_result["approved"]:
        trace[-1]["action"] = "❌ REJECTED - Query is out of scope"
        trace[-1]["status"] = "fail"
        return {
            "answer": guardrail_result["message"],
            "trace": trace,
            "validation_result": None,
            "agents_used": []
        }
    
    trace[-1]["action"] = "✅ APPROVED - Query is marketing-related"
    trace[-1]["status"] = "pass"
    
    # Step 2: Orchestrator
    trace.append({
        "step": "Orchestrator",
        "agent": "orchestrator",
        "action": "Selecting specialist agents...",
        "status": "running"
    })
    
    try:
        routing = run_orchestrator(llm, question, conversation_history)
        selected_agents = routing["selected_agents"]
        selected_agents = [a for a in selected_agents if a in specialist_agents]
        
        if not selected_agents:
            selected_agents = list(specialist_agents.keys())[:1]
        
        trace[-1]["action"] = f"Selected agents: **{', '.join(selected_agents)}**"
        trace[-1]["details"] = f"Reasoning: {routing['reasoning']}"
        trace[-1]["status"] = "pass"
        
    except Exception as e:
        selected_agents = list(specialist_agents.keys())[:1]
        trace[-1]["action"] = f"⚠️ Routing error, using fallback"
        trace[-1]["status"] = "warning"
    
    # Step 3: Run Specialist Agents
    agent_responses = {}
    
    contextual_question = question
    if uploaded_file_context:
        contextual_question = f"[Context:\n{uploaded_file_context[:5000]}]\n\nQuestion: {question}"
    
    for agent_name in selected_agents:
        trace.append({
            "step": f"{agent_name.title()} Agent",
            "agent": agent_name,
            "action": f"Processing query...",
            "status": "running"
        })
        
        try:
            agent = specialist_agents[agent_name]
            result = agent.invoke({"messages": [("user", contextual_question)]})
            
            if result.get("messages"):
                last_msg = result["messages"][-1]
                response = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
                agent_responses[agent_name] = response
                
                trace[-1]["action"] = f"✅ Generated response ({len(response)} chars)"
                trace[-1]["status"] = "pass"
            else:
                trace[-1]["action"] = "⚠️ No response generated"
                trace[-1]["status"] = "warning"
                
        except Exception as e:
            trace[-1]["action"] = f"❌ Error: {str(e)[:100]}"
            trace[-1]["status"] = "fail"
    
    if not agent_responses:
        return {
            "answer": "I apologize, but I couldn't generate a response. Please try rephrasing your question.",
            "trace": trace,
            "validation_result": None,
            "agents_used": selected_agents
        }
    
    # Step 4: Consolidate Responses
    trace.append({
        "step": "Consolidation",
        "agent": "consolidator",
        "action": "Merging agent responses...",
        "status": "running"
    })
    
    try:
        consolidated = consolidate_responses(llm, question, agent_responses, conversation_history)
        trace[-1]["action"] = f"✅ Consolidated {len(agent_responses)} agent responses"
        trace[-1]["status"] = "pass"
    except Exception:
        consolidated = list(agent_responses.values())[0]
        trace[-1]["action"] = f"⚠️ Using primary agent response"
        trace[-1]["status"] = "warning"
    
    # Step 5: Validation
    validation_result = None
    
    if enable_validation:
        trace.append({
            "step": "Validator",
            "agent": "validator",
            "action": "Validating response quality...",
            "status": "running"
        })
        
        try:
            validation_result = run_validator(llm, question, agent_responses)
            
            status_emoji = {"APPROVE": "✅", "REVISE": "⚠️", "REJECT": "❌"}
            
            trace[-1]["action"] = (
                f"{status_emoji.get(validation_result.recommendation, '?')} "
                f"**{validation_result.recommendation}** | "
                f"Accuracy: {validation_result.accuracy_score}"
            )
            trace[-1]["status"] = "pass" if validation_result.recommendation == "APPROVE" else "warning"
            
        except Exception as e:
            trace[-1]["action"] = f"⚠️ Validation skipped"
            trace[-1]["status"] = "warning"
    
    return {
        "answer": consolidated,
        "trace": trace,
        "validation_result": validation_result,
        "agents_used": selected_agents
    }


def run_direct_web_search(llm, question: str, conversation_history: str = "") -> Dict[str, Any]:
    """Run a direct web search bypassing other agents."""
    trace = []
    
    trace.append({
        "step": "Direct Web Search",
        "agent": "web",
        "action": "🌐 Searching the web directly...",
        "status": "running"
    })
    
    try:
        tavily_search = TavilySearchResults(
            max_results=5,
            search_depth='advanced',
            include_raw_content=True
        )
        
        results = tavily_search.invoke(question)
        
        formatted_results = []
        for r in results:
            title = r.get('title', 'No Title')
            content = r.get('content', 'No Content')
            url = r.get('url', '')
            formatted_results.append(f"**{title}**\n{content}\n*Source: {url}*")
        
        search_results = "\n\n---\n\n".join(formatted_results) if formatted_results else "No results found."
        
        trace[-1]["action"] = f"✅ Found {len(results)} web results"
        trace[-1]["status"] = "pass"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a marketing research assistant. Use the web search results to answer the question. Cite sources."),
            ("human", "Web Results:\n{results}\n\nQuestion: {question}\n\nAnswer:")
        ])
        
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({'results': search_results, 'question': question})
        
        trace.append({
            "step": "Generate Answer",
            "agent": "web",
            "action": f"✅ Generated answer from web results",
            "status": "pass"
        })
        
        return {
            "answer": answer,
            "trace": trace,
            "validation_result": None,
            "agents_used": ["web"]
        }
        
    except Exception as e:
        trace[-1]["action"] = f"❌ Web search error: {str(e)[:100]}"
        trace[-1]["status"] = "fail"
        return {
            "answer": f"Sorry, I encountered an error during web search: {str(e)}",
            "trace": trace,
            "validation_result": None,
            "agents_used": ["web"]
        }


# ============================================================================
# Login Page
# ============================================================================

def render_login():
    st.markdown("<style>section[data-testid='stSidebar']{display:none;}</style>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.0, 1.8, 1.0])
    with c2:
        st.markdown("""
        <div class="login-brand">
            <div class="lb-icon">🤖</div>
            <h1>Multi-Agent <span>Dentsu Buddy</span></h1>
            <div class="lb-tag">Agentic RAG Marketing Research System</div>
            <div class="lb-desc">
                Your AI-powered marketing research companion with multi-agent orchestration,
                specialized knowledge bases, response validation, and data analysis.
            </div>
            <div class="login-features">
                <div class="login-feat"><span class="lf-dot"></span> Multi-Agent RAG</div>
                <div class="login-feat"><span class="lf-dot"></span> Document Q&A</div>
                <div class="login-feat"><span class="lf-dot"></span> Data Analysis</div>
                <div class="login-feat"><span class="lf-dot"></span> Web Research</div>
                <div class="login-feat"><span class="lf-dot"></span> Response Validation</div>
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
                            st.session_state.update(authenticated=True, user=user, current_conv=None, messages=[])
                            st.rerun()
                        else:
                            st.error("Invalid credentials.")
                    else:
                        st.warning("Please fill in both fields.")
        with t2:
            with st.form("signup_form", clear_on_submit=True):
                nu = st.text_input("Choose a username", placeholder="e.g. john.doe", key="su")
                p1 = st.text_input("Create password", type="password", placeholder="Min 4 characters", key="sp1")
                p2 = st.text_input("Confirm password", type="password", placeholder="Re-enter password", key="sp2")
                if st.form_submit_button("Create Account", use_container_width=True):
                    nc = (nu or "").strip()
                    if not nc or not p1:
                        st.warning("Both fields required.")
                    elif len(nc) < 3:
                        st.warning("Username must be 3+ characters.")
                    elif len(p1) < 4:
                        st.warning("Password must be 4+ characters.")
                    elif p1 != p2:
                        st.error("Passwords don't match.")
                    else:
                        r = register_user(nc, p1)
                        if r:
                            st.success(f"Account created! Sign in as **{nc}**.")
                        else:
                            st.error("Username already taken.")
        
        st.markdown("<p style='text-align:center;color:var(--text-400);font-size:0.68rem;margin-top:1.5rem;'>Powered by Dentsu AI · Multi-Agent Architecture</p>", unsafe_allow_html=True)


# ============================================================================
# Sidebar
# ============================================================================

def render_sidebar():
    user = st.session_state["user"]
    with st.sidebar:
        st.markdown('<div class="brand-box"><div class="logo">🤖 Multi-Agent Dentsu Buddy</div><div class="sub">Agentic RAG System</div></div>', unsafe_allow_html=True)
        ini = user["display_name"][0].upper()
        st.markdown(f'<div class="user-pill"><div class="av">{ini}</div><div><div class="nm">{user["display_name"]}</div><div class="rl">{user["role"].title()}</div></div></div>', unsafe_allow_html=True)
        
        st.markdown('<hr class="sd">', unsafe_allow_html=True)
        
        # API Configuration (optional override)
        with st.expander("🔑 API Configuration (Optional)", expanded=False):
            st.caption("Override .env file credentials if needed")
            ep_val = st.text_input("Model Endpoint", value=st.session_state.get("azure_endpoint", ""),
                                   placeholder="https://your-resource.openai.azure.com/", key="input_endpoint")
            ak_val = st.text_input("Azure OpenAI API Key", value=st.session_state.get("azure_api_key", ""),
                                   placeholder="Enter your API key", key="input_api_key", type="password")
            if st.button("Save Credentials", use_container_width=True, key="save_creds"):
                if ep_val.strip() and ak_val.strip():
                    st.session_state["azure_endpoint"] = ep_val.strip()
                    st.session_state["azure_api_key"] = ak_val.strip()
                    st.session_state.pop("initialized", None)  # Force reinitialization
                    st.success("Credentials saved! Reinitializing...")
                    st.rerun()
                else:
                    st.warning("Both fields are required.")
        
        # Agent Settings
        with st.expander("⚙️ Agent Settings", expanded=False):
            st.session_state["show_trace"] = st.checkbox("Show Agent Execution Trace", value=True)
            st.session_state["enable_validation"] = st.checkbox("Enable Response Validation", value=True)
        
        if st.button("＋  New conversation", use_container_width=True, key="new_chat"):
            st.session_state["current_conv"] = None
            st.session_state["messages"] = []
            st.session_state.pop("pending_web_search", None)
            st.rerun()
        
        st.markdown('<hr class="sd">', unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.72rem;font-weight:700;color:var(--text-400);letter-spacing:0.1em;text-transform:uppercase;'>History</p>", unsafe_allow_html=True)
        
        for c in get_conversations(user["user_id"])[:20]:
            active = st.session_state.get("current_conv") == c["id"]
            cols = st.columns([6, 1])
            with cols[0]:
                if st.button(f"{'▸ ' if active else ''}{c['title'][:35]}", key=f"c_{c['id']}", use_container_width=True):
                    st.session_state["current_conv"] = c["id"]
                    st.session_state["messages"] = get_messages(c["id"])
                    st.rerun()
            with cols[1]:
                if st.button("×", key=f"d_{c['id']}"):
                    delete_conversation(c["id"])
                    if st.session_state.get("current_conv") == c["id"]:
                        st.session_state["current_conv"] = None
                        st.session_state["messages"] = []
                    st.rerun()
        
        if not get_conversations(user["user_id"]):
            st.caption("No conversations yet")
        
        st.markdown('<hr class="sd">', unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.72rem;font-weight:700;color:var(--text-400);letter-spacing:0.1em;text-transform:uppercase;'>Documents & Data</p>", unsafe_allow_html=True)
        
        if "processed_files" not in st.session_state:
            st.session_state["processed_files"] = []
        if "doc_texts" not in st.session_state:
            st.session_state["doc_texts"] = {}
        
        uploaded = st.file_uploader(
            "Upload PDF, DOCX, CSV, Excel, Images",
            type=["pdf", "docx", "txt", "png", "jpg", "jpeg", "gif", "webp", "csv", "xlsx", "xls"],
            accept_multiple_files=True,
            key="uploader"
        )
        
        if uploaded:
            for uf in uploaded:
                if uf.name not in st.session_state["processed_files"]:
                    with st.spinner(f"Reading {uf.name}..."):
                        txt = process_upload(uf)
                        if txt and not txt.startswith("["):
                            st.session_state["doc_texts"][uf.name] = txt
                            preview = txt[:500] if not (is_image_data(txt) or is_dataframe_data(txt)) else txt[:100]
                            save_doc_record(user["user_id"], uf.name, uf.name.split(".")[-1], preview)
                            st.session_state["processed_files"].append(uf.name)
                            st.rerun()
                        else:
                            st.error(f"Failed: {uf.name}")
        
        # URL input
        st.markdown('<hr class="sd">', unsafe_allow_html=True)
        url_input = st.text_input("🔗 Paste a blog/article URL", placeholder="https://example.com/article", key="url_input")
        if url_input and url_input.startswith("http"):
            if url_input not in st.session_state.get("url_docs", {}):
                with st.spinner("Loading URL content..."):
                    content = load_url_content(url_input)
                    if not content.startswith("["):
                        doc_name = store_url_as_doc(url_input, content)
                        st.success(f"Loaded: {doc_name}")
                        st.rerun()
                    else:
                        st.error(content)
        
        # Show docs
        user_docs = get_user_docs(user["user_id"])
        if user_docs:
            for d in user_docs[:10]:
                dc1, dc2 = st.columns([6, 1])
                with dc1:
                    fn_l = d["filename"].lower()
                    icon = "🖼️" if fn_l.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")) else "📊" if fn_l.endswith((".csv", ".xlsx", ".xls")) else "📄"
                    st.markdown(f'<div class="doc-item"><span class="di">{icon}</span>{d["filename"]}</div>', unsafe_allow_html=True)
                with dc2:
                    if st.button("×", key=f"deldoc_{d['id']}"):
                        fname = delete_document(d["id"], user["user_id"])
                        if fname and fname in st.session_state.get("doc_texts", {}):
                            del st.session_state["doc_texts"][fname]
                        if fname and fname in st.session_state.get("processed_files", []):
                            st.session_state["processed_files"].remove(fname)
                        st.rerun()
        
        for url, name in st.session_state.get("url_docs", {}).items():
            st.markdown(f'<div class="doc-item"><span class="di">🔗</span>{name}</div>', unsafe_allow_html=True)
        
        if not user_docs and not st.session_state.get("url_docs"):
            st.caption("No documents uploaded")
        
        st.markdown('<hr class="sd">', unsafe_allow_html=True)
        
        # Agent Architecture Info
        with st.expander("🤖 Agent Architecture"):
            st.markdown("""
            **Orchestrator**: Routes queries to specialists
            
            **Specialists**:
            - 📊 **Campaign**: Campaign database & benchmarks
            - 📚 **Research**: Marketing research PDFs
            - ⚖️ **Compliance**: Legal & regulatory guidance
            - 🌐 **Web**: Real-time news & updates
            - 📈 **Data**: CSV/Excel analysis with charts
            
            **Validator**: Quality assurance before delivery
            
            **Guardrails**: Marketing scope enforcement
            """)
        
        st.markdown('<hr class="sd">', unsafe_allow_html=True)
        if st.button("Sign out", use_container_width=True, key="logout"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
        st.markdown("<p style='color:var(--text-400);font-size:0.62rem;text-align:center;padding-top:0.8rem;'>Multi-Agent Dentsu Buddy v2.0</p>", unsafe_allow_html=True)


# ============================================================================
# Chat Interface
# ============================================================================

def render_chat():
    messages = st.session_state.get("messages", [])
    show_trace = st.session_state.get("show_trace", True)
    enable_validation = st.session_state.get("enable_validation", True)
    
    if not messages:
        st.markdown("""
        <div class="welcome-area">
            <div class="w-icon">🤖</div>
            <h2>Multi-Agent Dentsu Buddy</h2>
            <p>Ask me about marketing, advertising, or campaigns. I use multiple specialized agents to provide
            comprehensive, validated answers. Upload documents or paste URLs for additional context.</p>
        </div>
        """, unsafe_allow_html=True)
    
    for idx, msg in enumerate(messages):
        role = msg["role"]
        avatar = "🟢" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            # Clean and display content
            sources = msg.get("sources") or []
            display_text = clean_answer_urls(msg["content"], sources) if sources else msg["content"]
            st.markdown(display_text)
            
            if role == "assistant":
                # Show chart if present
                if msg.get("chart_key") and msg["chart_key"] in st.session_state.get("charts", {}):
                    st.pyplot(st.session_state["charts"][msg["chart_key"]])
                
                # Show agents used
                if msg.get("agents_used"):
                    agent_badges = " ".join([
                        f'<span class="agent-badge agent-{agent}">{agent.title()}</span>'
                        for agent in msg["agents_used"]
                    ])
                    st.markdown(f"**Agents:** {agent_badges}", unsafe_allow_html=True)
                
                # Show sources
                render_sources_and_tools(sources, msg.get("tool_calls"))
                
                # Show execution trace
                if msg.get("trace") and show_trace:
                    with st.expander("🔍 Agent Execution Trace", expanded=False):
                        for step in msg["trace"]:
                            status_icon = "✅" if step.get("status") == "pass" else "❌" if step.get("status") == "fail" else "⚠️"
                            agent_badge = f'<span class="agent-badge agent-{step.get("agent", "general")}">{step.get("agent", "").title()}</span>'
                            st.markdown(
                                f'<div class="agent-step">'
                                f'<strong>{step["step"]}</strong> {agent_badge}<br/>'
                                f'{step["action"]} {status_icon}'
                                f'</div>',
                                unsafe_allow_html=True
                            )
    
    # Chat input
    if prompt := st.chat_input("Ask me about marketing, advertising, or campaigns..."):
        user = st.session_state["user"]
        
        if not st.session_state.get("current_conv"):
            cid = create_conversation(user["user_id"], generate_title(prompt))
            st.session_state["current_conv"] = cid
        else:
            cid = st.session_state["current_conv"]
        
        save_message(cid, "user", prompt)
        st.session_state.setdefault("messages", []).append(
            {"role": "user", "content": prompt, "agents_used": None, "trace": None, "sources": None}
        )
        
        with st.chat_message("user", avatar="🟢"):
            st.markdown(prompt)
        
        # Handle pending web search confirmation
        pending = st.session_state.pop("pending_web_search", None)
        if pending:
            yes_words = {"yes", "y", "yeah", "sure", "ok", "go ahead", "yep", "please", "do it"}
            if prompt.strip().lower() in yes_words:
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("Running web search..."):
                        result = run_direct_web_search(st.session_state.llm, pending)
                    st.markdown(result["answer"])
                    if result["agents_used"]:
                        st.markdown('<span class="agent-badge agent-web">Web</span>', unsafe_allow_html=True)
                save_message(cid, "assistant", result["answer"], agents_used=result["agents_used"], trace=result["trace"])
                st.session_state["messages"].append({
                    "role": "assistant", "content": result["answer"],
                    "trace": result["trace"], "agents_used": result["agents_used"], "sources": None
                })
                return
            
            no_words = {"no", "n", "nope", "nah", "cancel"}
            if prompt.strip().lower() in no_words:
                reply = "No problem! Ask me anything about your uploaded documents or marketing-related topics."
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(reply)
                save_message(cid, "assistant", reply)
                st.session_state["messages"].append({
                    "role": "assistant", "content": reply, "sources": None, "agents_used": None, "trace": None
                })
                return
        
        # Check for URL in chat
        detected_url = detect_url(prompt)
        if detected_url and detected_url not in st.session_state.get("url_docs", {}):
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner(f"Loading content from URL..."):
                    content = load_url_content(detected_url)
                    if not content.startswith("["):
                        doc_name = store_url_as_doc(detected_url, content)
                        answer = f"I've loaded **{doc_name}** into memory. You can now ask me questions about this article!"
                    else:
                        answer = f"I couldn't load that URL: {content}"
                st.markdown(answer)
            save_message(cid, "assistant", answer)
            st.session_state["messages"].append({
                "role": "assistant", "content": answer, "sources": None,
                "agents_used": ["upload"], "trace": None, "tool_calls": "URL Content"
            })
            return
        
        # Check for data analysis queries
        image_entries, text_entries, df_entries = classify_doc_types()
        has_docs = bool(st.session_state.get("doc_texts"))
        
        is_data_query = bool(df_entries) and any(kw in prompt.lower() for kw in [
            "chart", "plot", "graph", "visuali", "bar", "pie", "line", "histogram",
            "scatter", "trend", "distribution", "average", "mean", "sum", "count",
            "total", "max", "min", "group", "compare", "analyze", "data"
        ])
        
        with st.chat_message("assistant", avatar="🤖"):
            if is_data_query:
                # Data analysis mode
                with st.status("📊 Analyzing data...", expanded=True) as status:
                    st.write("📊 Processing data analysis request...")
                    answer, chart_fig = run_dataframe_analysis(prompt, df_entries, st.session_state.llm)
                    status.update(label="✅ Analysis complete!", state="complete")
                
                st.markdown(answer)
                chart_key = None
                if chart_fig:
                    chart_key = f"chart_{uuid.uuid4().hex[:8]}"
                    st.session_state.setdefault("charts", {})[chart_key] = chart_fig
                    st.pyplot(chart_fig)
                
                st.markdown('<span class="agent-badge agent-data">Data Analysis</span>', unsafe_allow_html=True)
                
                result = {"answer": answer, "trace": [], "agents_used": ["data"], "validation_result": None}
                
                save_message(cid, "assistant", result["answer"], agents_used=result["agents_used"], trace=result["trace"])
                st.session_state["messages"].append({
                    "role": "assistant", "content": result["answer"],
                    "trace": result["trace"], "agents_used": result["agents_used"],
                    "sources": None, "chart_key": chart_key
                })
                
            elif has_docs and is_query_about_docs(prompt):
                # Query is about uploaded documents - use multi-agent with doc context
                with st.status("🤖 Multi-Agent System Processing...", expanded=True) as status:
                    conversation_history = format_conversation_history(
                        st.session_state.messages[:-1],
                        max_turns=5
                    )
                    
                    # Build uploaded file context
                    uploaded_context = ""
                    if text_entries:
                        uploaded_context += "\n\n".join([f"[Document: {fn}]\n{txt[:3000]}" for fn, txt in text_entries.items()])
                    
                    st.write("🛡️ Checking guardrails...")
                    
                    result = run_multi_agent_system(
                        st.session_state.llm,
                        st.session_state.specialist_agents,
                        prompt,
                        conversation_history=conversation_history,
                        uploaded_file_context=uploaded_context,
                        enable_validation=enable_validation
                    )
                    
                    if result["agents_used"]:
                        st.write(f"📊 Agents consulted: {', '.join(result['agents_used'])}")
                    
                    status.update(label="✅ Complete!", state="complete")
                
                st.markdown(result["answer"])
                
                if result["agents_used"]:
                    agent_badges = " ".join([
                        f'<span class="agent-badge agent-{agent}">{agent.title()}</span>'
                        for agent in result["agents_used"]
                    ])
                    st.markdown(f"**Agents:** {agent_badges}", unsafe_allow_html=True)
                
                if show_trace and result["trace"]:
                    with st.expander("🔍 Agent Execution Trace", expanded=False):
                        for step in result["trace"]:
                            status_icon = "✅" if step.get("status") == "pass" else "❌" if step.get("status") == "fail" else "⚠️"
                            agent_badge = f'<span class="agent-badge agent-{step.get("agent", "general")}">{step.get("agent", "").title()}</span>'
                            st.markdown(
                                f'<div class="agent-step">'
                                f'<strong>{step["step"]}</strong> {agent_badge}<br/>'
                                f'{step["action"]} {status_icon}'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                
                save_message(cid, "assistant", result["answer"], agents_used=result["agents_used"], trace=result["trace"])
                st.session_state["messages"].append({
                    "role": "assistant", "content": result["answer"],
                    "trace": result["trace"], "agents_used": result["agents_used"], "sources": None
                })
            
            elif has_docs:
                # Has docs but query doesn't seem related - ask user
                routing_msg = ("This query doesn't appear to be related to your uploaded documents.\n\n"
                              "Would you like me to **search the web** or use the **multi-agent system** for an answer?\n\n"
                              "Type **Yes** to proceed, or **No** to cancel.")
                st.markdown(routing_msg)
                st.session_state["pending_web_search"] = prompt
                save_message(cid, "assistant", routing_msg)
                st.session_state["messages"].append({
                    "role": "assistant", "content": routing_msg, "sources": None,
                    "agents_used": None, "trace": None, "tool_calls": "Smart Routing"
                })
            
            else:
                # No docs - use multi-agent system directly
                with st.status("🤖 Multi-Agent System Processing...", expanded=True) as status:
                    conversation_history = format_conversation_history(
                        st.session_state.messages[:-1],
                        max_turns=5
                    )
                    
                    st.write("🛡️ Checking guardrails...")
                    
                    result = run_multi_agent_system(
                        st.session_state.llm,
                        st.session_state.specialist_agents,
                        prompt,
                        conversation_history=conversation_history,
                        uploaded_file_context="",
                        enable_validation=enable_validation
                    )
                    
                    if result["agents_used"]:
                        st.write(f"📊 Agents consulted: {', '.join(result['agents_used'])}")
                    
                    status.update(label="✅ Complete!", state="complete")
                
                st.markdown(result["answer"])
                
                if result["agents_used"]:
                    agent_badges = " ".join([
                        f'<span class="agent-badge agent-{agent}">{agent.title()}</span>'
                        for agent in result["agents_used"]
                    ])
                    st.markdown(f"**Agents:** {agent_badges}", unsafe_allow_html=True)
                
                if show_trace and result["trace"]:
                    with st.expander("🔍 Agent Execution Trace", expanded=False):
                        for step in result["trace"]:
                            status_icon = "✅" if step.get("status") == "pass" else "❌" if step.get("status") == "fail" else "⚠️"
                            agent_badge = f'<span class="agent-badge agent-{step.get("agent", "general")}">{step.get("agent", "").title()}</span>'
                            st.markdown(
                                f'<div class="agent-step">'
                                f'<strong>{step["step"]}</strong> {agent_badge}<br/>'
                                f'{step["action"]} {status_icon}'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                
                save_message(cid, "assistant", result["answer"], agents_used=result["agents_used"], trace=result["trace"])
                st.session_state["messages"].append({
                    "role": "assistant", "content": result["answer"],
                    "trace": result["trace"], "agents_used": result["agents_used"], "sources": None
                })


# ============================================================================
# Main Application
# ============================================================================

def main():
    init_database()
    seed_defaults()
    
    if not st.session_state.get("authenticated"):
        render_login()
        return
    
    render_sidebar()
    
    # Initialize models and tools
    if not st.session_state.get("initialized"):
        try:
            with st.spinner("Loading multi-agent system..."):
                credentials = load_credentials()
                
                if not credentials["api_key"]:
                    st.error("❌ API key not found. Please create a .env file with your Azure OpenAI credentials.")
                    return
                
                llm, embeddings = initialize_models(credentials)
                
                tools_dict = {}
                
                campaign_tool = setup_campaign_db(embeddings)
                if campaign_tool:
                    tools_dict["campaign"] = campaign_tool
                
                pdf_tool = setup_pdf_db(embeddings)
                if pdf_tool:
                    tools_dict["research"] = pdf_tool
                
                web_articles_tool = setup_web_articles_db(embeddings)
                if web_articles_tool:
                    tools_dict["compliance"] = web_articles_tool
                
                web_search_tool = create_web_search_tool()
                tools_dict["web"] = web_search_tool
                
                specialist_agents = build_specialist_agents(llm, tools_dict)
                
                st.session_state.initialized = True
                st.session_state.llm = llm
                st.session_state.specialist_agents = specialist_agents
                st.session_state.tools_dict = tools_dict
                
        except Exception as e:
            st.error(f"❌ Error initializing: {str(e)}")
            st.info("Make sure your .env file is configured correctly.")
            return
    
    render_chat()


if __name__ == "__main__":
    main()
