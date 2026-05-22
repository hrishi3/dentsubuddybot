"""
Chef AI — Smart Recipe & Meal Planning Assistant
=================================================
A multi-model AI application for:
  1. Recipe Generation from available ingredients
  2. Weekly Meal Planning with dietary preferences
  3. Nutritional Analysis of meals
  4. Smart Shopping List generation
  5. Food Image Recognition
  6. Cooking Tips & Substitutions

Models Used:
  - GPT-4o: Recipe generation, meal planning, nutrition advice
  - GPT-4o Vision: Food image recognition
  - Code Generation: Nutrition calculations

Tech Stack:
  - Streamlit, LangChain, Azure OpenAI
  - SQLite for user data & saved recipes
  - pandas for nutrition calculations
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
from datetime import datetime, timedelta
from warnings import filterwarnings

filterwarnings("ignore")

st.set_page_config(
    page_title="Chef AI - Recipe Assistant",
    page_icon="👨‍🍳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS - Warm Kitchen Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --primary: #E85D04;
    --primary-dim: rgba(232,93,4,0.10);
    --primary-border: rgba(232,93,4,0.18);
    --primary-glow: rgba(232,93,4,0.25);
    --accent: #2EC4B6;
    --accent-dim: rgba(46,196,182,0.10);
    --bg-root: #1A1A2E;
    --bg-surface: #16213E;
    --bg-card: #1F2B47;
    --bg-elevated: #2A3A5C;
    --text-100: #F8F9FA;
    --text-200: #E9ECEF;
    --text-300: #ADB5BD;
    --text-400: #6C757D;
    --border: rgba(255,255,255,0.08);
    --border-focus: rgba(232,93,4,0.45);
    --gold: #FFD700;
    --green: #40C057;
    --red: #FA5252;
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --radius-xl: 24px;
}

html, body, .stApp {
    background: var(--bg-root) !important;
    font-family: 'Poppins', -apple-system, BlinkMacSystemFont, sans-serif !important;
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
    color: var(--text-200) !important; font-family: 'Poppins', sans-serif !important; font-size: 0.85rem !important;
}

.brand-box {
    padding: 1.5rem 1rem 1.2rem 1rem; text-align: center;
    border-bottom: 1px solid var(--border); margin-bottom: 1rem;
}
.brand-box .logo {
    font-size: 1.25rem; font-weight: 700; color: var(--primary);
    letter-spacing: 0.05em;
    display: flex; align-items: center; justify-content: center; gap: 0.5rem;
}
.brand-box .sub {
    font-size: 0.65rem; color: var(--text-400);
    letter-spacing: 0.12em; text-transform: uppercase; margin-top: 4px;
}

.user-pill {
    display: flex; align-items: center; gap: 0.6rem;
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius-sm); padding: 0.55rem 0.75rem; margin: 0.6rem 0;
}
.user-pill .av {
    width: 32px; height: 32px; border-radius: 50%;
    background: linear-gradient(135deg, var(--primary), var(--gold));
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.8rem; color: #fff; flex-shrink: 0;
}
.user-pill .nm { font-weight: 600; font-size: 0.85rem; color: var(--text-100); }
.user-pill .rl { font-size: 0.7rem; color: var(--text-400); }

.sd { border: none; border-top: 1px solid var(--border); margin: 0.9rem 0; }

/* Mode Selector */
.mode-btn {
    display: flex; align-items: center; gap: 0.5rem;
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius-sm); padding: 0.6rem 0.8rem; margin: 0.3rem 0;
    cursor: pointer; transition: all 0.2s; width: 100%;
}
.mode-btn:hover { border-color: var(--primary-border); background: var(--bg-elevated); }
.mode-btn.active { border-color: var(--primary); background: var(--primary-dim); }
.mode-btn .m-icon { font-size: 1.1rem; }
.mode-btn .m-name { font-size: 0.8rem; font-weight: 500; color: var(--text-100); }

/* Login */
.login-brand {
    text-align: center; margin-bottom: 2rem;
}
.login-brand .lb-icon {
    width: 72px; height: 72px; border-radius: 22px;
    background: linear-gradient(135deg, var(--primary), var(--gold));
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 2rem; margin-bottom: 1rem; box-shadow: 0 8px 24px rgba(232,93,4,0.3);
}
.login-brand h1 {
    font-size: 1.7rem; font-weight: 700; color: var(--text-100);
    letter-spacing: 0.03em; margin: 0 0 0.15rem 0;
}
.login-brand h1 span { color: var(--gold); }
.login-brand .lb-tag {
    font-size: 0.7rem; color: var(--text-400); letter-spacing: 0.18em;
    text-transform: uppercase; margin-top: 0.2rem;
}
.login-brand .lb-desc {
    font-size: 0.85rem; color: var(--text-300); margin-top: 0.8rem;
    line-height: 1.6; max-width: 380px; margin-left: auto; margin-right: auto;
}
.login-features {
    display: flex; justify-content: center; gap: 1rem; margin-top: 1.2rem; flex-wrap: wrap;
}
.login-feat {
    display: flex; align-items: center; gap: 0.35rem;
    font-size: 0.72rem; color: var(--text-300);
    background: var(--bg-elevated); border: 1px solid var(--border);
    border-radius: 20px; padding: 0.35rem 0.75rem;
}
.login-feat .lf-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); flex-shrink: 0; }

/* Chat */
[data-testid="stChatMessage"] {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important; padding: 0.9rem 1.1rem !important;
    margin-bottom: 0.6rem !important; color: var(--text-100) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(135deg, rgba(232,93,4,0.12), rgba(232,93,4,0.05)) !important;
    border: 1px solid var(--primary-border) !important;
}
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] td, [data-testid="stChatMessage"] span {
    color: var(--text-100) !important; font-family: 'Poppins', sans-serif !important;
}
[data-testid="stChatMessage"] h1, [data-testid="stChatMessage"] h2, [data-testid="stChatMessage"] h3 {
    color: var(--text-100) !important; font-weight: 600 !important;
}
[data-testid="stChatMessage"] code {
    background: rgba(255,255,255,0.08) !important; color: #FFD93D !important;
    font-family: 'JetBrains Mono', monospace !important; padding: 0.15em 0.4em !important;
}
[data-testid="stChatMessage"] a { color: var(--accent) !important; }

/* Recipe Card */
.recipe-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius-lg); padding: 1.2rem; margin: 0.5rem 0;
}
.recipe-card .rc-title {
    font-size: 1.1rem; font-weight: 600; color: var(--primary);
    margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.4rem;
}
.recipe-card .rc-meta {
    display: flex; gap: 1rem; margin-bottom: 0.8rem; flex-wrap: wrap;
}
.recipe-card .rc-tag {
    display: inline-flex; align-items: center; gap: 0.25rem;
    font-size: 0.72rem; color: var(--text-300);
    background: var(--bg-elevated); border-radius: 15px; padding: 0.25rem 0.6rem;
}
.recipe-card .rc-tag .rt-icon { font-size: 0.75rem; }

/* Nutrition Badge */
.nutr-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.6rem;
    margin: 0.8rem 0;
}
.nutr-item {
    background: var(--bg-elevated); border-radius: var(--radius-sm);
    padding: 0.6rem; text-align: center;
}
.nutr-item .ni-val { font-size: 1.1rem; font-weight: 700; color: var(--accent); }
.nutr-item .ni-label { font-size: 0.65rem; color: var(--text-400); text-transform: uppercase; letter-spacing: 0.08em; }

/* Welcome Area */
.welcome-area { text-align: center; padding: 8vh 2rem 3rem 2rem; }
.welcome-area .w-icon {
    width: 80px; height: 80px; border-radius: 24px;
    background: linear-gradient(135deg, var(--primary), var(--gold));
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 2.2rem; margin-bottom: 1.2rem; box-shadow: 0 12px 32px rgba(232,93,4,0.3);
}
.welcome-area h2 { font-size: 1.5rem; font-weight: 600; color: var(--text-100); margin-bottom: 0.5rem; }
.welcome-area p { font-size: 0.9rem; color: var(--text-400); max-width: 520px; margin: 0 auto; line-height: 1.65; }

/* Tech Cards */
.tech-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem; margin-top: 1.5rem; max-width: 750px; margin-left: auto; margin-right: auto;
}
.tech-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius-md); padding: 1rem; text-align: center; transition: all 0.2s;
}
.tech-card:hover { border-color: var(--primary-border); transform: translateY(-3px); }
.tech-card .tc-icon { font-size: 1.6rem; margin-bottom: 0.5rem; }
.tech-card .tc-title { font-size: 0.88rem; font-weight: 600; color: var(--text-100); margin-bottom: 0.2rem; }
.tech-card .tc-desc { font-size: 0.7rem; color: var(--text-400); line-height: 1.4; }

.tech-badges {
    display: flex; flex-wrap: wrap; justify-content: center; gap: 0.5rem; margin-top: 1.5rem;
}
.tech-badge {
    display: inline-flex; align-items: center; gap: 0.3rem;
    background: var(--bg-elevated); border: 1px solid var(--border);
    border-radius: 20px; padding: 0.3rem 0.7rem; font-size: 0.7rem; color: var(--text-300);
}
.tech-badge .tb-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); }

/* Ingredient Tags */
.ing-tags { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.5rem 0; }
.ing-tag {
    background: var(--accent-dim); border: 1px solid rgba(46,196,182,0.2);
    border-radius: 15px; padding: 0.25rem 0.6rem;
    font-size: 0.72rem; color: var(--accent);
}

/* Saved Recipes */
.saved-recipe {
    display: flex; align-items: center; gap: 0.5rem;
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 6px; padding: 0.4rem 0.6rem; margin: 0.25rem 0;
    font-size: 0.75rem; color: var(--text-200); cursor: pointer; transition: all 0.15s;
}
.saved-recipe:hover { border-color: var(--primary-border); }
.saved-recipe .sr-icon { color: var(--gold); }

/* Input styling */
.stTextInput > div > div > input, .stTextArea > div > div > textarea {
    background: var(--bg-elevated) !important; border: 1px solid var(--border) !important;
    color: var(--text-100) !important; border-radius: var(--radius-sm) !important;
}
.stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
    border-color: var(--border-focus) !important;
}
.stButton > button {
    font-family: 'Poppins', sans-serif !important; font-weight: 600 !important;
    border-radius: var(--radius-sm) !important; transition: all 0.2s !important;
}
form .stButton > button {
    background: linear-gradient(135deg, var(--primary), #FF7B00) !important;
    color: #fff !important; border: none !important;
}
form .stButton > button:hover { box-shadow: 0 4px 16px var(--primary-glow) !important; }
section[data-testid="stSidebar"] .stButton > button {
    background: var(--bg-card) !important; color: var(--text-200) !important;
    border: 1px solid var(--border) !important;
}
.stChatInput > div { background: var(--bg-card) !important; border: 1px solid var(--border) !important; }
.stChatInput textarea { color: var(--text-100) !important; }
.stSpinner > div > div { border-top-color: var(--accent) !important; }
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: var(--text-300) !important;
    font-weight: 600 !important; font-size: 0.85rem !important;
}
.stTabs [aria-selected="true"] { color: var(--primary) !important; border-bottom-color: var(--primary) !important; }
.stSelectbox > div > div { background: var(--bg-elevated) !important; }
.stMultiSelect > div > div { background: var(--bg-elevated) !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--bg-elevated); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════

DB_PATH = "chef_ai.db"

def init_database():
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY, username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL, display_name TEXT,
        dietary_prefs TEXT, allergies TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS recipes (
        recipe_id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
        title TEXT NOT NULL, ingredients TEXT, instructions TEXT,
        prep_time INTEGER, cook_time INTEGER, servings INTEGER,
        calories INTEGER, protein REAL, carbs REAL, fat REAL,
        cuisine TEXT, meal_type TEXT, tags TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS meal_plans (
        plan_id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
        week_start DATE, plan_data TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS shopping_lists (
        list_id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
        title TEXT, items TEXT, checked_items TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS conversations (
        conv_id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
        mode TEXT, title TEXT, messages TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()

def _hash(pw): return hashlib.sha256(f"chef_ai_2026_{pw}".encode()).hexdigest()

def register_user(username, password, dietary_prefs="", allergies=""):
    uid = str(uuid.uuid4()); display = username.strip().title()
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("INSERT INTO users VALUES (?,?,?,?,?,?,CURRENT_TIMESTAMP)",
                     (uid, username.lower().strip(), _hash(password), display, dietary_prefs, allergies))
        conn.commit(); conn.close()
        return {"user_id": uid, "username": username.lower().strip(), "display_name": display,
                "dietary_prefs": dietary_prefs, "allergies": allergies}
    except sqlite3.IntegrityError:
        conn.close(); return None

def authenticate(username, password):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT user_id,username,display_name,dietary_prefs,allergies FROM users WHERE username=? AND password_hash=?",
                       (username.lower().strip(), _hash(password))).fetchone()
    conn.close()
    return {"user_id": row[0], "username": row[1], "display_name": row[2],
            "dietary_prefs": row[3] or "", "allergies": row[4] or ""} if row else None

def seed_defaults():
    conn = sqlite3.connect(DB_PATH)
    if conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
        conn.execute("INSERT OR IGNORE INTO users VALUES (?,?,?,?,?,?,CURRENT_TIMESTAMP)",
                     (str(uuid.uuid4()), "chef", _hash("chef123"), "Chef", "", ""))
    conn.commit(); conn.close()

def save_recipe(uid, title, ingredients, instructions, prep_time, cook_time, servings,
                calories, protein, carbs, fat, cuisine, meal_type, tags):
    rid = str(uuid.uuid4()); conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO recipes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)",
                 (rid, uid, title, json.dumps(ingredients), json.dumps(instructions),
                  prep_time, cook_time, servings, calories, protein, carbs, fat, cuisine, meal_type, json.dumps(tags)))
    conn.commit(); conn.close(); return rid

def get_user_recipes(uid, limit=20):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT recipe_id,title,cuisine,meal_type,calories,prep_time,cook_time FROM recipes WHERE user_id=? ORDER BY created_at DESC LIMIT ?", (uid, limit)).fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1], "cuisine": r[2], "meal_type": r[3], "calories": r[4], "prep_time": r[5], "cook_time": r[6]} for r in rows]

def get_recipe(rid):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT * FROM recipes WHERE recipe_id=?", (rid,)).fetchone()
    conn.close()
    if not row: return None
    return {
        "id": row[0], "user_id": row[1], "title": row[2],
        "ingredients": json.loads(row[3]) if row[3] else [],
        "instructions": json.loads(row[4]) if row[4] else [],
        "prep_time": row[5], "cook_time": row[6], "servings": row[7],
        "calories": row[8], "protein": row[9], "carbs": row[10], "fat": row[11],
        "cuisine": row[12], "meal_type": row[13],
        "tags": json.loads(row[14]) if row[14] else []
    }

def delete_recipe(rid, uid):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM recipes WHERE recipe_id=? AND user_id=?", (rid, uid))
    conn.commit(); conn.close()

def update_user_prefs(uid, dietary_prefs, allergies):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE users SET dietary_prefs=?, allergies=? WHERE user_id=?", (dietary_prefs, allergies, uid))
    conn.commit(); conn.close()


# ═══════════════════════════════════════════════
# IMAGE PROCESSING
# ═══════════════════════════════════════════════

def extract_image(f):
    try:
        f.seek(0); data = f.read()
        if not data: return None
        b64 = base64.b64encode(data).decode("utf-8")
        ext = f.name.rsplit(".", 1)[-1].lower()
        mime_map = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif", "webp": "image/webp"}
        return {"mime": mime_map.get(ext, "image/png"), "b64": b64}
    except: return None


# ═══════════════════════════════════════════════
# LLM HELPER
# ═══════════════════════════════════════════════

def get_llm(max_tokens=2000, temperature=0.7):
    from langchain_openai import AzureChatOpenAI
    return AzureChatOpenAI(
        azure_deployment="gpt-4o", api_version="2024-12-01-preview",
        azure_endpoint=st.session_state.get("azure_endpoint", ""),
        api_key=st.session_state.get("azure_api_key", ""),
        temperature=temperature, max_tokens=max_tokens
    )


# ═══════════════════════════════════════════════
# AI MODES
# ═══════════════════════════════════════════════

MODES = {
    "recipe": {"name": "Recipe Generator", "icon": "🍳", "desc": "Create recipes from ingredients"},
    "meal_plan": {"name": "Meal Planner", "icon": "📅", "desc": "Plan your weekly meals"},
    "nutrition": {"name": "Nutrition Analyzer", "icon": "🥗", "desc": "Analyze nutritional content"},
    "shopping": {"name": "Shopping Assistant", "icon": "🛒", "desc": "Generate shopping lists"},
    "tips": {"name": "Cooking Tips", "icon": "💡", "desc": "Get cooking advice & substitutions"},
    "identify": {"name": "Food Identifier", "icon": "📸", "desc": "Identify food from images"},
}


def run_recipe_generator(prompt, user):
    """Generate recipes based on ingredients or preferences."""
    from langchain_core.messages import HumanMessage, SystemMessage
    
    dietary = user.get("dietary_prefs", "")
    allergies = user.get("allergies", "")
    
    system_prompt = f"""You are Chef AI, a world-class culinary assistant. Generate delicious, practical recipes.

User's Dietary Preferences: {dietary if dietary else "None specified"}
User's Allergies/Restrictions: {allergies if allergies else "None specified"}

When generating recipes, always include:
1. Recipe title with emoji
2. Brief description
3. Prep time, cook time, servings
4. Complete ingredient list with quantities
5. Step-by-step instructions
6. Nutritional info (approximate calories, protein, carbs, fat per serving)
7. Chef's tips or variations

Format output with clear markdown. Be creative but practical. Respect dietary restrictions."""

    try:
        llm = get_llm(max_tokens=2000, temperature=0.8)
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ])
        return response.content if response.content else "I couldn't generate a recipe. Please try again."
    except Exception as e:
        return f"Error generating recipe: {str(e)}"


def run_meal_planner(prompt, user):
    """Create weekly meal plans."""
    from langchain_core.messages import HumanMessage, SystemMessage
    
    dietary = user.get("dietary_prefs", "")
    allergies = user.get("allergies", "")
    today = datetime.now().strftime("%A, %B %d, %Y")
    
    system_prompt = f"""You are Chef AI, a meal planning expert. Create balanced, varied weekly meal plans.

Today's Date: {today}
User's Dietary Preferences: {dietary if dietary else "None specified"}
User's Allergies/Restrictions: {allergies if allergies else "None specified"}

When creating meal plans:
1. Plan for 7 days (Monday-Sunday)
2. Include breakfast, lunch, dinner, and snacks
3. Ensure nutritional balance across the week
4. Consider meal prep efficiency (use similar ingredients across meals)
5. Include variety in cuisines and cooking methods
6. Provide estimated daily calorie totals
7. Note which meals can be prepped ahead

Format as a clear, organized markdown table or list. Be practical and family-friendly."""

    try:
        llm = get_llm(max_tokens=3000, temperature=0.7)
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ])
        return response.content if response.content else "I couldn't create a meal plan. Please try again."
    except Exception as e:
        return f"Error creating meal plan: {str(e)}"


def run_nutrition_analyzer(prompt, user, image_data=None):
    """Analyze nutritional content of foods or meals."""
    from langchain_core.messages import HumanMessage, SystemMessage
    
    system_prompt = """You are Chef AI, a nutrition expert. Analyze foods and provide detailed nutritional information.

When analyzing nutrition:
1. Provide calorie count (kcal)
2. Macronutrients: Protein (g), Carbohydrates (g), Fat (g), Fiber (g)
3. Key micronutrients present
4. Health benefits and considerations
5. Portion size recommendations
6. Healthier alternatives if applicable
7. Dietary category (keto-friendly, vegan, gluten-free, etc.)

Use tables for nutritional data. Be accurate and evidence-based."""

    try:
        llm = get_llm(max_tokens=1500, temperature=0.3)
        
        if image_data:
            content = [
                {"type": "text", "text": f"{prompt}\n\nAnalyze the nutritional content of the food in this image."},
                {"type": "image_url", "image_url": {"url": f"data:{image_data['mime']};base64,{image_data['b64']}", "detail": "high"}}
            ]
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=content)
            ])
        else:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ])
        return response.content if response.content else "I couldn't analyze the nutrition. Please try again."
    except Exception as e:
        return f"Error analyzing nutrition: {str(e)}"


def run_shopping_assistant(prompt, user):
    """Generate organized shopping lists."""
    from langchain_core.messages import HumanMessage, SystemMessage
    
    system_prompt = """You are Chef AI, a shopping assistant. Create organized, efficient shopping lists.

When generating shopping lists:
1. Organize items by store section (Produce, Dairy, Meat, Pantry, Frozen, etc.)
2. Include quantities needed
3. Suggest best quality indicators for fresh items
4. Note any seasonal availability
5. Include estimated costs when possible
6. Suggest store brand vs name brand options
7. Add meal prep tips for purchased items

Format as a clear checklist with emojis for sections. Be thorough but organized."""

    try:
        llm = get_llm(max_tokens=2000, temperature=0.5)
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ])
        return response.content if response.content else "I couldn't generate a shopping list. Please try again."
    except Exception as e:
        return f"Error generating shopping list: {str(e)}"


def run_cooking_tips(prompt, user):
    """Provide cooking tips, techniques, and ingredient substitutions."""
    from langchain_core.messages import HumanMessage, SystemMessage
    
    dietary = user.get("dietary_prefs", "")
    allergies = user.get("allergies", "")
    
    system_prompt = f"""You are Chef AI, a culinary expert with knowledge of cooking techniques worldwide.

User's Dietary Preferences: {dietary if dietary else "None specified"}
User's Allergies/Restrictions: {allergies if allergies else "None specified"}

Provide helpful cooking advice including:
1. Cooking techniques and methods
2. Ingredient substitutions (especially for dietary restrictions)
3. Kitchen equipment recommendations
4. Food storage and safety tips
5. Flavor pairing suggestions
6. Time-saving cooking hacks
7. Common mistake fixes

Be practical, detailed, and encouraging. Share professional chef secrets when relevant."""

    try:
        llm = get_llm(max_tokens=1500, temperature=0.7)
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ])
        return response.content if response.content else "I couldn't provide tips. Please try again."
    except Exception as e:
        return f"Error: {str(e)}"


def run_food_identifier(prompt, image_data, user):
    """Identify food from images and provide recipe suggestions."""
    from langchain_core.messages import HumanMessage, SystemMessage
    
    if not image_data:
        return "Please upload a food image to identify."
    
    system_prompt = """You are Chef AI with expert food recognition abilities.

When identifying food:
1. Identify all visible dishes, ingredients, and foods
2. Estimate portion sizes
3. Provide the likely cuisine/origin
4. Suggest similar recipes to recreate the dish
5. List key ingredients you can identify
6. Estimate nutritional content
7. Rate the dish's apparent quality/presentation

Be detailed and accurate. If uncertain, provide your best assessment with confidence levels."""

    try:
        llm = get_llm(max_tokens=1500, temperature=0.5)
        content = [
            {"type": "text", "text": prompt if prompt else "Identify this food and provide details."},
            {"type": "image_url", "image_url": {"url": f"data:{image_data['mime']};base64,{image_data['b64']}", "detail": "high"}}
        ]
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=content)
        ])
        return response.content if response.content else "I couldn't identify the food. Please try again."
    except Exception as e:
        return f"Error identifying food: {str(e)}"


# ═══════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════

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
            <div class="lb-icon">👨‍🍳</div>
            <h1>Chef <span>AI</span></h1>
            <div class="lb-tag">Smart Recipe & Meal Assistant</div>
            <div class="lb-desc">
                Your AI-powered culinary companion — generate recipes, plan meals,
                analyze nutrition, and master cooking techniques.
            </div>
            <div class="login-features">
                <div class="login-feat"><span class="lf-dot"></span> Recipe Generation</div>
                <div class="login-feat"><span class="lf-dot"></span> Meal Planning</div>
                <div class="login-feat"><span class="lf-dot"></span> Nutrition Analysis</div>
                <div class="login-feat"><span class="lf-dot"></span> Food Recognition</div>
                <div class="login-feat"><span class="lf-dot"></span> Shopping Lists</div>
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
                            st.session_state.update(authenticated=True, user=user, current_mode="recipe", messages=[])
                            st.rerun()
                        else: st.error("Invalid credentials.")
                    else: st.warning("Please fill in both fields.")
        with t2:
            with st.form("signup_form", clear_on_submit=True):
                nu = st.text_input("Choose a username", placeholder="e.g. homechef", key="su")
                p1 = st.text_input("Create password", type="password", placeholder="Min 4 characters", key="sp1")
                p2 = st.text_input("Confirm password", type="password", placeholder="Re-enter password", key="sp2")
                dp = st.multiselect("Dietary Preferences", ["Vegetarian", "Vegan", "Pescatarian", "Keto", "Paleo", "Gluten-Free", "Dairy-Free", "Low-Carb", "Mediterranean"])
                al = st.text_input("Allergies (comma-separated)", placeholder="e.g. peanuts, shellfish", key="allergy")
                
                if st.form_submit_button("Create Account", use_container_width=True):
                    nc = (nu or "").strip()
                    if not nc or not p1: st.warning("Username and password required.")
                    elif len(nc) < 3: st.warning("Username must be 3+ characters.")
                    elif len(p1) < 4: st.warning("Password must be 4+ characters.")
                    elif p1 != p2: st.error("Passwords don't match.")
                    else:
                        r = register_user(nc, p1, ", ".join(dp), al)
                        if r: st.success(f"Account created! Sign in as **{nc}**.")
                        else: st.error("Username already taken.")
        
        st.markdown("<p style='text-align:center;color:var(--text-400);font-size:0.7rem;margin-top:1.5rem;'>Powered by Chef AI · GPT-4o Vision</p>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════

def render_sidebar():
    user = st.session_state["user"]
    with st.sidebar:
        st.markdown('<div class="brand-box"><div class="logo">👨‍🍳 Chef AI</div><div class="sub">Recipe & Meal Assistant</div></div>', unsafe_allow_html=True)
        ini = user["display_name"][0].upper()
        st.markdown(f'<div class="user-pill"><div class="av">{ini}</div><div><div class="nm">{user["display_name"]}</div><div class="rl">Home Chef</div></div></div>', unsafe_allow_html=True)
        
        st.markdown('<hr class="sd">', unsafe_allow_html=True)
        
        # API Configuration
        with st.expander("🔑 API Configuration", expanded=not st.session_state.get("azure_endpoint")):
            ep_val = st.text_input("Azure Endpoint", value=st.session_state.get("azure_endpoint", ""),
                                   placeholder="https://your-resource.openai.azure.com/", key="input_endpoint")
            ak_val = st.text_input("API Key", value=st.session_state.get("azure_api_key", ""),
                                   placeholder="Enter your API key", key="input_api_key", type="password")
            if st.button("Save", use_container_width=True, key="save_creds"):
                if ep_val.strip() and ak_val.strip():
                    st.session_state["azure_endpoint"] = ep_val.strip()
                    st.session_state["azure_api_key"] = ak_val.strip()
                    st.success("Saved!"); st.rerun()
                else: st.warning("Both fields required.")
        
        st.markdown('<hr class="sd">', unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.72rem;font-weight:700;color:var(--text-400);letter-spacing:0.1em;text-transform:uppercase;'>AI Mode</p>", unsafe_allow_html=True)
        
        # Mode Selection
        current_mode = st.session_state.get("current_mode", "recipe")
        for mode_id, mode_info in MODES.items():
            is_active = mode_id == current_mode
            if st.button(f"{mode_info['icon']} {mode_info['name']}", key=f"mode_{mode_id}", use_container_width=True,
                        type="primary" if is_active else "secondary"):
                st.session_state["current_mode"] = mode_id
                st.session_state["messages"] = []
                st.rerun()
        
        st.markdown('<hr class="sd">', unsafe_allow_html=True)
        
        # Dietary Preferences
        with st.expander("🥬 Dietary Preferences", expanded=False):
            dietary = st.multiselect("Preferences", 
                ["Vegetarian", "Vegan", "Pescatarian", "Keto", "Paleo", "Gluten-Free", "Dairy-Free", "Low-Carb"],
                default=[x.strip() for x in user.get("dietary_prefs", "").split(",") if x.strip()],
                key="diet_pref")
            allergies = st.text_input("Allergies", value=user.get("allergies", ""), key="allergy_input")
            if st.button("Update Preferences", use_container_width=True):
                update_user_prefs(user["user_id"], ", ".join(dietary), allergies)
                st.session_state["user"]["dietary_prefs"] = ", ".join(dietary)
                st.session_state["user"]["allergies"] = allergies
                st.success("Updated!")
        
        st.markdown('<hr class="sd">', unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.72rem;font-weight:700;color:var(--text-400);letter-spacing:0.1em;text-transform:uppercase;'>Saved Recipes</p>", unsafe_allow_html=True)
        
        # Saved Recipes
        recipes = get_user_recipes(user["user_id"], limit=10)
        if recipes:
            for r in recipes:
                cols = st.columns([6, 1])
                with cols[0]:
                    if st.button(f"🍽️ {r['title'][:25]}", key=f"rec_{r['id']}", use_container_width=True):
                        recipe = get_recipe(r['id'])
                        if recipe:
                            st.session_state["selected_recipe"] = recipe
                with cols[1]:
                    if st.button("×", key=f"del_{r['id']}"):
                        delete_recipe(r['id'], user["user_id"])
                        st.rerun()
        else:
            st.caption("No saved recipes yet")
        
        st.markdown('<hr class="sd">', unsafe_allow_html=True)
        
        # Models & Tools Info
        with st.expander("🛠️ Models & Tools", expanded=False):
            st.markdown("""
**🤖 AI Model**
- **GPT-4o** (Azure OpenAI)
- Vision-capable for food recognition
- Temperature: Varies by mode

**🔧 Features**
| Mode | Function |
|------|----------|
| 🍳 Recipe | Generate recipes |
| 📅 Meal Plan | Weekly planning |
| 🥗 Nutrition | Analyze foods |
| 🛒 Shopping | Create lists |
| 💡 Tips | Cooking advice |
| 📸 Identify | Food recognition |
            """)
        
        st.markdown('<hr class="sd">', unsafe_allow_html=True)
        if st.button("Sign out", use_container_width=True, key="logout"):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()
        st.markdown("<p style='color:var(--text-400);font-size:0.62rem;text-align:center;padding-top:0.8rem;'>Chef AI v1.0</p>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# MAIN CHAT
# ═══════════════════════════════════════════════

def render_chat():
    if not st.session_state.get("azure_endpoint") or not st.session_state.get("azure_api_key"):
        st.markdown("""
        <div class="welcome-area">
            <div class="w-icon">🔑</div>
            <h2>Configure API Credentials</h2>
            <p>Open <b>🔑 API Configuration</b> in the sidebar to get started.</p>
        </div>""", unsafe_allow_html=True)
        return
    
    current_mode = st.session_state.get("current_mode", "recipe")
    mode_info = MODES.get(current_mode, MODES["recipe"])
    messages = st.session_state.get("messages", [])
    user = st.session_state["user"]
    
    # Show selected recipe if any
    if st.session_state.get("selected_recipe"):
        recipe = st.session_state.pop("selected_recipe")
        st.markdown(f"""
        <div class="recipe-card">
            <div class="rc-title">🍽️ {recipe['title']}</div>
            <div class="rc-meta">
                <span class="rc-tag"><span class="rt-icon">⏱️</span> Prep: {recipe['prep_time']}min</span>
                <span class="rc-tag"><span class="rt-icon">🍳</span> Cook: {recipe['cook_time']}min</span>
                <span class="rc-tag"><span class="rt-icon">👥</span> Serves: {recipe['servings']}</span>
                <span class="rc-tag"><span class="rt-icon">🌍</span> {recipe['cuisine']}</span>
            </div>
            <div class="nutr-grid">
                <div class="nutr-item"><div class="ni-val">{recipe['calories']}</div><div class="ni-label">Calories</div></div>
                <div class="nutr-item"><div class="ni-val">{recipe['protein']}g</div><div class="ni-label">Protein</div></div>
                <div class="nutr-item"><div class="ni-val">{recipe['carbs']}g</div><div class="ni-label">Carbs</div></div>
                <div class="nutr-item"><div class="ni-val">{recipe['fat']}g</div><div class="ni-label">Fat</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("**Ingredients:**")
        for ing in recipe.get('ingredients', []):
            st.markdown(f"- {ing}")
        st.markdown("**Instructions:**")
        for i, step in enumerate(recipe.get('instructions', []), 1):
            st.markdown(f"{i}. {step}")
        st.markdown("---")
    
    # Welcome message for empty chat
    if not messages:
        st.markdown(f"""
        <div class="welcome-area">
            <div class="w-icon">{mode_info['icon']}</div>
            <h2>{mode_info['name']}</h2>
            <p>{get_mode_welcome(current_mode)}</p>
        </div>
        
        <div class="tech-grid">
            <div class="tech-card">
                <div class="tc-icon">🧠</div>
                <div class="tc-title">GPT-4o</div>
                <div class="tc-desc">Advanced language model with culinary expertise</div>
            </div>
            <div class="tech-card">
                <div class="tc-icon">👁️</div>
                <div class="tc-title">Vision AI</div>
                <div class="tc-desc">Identify foods from photos</div>
            </div>
            <div class="tech-card">
                <div class="tc-icon">📊</div>
                <div class="tc-title">Nutrition DB</div>
                <div class="tc-desc">Accurate nutritional calculations</div>
            </div>
            <div class="tech-card">
                <div class="tc-icon">🗄️</div>
                <div class="tc-title">Recipe Storage</div>
                <div class="tc-desc">Save and organize your favorites</div>
            </div>
        </div>
        
        <div class="tech-badges">
            <span class="tech-badge"><span class="tb-dot"></span>LangChain</span>
            <span class="tech-badge"><span class="tb-dot"></span>Streamlit</span>
            <span class="tech-badge"><span class="tb-dot"></span>Azure OpenAI</span>
            <span class="tech-badge"><span class="tb-dot"></span>SQLite</span>
            <span class="tech-badge"><span class="tb-dot"></span>GPT-4o Vision</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Display messages
    for msg in messages:
        avatar = "🧑‍🍳" if msg["role"] == "user" else "👨‍🍳"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
    
    # Image upload for vision modes
    uploaded_image = None
    if current_mode in ["nutrition", "identify"]:
        uploaded_image = st.file_uploader("📸 Upload food image (optional)", type=["png", "jpg", "jpeg", "webp"], key="food_img")
    
    # Chat input
    placeholder = get_mode_placeholder(current_mode)
    if prompt := st.chat_input(placeholder, key="chat_input"):
        # Add user message
        st.session_state.setdefault("messages", []).append({"role": "user", "content": prompt})
        
        with st.chat_message("user", avatar="🧑‍🍳"):
            st.markdown(prompt)
        
        # Process image if uploaded
        image_data = None
        if uploaded_image:
            image_data = extract_image(uploaded_image)
        
        # Generate response based on mode
        with st.chat_message("assistant", avatar="👨‍🍳"):
            with st.spinner(f"Chef AI is {get_mode_action(current_mode)}..."):
                if current_mode == "recipe":
                    response = run_recipe_generator(prompt, user)
                elif current_mode == "meal_plan":
                    response = run_meal_planner(prompt, user)
                elif current_mode == "nutrition":
                    response = run_nutrition_analyzer(prompt, user, image_data)
                elif current_mode == "shopping":
                    response = run_shopping_assistant(prompt, user)
                elif current_mode == "tips":
                    response = run_cooking_tips(prompt, user)
                elif current_mode == "identify":
                    response = run_food_identifier(prompt, image_data, user)
                else:
                    response = run_recipe_generator(prompt, user)
            
            st.markdown(response)
        
        st.session_state["messages"].append({"role": "assistant", "content": response})


def get_mode_welcome(mode):
    welcomes = {
        "recipe": "Tell me what ingredients you have, or describe a dish you'd like to make. I'll create a delicious recipe for you!",
        "meal_plan": "I'll help you plan your meals for the week. Tell me your goals, preferences, or how many people you're cooking for.",
        "nutrition": "Ask me about any food's nutritional content, or upload a photo of your meal for analysis.",
        "shopping": "Tell me what recipes you want to make, or describe your meal plan. I'll create an organized shopping list.",
        "tips": "Ask me anything about cooking techniques, ingredient substitutions, or kitchen hacks!",
        "identify": "Upload a photo of any food, and I'll identify it, estimate nutrition, and suggest recipes.",
    }
    return welcomes.get(mode, welcomes["recipe"])


def get_mode_placeholder(mode):
    placeholders = {
        "recipe": "I have chicken, rice, and broccoli. What can I make?",
        "meal_plan": "Plan healthy meals for a family of 4 this week",
        "nutrition": "How many calories in a Caesar salad?",
        "shopping": "Create a shopping list for Italian dinner for 6",
        "tips": "How do I substitute eggs in baking?",
        "identify": "What dish is this? (upload an image)",
    }
    return placeholders.get(mode, placeholders["recipe"])


def get_mode_action(mode):
    actions = {
        "recipe": "creating your recipe",
        "meal_plan": "planning your meals",
        "nutrition": "analyzing nutrition",
        "shopping": "building your list",
        "tips": "preparing advice",
        "identify": "identifying food",
    }
    return actions.get(mode, "thinking")


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
