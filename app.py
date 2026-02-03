import streamlit as st
import hashlib
import re
import io
import asyncio
import aiohttp
import requests
from dataclasses import dataclass
import fitz
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from difflib import SequenceMatcher
from fpdf import FPDF
import spacy
from supabase import create_client, Client
from scholarly import scholarly

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ─── Supabase ────────────────────────────────────────────────────────────────────
supabase_url = st.secrets["supabase"]["url"]
supabase_key = st.secrets["supabase"]["anon_key"]
supabase = create_client(supabase_url, supabase_key)

# ─── Session ─────────────────────────────────────────────────────────────────────
if "user" not in st.session_state:
    st.session_state.user = None

def logout():
    supabase.auth.sign_out()
    st.session_state.user = None
    st.rerun()

# ─── Lazy Models ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_cross_encoder():
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_tokenizer():
    return GPT2Tokenizer.from_pretrained("gpt2")

@st.cache_resource
def load_gpt_model():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    return model

# ─── Language ────────────────────────────────────────────────────────────────────
LANGUAGES = {
    "en": {
        "title": "AG PlagCheck",
        "subtitle": "Your Smart Academic Guardian",
        "login_title": "Login / Sign Up",
        "email": "Email",
        "send_magic": "Send Magic Link",
        "guest": "Continue as Guest",
        "logout": "Logout",
        "tab_check": "📤 Check Documents",
        "tab_db": "📚 Database",
        "tab_info": "ℹ️ How It Works",
        "upload_label": "Upload PDF, DOCX or TXT",
        "global_checkbox": "Include Global Search (slower)",
        "analyzing": "Analyzing...",
        "done": "Done! 🎉",
        "ai_likelihood": "AI Likelihood",
        "similarity": "Similarity",
        "matches": "Matches",
        "download_pdf": "Download PDF Report",
        "sidebar_info": "Private & secure"
    }
    # Add Kiswahili if needed later
}

if "lang" not in st.session_state:
    st.session_state.lang = "en"

lang = st.session_state.lang
txt = LANGUAGES[lang]

# ─── Styling ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AG PlagCheck", page_icon="🔍", layout="wide")

st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #f0f7ff, #e6f0ff); }
    .big-title { font-size: 2.8rem; font-weight: 800; background: linear-gradient(90deg, #00d4ff, #7c3aed, #ff6bcb); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; }
    .subtitle { font-size: 1.15rem; color: #4a5568; text-align: center; }
    </style>
""", unsafe_allow_html=True)

def show_logo():
    try:
        st.image("ag_logo.png", width=180)
    except:
        pass
    st.markdown(f'<h1 class="big-title">{txt["title"]}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="subtitle">{txt["subtitle"]}</p>', unsafe_allow_html=True)

show_logo()

# ─── Sidebar Auth ────────────────────────────────────────────────────────────────
with st.sidebar:
    if st.session_state.user:
        st.success(f"Logged in as: {st.session_state.user.get('email', 'Guest')}")
        if st.button(txt["logout"]):
            logout()
    else:
        st.subheader(txt["login_title"])
        email = st.text_input(txt["email"], key="login_email")

        col1, col2 = st.columns(2)
        with col1:
            if st.button(txt["send_magic"]):
                if not email:
                    st.error("Enter your email")
                else:
                    with st.spinner("Sending..."):
                        try:
                            supabase.auth.sign_in_with_otp({"email": email})
                            st.success("Magic link sent! Check your email and click the link.")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")

        with col2:
            if st.button(txt["guest"]):
                st.session_state.user = {"id": "guest", "email": "guest@local"}
                st.rerun()

    if not st.session_state.user:
        st.stop()

    st.markdown("---")
    st.info(txt["sidebar_info"])

# ─── Callback Handling (critical for magic link) ─────────────────────────────────
query_params = st.query_params.to_dict()

if query_params:
    st.sidebar.write("**Debug — URL params received:**")
    st.sidebar.json(query_params)

if "access_token" in query_params or "type" in query_params:
    with st.spinner("Completing login..."):
        try:
            user = supabase.auth.get_user()
            if user.user:
                st.session_state.user = user.user
                st.success("Logged in successfully! 🎉")
                st.query_params.clear()
                st.rerun()
            else:
                st.error("No user returned from Supabase")
        except Exception as e:
            st.error(f"Login failed: {str(e)}")
            st.sidebar.error("Detailed error: " + str(e))

# ─── Tabs ────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([txt["tab_check"], txt["tab_db"], txt["tab_info"]])

# ─── Core Functions (simplified) ─────────────────────────────────────────────────

def process(file_bytes, fn, enable_global):
    # Minimal version - just extract and save
    text = extract(file_bytes, fn)
    if len(text.strip()) < 200:
        return {"err": "File too short", "text": text}
    return {"text": text, "fn": fn}

with tab1:
    st.header(txt["tab_check"])
    files = st.file_uploader("Upload files", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    if files:
        for f in files:
            res = process(f.read(), f.name, False)
            if "err" in res:
                st.error(f"{f.name}: {res['err']}")
            else:
                st.success(f"{f.name} uploaded!")
                st.text_area("Extracted text", res["text"], height=200)

with tab2:
    st.header(txt["tab_db"])
    st.info("Database view coming soon")

with tab3:
    st.header(txt["tab_info"])
    st.markdown("Login works via magic link. Features added gradually.")

st.sidebar.info("Login test version")