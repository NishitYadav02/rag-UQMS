"""
Student University Query Management System
=========================================
Uses RAG (Retrieval-Augmented Generation) to answer student questions
from university documents using FAISS + Sentence Transformers + Gemini API.
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from rag_engine import build_index, search_chunks
import google.generativeai as genai

BASE_DIR = Path(__file__).resolve().parent

# ── Load environment variables from .env file ──────────────────────────────
load_dotenv(BASE_DIR / ".env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ── Configure Gemini ───────────────────────────────────────────────────────
model = None
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    # model = genai.GenerativeModel("gemini-1.5-flash")
    model = genai.GenerativeModel("gemini-2.5-flash")

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UniQuery — Student Help Desk",
    page_icon="🎓",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Sora:wght@700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Background */
.stApp { background: #0f1117; }

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #1a1f2e 0%, #0f1117 60%);
    border: 1px solid #2a2f3e;
    border-radius: 16px;
    padding: 36px 40px 28px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(99,102,241,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-eyebrow {
    font-size: 11px; font-weight: 600; letter-spacing: 2px;
    color: #6366f1; text-transform: uppercase; margin-bottom: 10px;
}
.hero-title {
    font-family: 'Sora', sans-serif;
    font-size: 30px; font-weight: 800;
    color: #f0f0f5; line-height: 1.2; margin-bottom: 12px;
}
.hero-title span { color: #6366f1; }
.hero-desc {
    font-size: 14px; color: #8b8fa8; line-height: 1.7; max-width: 520px;
}

/* How it works pills */
.steps-row {
    display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 28px;
}
.step-pill {
    background: #1a1f2e; border: 1px solid #2a2f3e;
    border-radius: 20px; padding: 6px 14px;
    font-size: 12px; color: #8b8fa8;
    display: flex; align-items: center; gap: 6px;
}
.step-pill b { color: #6366f1; font-size: 11px; }

/* Input label */
.q-label {
    font-size: 13px; font-weight: 600;
    color: #c5c7d4; margin-bottom: 8px; letter-spacing: 0.3px;
}

/* Answer card */
.answer-card {
    background: #1a1f2e;
    border: 1px solid #2a2f3e;
    border-left: 3px solid #6366f1;
    border-radius: 12px;
    padding: 22px 26px;
    margin-top: 20px;
    color: #d4d6e4;
    font-size: 15px;
    line-height: 1.75;
}
.answer-label {
    font-size: 11px; font-weight: 600; letter-spacing: 1.5px;
    color: #6366f1; text-transform: uppercase; margin-bottom: 10px;
}

/* Error card */
.error-card {
    background: #1f1520; border: 1px solid #3d1f2e;
    border-radius: 12px; padding: 16px 20px;
    color: #f87171; font-size: 14px; margin-top: 16px;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Build FAISS index once (cached so it doesn't reload on every interaction)
@st.cache_resource(show_spinner="Building knowledge index…")
def load_index():
    """Load the university doc and build FAISS vector index."""
    return build_index(str(BASE_DIR / "university_info.txt"))


def ask_gemini(context: str, question: str) -> str:
    """Send retrieved context + student question to Gemini and get an answer."""
    if not model:
        return "Gemini is not configured because the API key is missing."

    prompt = f"""You are a helpful university assistant.
Use ONLY the context below to answer the student's question.
If the answer is not in the context, say: "I don't have that information in the university documents."

Context:
{context}

Student Question: {question}

Answer in simple, clear language suitable for a student."""
    response = model.generate_content(prompt)
    return response.text


# ── UI ─────────────────────────────────────────────────────────────────────

# Hero
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">🎓 RAG-Powered Help Desk</div>
    <div class="hero-title">Student <span>University</span> Query System</div>
    <div class="hero-desc">
        Ask anything about admissions, fees, exam dates, syllabus, hostel rules,
        library timings, placements, or scholarships. Answers are pulled directly
        from official university documents.
    </div>
</div>
""", unsafe_allow_html=True)

# How it works
st.markdown("""
<div class="steps-row">
    <div class="step-pill"><b>01</b> You ask a question</div>
    <div class="step-pill"><b>02</b> FAISS finds relevant text</div>
    <div class="step-pill"><b>03</b> Gemini generates the answer</div>
</div>
""", unsafe_allow_html=True)

# Load index
try:
    chunks, index, embed_model = load_index()
except FileNotFoundError:
    st.markdown('<div class="error-card">⚠️ <b>university_info.txt</b> not found. Please place it in the project root.</div>', unsafe_allow_html=True)
    st.stop()

# Question input
st.markdown('<div class="q-label">Your Question</div>', unsafe_allow_html=True)
question = st.text_input(
    label="question_input",
    placeholder="e.g. What are the hostel rules? When is the last date to apply?",
    label_visibility="collapsed",
)

ask = st.button("Ask →", type="primary", use_container_width=True)

# Answer
if ask:
    if not question.strip():
        st.warning("Please type a question first.")
    elif not GEMINI_API_KEY:
        st.markdown('<div class="error-card">⚠️ GEMINI_API_KEY missing in .env file.</div>', unsafe_allow_html=True)
    else:
        with st.spinner("Searching documents and generating answer…"):
            # Step 1 — Retrieve relevant chunks from FAISS
            relevant_chunks = search_chunks(question, chunks, index, embed_model, top_k=4)
            context = "\n\n".join(relevant_chunks)

            # Step 2 — Ask Gemini with context
            answer = ask_gemini(context, question)

        st.markdown(f"""
        <div class="answer-card">
            <div class="answer-label">Answer</div>
            {answer}
        </div>
        """, unsafe_allow_html=True)

        # Expandable: show which chunks were used (useful for demo)
        with st.expander("📄 Source chunks retrieved from documents"):
            for i, chunk in enumerate(relevant_chunks, 1):
                st.markdown(f"**Chunk {i}:**\n\n{chunk}\n\n---")
