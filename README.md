# 🎓 Student University Query Management System

> A beginner-friendly RAG project using Python · Streamlit · FAISS · Sentence Transformers · Gemini API

---

## 📁 Folder Structure

```
student_query_system/
│
├── app.py                  ← Main Streamlit UI (run this)
├── rag_engine.py           ← RAG logic: chunking, embeddings, FAISS search
├── university_info.txt     ← University data (plain text document)
├── requirements.txt        ← All Python packages needed
├── .env                    ← Your secret Gemini API key (never share this)
└── .gitignore              ← Tells Git to ignore .env and cache files
```

---

## 🧠 What is RAG? (Simple Explanation)

**RAG = Retrieval-Augmented Generation**

Imagine you give an AI a textbook and say:
> "When a student asks a question, first search this textbook for the relevant pages, then answer using those pages."

That's exactly RAG! Instead of relying on the AI's general training, you make it answer **from your own documents**.

---

## 🔄 How This Project Works — Step by Step

```
┌─────────────────────────────────────────────────────────────┐
│                    PROJECT WORKFLOW                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  university_info.txt                                        │
│         │                                                   │
│         ▼                                                   │
│  [Step 1] Read the full document                           │
│         │                                                   │
│         ▼                                                   │
│  [Step 2] Split into small chunks (300 words each)         │
│           "Chunk 1: About University..."                    │
│           "Chunk 2: Admission Process..."                   │
│           "Chunk 3: Fee Structure..."   ... etc.            │
│         │                                                   │
│         ▼                                                   │
│  [Step 3] Convert each chunk into a number vector          │
│           (Sentence Transformers does this)                 │
│           Chunk 1 → [0.23, -0.11, 0.87, ...]              │
│           Chunk 2 → [0.55,  0.32, 0.14, ...]              │
│         │                                                   │
│         ▼                                                   │
│  [Step 4] Store all vectors in FAISS index                 │
│           (FAISS = fast search database for vectors)        │
│                                                             │
│  ─────────────── Above steps run ONCE at startup ──────── │
│                                                             │
│  Student types: "What is the hostel entry time?"           │
│         │                                                   │
│         ▼                                                   │
│  [Step 5] Convert question into a vector too               │
│                                                             │
│         ▼                                                   │
│  [Step 6] Search FAISS → find 4 most similar chunks       │
│           Returns: Hostel Rules chunk, Hostel Info chunk... │
│         │                                                   │
│         ▼                                                   │
│  [Step 7] Send to Gemini API:                              │
│           "Here is context: [chunks]. Answer: [question]"  │
│         │                                                   │
│         ▼                                                   │
│  [Step 8] Gemini returns a clear answer                    │
│         │                                                   │
│         ▼                                                   │
│  Display answer in Streamlit UI ✅                         │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Installation — Step by Step

### 1. Make sure Python is installed
```bash
python --version   # Should be 3.9 or higher
```

### 2. Create and activate a virtual environment (recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (Mac/Linux)
source venv/bin/activate
```

### 3. Install all required packages
```bash
pip install -r requirements.txt
```
> ⚠️ This will download ~500MB (sentence transformer model). Do this on a good internet connection.

### 4. Get your FREE Gemini API key
- Go to: https://aistudio.google.com/app/apikey
- Click **"Create API Key"**
- Copy the key

### 5. Set up the `.env` file
Open the `.env` file and replace the placeholder:
```
GEMINI_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXX   ← paste your key here
```

### 6. Run the app
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

---

## 🧪 Sample Questions to Try

| Question | Expected Topic |
|---|---|
| What is the last date for admission? | Admissions |
| How much is the B.Tech fee per year? | Fee Structure |
| When are the end semester exams? | Exam Schedule |
| What are the hostel rules? | Hostel |
| What are the library timings on Sunday? | Library |
| How can I apply for a scholarship? | Scholarship |
| What is the placement percentage? | Placement |
| What CGPA is required for placement? | Placement |
| What documents are needed for admission? | Admissions |

---

## 🛠️ Technology Stack Explained

| Technology | What it does in this project |
|---|---|
| **Python** | Main programming language |
| **Streamlit** | Creates the web UI — no HTML/CSS knowledge needed |
| **Sentence Transformers** | Converts text into numeric vectors (embeddings) |
| **FAISS** | Stores vectors and searches them quickly |
| **Gemini API** | Google's AI that generates the final answer |
| **python-dotenv** | Reads the API key from `.env` file securely |

---

## 📖 File-by-File Explanation (For Faculty Demo)

### `university_info.txt`
Plain text file with all university information. You can edit this to add real university data.

### `rag_engine.py`
Contains 4 simple functions:
- `load_and_chunk()` — reads file and splits into pieces
- `get_embeddings()` — converts text to numbers
- `build_faiss_index()` — builds a searchable database
- `search_chunks()` — finds the most relevant pieces for a question

### `app.py`
The Streamlit web application:
- Loads and indexes the document at startup (once)
- Shows a text input for the student's question
- On clicking "Ask →", retrieves relevant chunks and calls Gemini
- Displays the answer beautifully

---

## 📌 Key Concepts for Faculty Presentation

1. **Chunking** — Why? LLMs have a token limit. We can't send a 10-page doc all at once, so we break it into pieces.

2. **Embeddings** — Converting words to numbers captures their *meaning*, not just the letters. "fee" and "cost" will have similar vectors.

3. **FAISS** — Like a fast search engine, but for meaning. Google searches by keywords; FAISS searches by concept.

4. **Grounding** — By giving Gemini only the relevant chunks, we prevent hallucination (making up wrong answers).

---

## 🚀 Possible Extensions (Future Work)
- Upload multiple PDFs through the UI
- Add conversation history (chat mode)
- Support multiple departments with different documents
- Add a feedback button (thumbs up/down)
- Show confidence score of the retrieved chunks
