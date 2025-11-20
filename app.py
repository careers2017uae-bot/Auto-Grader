# app.py
"""
RAG-based Student Work Auto-Grader (Streamlit)
- Embeddings: sentence-transformers (local) -> all-MiniLM-L6-v2
- Simple RAG: embed model answer + student answer, compute cosine similarity
- Rubric JSON support (see examples). If rubric JSON provided, it is used.
- Optional: If GROQ_API_URL and GROQ_API_KEY are set, the app will call Groq chat completions
  to produce a more natural feedback explanation. If not set, deterministic feedback is produced.
- Run: streamlit run app.py
"""

import os
import json
import time
import streamlit as st
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# Optional imports
try:
    import docx2txt
except Exception:
    docx2txt = None

try:
    import language_tool_python
    lang_tool = language_tool_python.LanguageTool("en-US")
except Exception:
    lang_tool = None

st.set_page_config(page_title="RAG Auto-Grader (Local Embeddings)", layout="wide")

# Load embedding model once
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer("llama-3.3-70b-versatile")

embedding_model = load_embedding_model()

# ---------------------------
# Utilities
# ---------------------------
def read_text_file(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    name = uploaded_file.name.lower()
    try:
        content = uploaded_file.getvalue()
    except Exception:
        return ""
    if name.endswith(".txt"):
        try:
            return content.decode("utf-8")
        except Exception:
            return str(content)
    if name.endswith(".docx"):
        if docx2txt:
            # write to temp and process
            tmp_path = "/tmp/temp_upload.docx"
            with open(tmp_path, "wb") as f:
                f.write(content)
            return docx2txt.process(tmp_path)
        else:
            st.warning("docx2txt not installed; paste text or upload .txt instead.")
            return ""
    # fallback
    try:
        return content.decode("utf-8")
    except Exception:
        return str(content)

def safe_load_json(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except Exception:
        return None

def embed_texts(texts: List[str]) -> np.ndarray:
    # returns numpy array shape (n, dim)
    texts = [t if t is not None else "" for t in texts]
    vectors = embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return vectors

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(1, -1)
    b = b.reshape(1, -1)
    return float(cosine_similarity(a, b)[0][0])

def grammar_check(text: str) -> Dict[str, Any]:
    if not lang_tool:
        return {"available": False, "issues_count": None, "examples": []}
    matches = lang_tool.check(text)
    examples = []
    for m in matches[:6]:
        context = text[max(0, m.offset-30): m.offset+30]
        examples.append({"message": m.message, "context": context})
    return {"available": True, "issues_count": len(matches), "examples": examples}

# ---------------------------
# Grading logic
# ---------------------------
def apply_rubric_json(rubric: dict, model_ans: str, student_ans: str) -> Dict[str, Any]:
    """
    Expected rubric format:
    { "criteria":[ {"name":"Task Achievement","weight":0.25,"type":"similarity"}, ... ], "scale": {"min":0,"max":100} }
    Types supported: "similarity" (embedding similarity to model), "grammar_penalty" (penalize by grammar issues)
    """
    criteria = rubric.get("criteria", [])
    if not criteria:
        # fallback to heuristic
        return heuristic_grade(model_ans, student_ans)

    # get embeddings
    vecs = embed_texts([model_ans, student_ans])
    sim = cosine_sim(vecs[0], vecs[1])  # [-1,1]
    sim_norm = max(0.0, min((sim + 1) / 2.0, 1.0))  # to [0,1]
    g = grammar_check(student_ans)
    issues = g["issues_count"] if g.get("available") else None

    total_weight = sum(c.get("weight", 0) for c in criteria) or 1.0
    total_score = 0.0
    breakdown = []
    for c in criteria:
        name = c.get("name", "criterion")
        w = c.get("weight", 0) / total_weight
        t = c.get("type", "similarity")
        subscore = 0.0
        if t == "similarity":
            subscore = sim_norm * 100
        elif t == "grammar_penalty":
            if issues is None:
                subscore = 100.0
            else:
                penalty_per = c.get("penalty_per_issue", 1.5)
                subscore = max(0.0, 100.0 - penalty_per * issues)
        else:
            subscore = sim_norm * 100
        total_score += subscore * w
        breakdown.append({"criterion": name, "weight": round(w,3), "subscore": round(subscore,2)})

    final_score = round(total_score, 2)
    return {"final_score": final_score, "breakdown": breakdown, "similarity": sim_norm, "grammar": g}

def heuristic_grade(model_ans: str, student_ans: str) -> Dict[str, Any]:
    vecs = embed_texts([model_ans, student_ans])
    sim = cosine_sim(vecs[0], vecs[1])
    sim_norm = max(0.0, min((sim + 1) / 2.0, 1.0))
    base = sim_norm * 100
    g = grammar_check(student_ans)
    penalty = 0.0
    if g.get("available"):
        issues = g["issues_count"]
        penalty = min(40.0, issues * 1.5)
    final = round(max(0.0, base - penalty), 2)
    # short breakdown
    breakdown = [
        {"criterion": "Similarity (auto)", "weight": 0.8, "subscore": round(base,2)},
        {"criterion": "Grammar penalty", "weight": 0.2, "subscore": round(max(0, 100 - penalty),2)}
    ]
    return {"final_score": final, "breakdown": breakdown, "similarity": sim_norm, "grammar": g, "penalty": penalty}

# ---------------------------
# Optional Groq helper (for richer natural-language feedback)
# ---------------------------
def generate_feedback_with_groq(prompt_text: str) -> Optional[str]:
    """
    Sends a prompt to Groq Chat Completions if GROQ_API_URL and GROQ_API_KEY are set.
    Expects GROQ OpenAI-compatible chat endpoint: {GROQ_API_URL}/chat/completions
    If no creds set, returns None.
    """
    base = os.getenv("GROQ_API_URL")
    key = os.getenv("GROQ_API_KEY")
    if not base or not key:
        return None
    # Ensure the base URL doesn't duplicate /chat/completions
    if base.endswith("/"):
        url = base + "chat/completions"
    else:
        url = base + "/chat/completions"

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4o-mini" if "gpt-4" in os.getenv("GROQ_MODEL", "") else "gpt-4o-mini", 
        "messages": [
            {"role": "system", "content": "You are an objective IELTS grading assistant. Produce concise JSON with score reasoning and improvement steps."},
            {"role": "user", "content": prompt_text}
        ],
        "temperature": 0.2,
        "max_tokens": 500
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code != 200:
            st.warning(f"Groq call failed ({resp.status_code}): {resp.text}")
            return None
        data = resp.json()
        # Try to extract content
        # OpenAI-compatible: data['choices'][0]['message']['content']
        if isinstance(data, dict):
            choices = data.get("choices")
            if choices and isinstance(choices, list) and len(choices) > 0:
                msg = choices[0].get("message", {}).get("content") or choices[0].get("text")
                return msg
        return None
    except Exception as e:
        st.warning(f"Groq request exception: {e}")
        return None

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("📝 RAG Auto-Grader — Local Embeddings + Optional Groq Feedback")

st.markdown("Upload Exercise, Model solution, Rubric (optional, JSON) and student submissions. "
            "Embeddings are computed locally (no external embedding API).")

with st.sidebar:
    st.header("Settings")
    output_scale = st.selectbox("Output scale", ["numeric_100", "ielts_band_0-9"], index=0)
    show_grammar_examples = st.checkbox("Show grammar examples (if available)", value=True)
    st.markdown("---")
    st.info("Optional: set GROQ_API_URL and GROQ_API_KEY as environment variables to enable richer LLM feedback via Groq chat completions.")

# Inputs
st.header("Inputs")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Exercise Description")
    ex_file = st.file_uploader("Upload exercise (.txt or .docx) or paste", type=["txt","docx"])
    ex_text_paste = st.text_area("Or paste exercise description here", height=120)
with col2:
    st.subheader("Model Solution (Reference)")
    model_file = st.file_uploader("Upload model solution (.txt or .docx) or paste", type=["txt","docx"])
    model_text_paste = st.text_area("Or paste model solution here", height=120)

st.subheader("Grading Rubric (optional, JSON)")
rubric_file = st.file_uploader("Upload rubric.json (optional)", type=["json"])
rubric_text_paste = st.text_area("Or paste rubric JSON here", height=140)

st.subheader("Student Submissions")
st.markdown("Upload multiple student files (.txt/.docx) or paste multiple submissions separated by a line `---`.")
student_files = st.file_uploader("Upload student files (multiple)", accept_multiple_files=True, type=["txt","docx"])
student_paste = st.text_area("Or paste submissions here (separate by `---`)", height=180)

if st.button("Run Grader"):
    # gather inputs
    exercise_text = ""
    if ex_file:
        exercise_text = read_text_file(ex_file)
    if not exercise_text and ex_text_paste.strip():
        exercise_text = ex_text_paste.strip()
    if not exercise_text:
        st.error("Please provide the exercise description.")
        st.stop()

    model_text = ""
    if model_file:
        model_text = read_text_file(model_file)
    if not model_text and model_text_paste.strip():
        model_text = model_text_paste.strip()
    if not model_text:
        st.error("Please provide the model solution.")
        st.stop()

    rubric_obj = None
    rubric_text = ""
    if rubric_file:
        rubric_text = rubric_file.getvalue().decode("utf-8")
    if not rubric_text and rubric_text_paste.strip():
        rubric_text = rubric_text_paste.strip()
    if rubric_text:
        rubric_obj = safe_load_json(rubric_text)
        if rubric_obj is None:
            st.error("Rubric JSON invalid. Please provide valid JSON.")
            st.stop()

    # students
    student_texts = []
    student_names = []
    if student_files:
        for f in student_files:
            txt = read_text_file(f)
            if txt.strip():
                student_texts.append(txt.strip())
                student_names.append(f.name)
    if student_paste.strip():
        parts = [p.strip() for p in student_paste.split("\n---\n") if p.strip()]
        for i,p in enumerate(parts):
            student_texts.append(p)
            student_names.append(f"Pasted_{i+1}")

    if not student_texts:
        st.error("No student submissions provided.")
        st.stop()

    # grade each student
    results = []
    with st.spinner("Grading submissions..."):
        for idx, s in enumerate(student_texts):
            try:
                if rubric_obj:
                    res = apply_rubric_json(rubric_obj, model_text, s)
                else:
                    res = heuristic_grade(model_text, s)
                # Prepare short reasoning and feedback
                sim_pct = round(res.get("similarity", 0) * 100, 2)
                issues = res.get("grammar", {}).get("issues_count", "N/A")
                reasoning = f"Similarity to model: {sim_pct}%. Grammar issues (approx): {issues}."
                # Deterministic feedback
                feedback_lines = []
                if res.get("similarity", 0) >= 0.75:
                    feedback_lines.append("Good coverage of model answer and task achievement.")
                elif res.get("similarity", 0) >= 0.5:
                    feedback_lines.append("Partial coverage — some key points missing or underdeveloped.")
                else:
                    feedback_lines.append("Limited overlap with model answer; address main task points directly.")
                if res.get("grammar", {}).get("available"):
                    if res["grammar"]["issues_count"] > 6:
                        feedback_lines.append("Many grammar errors — focus on verb forms and sentence structure.")
                    elif res["grammar"]["issues_count"] > 2:
                        feedback_lines.append("Some grammar errors — proofread carefully.")
                feedback_lines.append("Actionable: (1) Map each paragraph to task points; (2) Use linking words for cohesion; (3) Keep sentences concise.")

                # Try to get richer feedback from Groq if configured
                groq_feedback = None
                groq_prompt = (
                    f"Grade this student's answer using the rubric and return a short JSON with "
                    f"final_score, reasoning (1-2 paragraphs), and 3 actionable steps. "
                    f"Rubric: {json.dumps(rubric_obj) if rubric_obj else 'None'}\n"
                    f"Model answer:\n{model_text}\n\nStudent answer:\n{s}"
                )
                groq_feedback = generate_feedback_with_groq(groq_prompt)

                results.append({
                    "name": student_names[idx],
                    "final_score": res.get("final_score"),
                    "reasoning": reasoning,
                    "feedback_lines": feedback_lines,
                    "details": res,
                    "groq_feedback": groq_feedback
                })
            except Exception as e:
                results.append({"name": student_names[idx], "error": str(e)})

    # Display results
    st.header("Results")
    for r in results:
        st.subheader(r.get("name", "Student"))
        if r.get("error"):
            st.error(f"Error: {r['error']}")
            continue
        st.metric("Final Score", f"{r['final_score']} / 100")
        st.markdown("**Reasoning (concise):**")
        st.write(r["reasoning"])
        st.markdown("**Actionable Feedback (deterministic):**")
        for line in r["feedback_lines"]:
            st.write(f"- {line}")
        if r.get("groq_feedback"):
            st.markdown("**Optional Groq-generated feedback:**")
            st.write(r["groq_feedback"])
        st.markdown("**Details / Breakdown:**")
        st.json(r["details"])
        if show_grammar_examples and r["details"].get("grammar", {}).get("available"):
            g = r["details"]["grammar"]
            st.markdown("**Grammar Examples:**")
            st.write(f"Issues found: {g['issues_count']}")
            for ex in g["examples"]:
                st.write(f"- {ex['message']} — ...{ex['context']}...")
        st.divider()

st.markdown("---")
st.markdown("Notes: Embeddings are computed locally using sentence-transformers `all-MiniLM-L6-v2`. "
            "If you want to enable Groq feedback, set GROQ_API_URL and GROQ_API_KEY in your environment. "
            "Edit rubric JSON to tune scoring.")
