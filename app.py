# app.py
"""
RAG-based Student Work Auto-Grader (Streamlit)
- Local embeddings: sentence-transformers "all-MiniLM-L6-v2"
- Optional Groq feedback (only if GROQ_API_URL and GROQ_API_KEY are valid)
- Rubric JSON support and multi-student grading
- Grammar checking optional (language-tool-python)
"""

import os
import json
import streamlit as st
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# Optional
try:
    import docx2txt
except:
    docx2txt = None

try:
    import language_tool_python
    lang_tool = language_tool_python.LanguageTool("en-US")
except:
    lang_tool = None

st.set_page_config(page_title="RAG Auto-Grader", layout="wide")

# Load embedding model
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_model()

# ---------------------------
# Utilities
# ---------------------------
def read_text_file(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    name = uploaded_file.name.lower()
    content = uploaded_file.getvalue()
    if name.endswith(".txt"):
        return content.decode("utf-8")
    if name.endswith(".docx") and docx2txt:
        tmp_path = "/tmp/temp_upload.docx"
        with open(tmp_path, "wb") as f:
            f.write(content)
        return docx2txt.process(tmp_path)
    # fallback
    try:
        return content.decode("utf-8")
    except:
        return str(content)

def safe_load_json(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except:
        return None

def embed_texts(texts: List[str]) -> np.ndarray:
    return embedding_model.encode([t if t else "" for t in texts], convert_to_numpy=True, show_progress_bar=False)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(cosine_similarity(a.reshape(1,-1), b.reshape(1,-1))[0][0])

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
def apply_rubric(rubric: dict, model_ans: str, student_ans: str) -> Dict[str, Any]:
    criteria = rubric.get("criteria", [])
    if not criteria:
        return heuristic_grade(model_ans, student_ans)
    
    vecs = embed_texts([model_ans, student_ans])
    sim_norm = max(0.0, min((cosine_sim(vecs[0], vecs[1]) + 1) / 2, 1.0))
    g = grammar_check(student_ans)
    issues = g.get("issues_count", 0) if g.get("available") else 0

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
            penalty_per = c.get("penalty_per_issue", 1.5)
            subscore = max(0.0, 100.0 - penalty_per * issues)
        total_score += subscore * w
        breakdown.append({"criterion": name, "weight": round(w,3), "subscore": round(subscore,2)})

    final_score = round(total_score, 2)
    return {"final_score": final_score, "breakdown": breakdown, "similarity": sim_norm, "grammar": g}

def heuristic_grade(model_ans: str, student_ans: str) -> Dict[str, Any]:
    vecs = embed_texts([model_ans, student_ans])
    sim_norm = max(0.0, min((cosine_sim(vecs[0], vecs[1]) + 1) / 2, 1.0))
    g = grammar_check(student_ans)
    penalty = 0.0
    if g.get("available"):
        penalty = min(40.0, g.get("issues_count", 0) * 1.5)
    final = round(max(0.0, sim_norm*100 - penalty), 2)
    breakdown = [
        {"criterion": "Similarity (auto)", "weight": 0.8, "subscore": round(sim_norm*100,2)},
        {"criterion": "Grammar penalty", "weight": 0.2, "subscore": round(max(0,100-penalty),2)}
    ]
    return {"final_score": final, "breakdown": breakdown, "similarity": sim_norm, "grammar": g}

# ---------------------------
# Optional Groq call (safe)
# ---------------------------
def generate_feedback_with_groq(prompt_text: str) -> Optional[str]:
    url = os.getenv("GROQ_API_URL")
    key = os.getenv("GROQ_API_KEY")
    if not url or not key:
        return None
    if url.endswith("/"):
        url = url + "chat/completions"
    else:
        url = url + "/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": os.getenv("GROQ_MODEL","gpt-3.5-mini"),
        "messages":[{"role":"system","content":"You are an objective IELTS grading assistant. Return JSON with final_score, reasoning, and 3 actionable improvement steps."},
                    {"role":"user","content": prompt_text}],
        "temperature":0.2,
        "max_tokens":500
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code != 200:
            return None
        data = resp.json()
        choices = data.get("choices")
        if choices and len(choices)>0:
            msg = choices[0].get("message",{}).get("content") or choices[0].get("text")
            return msg
        return None
    except:
        return None

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("📝 RAG Auto-Grader — Local Embeddings + Optional Groq Feedback")

st.markdown("Upload Exercise, Model Solution, Rubric JSON (optional), and Student Submissions. Local embeddings handle grading safely.")

with st.sidebar:
    st.header("Settings")
    show_grammar = st.checkbox("Show grammar examples", True)
    st.info("Optional: set GROQ_API_URL and GROQ_API_KEY to enable Groq LLM feedback.")

# Inputs
st.header("Inputs")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Exercise Description")
    ex_file = st.file_uploader("Upload exercise (.txt/.docx)", type=["txt","docx"])
    ex_text_paste = st.text_area("Or paste exercise description", height=120)
with col2:
    st.subheader("Model Solution")
    model_file = st.file_uploader("Upload model solution (.txt/.docx)", type=["txt","docx"])
    model_text_paste = st.text_area("Or paste model solution", height=120)

st.subheader("Grading Rubric (optional, JSON)")
rubric_file = st.file_uploader("Upload rubric.json", type=["json"])
rubric_text_paste = st.text_area("Or paste JSON", height=140)

st.subheader("Student Submissions")
st.markdown("Upload multiple files or paste multiple submissions separated by `---`.")
student_files = st.file_uploader("Upload student files (multiple)", accept_multiple_files=True, type=["txt","docx"])
student_paste = st.text_area("Or paste submissions here", height=180)

# Run
if st.button("Run Grader"):
    exercise_text = read_text_file(ex_file) or ex_text_paste.strip()
    model_text = read_text_file(model_file) or model_text_paste.strip()
    if not exercise_text or not model_text:
        st.error("Provide exercise and model solution.")
        st.stop()
    rubric_obj = None
    rubric_text = (rubric_file.getvalue().decode("utf-8") if rubric_file else "") or rubric_text_paste.strip()
    if rubric_text:
        rubric_obj = safe_load_json(rubric_text)
        if rubric_obj is None:
            st.error("Invalid rubric JSON.")
            st.stop()
    student_texts = []
    student_names = []
    if student_files:
        for f in student_files:
            txt = read_text_file(f).strip()
            if txt:
                student_texts.append(txt)
                student_names.append(f.name)
    if student_paste.strip():
        for i,s in enumerate([p.strip() for p in student_paste.split("\n---\n") if p.strip()]):
            student_texts.append(s)
            student_names.append(f"Pasted_{i+1}")
    if not student_texts:
        st.error("No student submissions.")
        st.stop()

    # Grading
    results = []
    with st.spinner("Grading..."):
        for idx, s in enumerate(student_texts):
            try:
                res = apply_rubric(rubric_obj, model_text, s) if rubric_obj else heuristic_grade(model_text, s)
                reasoning = f"Similarity: {round(res.get('similarity',0)*100,2)}%, Grammar issues: {res.get('grammar',{}).get('issues_count','N/A')}"
                feedback = ["Good coverage" if res.get('similarity',0)>=0.75 else "Partial coverage" if res.get('similarity',0)>=0.5 else "Limited overlap"]
                if res.get('grammar',{}).get('available') and res['grammar']['issues_count']>2:
                    feedback.append("Check grammar carefully.")
                # Optional Groq
                groq_prompt = f"Rubric: {json.dumps(rubric_obj) if rubric_obj else 'None'}\nModel answer:\n{model_text}\nStudent answer:\n{s}"
                groq_feedback = generate_feedback_with_groq(groq_prompt)
                results.append({"name": student_names[idx], "final_score": res['final_score'], "reasoning":reasoning, "feedback":feedback, "details":res, "groq_feedback": groq_feedback})
            except Exception as e:
                results.append({"name": student_names[idx], "error": str(e)})

    # Display results
    st.header("Results")
    for r in results:
        st.subheader(r.get("name"))
        if r.get("error"):
            st.error(r['error'])
            continue
        st.metric("Final Score", f"{r['final_score']} / 100")
        st.markdown("**Reasoning:**")
        st.write(r["reasoning"])
        st.markdown("**Feedback:**")
        for f in r["feedback"]:
            st.write(f"- {f}")
        if r.get("groq_feedback"):
            st.markdown("**Optional Groq feedback:**")
            st.write(r["groq_feedback"])
        st.markdown("**Details / Breakdown:**")
        st.json(r["details"])
        if show_grammar and r["details"].get("grammar",{}).get("available"):
            st.markdown("**Grammar Examples:**")
            st.write(f"Issues: {r['details']['grammar']['issues_count']}")
            for ex in r['details']['grammar']['examples']:
                st.write(f"- {ex['message']} ...{ex['context']}...")
        st.divider()
