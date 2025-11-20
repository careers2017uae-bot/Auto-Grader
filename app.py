# app.py
"""
Student Work Auto-Grader (RAG-style) - Streamlit app (one-shot)
- Supports uploading exercise description, model solution, grading rubric, and multiple student submissions.
- Uses Groq embeddings (you must set GROQ_API_URL and GROQ_API_KEY as env vars).
- If a JSON rubric is provided, it will be applied strictly. If no structured rubric is given,
  the app uses a configurable similarity->score mapping + grammar penalty to produce an explainable score.
- Outputs: final score, concise reasoning (1-2 paragraphs), actionable feedback, optional error highlights.

Run:
  streamlit run app.py
"""

import os
import json
import io
import time
from typing import List, Dict, Any, Optional

import streamlit as st
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Optional helpers for docx
try:
    import docx2txt
except Exception:
    docx2txt = None

# Optional grammar checker
try:
    import language_tool_python
    lang_tool = language_tool_python.LanguageTool('en-US')
except Exception:
    lang_tool = None

# ---------------------------
# Utility functions
# ---------------------------
st.set_page_config(page_title="RAG Auto-Grader (Groq)", layout="wide")

def read_text_file(uploaded_file) -> str:
    """Read uploaded txt/docx/plain files."""
    name = uploaded_file.name.lower()
    if name.endswith(".txt"):
        return uploaded_file.getvalue().decode("utf-8")
    if name.endswith(".docx"):
        if docx2txt:
            # Need to save to disk or to BytesIO
            b = uploaded_file.getvalue()
            with open("/tmp/temp_doc.docx", "wb") as f:
                f.write(b)
            return docx2txt.process("/tmp/temp_doc.docx")
        else:
            st.warning("docx2txt not installed; please paste the text or upload .txt instead.")
            return ""
    # fallback: try decode
    try:
        return uploaded_file.getvalue().decode("utf-8")
    except Exception:
        return ""

def safe_load_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        return None

# ---------------------------
# Embedding + RAG functions
# ---------------------------

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Get embeddings from Groq endpoint.
    Expects env vars: GROQ_API_URL, GROQ_API_KEY.
    If you have a different request shape, edit this function accordingly.
    """
    url = os.environ.get("GROQ_API_URL")
    key = os.environ.get("GROQ_API_KEY")
    if not url or not key:
        st.error("GROQ_API_URL or GROQ_API_KEY not set. Set them as environment variables before running.")
        raise RuntimeError("Missing Groq config")

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }

    payload = {"inputs": texts}  # common shape; adapt if your Groq endpoint expects different field names
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        st.error(f"Groq embeddings request failed: {resp.status_code} {resp.text}")
        raise RuntimeError("Groq embeddings error")
    data = resp.json()

    # Try to handle different response shapes:
    if isinstance(data, dict) and "embeddings" in data:
        return data["embeddings"]
    if isinstance(data, list) and isinstance(data[0], dict) and "embedding" in data[0]:
        return [item["embedding"] for item in data]
    # If the API returns a list of vectors directly:
    if isinstance(data, list) and isinstance(data[0], list):
        return data
    # Last resort:
    raise RuntimeError("Unexpected Groq response shape. Inspect `resp.json()` and adapt embed_texts().")

def cosine_sim(a: List[float], b: List[float]) -> float:
    a = np.array(a).reshape(1, -1)
    b = np.array(b).reshape(1, -1)
    return float(cosine_similarity(a, b)[0][0])

# ---------------------------
# Grading logic
# ---------------------------

def grammar_issues(text: str) -> Dict[str, Any]:
    """Return count and short list of grammar issues using language_tool_python (if available)."""
    if not lang_tool:
        return {"available": False, "issues_count": None, "examples": []}
    matches = lang_tool.check(text)
    examples = []
    for m in matches[:6]:
        examples.append({"message": m.message, "context": text[max(0, m.offset-20):m.offset+40]})
    return {"available": True, "issues_count": len(matches), "examples": examples}

def apply_json_rubric(rubric: dict, model_ans: str, student_ans: str, score_scale: int = 100) -> Dict[str, Any]:
    """
    Simple rubric application:
    Expect rubric format like:
    {
      "criteria": [
         {"name": "Task Achievement", "weight": 0.4, "type":"similarity"},
         {"name": "Coherence", "weight": 0.3, "type":"similarity"},
         {"name": "Grammar", "weight": 0.3, "type":"grammar_penalty"}
      ],
      "scale": {"min":0, "max":100}
    }
    - 'type' controls how we compute: 'similarity' uses embedding cosine similarity to model answer,
      'grammar_penalty' uses grammar errors to reduce the subscore.
    """
    # Compute embeddings for small set
    try:
        emb_model, emb_student = embed_texts([model_ans, student_ans])
    except Exception as e:
        st.error(f"Embedding error while applying rubric: {e}")
        raise

    sim = cosine_sim(emb_model, emb_student)  # in [-1,1] but typically [0,1]
    sim = max(min((sim + 1) / 2.0, 1.0), 0.0)  # normalize if necessary -> [0,1]
    g = grammar_issues(student_ans)
    issues = g.get("issues_count") if g.get("available") else None

    criteria = rubric.get("criteria", [])
    total_weight = sum(c.get("weight", 0) for c in criteria) or 1.0
    total_score = 0.0
    breakdown = []
    for c in criteria:
        name = c.get("name", "criterion")
        w = c.get("weight", 0) / total_weight
        t = c.get("type", "similarity")
        subscore = 0.0
        if t == "similarity":
            # map sim [0,1] -> [0,100]
            subscore = sim * 100
        elif t == "grammar_penalty":
            # base 100, subtract penalty per error (or relative)
            if issues is None:
                subscore = 100.0
            else:
                penalty_per = c.get("penalty_per_issue", 2.0)
                subscore = max(0.0, 100.0 - issues * penalty_per)
        else:
            # unknown: fallback to similarity
            subscore = sim * 100
        total_score += subscore * w
        breakdown.append({"criterion": name, "weight": w, "subscore": round(subscore, 2)})
    final_score = round(total_score, 2)
    return {"final_score": final_score, "breakdown": breakdown, "similarity": sim, "grammar": g}

def heuristic_grade(model_ans: str, student_ans: str, scale_to_band: Optional[str] = None) -> Dict[str, Any]:
    """Fallback grading when no JSON rubric provided."""
    try:
        emb_model, emb_student = embed_texts([model_ans, student_ans])
    except Exception as e:
        st.error(f"Embedding error in heuristic grading: {e}")
        raise
    sim = cosine_sim(emb_model, emb_student)
    # Normalize similarity from -1..1 to 0..1
    sim_norm = max(min((sim + 1) / 2.0, 1.0), 0.0)
    # Base score out of 100
    base = sim_norm * 100
    # Grammar penalty (if available)
    g = grammar_issues(student_ans)
    if g.get("available"):
        issues = g["issues_count"]
        penalty = min(40, issues * 1.5)  # cap penalty
    else:
        penalty = 0
    final = round(max(0.0, base - penalty), 2)

    # If user wants IELTS-like band mapping (0-9)
    band = None
    if scale_to_band == "ielts" or scale_to_band == "band_9":
        # simple mapping
        band = round((final / 100) * 9, 1)

    return {"final_score": final, "band": band, "similarity": sim_norm, "grammar": g, "penalty": penalty}

# ---------------------------
# Streamlit UI
# ---------------------------

st.title("📝 RAG-based Student Work Auto-Grader (Groq)")
st.markdown(
    """
    Upload: Exercise description, Model solution, Grading rubric (optional, JSON), and student submissions.
    The app will use a RAG-style similarity check (Groq embeddings) plus rubric rules (if provided) to grade.
    """
)

with st.sidebar:
    st.header("Run Configuration")
    st.info("Set GROQ_API_URL and GROQ_API_KEY as environment variables before running the app.")
    col1, col2 = st.columns(2)
    with col1:
        test_band = st.selectbox("Output scale", ["numeric_100", "ielts_band_0-9"], index=0)
    with col2:
        show_examples = st.checkbox("Show grammar examples in output", value=True)
    st.markdown("---")
    if st.button("Quick Test (sample)"):
        st.experimental_set_query_params(sample="1")
        st.success("You can paste example text into the boxes below.")

# File inputs / textareas
st.header("Inputs")

colA, colB = st.columns(2)

with colA:
    st.subheader("Exercise Description")
    ex_upload = st.file_uploader("Upload exercise description (.txt or .docx) or paste below", type=["txt", "docx"])
    ex_paste = st.text_area("Or paste exercise description here", height=120)

with colB:
    st.subheader("Model Solution (Reference)")
    model_upload = st.file_uploader("Upload model solution (.txt or .docx) or paste below", type=["txt", "docx"])
    model_paste = st.text_area("Or paste model solution here", height=120)

st.subheader("Grading Rubric (optional, JSON)")
rubric_upload = st.file_uploader("Upload rubric.json (optional), or paste JSON below", type=["json"])
rubric_paste = st.text_area("Or paste rubric JSON here", height=140)

st.subheader("Student Submissions (multiple)")
st.markdown("Upload multiple files or paste multiple submissions separated by `---` lines.")
student_files = st.file_uploader("Upload student files (.txt, .docx). You can upload multiple.", accept_multiple_files=True, type=["txt", "docx"])
student_paste = st.text_area("Or paste multiple student submissions here (separate with a line `---`)", height=180)

if st.button("Run Grader"):
    # Read texts
    exercise_text = ""
    if ex_upload:
        exercise_text = read_text_file(ex_upload)
    if not exercise_text and ex_paste.strip():
        exercise_text = ex_paste.strip()
    if not exercise_text:
        st.error("Missing: Exercise Description. Please upload/paste it.")
        st.stop()

    model_text = ""
    if model_upload:
        model_text = read_text_file(model_upload)
    if not model_text and model_paste.strip():
        model_text = model_paste.strip()
    if not model_text:
        st.error("Missing: Model Solution. Please upload/paste it.")
        st.stop()

    # Rubric
    rubric_text = ""
    rubric_obj = None
    if rubric_upload:
        rubric_text = rubric_upload.getvalue().decode("utf-8")
    if not rubric_text and rubric_paste.strip():
        rubric_text = rubric_paste.strip()
    if rubric_text:
        rubric_obj = safe_load_json(rubric_text)
        if rubric_obj is None:
            st.error("Rubric JSON appears invalid. Please provide valid JSON.")
            st.stop()

    # Students: from files
    student_texts = []
    student_names = []
    if student_files:
        for f in student_files:
            txt = read_text_file(f)
            if txt.strip():
                student_texts.append(txt.strip())
                student_names.append(f.name)
    # from paste
    if student_paste.strip():
        parts = [p.strip() for p in student_paste.split("\n---\n") if p.strip()]
        for i, p in enumerate(parts):
            student_texts.append(p)
            student_names.append(f"Pasted_{i+1}")

    if not student_texts:
        st.error("No student submissions provided. Upload or paste at least one student's answer.")
        st.stop()

    # Show progress
    with st.spinner("Computing embeddings and grading..."):
        results = []
        for idx, s in enumerate(student_texts):
            try:
                if rubric_obj:
                    res = apply_json_rubric(rubric_obj, model_text, s)
                    final = res["final_score"]
                    band = None
                else:
                    h = heuristic_grade(model_text, s, scale_to_band="ielts" if test_band.startswith("ielts") else None)
                    final = h["final_score"]
                    band = h.get("band")
                    res = h
                # Compose explanation (concise)
                reasoning = (
                    f"The student's answer has a similarity score to the model reference of "
                    f"{round(res.get('similarity', 0)*100,2)}%. "
                    f"Grammar checks: {res.get('grammar',{}).get('issues_count','N/A')} issues."
                )
                feedback = []
                # Strengths/weaknesses
                if res.get("similarity", 0) > 0.75:
                    feedback.append("Good alignment with the model answer and strong task achievement.")
                else:
                    feedback.append("Limited overlap with the model answer; consider addressing key points more directly.")
                if res.get("grammar",{}).get("available"):
                    if res["grammar"]["issues_count"] > 6:
                        feedback.append("Multiple grammar errors — review sentence structure and verb forms.")
                # actionable steps
                feedback.append("Actionable: (1) Re-read the task and ensure each required point is answered; (2) Use simpler sentence structures to avoid grammar mistakes; (3) Review cohesion devices (linking words) to improve coherence.")
                results.append({
                    "name": student_names[idx],
                    "final_score": final,
                    "band": band,
                    "reasoning": reasoning,
                    "feedback": feedback,
                    "details": res
                })
            except Exception as e:
                results.append({
                    "name": student_names[idx],
                    "error": str(e)
                })
        time.sleep(0.2)

    # UI: present results in a nice format
    st.header("Results")
    cols = st.columns([1, 3])
    for r in results:
        with st.container():
            st.subheader(r.get("name", "Student"))
            if r.get("error"):
                st.error(f"Error grading: {r['error']}")
                continue
            score_text = f"Score: {r['final_score']} / 100"
            if r.get("band") is not None:
                score_text += f"  — Band (approx): {r['band']}/9"
            st.metric("Final Score", score_text)
            st.markdown("**Reasoning (concise):**")
            st.write(r["reasoning"])
            st.markdown("**Actionable Feedback:**")
            for f in r["feedback"]:
                st.write(f"- {f}")
            st.markdown("**Details / Breakdown:**")
            try:
                st.json(r["details"])
            except Exception:
                st.write(r["details"])
            # Optional grammar examples
            if show_examples and r["details"].get("grammar",{}).get("available"):
                g = r["details"]["grammar"]
                st.markdown("**Grammar Examples:**")
                st.write(f"Issues found: {g['issues_count']}")
                for ex in g["examples"]:
                    st.write(f"- {ex['message']} — ...{ex['context']}...")
            st.divider()

st.markdown("---")
st.markdown(
    """
    **Notes & Next Steps**
    - Provide a structured rubric JSON (recommended) for strict, reproducible grading.
    - Edit `embed_texts()` to match your exact Groq endpoint request/response format.
    - For production: persist embeddings in a vector DB (FAISS, Milvus) and implement chunked retrieval for longer model answers.
    """
)
