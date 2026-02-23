import streamlit as st
import pickle
import re
from typing import List, Dict

MODEL_PATH = "gu_qa_model.pkl"

# =========================
# Load model (.pkl)
# =========================
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

M = load_model()

GUJ_DIGITS = M["GUJ_DIGITS"]
STOP_WORDS = set(M["STOP_WORDS"])
BAD_IN_WORD_IN = set(M["BAD_IN_WORD_IN"])
T = M["TEMPLATES"]

NUM  = rf"[{GUJ_DIGITS}0-9][{GUJ_DIGITS}0-9,]*"
YEAR = rf"(?:{NUM})\s*માં"
DATE = rf"(?:{NUM}\s+\S+\s+{NUM}\s*માં)"

# =========================
# Helpers
# =========================
def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()

def sent_split(ctx: str) -> List[str]:
    sents = re.split(r"[।!?]\s*", norm(ctx))
    return [s for s in sents if s]

# =========================
# Extractors (STRICT)
# =========================
def extract_quoted(ctx: str) -> List[str]:
    out = []
    for m in re.finditer(r"[\"'‘’“”](.+?)[\"'‘’“”]", ctx):
        q = norm(m.group(1))
        if 2 <= len(q) <= 50 and q in ctx:
            out.append(q)
    return list(dict.fromkeys(out))

def extract_when(ctx: str) -> List[str]:
    out = []
    for m in re.finditer(DATE, ctx):
        out.append(norm(m.group(0)))
    for m in re.finditer(YEAR, ctx):
        out.append(norm(m.group(0)))
    out = list(dict.fromkeys(out))
    out.sort(key=len, reverse=True)
    return [x for x in out if x in ctx]

def extract_who(ctx: str) -> List[str]:
    out = []
    for s in sent_split(ctx):
        m = re.search(r"([^\s,।]+(?:\s+[^\s,।]+){0,8}\s+શાહે)", s)
        if not m:
            continue
        chunk = norm(m.group(1))
        toks = chunk.split()

        if len(toks) >= 4 and toks[-1] == "શાહે":
            cand = " ".join(toks[-4:])
        elif len(toks) >= 3 and toks[-1] == "શાહે":
            cand = " ".join(toks[-3:])
        else:
            continue

        ct = cand.split()
        if ct and ct[0] in STOP_WORDS and len(ct) > 1:
            cand = " ".join(ct[1:])

        if cand in ctx:
            out.append(cand)

    out = list(dict.fromkeys(out))
    out.sort(key=len, reverse=True)
    return out

def extract_river(ctx: str) -> List[str]:
    out = []
    for s in sent_split(ctx):
        m = re.search(r"(?<!\S)([^\s,।]{2,20}(?:\s+[^\s,।]{2,20}){0,1}\s+નદી)(?!\S)", s)
        if m:
            a = norm(m.group(1))
            if 3 <= len(a) <= 30 and a in ctx:
                out.append(a)
    out = list(dict.fromkeys(out))
    out.sort(key=len, reverse=True)
    return out

def extract_where(ctx: str) -> List[str]:
    out = []
    for s in sent_split(ctx):
        # "ગુજરાતના જુનાગઢ જિલ્લામાં" => "જુનાગઢ જિલ્લામાં"
        for m in re.finditer(r"(?:ગુજરાતના\s+)?([^\s,।]{2,30})\s+જિલ્લામાં", s):
            dist = norm(m.group(1) + " જિલ્લામાં")
            if dist in ctx:
                out.append(dist)

        m2 = re.search(r"([^\s,।]{2,30}\s+નદીના\s+કિનારે)", s)
        if m2:
            a = norm(m2.group(1))
            if a in ctx:
                out.append(a)

        # optional "<X>માં" (very strict)
        m3 = re.search(r"(?<!\S)([^\s,।]{3,25}\s*માં)(?!\S)", s)
        if m3:
            a = norm(m3.group(1))
            if a not in BAD_IN_WORD_IN and 3 <= len(a) <= 15 and a in ctx:
                out.append(a)

    out = list(dict.fromkeys(out))
    out.sort(key=len, reverse=True)
    return out

def extract_lists(ctx: str) -> List[str]:
    out = []
    for s in sent_split(ctx):
        # commas list
        if "," in s or "،" in s:
            parts = [norm(p) for p in re.split(r"[،,]", s) if norm(p)]
            if len(parts) >= 3:
                joined = ", ".join(parts[:3])
                if 6 <= len(joined) <= 140 and joined in ctx:
                    out.append(joined)

        # "A, B અને C" style (Gujarati)
        if " અને " in s:
            # crude split: replace " અને " with comma and re-use
            tmp = s.replace(" અને ", ", ")
            parts = [norm(p) for p in re.split(r"[،,]", tmp) if norm(p)]
            if len(parts) >= 3:
                joined = ", ".join(parts[:3])
                if 6 <= len(joined) <= 140 and joined in ctx:
                    out.append(joined)

    out = list(dict.fromkeys(out))
    out.sort(key=len, reverse=True)
    return out

# =========================
# QA Generator (STRICT)
# =========================
def generate_strict_qas(ctx: str, max_qas=12) -> List[Dict]:
    ctx = norm(ctx)
    qas = []
    used = set()

    def add(q, a, qtype):
        a = norm(a)
        if not a or (a not in ctx):
            return
        if a in used:
            return
        used.add(a)
        qas.append({"question": q, "answer": a, "question_type": qtype})

    # WHO
    for a in extract_who(ctx):
        add(T["who"], a, "factual")
        if len(qas) >= max_qas: return qas

    # WHEN
    for a in extract_when(ctx):
        add(T["when"], a, "numerical/date")
        if len(qas) >= max_qas: return qas

    # WHERE
    for a in extract_where(ctx):
        add(T["where"], a, "factual")
        if len(qas) >= max_qas: return qas

    # RIVER
    for a in extract_river(ctx):
        add(T["river"], a, "factual")
        if len(qas) >= max_qas: return qas

    # QUOTED
    for a in extract_quoted(ctx):
        add(T["quoted"].replace("{X}", a), a, "definition")
        if len(qas) >= max_qas: return qas

    # LIST
    for a in extract_lists(ctx):
        add(T["list"], a, "list")
        if len(qas) >= max_qas: return qas

    # TOPIC fallback
    first = sent_split(ctx)[0] if sent_split(ctx) else ctx
    words = first.split()
    if len(words) >= 3:
        a = " ".join(words[:3])
        if a in ctx:
            add(T["topic"], a, "factual")

    return qas

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Gujarati Strict QA Generator", layout="wide")
st.title("Gujarati Strict QA Generator (Paragraph-only ✅)")

st.markdown(
    "✅ Generates questions & answers **ONLY from the paragraph** (no outside info). "
    "Answers are kept only if they are exact substrings of your input."
)

ctx = st.text_area("Paste Gujarati paragraph here:", height=220)

col1, col2 = st.columns([1, 1])
with col1:
    max_qas = st.slider("Max QAs", 3, 25, 12)
with col2:
    run_btn = st.button("Generate Q/A")

if run_btn:
    if not ctx.strip():
        st.warning("Please paste a paragraph first.")
    else:
        qas = generate_strict_qas(ctx, max_qas=max_qas)
        if not qas:
            st.error("No strict QAs found from this paragraph. (Try adding clear dates/names/locations.)")
        else:
            st.subheader("Generated Q/A")
            for i, qa in enumerate(qas, 1):
                st.markdown(f"**{i}. [{qa['question_type']}] {qa['question']}**")
                st.write(f"✅ **Answer:** {qa['answer']}")
                st.divider()
