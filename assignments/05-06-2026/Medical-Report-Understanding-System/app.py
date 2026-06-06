"""
╔══════════════════════════════════════════════════════════╗
║   INTELLIGENT MEDICAL REPORT UNDERSTANDING SYSTEM       ║
║   Explainable Healthcare Dashboard — Task 7             ║
║   Python 3.11+ | PyTorch (no TensorFlow)               ║
╚══════════════════════════════════════════════════════════╝
Run:  streamlit run app.py
"""

import re, math, warnings, io, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical NLP Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark gradient header */
.main-header {
    background: linear-gradient(135deg, #0D0D2B 0%, #1a1a4e 40%, #2d0a5e 100%);
    padding: 2rem 2.5rem;
    border-radius: 18px;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(67,97,238,0.4);
    text-align: center;
}
.main-header h1 {
    color: #F72585; font-size: 2.4rem; font-weight: 700;
    text-shadow: 0 0 30px rgba(247,37,133,0.6);
    margin: 0; letter-spacing: 1px;
}
.main-header p { color: #b0b8ff; font-size: 1.05rem; margin-top: .5rem; }

/* Metric cards */
.metric-card {
    background: linear-gradient(145deg, #1e1e4f, #2a1060);
    border: 1px solid rgba(67,97,238,0.4);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(67,97,238,0.2);
    transition: transform .2s;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-value { font-size: 2.2rem; font-weight: 700; color: #4CC9F0; }
.metric-label { font-size: .88rem; color: #a0a8e8; margin-top: .3rem; }

/* Section headers */
.section-title {
    font-size: 1.35rem; font-weight: 700;
    color: #4CC9F0;
    border-left: 4px solid #F72585;
    padding-left: 12px;
    margin: 1.5rem 0 1rem 0;
}

/* Prediction box */
.pred-box {
    background: linear-gradient(135deg, #1a0040, #2d0a5e);
    border: 2px solid #7209B7;
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(114,9,183,0.4);
}
.pred-specialty { font-size: 1.9rem; font-weight: 700; color: #4CC9F0; }
.pred-confidence { font-size: 1.1rem; color: #F72585; margin-top: .5rem; }

/* Tag badges */
.badge {
    display: inline-block;
    background: rgba(76,201,240,0.15);
    border: 1px solid #4CC9F0;
    border-radius: 20px;
    padding: 3px 12px;
    margin: 3px;
    font-size: .82rem;
    color: #4CC9F0;
}

/* Streamlit overrides */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D0D2B 0%, #1a1a4e 100%);
}
[data-testid="stSidebar"] * { color: #d0d8ff !important; }
.stButton>button {
    background: linear-gradient(135deg, #4361EE, #7209B7);
    color: white; border: none; border-radius: 10px;
    padding: .6rem 1.6rem; font-weight: 600;
    transition: all .3s;
    width: 100%;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #F72585, #4361EE);
    transform: scale(1.02);
}
div[data-testid="stTextArea"] textarea {
    background: #0f0f2e; color: #e0e8ff;
    border: 1px solid #4361EE; border-radius: 10px;
    font-family: 'Inter', monospace;
}
div.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #0D0D2B;
    border-radius: 12px;
    padding: 4px;
}
div.stTabs [data-baseweb="tab"] {
    background: transparent; color: #a0a8e8;
    border-radius: 8px; padding: 8px 20px;
    font-weight: 600;
}
div.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #4361EE, #7209B7) !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# CONSTANTS & HELPERS
# ─────────────────────────────────────────────────────────
MAX_LEN    = 200
VOCAB_SIZE = 10000
PALETTE    = ["#4361EE","#3A0CA3","#7209B7","#F72585","#4CC9F0",
              "#06D6A0","#FFB703","#FB8500","#E63946","#2EC4B6"]
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# Hard-coded NLTK stop-set so we don't need corpus downloads at runtime
STOP = {
    "i","me","my","myself","we","our","ours","ourselves","you","your",
    "yours","yourself","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs",
    "themselves","what","which","who","whom","this","that","these",
    "those","am","is","are","was","were","be","been","being","have",
    "has","had","having","do","does","did","doing","would","should",
    "could","might","may","shall","will","can","a","an","the","and",
    "but","if","or","because","as","until","while","of","at","by",
    "for","with","about","against","between","into","through","during",
    "before","after","above","below","to","from","up","down","in",
    "out","on","off","over","under","again","further","then","once",
    "here","there","when","where","why","how","all","each","every",
    "both","few","more","most","other","some","such","no","nor","not",
    "only","own","same","so","than","too","very","s","t","just","don",
    "patient","procedure","history","right","left","normal","noted",
    "well","also","would","placed","used","pain","performed","using",
    "without","within","taken","given","showed","including",
    "however","upon","following","area","good","time","day","days",
    "one","two","three","four","five","able","findings","dr","cc",
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return [t for t in text.split() if len(t) > 2 and t not in STOP]

def get_pe(max_len, embed_dim):
    pe = np.zeros((max_len, embed_dim))
    for pos in range(max_len):
        for i in range(0, embed_dim, 2):
            angle = pos / (10000 ** (i / embed_dim))
            pe[pos, i]   = math.sin(angle)
            if i + 1 < embed_dim:
                pe[pos, i+1] = math.cos(angle)
    return pe

# ─────────────────────────────────────────────────────────
# MODELS (same architecture as notebook)
# ─────────────────────────────────────────────────────────
class BaselineModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout    = nn.Dropout(dropout)
        self.fc1        = nn.Linear(embed_dim, hidden_dim)
        self.bn1        = nn.BatchNorm1d(hidden_dim)
        self.fc2        = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3        = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x):
        emb = self.embedding(x).mean(dim=1)
        emb = self.dropout(emb)
        h   = F.relu(self.bn1(self.fc1(emb)))
        h   = self.dropout(h)
        h   = F.relu(self.fc2(h))
        return self.fc3(h)

class MedicalAttentionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim,
                 num_classes, dropout=0.3, max_len=MAX_LEN):
        super().__init__()
        self.embedding    = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = nn.Embedding(max_len, embed_dim)
        self.attn   = nn.MultiheadAttention(embed_dim, num_heads,
                                             dropout=dropout, batch_first=True)
        self.norm1  = nn.LayerNorm(embed_dim)
        self.ff     = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, embed_dim))
        self.norm2      = nn.LayerNorm(embed_dim)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def forward(self, x, return_attn=False):
        B, L      = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        emb       = self.embedding(x) + self.pos_encoding(positions)
        emb       = self.dropout(emb)
        kpm       = (x == 0)
        ao, aw    = self.attn(emb, emb, emb, key_padding_mask=kpm)
        emb       = self.norm1(emb + ao)
        emb       = self.norm2(emb + self.ff(emb))
        pooled    = emb[:, 0, :]
        if return_attn:
            return self.classifier(pooled), aw
        return self.classifier(pooled)

# ─────────────────────────────────────────────────────────
# DATA & MODEL LOADING (cached)
# ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading dataset & building vocabulary…")
def load_everything():
    DATA_PATH = "mtsamples.csv"
    if not os.path.exists(DATA_PATH):
        st.error("❌ 'mtsamples.csv' not found in the same folder as app.py")
        st.stop()

    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    df["medical_specialty"] = df["medical_specialty"].str.strip()
    df.dropna(subset=["transcription"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    TOP_N     = 10
    top_specs = df["medical_specialty"].value_counts().head(TOP_N).index.tolist()
    df_cls    = df[df["medical_specialty"].isin(top_specs)].copy()
    df_cls.reset_index(drop=True, inplace=True)

    le = LabelEncoder()
    df_cls["label"] = le.fit_transform(df_cls["medical_specialty"])

    df_cls["tokens"] = df_cls["transcription"].apply(clean_text)
    all_tokens = [t for toks in df_cls["tokens"] for t in toks]
    tf         = Counter(all_tokens)
    vocab      = ["<PAD>","<UNK>"] + [t for t,_ in tf.most_common(VOCAB_SIZE-2)]
    word2idx   = {w: i for i, w in enumerate(vocab)}

    NUM_CLASSES = len(le.classes_)

    # ── Load / initialise models ──
    baseline = BaselineModel(VOCAB_SIZE, 128, 256, NUM_CLASSES)
    attn_m   = MedicalAttentionModel(VOCAB_SIZE, 128, 4, 256, NUM_CLASSES)

    for model, path in [(baseline, "baseline_model.pt"),
                        (attn_m,   "attn_model.pt")]:
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()

    return df, df_cls, le, word2idx, NUM_CLASSES, baseline, attn_m, tf

df, df_cls, le, word2idx, NUM_CLASSES, baseline, attn_m, term_freq = load_everything()

def encode_text(text, max_len=MAX_LEN):
    tokens = clean_text(text)
    ids    = [word2idx.get(t, 1) for t in tokens[:max_len]]
    ids   += [0] * (max_len - len(ids))
    return torch.tensor([ids], dtype=torch.long), tokens[:max_len]

def predict(text, model):
    model.eval()
    x, tokens = encode_text(text)
    with torch.no_grad():
        logits = model(x)
    probs  = F.softmax(logits, dim=-1)[0].numpy()
    top_id = int(np.argmax(probs))
    return le.classes_[top_id], probs[top_id], probs, tokens

def get_attention_map(text):
    attn_m.eval()
    x, tokens = encode_text(text)
    with torch.no_grad():
        _, attn_w = attn_m(x, return_attn=True)    # (1, L, L)
    return attn_w[0].cpu().numpy(), tokens

# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 .5rem 0;'>
        <span style='font-size:3rem'>🏥</span>
        <div style='font-size:1.1rem; font-weight:700; color:#F72585; margin-top:.3rem'>
            MedNLP Suite
        </div>
        <div style='font-size:.78rem; color:#8898cc; margin-top:.2rem'>
            Healthcare AI Dashboard
        </div>
    </div>
    <hr style='border-color:#2a2a6e; margin:.8rem 0'>
    """, unsafe_allow_html=True)

    nav = st.radio(
        "Navigate",
        ["🏠  Overview", "🔬 Text Analysis", "📚 Vocabulary",
         "🤖 Model Comparison", "🌀 Positional Encoding",
         "🩺 Live Diagnosis", "📄 PDF Report"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#2a2a6e; margin:.8rem 0'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:.8rem; color:#8898cc; padding:.4rem'>
        📊 Dataset: <b style='color:#4CC9F0'>{len(df):,}</b> reports<br>
        🏷️ Specialties: <b style='color:#4CC9F0'>{df['medical_specialty'].nunique()}</b><br>
        🖥️ Device: <b style='color:#4CC9F0'>{DEVICE.upper()}</b>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1>🏥 Intelligent Medical Report Understanding System</h1>
    <p>Healthcare NLP • Multi-Head Attention • Explainable AI • Positional Encoding</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════
if "Overview" in nav:
    c1, c2, c3, c4 = st.columns(4)
    stats = [
        ("📋", f"{len(df):,}",        "Total Reports"),
        ("🏷️", f"{df['medical_specialty'].nunique()}", "Specialties"),
        ("📝", f"{int(df['transcription'].apply(lambda x: len(str(x).split())).mean()):,}", "Avg Words"),
        ("🔬", f"{len(term_freq):,}", "Unique Terms"),
    ]
    for col, (icon, val, lbl) in zip([c1,c2,c3,c4], stats):
        col.markdown(f"""
        <div class='metric-card'>
            <div style='font-size:2rem'>{icon}</div>
            <div class='metric-value'>{val}</div>
            <div class='metric-label'>{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>📊 Specialty Distribution</div>", unsafe_allow_html=True)
    spec_counts = df["medical_specialty"].value_counts().head(15).reset_index()
    spec_counts.columns = ["Specialty", "Count"]
    fig = px.bar(spec_counts, x="Count", y="Specialty", orientation="h",
                 color="Count", color_continuous_scale="Viridis",
                 title="Top 15 Medical Specialties",
                 template="plotly_dark")
    fig.update_layout(paper_bgcolor="#0D0D2B", plot_bgcolor="#0D0D2B",
                      font_color="#d0d8ff", height=480,
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        # Pie
        top10 = df["medical_specialty"].value_counts().head(10).reset_index()
        top10.columns = ["Specialty","Count"]
        fig2 = px.pie(top10, values="Count", names="Specialty",
                      hole=0.45, color_discrete_sequence=PALETTE,
                      title="Top 10 Share", template="plotly_dark")
        fig2.update_layout(paper_bgcolor="#0D0D2B", font_color="#d0d8ff", height=380)
        st.plotly_chart(fig2, use_container_width=True)
    with col_b:
        # Word length box
        df["word_count"] = df["transcription"].apply(lambda x: len(str(x).split()))
        fig3 = px.box(df[df["medical_specialty"].isin(
                        df["medical_specialty"].value_counts().head(8).index)],
                      x="word_count", y="medical_specialty",
                      color="medical_specialty", color_discrete_sequence=PALETTE,
                      title="Report Length by Specialty", template="plotly_dark")
        fig3.update_layout(paper_bgcolor="#0D0D2B", font_color="#d0d8ff",
                           height=380, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# PAGE: TEXT ANALYSIS
# ═══════════════════════════════════════════════════════════
elif "Text Analysis" in nav:
    st.markdown("<div class='section-title'>🔬 Task 1 — Medical Text Analysis</div>",
                unsafe_allow_html=True)

    # Term frequency
    all_text = " ".join(df["transcription"].fillna("").tolist())
    tf_counts = Counter(clean_text(all_text)).most_common(40)
    tf_df = pd.DataFrame(tf_counts, columns=["Term","Frequency"])

    fig = px.bar(tf_df, x="Frequency", y="Term", orientation="h",
                 color="Frequency", color_continuous_scale="Plasma",
                 title="Top 40 Medical Terms", template="plotly_dark")
    fig.update_layout(paper_bgcolor="#0D0D2B", plot_bgcolor="#0D0D2B",
                      font_color="#d0d8ff", height=900, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-title'>📋 Report Statistics</div>",
                unsafe_allow_html=True)
    df["word_count"] = df["transcription"].apply(lambda x: len(str(x).split()))
    fig2 = px.histogram(df, x="word_count", nbins=60,
                        color_discrete_sequence=["#4361EE"],
                        title="Report Word-Count Distribution",
                        template="plotly_dark")
    fig2.add_vline(x=df["word_count"].median(), line_color="#F72585", line_dash="dash",
                   annotation_text=f"Median={df['word_count'].median():.0f}",
                   annotation_font_color="#F72585")
    fig2.update_layout(paper_bgcolor="#0D0D2B", plot_bgcolor="#0D0D2B",
                       font_color="#d0d8ff", height=350)
    st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# PAGE: VOCABULARY
# ═══════════════════════════════════════════════════════════
elif "Vocabulary" in nav:
    st.markdown("<div class='section-title'>📚 Task 2 — Medical Vocabulary Builder</div>",
                unsafe_allow_html=True)

    top_n = st.slider("Number of terms to display", 20, 200, 50, 10)
    vocab_df = pd.DataFrame(term_freq.most_common(top_n), columns=["Term","Frequency"])

    col1, col2 = st.columns([3, 2])
    with col1:
        fig = px.treemap(vocab_df, path=["Term"], values="Frequency",
                         color="Frequency", color_continuous_scale="Viridis",
                         title="Medical Vocabulary Treemap",
                         template="plotly_dark")
        fig.update_layout(paper_bgcolor="#0D0D2B", font_color="#d0d8ff",
                          height=500, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.bar(vocab_df.head(25), x="Frequency", y="Term",
                      orientation="h", color="Frequency",
                      color_continuous_scale="Magma",
                      template="plotly_dark",
                      title="Top 25 Frequencies")
        fig2.update_layout(paper_bgcolor="#0D0D2B", plot_bgcolor="#0D0D2B",
                           font_color="#d0d8ff", height=500,
                           coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='section-title'>🔍 Search Term Frequency</div>",
                unsafe_allow_html=True)
    search = st.text_input("Search a medical term", placeholder="e.g. fracture")
    if search:
        freq = term_freq.get(search.strip().lower(), 0)
        rank = sorted(term_freq, key=term_freq.get, reverse=True)
        rank_pos = rank.index(search.strip().lower())+1 if search.strip().lower() in rank else "N/A"
        col_a, col_b = st.columns(2)
        col_a.metric("Frequency", f"{freq:,}")
        col_b.metric("Global Rank", f"#{rank_pos}")

    st.markdown("<div class='section-title'>📋 Full Vocabulary Table</div>",
                unsafe_allow_html=True)
    st.dataframe(vocab_df.style.background_gradient(cmap="Blues", subset=["Frequency"]),
                 use_container_width=True, height=350)

# ═══════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ═══════════════════════════════════════════════════════════
elif "Model Comparison" in nav:
    st.markdown("<div class='section-title'>🤖 Tasks 3 & 4 — Model Architecture Comparison</div>",
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(145deg, #0f1040, #1a1060);
                    border: 1px solid #4361EE; border-radius: 14px; padding: 1.5rem;'>
            <div style='color:#4CC9F0; font-size:1.1rem; font-weight:700;
                        border-bottom:1px solid #333; padding-bottom:.5rem; margin-bottom:.8rem'>
                📦 Baseline Dense Model (Task 3)
            </div>
            <div style='font-family:monospace; font-size:.9rem; color:#a0f0c0; line-height:2'>
                Input → Embedding(10k, 128)<br>
                ↓  Mean Pooling<br>
                ↓  Dropout(0.3)<br>
                ↓  Dense(128→256) + BN + ReLU<br>
                ↓  Dense(256→128) + ReLU<br>
                ↓  Dense(128→10)<br>
                Output → Softmax
            </div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background: linear-gradient(145deg, #0f0030, #1a0050);
                    border: 1px solid #7209B7; border-radius: 14px; padding: 1.5rem;'>
            <div style='color:#F72585; font-size:1.1rem; font-weight:700;
                        border-bottom:1px solid #333; padding-bottom:.5rem; margin-bottom:.8rem'>
                🧠 Self-Attention Model (Task 4)
            </div>
            <div style='font-family:monospace; font-size:.9rem; color:#a0c0ff; line-height:2'>
                Input → Embedding(10k, 128)<br>
                ↓  + Positional Encoding<br>
                ↓  MultiHeadAttention(4 heads)<br>
                ↓  LayerNorm + Residual<br>
                ↓  FeedForward (GELU)<br>
                ↓  LayerNorm + Residual<br>
                ↓  CLS Pooling → Dense(10)<br>
                Output → Softmax
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>📈 Simulated Training Curves</div>",
                unsafe_allow_html=True)

    epochs = list(range(1, 9))
    # representative curves (actual training happens in notebook)
    bl_train = [0.42, 0.55, 0.63, 0.68, 0.72, 0.74, 0.76, 0.77]
    bl_val   = [0.38, 0.50, 0.58, 0.63, 0.66, 0.68, 0.69, 0.70]
    at_train = [0.46, 0.60, 0.68, 0.74, 0.78, 0.81, 0.83, 0.84]
    at_val   = [0.42, 0.55, 0.63, 0.68, 0.72, 0.74, 0.76, 0.77]

    fig = go.Figure()
    for name, tr, va, col in [
        ("Baseline Train", bl_train, None, "#4361EE"),
        ("Baseline Val",   None, bl_val,  "#4361EE"),
        ("Attention Train",at_train, None, "#F72585"),
        ("Attention Val",  None, at_val,  "#F72585"),
    ]:
        vals = tr if tr else va
        dash = "solid" if tr else "dash"
        fig.add_trace(go.Scatter(x=epochs, y=vals, name=name,
                                 mode="lines+markers",
                                 line=dict(color=col, dash=dash, width=2.5),
                                 marker=dict(size=7)))
    fig.update_layout(
        paper_bgcolor="#0D0D2B", plot_bgcolor="#111135",
        font_color="#d0d8ff", height=380,
        legend=dict(bgcolor="#111135", bordercolor="#333"),
        xaxis_title="Epoch", yaxis_title="Accuracy",
        title="Training & Validation Accuracy",
        yaxis=dict(range=[0.3, 0.9])
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info("💡 Train actual model weights by running **medical_nlp_notebook.py** first."
            " The app will auto-load the saved `.pt` files.", icon="ℹ️")

# ═══════════════════════════════════════════════════════════
# PAGE: POSITIONAL ENCODING
# ═══════════════════════════════════════════════════════════
elif "Positional" in nav:
    st.markdown("<div class='section-title'>🌀 Task 5 — Positional Encoding (from Scratch)</div>",
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        pe_len = st.slider("Max sequence length", 20, 200, 100, 10)
    with c2:
        pe_dim = st.slider("Embedding dimension", 16, 128, 64, 8)

    pe = get_pe(pe_len, pe_dim)

    # Heatmap
    fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0D0D2B")
    ax.set_facecolor("#0D0D2B")
    im = ax.imshow(pe, aspect="auto", cmap="RdYlBu_r", vmin=-1, vmax=1)
    ax.set_xlabel("Dimension", color="#d0d8ff")
    ax.set_ylabel("Position", color="#d0d8ff")
    ax.set_title("Sinusoidal Positional Encoding Heatmap",
                 color="#4CC9F0", fontsize=13)
    ax.tick_params(colors="#a0a8e8")
    plt.colorbar(im, ax=ax).ax.yaxis.set_tick_params(color="#a0a8e8")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Waveforms
    st.markdown("<div class='section-title'>〰️ PE Waveforms (first 10 positions)</div>",
                unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(12, 4), facecolor="#0D0D2B")
    ax2.set_facecolor("#111135")
    cols8 = plt.cm.plasma(np.linspace(0.1, 0.9, 10))
    for i in range(10):
        ax2.plot(pe[i, :pe_dim//2], lw=2, color=cols8[i],
                 label=f"pos {i}", alpha=0.9)
    ax2.set_xlabel("Dimension Index", color="#d0d8ff")
    ax2.set_ylabel("Value", color="#d0d8ff")
    ax2.set_title("PE Waveforms", color="#F72585", fontsize=13)
    ax2.legend(fontsize=8, ncol=2, frameon=False,
               labelcolor="#d0d8ff")
    ax2.axhline(0, color="#555", lw=0.8, ls="--")
    ax2.tick_params(colors="#a0a8e8")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # Similarity
    st.markdown("<div class='section-title'>🔁 Position Similarity Matrix</div>",
                unsafe_allow_html=True)
    n_sim = min(pe_len, 60)
    norm  = pe[:n_sim] / (np.linalg.norm(pe[:n_sim], axis=1, keepdims=True) + 1e-9)
    sim   = norm @ norm.T
    fig3, ax3 = plt.subplots(figsize=(8, 7), facecolor="#0D0D2B")
    ax3.set_facecolor("#0D0D2B")
    sns.heatmap(sim, ax=ax3, cmap="coolwarm", center=0, square=True,
                xticklabels=5, yticklabels=5,
                cbar_kws={"label": "Cosine Similarity"})
    ax3.set_title(f"Position Similarity ({n_sim} tokens)",
                  color="#4CC9F0", fontsize=13)
    ax3.tick_params(colors="#a0a8e8")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

# ═══════════════════════════════════════════════════════════
# PAGE: LIVE DIAGNOSIS
# ═══════════════════════════════════════════════════════════
elif "Live Diagnosis" in nav:
    st.markdown("<div class='section-title'>🩺 Task 7 — Live Medical Report Diagnosis</div>",
                unsafe_allow_html=True)

    SAMPLE_REPORTS = {
        "— Select a sample —": "",
        "Cardiology": (
            "The patient presents with severe chest pain radiating to the left arm. "
            "ECG shows ST elevation in leads II, III, aVF consistent with inferior "
            "myocardial infarction. Troponin levels are elevated. Echocardiogram "
            "reveals reduced ejection fraction. Initiated aspirin, heparin and "
            "nitroglycerin. Cardiac catheterization recommended urgently."),
        "Neurology": (
            "Patient is a 58-year-old male presenting with sudden onset weakness "
            "on the right side of the body with aphasia. MRI brain shows ischemic "
            "stroke in the left middle cerebral artery territory. CT angiography "
            "demonstrates occlusion of the left internal carotid artery. tPA "
            "administered within the thrombolysis window. Neurological consultation "
            "for further management of cerebrovascular accident."),
        "Orthopedic": (
            "A 45-year-old female presented with acute right knee pain following "
            "a sports injury. X-ray reveals no fracture but MRI demonstrates "
            "complete tear of the anterior cruciate ligament with bone bruising. "
            "Physical examination confirms Lachman test positive. Arthroscopic "
            "reconstruction recommended with post-operative rehabilitation protocol."),
        "Radiology": (
            "CT chest with contrast performed for evaluation of pulmonary nodule "
            "identified on prior imaging. Current study demonstrates 1.2 cm "
            "spiculated nodule in the right upper lobe with mediastinal lymph "
            "node enlargement. Findings are suspicious for primary lung malignancy. "
            "PET-CT and tissue biopsy recommended for further characterization."),
    }

    col_sel, col_model = st.columns(2)
    with col_sel:
        sel = st.selectbox("Load a sample report", list(SAMPLE_REPORTS.keys()))
    with col_model:
        model_choice = st.radio("Model", ["Attention (Task 4)", "Baseline (Task 3)"],
                                horizontal=True)

    report_text = st.text_area(
        "Medical Report Text",
        value=SAMPLE_REPORTS.get(sel, ""),
        height=180,
        placeholder="Paste or type a medical transcription here…"
    )

    analyze_btn = st.button("🔍  Analyse Report")

    if analyze_btn and report_text.strip():
        model = attn_m if "Attention" in model_choice else baseline
        specialty, confidence, all_probs, tokens = predict(report_text, model)

        # ── Prediction box ──
        st.markdown(f"""
        <div class='pred-box' style='margin:1.2rem 0'>
            <div style='font-size:.9rem; color:#a0b0ff; margin-bottom:.4rem'>
                Predicted Specialty
            </div>
            <div class='pred-specialty'>🏥 {specialty}</div>
            <div class='pred-confidence'>Confidence: {confidence*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📊 Confidence Scores",
                                     "🗺️ Attention Map",
                                     "🌀 PE Heatmap"])

        with tab1:
            prob_df = pd.DataFrame({
                "Specialty": le.classes_,
                "Probability": all_probs
            }).sort_values("Probability", ascending=True)
            fig = px.bar(prob_df, x="Probability", y="Specialty",
                         orientation="h",
                         color="Probability",
                         color_continuous_scale="Plasma",
                         template="plotly_dark",
                         title="Confidence Score per Specialty")
            fig.update_layout(paper_bgcolor="#0D0D2B", plot_bgcolor="#111135",
                              font_color="#d0d8ff", height=420,
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            attn_w, tokens_used = get_attention_map(report_text)
            n_tok = min(len(tokens_used), 40)
            attn_slice = attn_w[:n_tok, :n_tok]

            fig, ax = plt.subplots(figsize=(max(10, n_tok//3), max(8, n_tok//3)),
                                   facecolor="#0D0D2B")
            ax.set_facecolor("#0D0D2B")
            im = ax.imshow(attn_slice, cmap="inferno", aspect="auto")
            ax.set_xticks(range(n_tok))
            ax.set_yticks(range(n_tok))
            ax.set_xticklabels(tokens_used[:n_tok], rotation=90,
                               fontsize=7, color="#a0a8e8")
            ax.set_yticklabels(tokens_used[:n_tok], fontsize=7, color="#a0a8e8")
            ax.set_title("Self-Attention Map", color="#F72585", fontsize=13)
            plt.colorbar(im, ax=ax).ax.yaxis.set_tick_params(color="#a0a8e8")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # CLS token attention (diagnostic importance)
            cls_attn = attn_w[0, :n_tok]
            top_ids  = np.argsort(cls_attn)[::-1][:15]
            imp_df   = pd.DataFrame({
                "Word":   [tokens_used[i] for i in top_ids],
                "Weight": cls_attn[top_ids]
            })
            fig2 = px.bar(imp_df, x="Weight", y="Word", orientation="h",
                          color="Weight", color_continuous_scale="Plasma",
                          title="Most Diagnostically Important Words",
                          template="plotly_dark")
            fig2.update_layout(paper_bgcolor="#0D0D2B", plot_bgcolor="#111135",
                               font_color="#d0d8ff", height=420,
                               coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

            # Display top words as badges
            top_words_html = " ".join(
                f"<span class='badge'>{tokens_used[i]}</span>"
                for i in top_ids[:12]
            )
            st.markdown(f"""
            <div style='background:#0f0f2e; border-radius:10px; padding:1rem; margin:.5rem 0'>
                <div style='color:#a0b0ff; font-size:.85rem; margin-bottom:.5rem'>
                    🔑 Key Diagnostic Terms
                </div>
                {top_words_html}
            </div>""", unsafe_allow_html=True)

        with tab3:
            pe_vis = get_pe(MAX_LEN, 64)
            n_show = min(len(tokens_used), 50)
            pe_slice = pe_vis[:n_show, :]

            fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0D0D2B")
            ax.set_facecolor("#0D0D2B")
            im = ax.imshow(pe_slice, aspect="auto", cmap="RdYlBu_r",
                           vmin=-1, vmax=1)
            ax.set_xlabel("Dimension", color="#d0d8ff")
            ax.set_ylabel("Token Position", color="#d0d8ff")
            ax.set_yticks(range(n_show))
            ax.set_yticklabels(tokens_used[:n_show], fontsize=7, color="#a0a8e8")
            ax.set_title("Positional Encoding for This Report",
                         color="#4CC9F0", fontsize=13)
            plt.colorbar(im, ax=ax).ax.yaxis.set_tick_params(color="#a0a8e8")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    elif analyze_btn:
        st.warning("⚠️ Please enter some medical text to analyse.")

# ═══════════════════════════════════════════════════════════
# PAGE: PDF REPORT  (BONUS)
# ═══════════════════════════════════════════════════════════
elif "PDF" in nav:
    st.markdown("<div class='section-title'>📄 Bonus — Generate Medical Analysis PDF</div>",
                unsafe_allow_html=True)

    report_text = st.text_area(
        "Medical Report Text",
        height=150,
        placeholder="Paste a medical report here to generate a PDF analysis…"
    )
    patient_name = st.text_input("Patient / Report ID (optional)", "P-001")

    gen_btn = st.button("📥  Generate PDF Report")

    if gen_btn and report_text.strip():
        try:
            from fpdf import FPDF
        except ImportError:
            st.error("fpdf2 not installed. Run: `pip install fpdf2`")
            st.stop()

        specialty, confidence, all_probs, tokens = predict(report_text, attn_m)
        attn_w, tokens_used = get_attention_map(report_text)
        top_ids   = np.argsort(attn_w[0, :len(tokens_used)])[::-1][:10]
        top_words = [tokens_used[i] for i in top_ids]

        # ── Build PDF ──
        pdf = FPDF()
        pdf.add_page()

        # Header
        pdf.set_fill_color(13, 13, 43)
        pdf.rect(0, 0, 210, 40, "F")
        pdf.set_font("Helvetica", "B", 20)
        pdf.set_text_color(247, 37, 133)
        pdf.cell(0, 15, "", ln=True)
        pdf.cell(0, 12, "Medical Report Analysis", ln=True, align="C")
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(176, 184, 255)
        pdf.cell(0, 8, "Intelligent Medical NLP System", ln=True, align="C")
        pdf.ln(10)

        # Meta
        pdf.set_text_color(50, 50, 50)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_fill_color(230, 235, 255)
        pdf.cell(0, 8, f"  Report ID: {patient_name}", ln=True, fill=True)
        pdf.ln(4)

        # Prediction
        pdf.set_fill_color(67, 97, 238)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 12, f"  Predicted Specialty: {specialty}", ln=True, fill=True)
        pdf.set_fill_color(114, 9, 183)
        pdf.cell(0, 10, f"  Confidence Score: {confidence*100:.1f}%", ln=True, fill=True)
        pdf.ln(6)

        # All scores
        pdf.set_text_color(30, 30, 80)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Classification Confidence by Specialty:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for cls, prob in sorted(zip(le.classes_, all_probs),
                                key=lambda x: -x[1]):
            bar_w = int(prob * 80)
            pdf.set_fill_color(230, 235, 255)
            pdf.cell(70, 7, f"  {cls}", fill=True)
            pdf.set_fill_color(67, 97, 238)
            pdf.cell(bar_w, 7, "", fill=True)
            pdf.set_fill_color(200, 210, 255)
            pdf.cell(80 - bar_w, 7, f" {prob*100:.1f}%", fill=True)
            pdf.ln()
        pdf.ln(5)

        # Key terms
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(30, 30, 80)
        pdf.cell(0, 8, "Key Diagnostic Terms (Attention-based):", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 7, "  " + ", ".join(top_words))
        pdf.ln(5)

        # Report excerpt
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Report Excerpt:", ln=True)
        pdf.set_font("Helvetica", "", 9)
        excerpt = report_text[:800].replace("\n", " ")
        pdf.multi_cell(0, 5.5, excerpt)

        # Footer
        pdf.set_y(-20)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(150, 150, 180)
        pdf.cell(0, 6,
                 "Generated by Medical NLP System | For Research Use Only",
                 align="C")

        buf = io.BytesIO()
        pdf_bytes = pdf.output()
        buf.write(pdf_bytes)
        buf.seek(0)

        st.success("✅ PDF generated!")
        st.download_button(
            label="📥 Download Medical Analysis PDF",
            data=buf,
            file_name=f"medical_analysis_{patient_name}.pdf",
            mime="application/pdf"
        )
    elif gen_btn:
        st.warning("⚠️ Please enter a medical report first.")

# ─────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:#2a2a6e; margin-top: 3rem'>
<div style='text-align:center; color:#4a4a8a; font-size:.82rem; padding-bottom:1.5rem'>
    🏥 Medical NLP Dashboard  •  Built with PyTorch & Streamlit  •
    For Research &amp; Educational Use Only
</div>
""", unsafe_allow_html=True)