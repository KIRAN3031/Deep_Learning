# 🏥 Intelligent Medical Report Understanding System

> **An Explainable Healthcare NLP Dashboard** — powered by PyTorch, Multi-Head Self-Attention, and Streamlit.

🔗 **Live Demo:** [https://deeplearning-bnoy2ucap44a5yfu662lzp.streamlit.app/](https://deeplearning-bnoy2ucap44a5yfu662lzp.streamlit.app/)

---

## 📌 About the Project

This project builds an end-to-end **Natural Language Processing (NLP) pipeline** for understanding medical transcription reports. It classifies reports into medical specialties using deep learning, provides explainable AI through attention visualizations, and packages everything into a production-ready interactive dashboard.

The system processes **4,966 real medical transcription reports** spanning **40 specialties**, demonstrating the full ML lifecycle — from data exploration through model training to deployment.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔬 **Text Analysis** | Frequency analysis, word-count distributions, and medical term extraction |
| 📚 **Vocabulary Builder** | Interactive treemaps, search, and ranked medical terminology |
| 🤖 **Dual Model Architecture** | Baseline Dense Network vs. Self-Attention Transformer |
| 🌀 **Positional Encoding** | From-scratch sinusoidal PE with heatmaps, waveforms & similarity matrices |
| 🩺 **Live Diagnosis** | Real-time report classification with attention-based explainability |
| 📄 **PDF Report Generation** | Downloadable medical analysis reports with confidence scores |
| 🎨 **Premium Dashboard** | Dark-themed, responsive UI with interactive Plotly visualizations |

---

## 🧠 Architecture

### Baseline Dense Model (Task 3)
```
Input → Embedding(10000, 128)
    ↓  Mean Pooling
    ↓  Dropout(0.3)
    ↓  Dense(128→256) + BatchNorm + ReLU
    ↓  Dense(256→128) + ReLU
    ↓  Dense(128→10)
    Output → Softmax
```

### Self-Attention Model (Task 4)
```
Input → Embedding(10000, 128)
    ↓  + Sinusoidal Positional Encoding
    ↓  MultiHeadAttention(4 heads)
    ↓  LayerNorm + Residual Connection
    ↓  FeedForward Network (GELU)
    ↓  LayerNorm + Residual Connection
    ↓  CLS Token Pooling → Dense(10)
    Output → Softmax
```

---

## 📋 Task Breakdown

| Task | Title | Description |
|------|-------|-------------|
| **Task 1** | Medical Text Analysis | Exploratory data analysis — specialty distribution, term frequency, report length statistics |
| **Task 2** | Vocabulary Builder | Build domain-specific medical vocabulary with frequency ranking, treemaps, and search |
| **Task 3** | Baseline Classifier | Train a Dense Neural Network with mean-pooled embeddings for specialty classification |
| **Task 4** | Attention Classifier | Train a Multi-Head Self-Attention model with positional encoding and CLS pooling |
| **Task 5** | Positional Encoding | Implement sinusoidal positional encoding from scratch with interactive visualizations |
| **Task 6** | Explainability | Attention-based word importance, TF-IDF analysis, and diagnostic term identification |
| **Task 7** | Dashboard & Deployment | Full Streamlit dashboard integrating all tasks with live inference and PDF export |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+**
- **pip** (Python package manager)
- **Git** (to clone the repository)

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Medical-Report-Understanding-System.git
cd Medical-Report-Understanding-System
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
| Package | Purpose |
|---------|---------|
| `torch` | Deep learning framework (PyTorch) |
| `scikit-learn` | Label encoding, metrics |
| `numpy` | Numerical operations |
| `pandas` | Data manipulation |
| `matplotlib` | Static visualizations |
| `seaborn` | Statistical plots |
| `plotly` | Interactive charts |
| `streamlit` | Web dashboard framework |
| `fpdf2` | PDF report generation |
| `nltk` | NLP utilities (notebook) |
| `wordcloud` | Word cloud generation (notebook) |

### 3. Run the Jupyter Notebook *(Optional — for training)*

```bash
jupyter notebook medical_report.ipynb
```

> The notebook trains both models and saves `baseline_model.pt` and `attn_model.pt`. Pre-trained weights are already included in this repository.

### 4. Launch the Dashboard

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501** in your browser.

---

## 📁 Project Structure

```
Medical-Report-Understanding-System/
│
├── app.py                      # Streamlit dashboard (Task 7 — main application)
├── medical_report.ipynb        # Jupyter notebook (Tasks 1–6 — training & analysis)
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── mtsamples.csv               # Dataset — 4,966 medical transcription reports
├── medical_vocabulary.csv      # Extracted medical vocabulary
│
├── baseline_model.pt           # Pre-trained Baseline Dense model weights
├── attn_model.pt               # Pre-trained Self-Attention model weights
│
├── task1_distribution.png      # Task 1 — Specialty distribution chart
├── task1_lengths.png           # Task 1 — Report length distribution
├── task2_top50.png             # Task 2 — Top 50 medical terms
├── task2_wordcloud.png         # Task 2 — Medical vocabulary word cloud
├── task3_baseline_training.png # Task 3 — Baseline model training curves
├── task3_confusion.png         # Task 3 — Confusion matrix
├── task4_attn_training.png     # Task 4 — Attention model training curves
├── task4_comparison.png        # Task 4 — Model comparison chart
├── task5_pe_full.png           # Task 5 — Positional encoding heatmap
├── task5_pe_similarity.png     # Task 5 — Position similarity matrix
├── task6_attention_words.png   # Task 6 — Attention-based word importance
└── task6_tfidf_terms.png       # Task 6 — TF-IDF term analysis
```

---

## 📊 Dataset

- **Source:** [MTSamples](https://www.mtsamples.com/) — Medical Transcription Samples
- **Records:** 4,966 medical transcription reports
- **Specialties:** 40 unique medical specialties
- **Top 10 Specialties Used for Classification:**
  Surgery, Consult - History and Phy., Cardiovascular / Pulmonary, Orthopedic, Radiology, General Medicine, Gastroenterology, Neurology, SOAP / Chart / Progress Notes, Urology

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Deep Learning** | PyTorch |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Web Framework** | Streamlit |
| **PDF Generation** | fpdf2 |
| **Language** | Python 3.11+ |

---

## 🧪 How It Works

1. **Data Loading** — Reads `mtsamples.csv`, filters top-10 specialties, builds a vocabulary of 10,000 tokens
2. **Text Preprocessing** — Lowercasing, stopword removal, medical-domain tokenization
3. **Model Inference** — Encodes input text as token IDs, pads/truncates to 200 tokens, runs through the selected model
4. **Attention Explainability** — Extracts self-attention weights to identify diagnostically important words
5. **Visualization** — Renders interactive charts, attention heatmaps, and positional encoding visualizations

---

## 📸 Dashboard Pages

| Page | What You'll See |
|------|-----------------|
| **🏠 Overview** | Dataset statistics, specialty distribution bar chart, pie chart, box plots |
| **🔬 Text Analysis** | Top 40 medical terms, word-count histogram with median line |
| **📚 Vocabulary** | Interactive treemap, frequency bar chart, term search, full vocabulary table |
| **🤖 Model Comparison** | Side-by-side architecture cards, simulated training curves |
| **🌀 Positional Encoding** | Adjustable PE heatmap, waveform plots, cosine similarity matrix |
| **🩺 Live Diagnosis** | Real-time classification, confidence scores, attention map, key diagnostic terms |
| **📄 PDF Report** | Generate and download a formatted medical analysis PDF |

---

## 📝 Usage Examples

### Live Diagnosis
1. Navigate to **🩺 Live Diagnosis** in the sidebar
2. Select a sample report or paste your own medical transcription
3. Choose between the **Attention** or **Baseline** model
4. Click **🔍 Analyse Report**
5. View the predicted specialty, confidence scores, attention heatmap, and key terms

### PDF Report Generation
1. Navigate to **📄 PDF Report** in the sidebar
2. Paste a medical report and enter a patient/report ID
3. Click **📥 Generate PDF Report**
4. Download the formatted analysis PDF

---

## ⚠️ Disclaimer

> This system is built for **research and educational purposes only**. It is NOT intended for clinical diagnosis or medical decision-making. Always consult qualified healthcare professionals for medical advice.

---

## 📜 License

This project is for academic/educational use as part of a Deep Learning assignment.

---

🔗 **Live Demo:** [https://deeplearning-bnoy2ucap44a5yfu662lzp.streamlit.app/](https://deeplearning-bnoy2ucap44a5yfu662lzp.streamlit.app/)
