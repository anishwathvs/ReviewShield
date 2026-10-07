# ReviewShield — AI-Powered Fake Product Review Detection Using NLP

**Tagline:** *"Shop Smarter. Trust Reviews with Confidence."*

ReviewShield is an AI-powered consumer review authenticity assistant built on a rigorous Natural Language Processing (NLP) foundation. It helps consumers detect suspicious, computer-generated, or hyper-promotional online product reviews before making purchasing decisions.

---

## 🌟 Product Features & Architecture

```text
                                  ReviewShield Platform
                                           │
         ┌─────────────────────────────────┴─────────────────────────────────┐
         ▼                                                                   ▼
   PUBLIC EXPERIENCE                                              CONSUMER SAAS APP
   ├── Modern Landing Page (Hero, Mockups, FAQ)                    ├── Personal Dashboard (Session Activity)
   ├── Features & Capability Showcase                             ├── Review Analyzer (Real Inference)
   ├── How It Works (4-Step Visual Workflow)                       ├── Side-by-Side Review Comparison
   ├── About ReviewShield (Mission, Privacy, Scope)               ├── Review Language Insights
   └── Split-Screen Login & Registration                          ├── Session Analysis History (Filter & Search)
                                                                  ├── Bookmarked Reviews (Saved Reviews)
                                                                  └── Settings & Technical Specifications
```

---

## ⚙️ Background Machine Learning System

The consumer-facing web application is powered silently in the background by a validated ML pipeline:

- **Dataset:** Benchmark Amazon Reviews Dataset (`dataset/fake reviews dataset.csv`, 40,432 records)
- **Feature Extraction:** TF-IDF Vectorizer (`ngram_range=(1, 2)`, `min_df=3`, 25,000 features, fit strictly on training data)
- **Classification Model:** Multinomial Naive Bayes (`MultinomialNB(alpha=1.0)`) with Laplace smoothing
- **Train/Test Split:** Stratified 80% train (32,328 samples) / 20% test (8,083 samples, `random_state=42`)

### 📈 Verified Test Set Performance (8,083 Unseen Samples)

| Metric | Real Test Value | Percentage / Context |
| :--- | :--- | :--- |
| **Accuracy** | **0.8937** | **89.37%** |
| **Precision (Fake)** | **0.8961** | **89.61%** |
| **Recall (Fake)** | **0.8906** | **89.06%** |
| **F1-Score (Fake)** | **0.8934** | **89.34%** |
| **Macro Average F1** | **0.8937** | **89.37%** |
| **Weighted Average F1** | **0.8937** | **89.37%** |

#### Confusion Matrix
```text
                  Predicted Fake    Predicted Genuine
Actual Fake           3,598                442
Actual Genuine          417              3,626
```

---

## 📁 Project Directory Structure

```text
fakeproduct/
│
├── dataset/                     # Real Kaggle CSV dataset
│   ├── fake reviews dataset.csv # 40,432 records
│   └── .gitkeep
│
├── models/                      # Serialized ML artifacts
│   ├── model.pkl                # Trained Multinomial Naive Bayes model (800.8 KB)
│   ├── vectorizer.pkl           # Fitted TF-IDF Vectorizer (989.6 KB)
│   ├── metrics.json             # Real test set evaluation metrics (1.7 KB)
│   └── .gitkeep
│
├── src/                         # Reusable ML & NLP source modules
│   ├── __init__.py              # Package identifier
│   ├── data_loader.py           # Dataset loader, schema validator & deduplication
│   ├── preprocessing.py         # Text cleaning (URLs, lowercase, punctuation, stopwords)
│   ├── train_model.py           # Training pipeline (TF-IDF + Multinomial Naive Bayes)
│   ├── predict.py               # Inference engine with probability outputs
│   └── evaluate.py              # Real metrics calculation (Accuracy, Precision, Recall, F1)
│
├── app.py                       # Modern ReviewShield Consumer SaaS Web Application
├── requirements.txt             # Pinned library dependencies
├── README.md                    # Project documentation
└── .gitignore                   # Git ignore configurations
```

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the ReviewShield Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

*(For academic demonstration, you can immediately click **"⚡ Continue as Demo User"** on the Login screen for instant access).*

### 3. (Optional) Re-run ML Pipeline Commands
```bash
# Validate dataset integrity
python -m src.data_loader

# Re-run model training and evaluation
python -m src.train_model
```

---

## 🛡️ Consumer Trust & Ethical AI Guidelines
1. **Probabilistic Guidance:** ReviewShield presents model likelihoods (*"Likely Genuine"* or *"Likely Fake"*); it never makes absolute legal claims of fraud.
2. **Contextual Language Signals:** Heuristic surface indicators (word counts, punctuation stress, hype patterns) provide auxiliary reading context and do not independently override the statistical ML classifier.
3. **Session Privacy:** Review analyses are conducted in-memory and stored exclusively within the user's active session state.
