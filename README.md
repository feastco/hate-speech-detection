# Hate Speech Detection (Indonesian Text)

> NLP/ML pipeline for detecting hate speech in Indonesian-language text

[![Stack](https://img.shields.io/badge/Stack-Python%20%7C%20scikit--learn%20%7C%20NLP-blue?style=flat-square)](#tech-stack)
[![GitHub](https://img.shields.io/badge/GitHub-feastco%2Fhate--speech--detection-181717?style=for-the-badge&logo=github)](https://github.com/feastco/hate-speech-detection)

Applied AI / NLP project for **classifying potentially hateful or abusive Indonesian-language text** using classical machine learning, statistical tests, and a FastAPI backend.

## ✨ Features

- **Indonesian hate speech classification** — multi-class (HS / abusive / neutral)
- **Classical ML models** — SVM, Logistic Regression, Random Forest with TF-IDF
- **Statistical validation** — McNemar, Chi-square, Wilson score, IRR analysis
- **Backend API** — FastAPI for serving predictions (`backend/main.py`)
- **Reproducible pipeline** — preprocessing → training → evaluation
- **Dataset analysis** — annotation inter-rater reliability (Cohen's kappa)

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| ML | scikit-learn, pandas, numpy, scipy |
| NLP | TF-IDF, custom preprocessing |
| Backend | FastAPI, uvicorn |
| Stats | McNemar, Chi-square, Wilson CI |
| Notebooks | Jupyter (analysis & experiments) |

## 📂 Project Structure

```
hate-speech-detection/
├── backend/                 # FastAPI service
│   ├── main.py              # API entry
│   ├── schemas.py           # Pydantic models
│   └── app.log
├── notebooks/               # Jupyter experiments
├── pipeline/                # Preprocessing & training scripts
├── models/                  # Trained model artifacts
├── data/                    # Datasets & annotation forms
├── tests/                   # Latency & unit tests
├── requirements.txt
├── preprocessing.py
├── calculate_irr.py
└── latency_test.py
```

## ⚡ Quick Start

```bash
git clone https://github.com/feastco/hate-speech-detection.git
cd hate-speech-detection
python -m venv .venv && source .venv/bin/activate   # or: .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Train / reproduce
python pipeline/train.py

# Run API
uvicorn backend.main:app --reload --port 8000
# Visit http://localhost:8000/docs
```

## 📊 Research Notes

- **IRR** (Inter-Rater Reliability) for annotation quality
- **McNemar's test** for pairwise model comparison
- **Wilson confidence interval** for imbalanced-class metrics
- Full analysis in `notebooks/`

## 👤 Author

**Fisco Maulana Ikhwan** — Informatics Engineering (D3), Universitas Dian Nuswantoro
- GitHub: [@feastco](https://github.com/feastco)
- LinkedIn: [fiscomaulanaikhwan](https://www.linkedin.com/in/fiscomaulanaikhwan)
