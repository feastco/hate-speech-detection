# D:\www\hate-speech\backend\main.py
import sys
import time
import logging
import os
import numpy as np
import joblib
from pathlib import Path
from typing import Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# ── Path Setup (penting di Windows!) ─────────────────────────────────────────
# Tambahkan root proyek ke sys.path agar 'preprocessing' bisa diimport
ROOT_DIR = Path(__file__).parent.parent  # D:\www\hate-speech
sys.path.insert(0, str(ROOT_DIR))

from preprocessing import preprocess_to_string
from schemas import (
    PredictRequest, PredictResponse,
    BatchPredictRequest, BatchPredictResponse,
    HealthResponse, ModelInfoResponse
)

# ── Konfigurasi Logging ───────────────────────────────────────────────────────
LOG_FILE = Path(__file__).parent / 'app.log'

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_FILE), mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger("hate_detect")

# ── Path Model ────────────────────────────────────────────────────────────────
MODELS_DIR = ROOT_DIR / 'models'
AVAILABLE_MODELS = {
    'svm': {
        'file': 'svm_model.pkl',
        'name': 'LinearSVC (SVM Linear Kernel)'
    },
    'mnb': {
        'file': 'mnb_model.pkl',
        'name': 'Multinomial Naive Bayes'
    },
    'cnb': {
        'file': 'cnb_model.pkl',
        'name': 'Complement Naive Bayes'
    }
}

INDOBERT_DIR_CANDIDATES = [
    Path(os.getenv('INDOBERT_MODEL_DIR', '')).expanduser() if os.getenv('INDOBERT_MODEL_DIR') else None,
    ROOT_DIR / 'indobert_hate_speech_model',
    ROOT_DIR / 'models' / 'indobert_hate_speech_model',
]

# ── Cache Global ──────────────────────────────────────────────────────────────
MODEL_CACHE = {}


def _resolve_indobert_dir() -> Path | None:
    for candidate in INDOBERT_DIR_CANDIDATES:
        if candidate and candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _build_indobert_label_map(indobert_model) -> Dict[int, str]:
    id2label = getattr(indobert_model.config, 'id2label', {}) or {}
    normalized = {}
    for k, v in id2label.items():
        try:
            idx = int(k)
        except (TypeError, ValueError):
            continue
        label_text = str(v).strip()
        label_norm = label_text.lower().replace('_', ' ')
        
        # Abaikan label default HuggingFace (LABEL_0, dst) agar fallback yang menangani
        if label_norm.startswith('label'):
            continue
            
        if label_norm in {'normal', 'neutral'}:
            normalized[idx] = 'Normal'
        elif label_norm in {'hate speech', 'hatespeech', 'hs'}:
            normalized[idx] = 'Hate Speech'
        elif label_norm in {'abusive', 'abuse'}:
            normalized[idx] = 'Abusive'
        else:
            normalized[idx] = label_text

    fallback = {
        0: 'Normal',
        1: 'Hate Speech',
        2: 'Abusive',
    }
    for idx, lbl in fallback.items():
        normalized.setdefault(idx, lbl)
    return normalized


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model saat startup, bersihkan saat shutdown."""
    logger.info("Server starting — loading models...")
    try:
        for model_key, model_cfg in AVAILABLE_MODELS.items():
            MODEL_CACHE[model_key] = joblib.load(MODELS_DIR / model_cfg['file'])
        MODEL_CACHE['vectorizer'] = joblib.load(MODELS_DIR / 'tfidf_vectorizer.pkl')
        MODEL_CACHE['top_terms'] = joblib.load(MODELS_DIR / 'top_terms.pkl')

        indobert_dir = _resolve_indobert_dir()
        if indobert_dir:
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                indobert_tokenizer = AutoTokenizer.from_pretrained(str(indobert_dir))
                indobert_model = AutoModelForSequenceClassification.from_pretrained(str(indobert_dir))
                indobert_model.eval()

                MODEL_CACHE['indobert'] = indobert_model
                MODEL_CACHE['indobert_tokenizer'] = indobert_tokenizer
                MODEL_CACHE['indobert_model'] = indobert_model
                MODEL_CACHE['indobert_label_map'] = _build_indobert_label_map(indobert_model)

                AVAILABLE_MODELS['indobert'] = {
                    'file': str(indobert_dir),
                    'name': 'IndoBERT Fine-Tuned'
                }
                logger.info(f"IndoBERT loaded from: {indobert_dir}")
            except ImportError:
                logger.warning("Transformers/Torch belum terinstall. Opsi model IndoBERT dinonaktifkan.")
            except Exception as e:
                logger.warning(f"Gagal load IndoBERT dari '{indobert_dir}': {e}")
        else:
            logger.info("Folder model IndoBERT tidak ditemukan. Opsi IndoBERT tidak diaktifkan.")

        n_feat = len(MODEL_CACHE['vectorizer'].vocabulary_)
        logger.info(
            f"Models loaded: {', '.join(AVAILABLE_MODELS.keys())}. "
            f"Vocabulary: {n_feat} features."
        )
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        logger.error("Pastikan notebook 03_modeling sudah dijalankan sampai selesai!")
        raise
    yield
    MODEL_CACHE.clear()
    logger.info("Server shutdown complete.")


# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="HateDetect ID API",
    description="Deteksi ujaran kebencian bahasa Indonesia — SVM + TF-IDF",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",    # Vite dev server
        "http://localhost:3000",    # alternatif
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper ────────────────────────────────────────────────────────────────────
def _resolve_model(model_type: str):
    model_key = (model_type or 'svm').strip().lower()
    if model_key not in AVAILABLE_MODELS:
        supported = ', '.join(AVAILABLE_MODELS.keys())
        raise HTTPException(status_code=422, detail=f"model_type tidak valid. Gunakan: {supported}")
    if model_key not in MODEL_CACHE:
        raise HTTPException(status_code=503, detail=f"Model '{model_key}' belum ter-load")
    return model_key, MODEL_CACHE[model_key]


def _get_confidence(model, X, predicted_label: str) -> float:
    """Hitung confidence dari API model yang tersedia (proba / decision function)."""
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)[0]
        return float(np.max(proba))

    if hasattr(model, 'decision_function'):
        decision = model.decision_function(X)
        if np.ndim(decision) > 1:
            decision = decision[0]
            exp_d = np.exp(decision - np.max(decision))
            return float(np.max(exp_d) / np.sum(exp_d))
        margin = float(abs(np.ravel(decision)[0]))
        return float(1 / (1 + np.exp(-margin)))

    # Fallback konservatif jika model tidak menyediakan score.
    return 0.5


def _predict_one(text: str, model_type: str = 'svm') -> PredictResponse:
    """Shared prediction logic untuk /predict dan /batch."""
    t_start = time.time()
    model_key, model = _resolve_model(model_type)

    if model_key == 'indobert':
        try:
            import torch
        except ImportError as e:
            raise HTTPException(status_code=503, detail="Torch belum tersedia untuk inference IndoBERT") from e

        raw_text = (text or '').strip()
        if not raw_text:
            return PredictResponse(
                label="Normal", confidence=0.5,
                top_terms=[], processed_text="",
                inference_time_ms=0.0
            )

        tokenizer = MODEL_CACHE['indobert_tokenizer']
        label_map = MODEL_CACHE.get('indobert_label_map', {})
        encoded = tokenizer(raw_text, return_tensors='pt', truncation=True, padding=True, max_length=128)

        with torch.no_grad():
            logits = model(**encoded).logits[0]
            probs = torch.softmax(logits, dim=-1)
            pred_idx = int(torch.argmax(probs).item())
            confidence = float(torch.max(probs).item())

        pred_label = label_map.get(pred_idx, f"LABEL_{pred_idx}")
        return PredictResponse(
            label=pred_label,
            confidence=round(confidence, 4),
            top_terms=[],
            processed_text=raw_text,
            inference_time_ms=round((time.time() - t_start) * 1000, 1)
        )

    processed = preprocess_to_string(text)

    if not processed:
        return PredictResponse(
            label="Normal", confidence=0.5,
            top_terms=[], processed_text="",
            inference_time_ms=0.0
        )

    X = MODEL_CACHE['vectorizer'].transform([processed])
    label = model.predict(X)[0]
    confidence = _get_confidence(model, X, label)

    return PredictResponse(
        label=label,
        confidence=round(confidence, 4),
        top_terms=MODEL_CACHE['top_terms'].get(f"{model_key}:{label}", MODEL_CACHE['top_terms'].get(label, [])),
        processed_text=processed,
        inference_time_ms=round((time.time() - t_start) * 1000, 1)
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    return HealthResponse(status="ok", model_loaded=bool(MODEL_CACHE))


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Monitoring"])
async def model_info(model_type: str = 'svm'):
    if not MODEL_CACHE:
        raise HTTPException(status_code=503, detail="Model belum ter-load")
    model_key, model = _resolve_model(model_type)
    if model_key == 'indobert':
        label_map = MODEL_CACHE.get('indobert_label_map', {})
        return ModelInfoResponse(
            model_name=AVAILABLE_MODELS[model_key]['name'],
            n_features=0,
            classes=list(label_map.values()) or ['Normal', 'Hate Speech', 'Abusive'],
            version="1.0.0"
        )

    return ModelInfoResponse(
        model_name=AVAILABLE_MODELS[model_key]['name'],
        n_features=len(MODEL_CACHE['vectorizer'].vocabulary_),
        classes=list(model.classes_),
        version="1.0.0"
    )


@app.get("/models", tags=["Monitoring"])
async def list_models() -> Dict[str, object]:
    loaded = sorted([k for k in AVAILABLE_MODELS.keys() if k in MODEL_CACHE])
    return {
        'default_model': 'svm',
        'available_models': list(AVAILABLE_MODELS.keys()),
        'loaded_models': loaded
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(req: PredictRequest):
    if not MODEL_CACHE:
        raise HTTPException(status_code=503, detail="Model belum ter-load")
    try:
        result = _predict_one(req.text, req.model_type)
        logger.info(
            f"PREDICT | model={req.model_type} | text='{req.text[:60]}' | "
            f"label={result.label} | conf={result.confidence:.3f} | "
            f"latency={result.inference_time_ms}ms"
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch", response_model=BatchPredictResponse, tags=["Prediction"])
async def batch_predict(req: BatchPredictRequest):
    if not MODEL_CACHE:
        raise HTTPException(status_code=503, detail="Model belum ter-load")
    if len(req.texts) == 0:
        raise HTTPException(status_code=422, detail="Array texts tidak boleh kosong")

    t_batch = time.time()
    results = [_predict_one(text, req.model_type) for text in req.texts]
    total_time = round((time.time() - t_batch) * 1000, 1)

    labels_summary = {}
    for r in results:
        labels_summary[r.label] = labels_summary.get(r.label, 0) + 1

    logger.info(
        f"BATCH | model={req.model_type} | n={len(req.texts)} | "
        f"summary={labels_summary} | total={total_time}ms"
    )

    return BatchPredictResponse(
        results=results,
        total=len(results),
        total_time_ms=total_time
    )
