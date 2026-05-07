"""
NSL-KDD Network Intrusion Scoring API
"""
import os, logging
from contextlib import asynccontextmanager
from typing import List, Literal

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("nsl-kdd")

# Live feed (50 dernières analyses)
from collections import deque
from datetime import datetime
live_feed = deque(maxlen=50)

# ── Feature definitions ────────────────────────────────────────────────────
FEATURES_41 = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate",
]

FEATURES_15 = [
    "duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","wrong_fragment","hot","logged_in","num_compromised",
    "count","srv_count","serror_rate","srv_serror_rate","rerror_rate"
]

CATEGORICALS = ["protocol_type", "service", "flag"]

# ── Model registry ─────────────────────────────────────────────────────────
class Registry:
    model = None
    scaler = None
    label_encoders = {}
    class_order = [0, 1]   # [normal, attack]
    version = "unknown"
    accuracy = None

def load_model():
    path = os.getenv("MODEL_PATH", "model.pkl")
    if not os.path.exists(path):
        raise RuntimeError(f"model.pkl introuvable. Lance train_model.py d'abord.")
    
    art = joblib.load(path)
    Registry.model         = art["model"]
    Registry.scaler        = art.get("scaler")
    Registry.label_encoders = art.get("label_encoders", {})
    Registry.class_order   = art.get("class_order", [0, 1])
    Registry.version       = art.get("version", "1.0.0")
    Registry.accuracy      = art.get("accuracy")
    logger.info(f"✅ Modèle chargé v{Registry.version} | accuracy={Registry.accuracy}")
    logger.info(f"   class_order: {Registry.class_order} → index 1 = attack")

# ── Preprocessing ──────────────────────────────────────────────────────────
def extract_15(features_41: list) -> list:
    d = dict(zip(FEATURES_41, features_41))
    return [d[f] for f in FEATURES_15]

def preprocess(features_15: list) -> np.ndarray:
    arr = np.array(features_15, dtype=object).reshape(1, -1)
    
    for i, feat in enumerate(FEATURES_15):
        if feat in CATEGORICALS:
            val = str(arr[0, i])
            le = Registry.label_encoders.get(feat)
            if le:
                try:
                    arr[0, i] = le.transform([val])[0]
                except ValueError:
                    logger.warning(f"Valeur inconnue {feat}={val}, fallback=0")
                    arr[0, i] = 0
            else:
                arr[0, i] = 0
    
    arr = arr.astype(float)
    if Registry.scaler:
        arr = Registry.scaler.transform(arr)
    return arr

def predict(arr: np.ndarray) -> dict:
    model = Registry.model
    proba = model.predict_proba(arr)[0]
    
    # class_order = [0, 1] → proba[1] est toujours P(attack=1)
    # On cherche l'index de la classe "1" (attack)
    attack_idx = list(Registry.class_order).index(1)
    attack_prob = float(proba[attack_idx])
    
    score = int(round(attack_prob * 100))
    
    if score < 30:
        label = "normal"
    elif score < 60:
        label = "suspect"
    else:
        label = "attaque"
    
    confidence = round(float(max(proba)), 4)
    return {"score": score, "label": label, "confidence": confidence}

# ── Lifespan ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(title="NSL-KDD Scorer", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Schemas ────────────────────────────────────────────────────────────────
class ScoreRequest(BaseModel):
    features: List = Field(..., min_length=41, max_length=41)

class ScoreResponse(BaseModel):
    score: int
    label: Literal["normal", "suspect", "attaque"]
    confidence: float
    model_version: str

class BatchRequest(BaseModel):
    connections: List[List] = Field(..., min_length=1, max_length=500)

class BatchResponse(BaseModel):
    results: List[ScoreResponse]
    total: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    accuracy: float | None

# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/", response_class=FileResponse)
async def ui():
    return FileResponse("static/index.html")

@app.get("/landing", response_class=FileResponse)
async def landing():
    return FileResponse("static/landing.html")

@app.get("/v1/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        model_loaded=Registry.model is not None,
        model_version=Registry.version,
        accuracy=Registry.accuracy,
    )

@app.post("/v1/score", response_model=ScoreResponse)
async def score(req: ScoreRequest):
    try:
        f15  = extract_15(req.features)
        arr  = preprocess(f15)
        res  = predict(arr)
        logger.info(f"score={res['score']} label={res['label']}")
        # Ajouter au live feed
        live_feed.appendleft({
            "time": datetime.now().strftime("%H:%M:%S"),
            "score": res["score"],
            "label": res["label"],
            "confidence": res["confidence"],
        })
        return ScoreResponse(**res, model_version=Registry.version)
    except Exception as e:
        logger.error(e)
        raise HTTPException(500, str(e))

@app.get("/v1/feed")
async def feed():
    return {"events": list(live_feed)}

@app.post("/v1/batch", response_model=BatchResponse)
async def batch(req: BatchRequest):
    results = []
    for i, f41 in enumerate(req.connections):
        if len(f41) != 41:
            raise HTTPException(400, f"Connexion #{i+1}: 41 features requises")
        try:
            arr = preprocess(extract_15(f41))
            res = predict(arr)
            results.append(ScoreResponse(**res, model_version=Registry.version))
        except Exception as e:
            raise HTTPException(500, f"#{i+1}: {e}")
    return BatchResponse(results=results, total=len(results))