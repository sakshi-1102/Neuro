import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asyncio
import io
import json
from datetime import datetime

import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from database import DecodedMessage
from database import Session as DBSession
from database import Patient, get_db
from model import load_model, predict, preprocess_signal

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="NeuroVoice API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

eeg_model = load_model("eeg_model.h5")

# ── Health ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "NeuroVoice API running ✅", "version": "1.0.0"}

# ── Patients ───────────────────────────────────────────────────────────────
@app.post("/patients")
def create_patient(name: str, age: int, condition: str, ward: str,
                   db: Session = Depends(get_db)):
    p = Patient(name=name, age=age, condition=condition, ward=ward)
    db.add(p); db.commit(); db.refresh(p)
    return p

@app.get("/patients")
def list_patients(db: Session = Depends(get_db)):
    return db.query(Patient).all()

@app.get("/patients/{patient_id}")
def get_patient(patient_id: int, db: Session = Depends(get_db)):
    p = db.query(Patient).filter(Patient.id == patient_id).first()
    if not p:
        raise HTTPException(404, "Patient not found")
    return p

@app.delete("/patients/{patient_id}")
def delete_patient(patient_id: int, db: Session = Depends(get_db)):
    p = db.query(Patient).filter(Patient.id == patient_id).first()
    if not p:
        raise HTTPException(404, "Patient not found")
    db.delete(p); db.commit()
    return {"message": "Patient deleted"}

# ── Sessions ───────────────────────────────────────────────────────────────
@app.post("/sessions/start/{patient_id}")
def start_session(patient_id: int, db: Session = Depends(get_db)):
    s = DBSession(patient_id=patient_id)
    db.add(s); db.commit(); db.refresh(s)
    return {"session_id": s.id, "started_at": str(s.started_at)}

@app.post("/sessions/end/{session_id}")
def end_session(session_id: int, db: Session = Depends(get_db)):
    s = db.query(DBSession).filter(DBSession.id == session_id).first()
    if not s:
        raise HTTPException(404, "Session not found")
    s.ended_at = datetime.utcnow()
    db.commit()
    return {"message": "Session ended", "total_decoded": s.total_decoded}

# ── Predict ────────────────────────────────────────────────────────────────
@app.post("/predict")
async def predict_from_file(
    session_id: int,
    patient_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    contents = await file.read()
    try:
        if file.filename.endswith(".npy"):
            arr = np.load(io.BytesIO(contents))
        elif file.filename.endswith(".csv"):
            arr = np.loadtxt(io.StringIO(contents.decode()), delimiter=",")
        else:
            arr = np.random.randn(64, 256) * 10   # demo signal
    except Exception:
        arr = np.random.randn(64, 256) * 10

    X      = preprocess_signal(arr)
    result = predict(eeg_model, X)

    msg = DecodedMessage(
        session_id=session_id,
        patient_id=patient_id,
        word=result["word"],
        confidence=result["confidence"],
        all_probabilities=json.dumps(result["all_probabilities"]),
    )
    db.add(msg)

    s = db.query(DBSession).filter(DBSession.id == session_id).first()
    if s:
        s.total_decoded = (s.total_decoded or 0) + 1
    db.commit()
    return result

# ── History ────────────────────────────────────────────────────────────────
@app.get("/history/{patient_id}")
def get_history(patient_id: int, limit: int = 20,
                db: Session = Depends(get_db)):
    msgs = (db.query(DecodedMessage)
              .filter(DecodedMessage.patient_id == patient_id)
              .order_by(DecodedMessage.timestamp.desc())
              .limit(limit).all())
    return msgs

@app.get("/stats/{patient_id}")
def get_stats(patient_id: int, db: Session = Depends(get_db)):
    msgs = (db.query(DecodedMessage)
              .filter(DecodedMessage.patient_id == patient_id).all())
    if not msgs:
        return {"total": 0, "avg_confidence": 0, "most_frequent": "—"}
    from collections import Counter
    words = [m.word for m in msgs]
    return {
        "total":          len(msgs),
        "avg_confidence": round(sum(m.confidence for m in msgs) / len(msgs), 1),
        "most_frequent":  Counter(words).most_common(1)[0][0],
    }

# ── WebSocket ──────────────────────────────────────────────────────────────
@app.websocket("/ws/{patient_id}")
async def websocket_endpoint(websocket: WebSocket, patient_id: int):
    await websocket.accept()
    print(f"🔌 WebSocket connected — patient {patient_id}")
    try:
        while True:
            await asyncio.sleep(3)
            arr    = np.random.randn(64, 256) * 10
            X      = preprocess_signal(arr)
            result = predict(eeg_model, X)
            result["patient_id"] = patient_id
            result["timestamp"]  = datetime.utcnow().isoformat()
            await websocket.send_json(result)
    except Exception as e:
        print(f"WebSocket closed: {e}")