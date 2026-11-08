from __future__ import annotations

import os  # <--- AGREGADO: Necesario para leer variables de entorno
import joblib # <--- AGREGADO: Para cargar el modelo de ML
import time # <--- AGREGADO: Para medir el tiempo de inferencia

from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, String, Float
from sqlalchemy.orm import declarative_base, sessionmaker

app = FastAPI(
    title="ACI606 Cloud Fleet Challenge",
    version="0.1.0-starter",
    description="Starter API for the ACI606 Cloud Fleet Challenge.",
)

Status = Literal["NORMAL", "WARNING", "CRITICAL"]

# --- CONFIGURACIÓN DE LA BASE DE DATOS (NEON) ---
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./local.db")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class TelemetryDB(Base):
    __tablename__ = "telemetry"
    id = Column(String, primary_key=True, index=True)
    robot_id = Column(String, index=True)
    timestamp = Column(Float)
    x = Column(Float)
    y = Column(Float)
    speed = Column(Float)
    battery = Column(Float)
    motor_temp = Column(Float)
    vibration = Column(Float)
    motor_current = Column(Float)
    obstacle_distance = Column(Float)

Base.metadata.create_all(bind=engine)

# --- CARGAR EL MODELO DE ML ---
MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.joblib")
try:
    ml_model = joblib.load(MODEL_PATH)
    print("✅ Modelo cargado exitosamente.")
except Exception as e:
    print(f"⚠️ Error cargando el modelo: {e}")
    ml_model = None
# -------------------------------

class Telemetry(BaseModel):
    robot_id: str = Field(min_length=1, max_length=20)
    timestamp: float
    x: float
    y: float
    speed: float = Field(ge=0)
    battery: float = Field(ge=0, le=100)
    motor_temp: float
    vibration: float = Field(ge=0)
    motor_current: float = Field(ge=0)
    obstacle_distance: float = Field(ge=0)

class PredictionRequest(Telemetry):
    pass

LATEST: dict[str, Telemetry] = {}
HISTORY: dict[str, deque[Telemetry]] = defaultdict(lambda: deque(maxlen=100))
REQUEST_COUNT = 0

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "aci606-cloud-fleet",
        "version": app.version,
        "robots_in_memory": len(LATEST),
        "model_loaded": ml_model is not None, # <--- AGREGADO: Para verificar en Render
        "utc": datetime.now(timezone.utc).isoformat(),
    }

@app.post("/telemetry")
def ingest_telemetry(sample: Telemetry) -> dict:
    global REQUEST_COUNT
    REQUEST_COUNT += 1
    
    db = SessionLocal()
    try:
        unique_id = f"{sample.robot_id}_{sample.timestamp}"
        db_telemetry = TelemetryDB(
            id=unique_id,
            robot_id=sample.robot_id,
            timestamp=sample.timestamp,
            x=sample.x, y=sample.y, speed=sample.speed,
            battery=sample.battery, motor_temp=sample.motor_temp,
            vibration=sample.vibration, motor_current=sample.motor_current,
            obstacle_distance=sample.obstacle_distance
        )
        db.add(db_telemetry)
        db.commit()
    finally:
        db.close()

    LATEST[sample.robot_id] = sample
    HISTORY[sample.robot_id].append(sample)
    
    return {
        "accepted": True,
        "robot_id": sample.robot_id,
        "server_time": datetime.now(timezone.utc).isoformat(),
        "note": "Saved to database successfully.",
    }

@app.get("/robots")
def list_robots() -> list[dict]:
    return [sample.model_dump() for sample in LATEST.values()]

@app.get("/robots/{robot_id}/history")
def robot_history(robot_id: str) -> list[dict]:
    if robot_id not in HISTORY:
        raise HTTPException(status_code=404, detail="Robot not found")
    return [sample.model_dump() for sample in HISTORY[robot_id]]

@app.post("/predict")
def predict(sample: PredictionRequest) -> dict:
    start_time = time.time()

    if ml_model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado en el servidor.")

    features = [[
        sample.speed, sample.battery, sample.motor_temp, 
        sample.vibration, sample.motor_current, sample.obstacle_distance
    ]]
    
    prediction = ml_model.predict(features)[0]
    
    try:
        probabilities = ml_model.predict_proba(features)[0]
        confidence = float(max(probabilities))
    except:
        confidence = 1.0 
        
    inference_ms = (time.time() - start_time) * 1000
    
    return {
        "robot_id": sample.robot_id,
        "predicted_status": prediction,
        "confidence": round(confidence, 4),
        "model_version": "v1",
        "inference_ms": round(inference_ms, 2)
    }

@app.get("/metrics")
def metrics() -> dict:
    return {
        "requests_received": REQUEST_COUNT,
        "robots_in_memory": len(LATEST),
        "persistence": "POSTGRESQL",
        "prediction_service": "ACTIVE" if ml_model else "ERROR",
    }

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    path = Path(__file__).resolve().parents[1] / "dashboard" / "index.html"
    return path.read_text(encoding="utf-8")