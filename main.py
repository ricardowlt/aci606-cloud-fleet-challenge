from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
# --- NUEVOS IMPORTS PARA LA BASE DE DATOS ---
from sqlalchemy import create_engine, Column, String, Float
from sqlalchemy.orm import declarative_base, sessionmaker
# --------------------------------------------

app = FastAPI(
    title="ACI606 Cloud Fleet Challenge",
    version="0.1.0-starter",
    description="Starter API for the ACI606 Cloud Fleet Challenge.",
)

Status = Literal["NORMAL", "WARNING", "CRITICAL"]
# --- CONFIGURACIÓN DE LA BASE DE DATOS (NEON) ---
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./local.db")
# Si no encuentra la variable de entorno (como en Render), usará SQLite local por si acaso.

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Definimos la tabla donde guardaremos la telemetría
class TelemetryDB(Base):
    __tablename__ = "telemetry"
    id = Column(String, primary_key=True, index=True) # Usaremos robot_id + timestamp
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

# Creamos las tablas en Neon automáticamente
Base.metadata.create_all(bind=engine)
# ------------------------------------------------


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


# Starter storage: intentionally volatile.
# Students must replace this with persistent storage for the final challenge.
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
        "utc": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/telemetry")
def ingest_telemetry(sample: Telemetry) -> dict:
    global REQUEST_COUNT
    REQUEST_COUNT += 1
    
    # Abrimos la conexión a la base de datos
    db = SessionLocal()
    try:
        # Creamos un ID único combinando el robot y el tiempo
        unique_id = f"{sample.robot_id}_{sample.timestamp}"
        
        # Creamos el registro
        db_telemetry = TelemetryDB(
            id=unique_id,
            robot_id=sample.robot_id,
            timestamp=sample.timestamp,
            x=sample.x, y=sample.y, speed=sample.speed,
            battery=sample.battery, motor_temp=sample.motor_temp,
            vibration=sample.vibration, motor_current=sample.motor_current,
            obstacle_distance=sample.obstacle_distance
        )
        # Lo guardamos en Postgres (Neon)
        db.add(db_telemetry)
        db.commit()
    finally:
        db.close()

    # Seguimos guardando en memoria solo para que el dashboard no se rompa por ahora
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
def predict(_: PredictionRequest) -> dict:
    # TODO E4: load the trained model and return a real prediction.
    # Required contract is documented in README.md and the student guide.
    raise HTTPException(
        status_code=501,
        detail="/predict is part of the challenge. Train and integrate your model.",
    )


@app.get("/metrics")
def metrics() -> dict:
    # TODO E4: extend with latency, errors, throughput and model metrics.
    return {
        "requests_received": REQUEST_COUNT,
        "robots_in_memory": len(LATEST),
        "persistence": "RAM_ONLY",
        "prediction_service": "NOT_IMPLEMENTED",
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    path = Path(__file__).resolve().parents[1] / "dashboard" / "index.html"
    return path.read_text(encoding="utf-8")
