from contextlib import asynccontextmanager
from threading import Event, Thread

from fastapi import Depends, FastAPI
from backend.app.services.kafka_consumer import run_consumer, run_explanation_consumer
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.admin import router as admin_router
from backend.app.api.auth import router as auth_router
from backend.app.api.doctor_availability import router as doctor_availability_router
from backend.app.api.handover import router as handover_router
from backend.app.api.patients import router as patients_router
from backend.app.api.risk import router as risk_router
from backend.app.api.stays import router as stays_router
from backend.app.core.dependencies import get_current_user
from backend.app.models.user import User


@asynccontextmanager
async def lifespan(app: FastAPI):
    stop_event = Event()

    risk_consumer_thread = Thread(
        target=run_consumer,
        args=(stop_event,),
        name="vitalis-risk-consumer",
        daemon=True,
    )
    explanation_consumer_thread = Thread(
        target=run_explanation_consumer,
        args=(stop_event,),
        name="vitalis-explanation-consumer",
        daemon=True,
    )

    risk_consumer_thread.start()
    explanation_consumer_thread.start()

    try:
        yield
    finally:
        stop_event.set()
        risk_consumer_thread.join(timeout=5)
        explanation_consumer_thread.join(timeout=5)


app = FastAPI(
    title="VITALIS",
    version="1.0.0",
    description="VITALIS ICU Mortality Risk Prediction Platform",
    lifespan=lifespan,
)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://localhost:5176",
        "http://localhost:5177",
        "http://localhost:5178",
        "http://localhost:5179",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
        "http://127.0.0.1:5176",
        "http://127.0.0.1:5177",
        "http://127.0.0.1:5178",
        "http://127.0.0.1:5179",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# API Routers
# ---------------------------------------------------------

app.include_router(admin_router)
app.include_router(auth_router)
app.include_router(doctor_availability_router)
app.include_router(handover_router)
app.include_router(patients_router)
app.include_router(risk_router)
app.include_router(stays_router)


# ---------------------------------------------------------
# Root
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "name": "VITALIS",
        "version": "1.0.0",
        "status": "running",
    }


# ---------------------------------------------------------
# Health
# ---------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "VITALIS backend",
    }


# ---------------------------------------------------------
# Current User
# ---------------------------------------------------------

@app.get("/auth/me")
def get_me(
    current_user: User = Depends(get_current_user),
):
    return {
        "id": current_user.id,
        "name": current_user.name,
        "email": current_user.email,
        "role": current_user.role.value,
        "is_active": current_user.is_active,
    }