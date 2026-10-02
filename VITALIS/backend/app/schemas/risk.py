from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class RiskEventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    patient_id: int
    stay_id: int
    window_id: int
    status: str
    event_time: datetime | None
    window_start: datetime | None
    window_end: datetime | None
    sequence_length: int | None
    prediction: int | None
    threshold: float | None
    xgb_raw_probability: float | None
    xgb_probability: float | None
    lstm_raw_probability: float | None
    lstm_probability: float | None
    ensemble_probability: float | None
    model_version: str | None
    latency_ms: float | None
    shap_status: str | None
    top_features: list[dict[str, Any]] | None
    created_at: datetime
