from datetime import datetime
from pydantic import BaseModel
class PatientStayCreate(BaseModel):
    stay_id: int
    intime: datetime | None = None
    outtime: datetime | None = None
class PatientStayResponse(BaseModel):
    id: int
    patient_id: int
    stay_id: int
    intime: datetime | None
    outtime: datetime | None
    created_at: datetime
    model_config = {"from_attributes": True}
