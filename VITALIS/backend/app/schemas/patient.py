from datetime import datetime

from pydantic import BaseModel, Field


class PatientCreate(BaseModel):
    patient_identifier: str = Field(
        min_length=2,
        max_length=50,
    )

    name: str = Field(
        min_length=2,
        max_length=150,
    )

    age: int = Field(
        ge=0,
        le=120,
    )

    gender: str = Field(
        min_length=1,
        max_length=30,
    )

    race: str | None = Field(
        default=None,
        max_length=100,
    )

    insurance: str | None = Field(
        default=None,
        max_length=100,
    )

    admission_type: str = Field(
        min_length=1,
        max_length=50,
    )

    admission_location: str | None = Field(
        default=None,
        max_length=200,
    )

    icu_unit: str = Field(
        min_length=1,
        max_length=100,
    )

    bed_number: str | None = Field(
        default=None,
        max_length=30,
    )

    admission_diagnosis: str | None = None

    height_cm: float | None = Field(
        default=None,
        gt=0,
        le=300,
    )

    weight_kg: float | None = Field(
        default=None,
        gt=0,
        le=500,
    )


class PatientResponse(BaseModel):
    id: int
    patient_identifier: str
    name: str
    age: int
    gender: str
    race: str | None
    insurance: str | None
    admission_type: str
    admission_location: str | None
    icu_unit: str
    bed_number: str | None
    admission_diagnosis: str | None
    height_cm: float | None
    weight_kg: float | None
    status: str
    created_by: int
    created_at: datetime
    updated_at: datetime

    model_config = {
        "from_attributes": True,
    }


class AssignmentCreate(BaseModel):
    doctor_id: int
    notes: str | None = None


class AssignmentResponse(BaseModel):
    id: int
    patient_id: int
    doctor_id: int
    assigned_by: int
    notes: str | None
    assigned_at: datetime
    unassigned_at: datetime | None

    model_config = {
        "from_attributes": True,
    }


class DoctorOption(BaseModel):
    id: int
    name: str
    email: str

    model_config = {
        "from_attributes": True,
    }
