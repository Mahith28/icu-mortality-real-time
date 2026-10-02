from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session
from backend.app.core.dependencies import get_current_user, require_nurse
from backend.app.db.database import get_db
from backend.app.models.patient import Patient
from backend.app.models.patient_stay import PatientStay
from backend.app.models.user import User
from backend.app.schemas.stay import (
    PatientStayCreate,
    PatientStayResponse,
)
router = APIRouter(
    prefix="/patients",
    tags=["Patient Stays"],
)
@router.post(
    "/{patient_id}/stays",
    response_model=PatientStayResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_stay(
    patient_id: int,
    data: PatientStayCreate,
    current_user: User = Depends(require_nurse),
    db: Session = Depends(get_db),
):
    patient = db.get(Patient, patient_id)
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found.",
        )
    existing = db.scalar(
        select(PatientStay).where(
            PatientStay.stay_id == data.stay_id
        )
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This ICU stay is already mapped.",
        )
    stay = PatientStay(
        patient_id=patient_id,
        stay_id=data.stay_id,
        intime=data.intime,
        outtime=data.outtime,
    )
    db.add(stay)
    db.commit()
    db.refresh(stay)
    return stay
@router.get(
    "/{patient_id}/stays",
    response_model=list[PatientStayResponse],
)
def list_stays(
    patient_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    patient = db.get(Patient, patient_id)
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found.",
        )
    stays = db.scalars(
        select(PatientStay)
        .where(PatientStay.patient_id == patient_id)
        .order_by(PatientStay.created_at.desc())
    ).all()
    return list(stays)
