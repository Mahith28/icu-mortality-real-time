from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from backend.app.db.database import get_db
from backend.app.models.patient import Patient
from backend.app.models.patient_stay import PatientStay
from backend.app.models.assignment import PatientAssignment
from backend.app.models.risk_event import RiskEvent
from backend.app.models.user import User
from backend.app.schemas.risk import RiskEventResponse
from backend.app.core.dependencies import get_current_user

router = APIRouter(
    prefix="/patients",
    tags=["Risk"]
)


def get_patient_for_user(
    patient_id: int,
    current_user: User,
    db: Session,
) -> Patient:
    patient = db.get(Patient, patient_id)

    if patient is None:
        raise HTTPException(
            status_code=404,
            detail="Patient not found",
        )

    if current_user.role == "NURSE":
        return patient

    if current_user.role == "DOCTOR":
        assignment = db.execute(
            select(PatientAssignment).where(
                PatientAssignment.patient_id == patient_id,
                PatientAssignment.doctor_id == current_user.id,
                PatientAssignment.unassigned_at.is_(None),
            )
        ).scalar_one_or_none()

        if assignment is None:
            raise HTTPException(
                status_code=403,
                detail="Patient is not assigned to this doctor",
            )

        return patient

    raise HTTPException(
        status_code=403,
        detail="Access denied",
    )


@router.get(
    "/{patient_id}/risk/latest",
    response_model=RiskEventResponse,
)
def get_latest_risk(
    patient_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    get_patient_for_user(patient_id, current_user, db)

    event = db.execute(
        select(RiskEvent)
        .where(RiskEvent.patient_id == patient_id)
        .order_by(
            desc(RiskEvent.event_time),
            desc(RiskEvent.id),
        )
        .limit(1)
    ).scalar_one_or_none()

    if event is None:
        raise HTTPException(
            status_code=404,
            detail="No risk prediction available for this patient",
        )

    return event


@router.get(
    "/{patient_id}/risk/history",
    response_model=list[RiskEventResponse],
)
def get_risk_history(
    patient_id: int,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    get_patient_for_user(patient_id, current_user, db)

    limit = min(max(limit, 1), 500)

    events = db.execute(
        select(RiskEvent)
        .where(RiskEvent.patient_id == patient_id)
        .order_by(
            desc(RiskEvent.event_time),
            desc(RiskEvent.id),
        )
        .limit(limit)
    ).scalars().all()

    return events


@router.get(
    "/{patient_id}/risk/explanation",
    response_model=RiskEventResponse,
)
def get_latest_risk_explanation(
    patient_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    get_patient_for_user(patient_id, current_user, db)

    event = db.execute(
        select(RiskEvent)
        .where(
            RiskEvent.patient_id == patient_id,
            RiskEvent.shap_status == "ok",
        )
        .order_by(
            desc(RiskEvent.event_time),
            desc(RiskEvent.id),
        )
        .limit(1)
    ).scalar_one_or_none()

    if event is None:
        raise HTTPException(
            status_code=404,
            detail="No SHAP explanation available for this patient",
        )

    return event
