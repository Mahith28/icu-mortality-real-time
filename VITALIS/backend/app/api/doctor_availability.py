from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from backend.app.core.dependencies import require_doctor
from backend.app.db.database import get_db
from backend.app.models.assignment import PatientAssignment
from backend.app.models.branch import Branch
from backend.app.models.doctor_availability import DoctorAvailability
from backend.app.models.handover_alert import HandoverAlert
from backend.app.models.patient import Patient, PatientStatus
from backend.app.models.staff_branch import StaffBranch
from backend.app.models.user import User


router = APIRouter(
    prefix="/doctors/me",
    tags=["Doctor Availability"],
)


def get_or_create_availability(
    doctor: User,
    db: Session,
) -> DoctorAvailability:
    availability = db.scalar(
        select(DoctorAvailability).where(
            DoctorAvailability.doctor_id == doctor.id
        )
    )

    if availability is None:
        availability = DoctorAvailability(
            doctor_id=doctor.id,
            is_online=True,
        )
        db.add(availability)
        db.flush()

    return availability


@router.get("/availability")
def get_availability(
    current_user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
):
    availability = get_or_create_availability(current_user, db)
    db.commit()

    return {
        "doctor_id": current_user.id,
        "is_online": availability.is_online,
        "updated_at": availability.updated_at,
    }


@router.post("/availability")
def set_availability(
    is_online: bool,
    current_user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
):
    availability = get_or_create_availability(current_user, db)

    if availability.is_online == is_online:
        db.commit()
        return {
            "doctor_id": current_user.id,
            "is_online": availability.is_online,
            "updated_at": availability.updated_at,
            "new_alerts": 0,
        }

    now = datetime.utcnow()

    if not is_online:
        assignments = db.scalars(
            select(PatientAssignment)
            .join(Patient, Patient.id == PatientAssignment.patient_id)
            .where(
                and_(
                    PatientAssignment.doctor_id == current_user.id,
                    PatientAssignment.unassigned_at.is_(None),
                    Patient.status == PatientStatus.ACTIVE,
                )
            )
            .with_for_update()
        ).all()

        for assignment in assignments:
            patient = db.get(Patient, assignment.patient_id)

            if patient is None or patient.branch_id is None:
                continue

            branch = db.get(Branch, patient.branch_id)
            membership = db.scalar(
                select(StaffBranch).where(
                    StaffBranch.staff_id == current_user.id,
                    StaffBranch.branch_id == patient.branch_id,
                )
            )

            if branch is None or not branch.is_active or membership is None:
                continue

            existing_alert = db.scalar(
                select(HandoverAlert).where(
                    HandoverAlert.assignment_id == assignment.id,
                    HandoverAlert.alert_type == "DOCTOR_OFFLINE",
                    HandoverAlert.status == "OPEN",
                )
            )

            if existing_alert is not None:
                continue

            db.add(
                HandoverAlert(
                    assignment_id=assignment.id,
                    patient_id=patient.id,
                    branch_id=patient.branch_id,
                    doctor_id=current_user.id,
                    alert_type="DOCTOR_OFFLINE",
                    status="OPEN",
                    message=(
                        f"Assigned doctor {current_user.name} is offline. "
                        f"Patient {patient.patient_identifier} needs handover."
                    ),
                )
            )

    availability.is_online = is_online
    availability.updated_at = now

    db.commit()

    return {
        "doctor_id": current_user.id,
        "is_online": availability.is_online,
        "updated_at": availability.updated_at,
        "message": (
            "Doctor is online."
            if is_online
            else "Doctor is offline. Review handover alerts for assigned patients."
        ),
    }
