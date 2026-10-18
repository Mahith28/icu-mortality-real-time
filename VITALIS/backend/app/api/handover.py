from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from backend.app.core.dependencies import require_nurse
from backend.app.db.database import get_db
from backend.app.models.assignment import PatientAssignment
from backend.app.models.branch import Branch
from backend.app.models.handover_alert import HandoverAlert
from backend.app.models.patient import Patient
from backend.app.models.staff_branch import StaffBranch
from backend.app.models.user import User


router = APIRouter(prefix="/handover", tags=["Handover"])


def authorized_branch_ids(nurse: User, db: Session) -> list[int]:
    return list(
        db.scalars(
            select(StaffBranch.branch_id)
            .join(Branch, Branch.id == StaffBranch.branch_id)
            .where(
                StaffBranch.staff_id == nurse.id,
                Branch.is_active.is_(True),
            )
        ).all()
    )


@router.get("/alerts")
def list_handover_alerts(
    current_user: User = Depends(require_nurse),
    db: Session = Depends(get_db),
):
    branch_ids = authorized_branch_ids(current_user, db)
    if not branch_ids:
        return {"alerts": [], "count": 0}

    alerts = db.scalars(
        select(HandoverAlert)
        .where(
            HandoverAlert.branch_id.in_(branch_ids),
            HandoverAlert.status == "OPEN",
        )
        .order_by(HandoverAlert.created_at.desc(), HandoverAlert.id.desc())
    ).all()

    results = []
    for alert in alerts:
        patient = db.get(Patient, alert.patient_id)
        assignment = db.get(PatientAssignment, alert.assignment_id)

        # Hide stale alerts if the patient was reassigned or is no longer active.
        if (
            assignment is None
            or assignment.unassigned_at is not None
            or patient is None
            or patient.status != "ACTIVE"
        ):
            continue

        results.append({
            "id": alert.id,
            "alert_type": alert.alert_type,
            "status": alert.status,
            "message": alert.message,
            "patient_id": patient.id,
            "patient_identifier": patient.patient_identifier,
            "branch_id": alert.branch_id,
            "doctor_id": alert.doctor_id,
            "assignment_id": alert.assignment_id,
            "created_at": alert.created_at,
        })

    return {"alerts": results, "count": len(results)}


@router.post("/alerts/{alert_id}/resolve")
def resolve_handover_alert(
    alert_id: int,
    current_user: User = Depends(require_nurse),
    db: Session = Depends(get_db),
):
    branch_ids = authorized_branch_ids(current_user, db)

    alert = db.scalar(
        select(HandoverAlert)
        .where(HandoverAlert.id == alert_id)
        .with_for_update()
    )

    if alert is None or alert.branch_id not in branch_ids:
        raise HTTPException(status_code=404, detail="Handover alert not found.")

    if alert.status != "OPEN":
        raise HTTPException(status_code=409, detail="Handover alert is already resolved.")

    assignment = db.scalar(
        select(PatientAssignment)
        .where(PatientAssignment.id == alert.assignment_id)
        .with_for_update()
    )
    patient = db.get(Patient, alert.patient_id)

    if (
        assignment is None
        or assignment.unassigned_at is not None
        or assignment.patient_id != alert.patient_id
        or patient is None
        or patient.status != "ACTIVE"
    ):
        raise HTTPException(
            status_code=409,
            detail="The patient assignment has changed or the patient is no longer active.",
        )

    # An alert is resolved only after the patient has a different active assignment.
    replacement = db.scalar(
        select(PatientAssignment).where(
            and_(
                PatientAssignment.patient_id == patient.id,
                PatientAssignment.unassigned_at.is_(None),
                PatientAssignment.id != alert.assignment_id,
            )
        )
    )
    if replacement is None:
        raise HTTPException(
            status_code=409,
            detail="Reassign the patient to another doctor before resolving this alert.",
        )

    alert.status = "RESOLVED"
    alert.resolved_at = datetime.utcnow()
    alert.resolved_by = current_user.id
    db.commit()

    return {
        "message": "Handover alert resolved.",
        "alert_id": alert.id,
        "status": alert.status,
        "resolved_by": current_user.id,
        "resolved_at": alert.resolved_at,
    }
