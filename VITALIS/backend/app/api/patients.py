from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from backend.app.core.dependencies import (
    get_current_user,
    require_nurse,
)
from backend.app.db.database import get_db
from backend.app.models.assignment import PatientAssignment
from backend.app.models.handover_alert import HandoverAlert
from backend.app.models.staff_branch import StaffBranch
from backend.app.models.branch import Branch
from backend.app.models.patient import Patient, PatientStatus
from backend.app.models.patient_stay import PatientStay
from backend.app.models.user import User, UserRole
from backend.app.schemas.patient import (
    AssignmentCreate,
    AssignmentResponse,
    DoctorOption,
    PatientCreate,
    PatientResponse,
)
from backend.app.services.realtime import realtime_kafka
from backend.app.services.stay_id import generate_stay_id

router = APIRouter(
    prefix="/patients",
    tags=["Patients"],
)


def get_patient_or_404(
    patient_id: int,
    db: Session,
) -> Patient:
    patient = db.get(Patient, patient_id)

    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found.",
        )

    return patient


def verify_patient_access(
    patient: Patient,
    current_user: User,
    db: Session,
) -> None:
    if current_user.role == UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Administrators cannot access patient records.")

    if current_user.role == UserRole.NURSE:
        if patient.branch_id is None:
            raise HTTPException(status_code=403, detail="The patient's branch cannot be verified.")
        branch = db.get(Branch, patient.branch_id)
        membership = db.scalar(select(StaffBranch).where(
            StaffBranch.staff_id == current_user.id,
            StaffBranch.branch_id == patient.branch_id,
        ))
        if branch is None or not branch.is_active or membership is None:
            raise HTTPException(status_code=403, detail="This patient is outside your authorized branch.")
        return

    if current_user.role == UserRole.DOCTOR:
        if patient.status != PatientStatus.ACTIVE:
            raise HTTPException(status_code=403, detail="Doctors can only access assigned active patients.")

        assignment = db.scalar(select(PatientAssignment).where(and_(
            PatientAssignment.patient_id == patient.id,
            PatientAssignment.doctor_id == current_user.id,
            PatientAssignment.unassigned_at.is_(None),
        )))
        if assignment is None:
            raise HTTPException(status_code=403, detail="You are not assigned to this active patient.")

        branch = db.get(Branch, patient.branch_id)
        membership = db.scalar(select(StaffBranch).where(
            StaffBranch.staff_id == current_user.id,
            StaffBranch.branch_id == patient.branch_id,
        ))
        if branch is None or not branch.is_active or membership is None:
            raise HTTPException(status_code=403, detail="This patient is outside your authorized branch.")
        return

    raise HTTPException(status_code=403, detail="Access denied.")

@router.post(
    "",
    response_model=PatientResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_patient(
    data: PatientCreate,
    current_user: User = Depends(require_nurse),
    db: Session = Depends(get_db),
):
    branch = db.get(Branch, data.branch_id)
    if branch is None or not branch.is_active:
        raise HTTPException(status_code=400, detail="Select an existing active branch.")

    branch_membership = db.scalar(select(StaffBranch).where(
        StaffBranch.staff_id == current_user.id,
        StaffBranch.branch_id == data.branch_id,
    ))
    if branch_membership is None:
        raise HTTPException(status_code=403, detail="You are not assigned to this branch.")

    existing = db.scalar(select(Patient).where(
        Patient.patient_identifier == data.patient_identifier.strip()
    ))
    if existing:
        raise HTTPException(status_code=409, detail="A patient with this identifier already exists.")

    entered_intime = data.admission_datetime
    if entered_intime.tzinfo is None:
        entered_intime = entered_intime.replace(
            tzinfo=ZoneInfo("Asia/Kolkata")
        )
    intime = entered_intime.astimezone(timezone.utc).replace(
        tzinfo=None
    )
    stay_id = generate_stay_id()
    while db.scalar(select(PatientStay).where(PatientStay.stay_id == stay_id)):
        stay_id = generate_stay_id()

    patient = Patient(
        patient_identifier=data.patient_identifier.strip(),
        name=data.name.strip(),
        age=data.age,
        gender=data.gender.strip(),
        race=data.race.strip() if data.race else None,
        insurance=data.insurance.strip() if data.insurance else None,
        admission_type=data.admission_type.strip(),
        admission_location=data.admission_location.strip() if data.admission_location else None,
        icu_unit=data.icu_unit.strip(),
        bed_number=data.bed_number.strip() if data.bed_number else None,
        admission_diagnosis=data.admission_diagnosis.strip() if data.admission_diagnosis else None,
        height_cm=data.height_cm,
        weight_kg=data.weight_kg,
        status=PatientStatus.ACTIVE,
        created_by=current_user.id,
        branch_id=data.branch_id,
    )
    db.add(patient)
    db.flush()

    stay = PatientStay(
        patient_id=patient.id,
        stay_id=stay_id,
        intime=intime,
        outtime=None,
    )
    db.add(stay)
    db.commit()
    db.refresh(patient)

    try:
        realtime_kafka.publish_static_patient(
            stay_id=stay_id,
            anchor_age=data.age,
            gender=data.gender.strip(),
            race=data.race,
            insurance=data.insurance,
            admission_type=data.admission_type.strip(),
            admission_location=data.admission_location,
            first_careunit=data.icu_unit.strip(),
            intime=intime,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Patient was created, but the realtime ICU static record could not be published: {exc}",
        )

    return patient

@router.get(
    "",
    response_model=list[PatientResponse],
)
def list_patients(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role == UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Administrators cannot access patient records.")

    if current_user.role == UserRole.NURSE:
        branch_ids = select(StaffBranch.branch_id).where(
            StaffBranch.staff_id == current_user.id
        )
        patients = db.scalars(
            select(Patient)
            .join(Branch, Branch.id == Patient.branch_id)
            .where(
                Patient.branch_id.in_(branch_ids),
                Branch.is_active.is_(True),
            )
            .order_by(Patient.created_at.desc())
        ).unique().all()
        return list(patients)

    if current_user.role == UserRole.DOCTOR:
        patients = db.scalars(
            select(Patient)
            .join(PatientAssignment, PatientAssignment.patient_id == Patient.id)
            .join(Branch, Branch.id == Patient.branch_id)
            .join(StaffBranch, and_(
                StaffBranch.branch_id == Patient.branch_id,
                StaffBranch.staff_id == current_user.id,
            ))
            .where(
                PatientAssignment.doctor_id == current_user.id,
                PatientAssignment.unassigned_at.is_(None),
                Patient.status == PatientStatus.ACTIVE,
                Branch.is_active.is_(True),
            )
            .order_by(Patient.created_at.desc())
        ).unique().all()
        return list(patients)

    raise HTTPException(status_code=403, detail="Access denied.")

@router.get("/branches")
def list_nurse_branches(
    current_user: User = Depends(require_nurse),
    db: Session = Depends(get_db),
):
    branches = db.scalars(
        select(Branch)
        .join(StaffBranch, StaffBranch.branch_id == Branch.id)
        .where(
            StaffBranch.staff_id == current_user.id,
            Branch.is_active.is_(True),
        )
        .order_by(Branch.name)
    ).unique().all()

    return [
        {
            "id": branch.id,
            "code": branch.code,
            "name": branch.name,
            "location": branch.location,
        }
        for branch in branches
    ]


@router.get(
    "/doctors",
    response_model=list[DoctorOption],
)
def list_doctors(
    current_user: User = Depends(require_nurse),
    db: Session = Depends(get_db),
):
    from sqlalchemy import func

    nurse_branch_ids = select(StaffBranch.branch_id).where(
        StaffBranch.staff_id == current_user.id
    )

    active_patient_counts = (
        select(
            PatientAssignment.doctor_id.label("doctor_id"),
            func.count(
                func.distinct(PatientAssignment.patient_id)
            ).label("active_patient_count"),
        )
        .join(Patient, Patient.id == PatientAssignment.patient_id)
        .where(
            Patient.status == PatientStatus.ACTIVE,
            PatientAssignment.unassigned_at.is_(None),
        )
        .group_by(PatientAssignment.doctor_id)
        .subquery()
    )

    rows = db.execute(
        select(
            User,
            func.coalesce(
                active_patient_counts.c.active_patient_count, 0
            ).label("active_patient_count"),
        )
        .join(StaffBranch, StaffBranch.staff_id == User.id)
        .join(Branch, Branch.id == StaffBranch.branch_id)
        .outerjoin(
            active_patient_counts,
            active_patient_counts.c.doctor_id == User.id,
        )
        .where(
            and_(
                User.role == UserRole.DOCTOR,
                User.is_active.is_(True),
                Branch.is_active.is_(True),
                StaffBranch.branch_id.in_(nurse_branch_ids),
            )
        )
        .distinct()
        .order_by(User.name.asc())
    ).all()

    return [
        DoctorOption(
            id=doctor.id,
            name=doctor.name,
            email=doctor.email,
            active_patient_count=int(count or 0),
        )
        for doctor, count in rows
    ]


@router.get(
    "/{patient_id}",
    response_model=PatientResponse,
)
def get_patient(
    patient_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    patient = get_patient_or_404(patient_id, db)

    verify_patient_access(
        patient,
        current_user,
        db,
    )

    return patient


@router.post(
    "/{patient_id}/assign",
    response_model=AssignmentResponse,
)
def assign_doctor(
    patient_id: int,
    data: AssignmentCreate,
    current_user: User = Depends(require_nurse),
    db: Session = Depends(get_db),
):
    patient = get_patient_or_404(patient_id, db)
    verify_patient_access(patient, current_user, db)

    if patient.status != PatientStatus.ACTIVE:
        raise HTTPException(status_code=409, detail="Only active patients can be assigned to a doctor.")

    doctor = db.scalar(select(User).where(and_(
        User.id == data.doctor_id,
        User.role == UserRole.DOCTOR,
        User.is_active.is_(True),
    )))
    if doctor is None:
        raise HTTPException(status_code=404, detail="Active doctor not found.")

    branch = db.get(Branch, patient.branch_id)
    if branch is None or not branch.is_active:
        raise HTTPException(status_code=409, detail="The patient's branch is inactive or missing.")

    doctor_membership = db.scalar(select(StaffBranch).where(
        StaffBranch.staff_id == doctor.id,
        StaffBranch.branch_id == patient.branch_id,
    ))
    if doctor_membership is None:
        raise HTTPException(status_code=400, detail="The doctor must be assigned to the patient's branch first.")

    current_assignment = db.scalar(select(PatientAssignment).where(and_(
        PatientAssignment.patient_id == patient_id,
        PatientAssignment.unassigned_at.is_(None),
    )))
    if current_assignment:
        if current_assignment.doctor_id == doctor.id:
            raise HTTPException(status_code=409, detail="This doctor is already assigned to the patient.")

        # Resolve handover alerts for the previous assignment as part of
        # the same transaction as the successful reassignment.
        now = datetime.utcnow()
        current_assignment.unassigned_at = now
        db.query(HandoverAlert).filter(
            HandoverAlert.assignment_id == current_assignment.id,
            HandoverAlert.patient_id == patient.id,
            HandoverAlert.status == "OPEN",
        ).update(
            {
                HandoverAlert.status: "RESOLVED",
                HandoverAlert.resolved_at: now,
                HandoverAlert.resolved_by: current_user.id,
            },
            synchronize_session=False,
        )

    assignment = PatientAssignment(
        patient_id=patient_id,
        doctor_id=doctor.id,
        assigned_by=current_user.id,
        notes=data.notes,
    )
    db.add(assignment)
    db.commit()
    db.refresh(assignment)
    return assignment

@router.post(
    "/{patient_id}/discharge",
    response_model=PatientResponse,
)
def discharge_patient(
    patient_id: int,
    current_user: User = Depends(require_nurse),
    db: Session = Depends(get_db),
):
    patient = get_patient_or_404(patient_id, db)

    verify_patient_access(patient, current_user, db)

    if patient.status != PatientStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Patient is already inactive.",
        )

    now = datetime.utcnow()

    active_assignment = db.scalar(
        select(PatientAssignment).where(
            and_(
                PatientAssignment.patient_id == patient_id,
                PatientAssignment.unassigned_at.is_(None),
            )
        )
    )

    active_stay = db.scalar(
        select(PatientStay)
        .where(
            and_(
                PatientStay.patient_id == patient_id,
                PatientStay.outtime.is_(None),
            )
        )
        .order_by(PatientStay.intime.desc())
    )

    if not active_stay:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No active ICU stay exists for this patient.",
        )

    stay_id = active_stay.stay_id

    patient.status = PatientStatus.DISCHARGED
    patient.updated_at = now

    if active_assignment:
        active_assignment.unassigned_at = now

    active_stay.outtime = now

    db.commit()
    db.refresh(patient)

    try:
        realtime_kafka.publish_discharge(
            stay_id=stay_id,
            event_time=now,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Patient was discharged in the database, but the "
                f"realtime discharge event could not be published: {exc}"
            ),
        )

    return patient

