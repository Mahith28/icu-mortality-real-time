from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from backend.app.core.dependencies import (
    get_current_user,
    require_nurse,
)
from backend.app.db.database import get_db
from backend.app.models.assignment import PatientAssignment
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


def verify_doctor_access(
    patient_id: int,
    current_user: User,
    db: Session,
) -> None:
    if current_user.role != UserRole.DOCTOR:
        return

    assignment = db.scalar(
        select(PatientAssignment).where(
            and_(
                PatientAssignment.patient_id == patient_id,
                PatientAssignment.doctor_id == current_user.id,
                PatientAssignment.unassigned_at.is_(None),
            )
        )
    )

    if not assignment:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not assigned to this patient.",
        )


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
    existing = db.scalar(
        select(Patient).where(
            Patient.patient_identifier == data.patient_identifier.strip()
        )
    )

    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A patient with this identifier already exists.",
        )

    intime = datetime.now(timezone.utc).replace(tzinfo=None)

    stay_id = generate_stay_id()

    while db.scalar(
        select(PatientStay).where(PatientStay.stay_id == stay_id)
    ):
        stay_id = generate_stay_id()

    patient = Patient(
        patient_identifier=data.patient_identifier.strip(),
        name=data.name.strip(),
        age=data.age,
        gender=data.gender.strip(),
        race=data.race.strip() if data.race else None,
        insurance=data.insurance.strip() if data.insurance else None,
        admission_type=data.admission_type.strip(),
        admission_location=(
            data.admission_location.strip()
            if data.admission_location
            else None
        ),
        icu_unit=data.icu_unit.strip(),
        bed_number=(
            data.bed_number.strip()
            if data.bed_number
            else None
        ),
        admission_diagnosis=(
            data.admission_diagnosis.strip()
            if data.admission_diagnosis
            else None
        ),
        height_cm=data.height_cm,
        weight_kg=data.weight_kg,
        status=PatientStatus.ACTIVE,
        created_by=current_user.id,
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
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Patient was created, but the realtime ICU static "
                f"record could not be published: {exc}"
            ),
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
    if current_user.role == UserRole.NURSE:
        patients = db.scalars(
            select(Patient)
            .order_by(Patient.created_at.desc())
        ).all()

        return list(patients)

    patients = db.scalars(
        select(Patient)
        .join(
            PatientAssignment,
            PatientAssignment.patient_id == Patient.id,
        )
        .where(
            and_(
                PatientAssignment.doctor_id == current_user.id,
                PatientAssignment.unassigned_at.is_(None),
                Patient.status == PatientStatus.ACTIVE,
            )
        )
        .order_by(Patient.created_at.desc())
    ).all()

    return list(patients)


@router.get(
    "/doctors",
    response_model=list[DoctorOption],
)
def list_doctors(
    current_user: User = Depends(require_nurse),
    db: Session = Depends(get_db),
):
    doctors = db.scalars(
        select(User)
        .where(
            and_(
                User.role == UserRole.DOCTOR,
                User.is_active.is_(True),
            )
        )
        .order_by(User.name.asc())
    ).all()

    return list(doctors)


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

    verify_doctor_access(
        patient_id,
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

    if patient.status != PatientStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only active patients can be assigned to a doctor.",
        )

    doctor = db.scalar(
        select(User).where(
            and_(
                User.id == data.doctor_id,
                User.role == UserRole.DOCTOR,
                User.is_active.is_(True),
            )
        )
    )

    if not doctor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Active doctor not found.",
        )

    current_assignment = db.scalar(
        select(PatientAssignment).where(
            and_(
                PatientAssignment.patient_id == patient_id,
                PatientAssignment.unassigned_at.is_(None),
            )
        )
    )

    if current_assignment:
        if current_assignment.doctor_id == doctor.id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This doctor is already assigned to the patient.",
            )

        current_assignment.unassigned_at = datetime.utcnow()

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


@router.get(
    "/{patient_id}/assignments",
    response_model=list[AssignmentResponse],
)
def get_assignments(
    patient_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    get_patient_or_404(patient_id, db)

    verify_doctor_access(
        patient_id,
        current_user,
        db,
    )

    assignments = db.scalars(
        select(PatientAssignment)
        .where(
            PatientAssignment.patient_id == patient_id
        )
        .order_by(
            PatientAssignment.assigned_at.desc()
        )
    ).all()

    return list(assignments)


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

