from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.dependencies import require_role
from backend.app.core.security import hash_password
from backend.app.db.database import get_db
from backend.app.models.user import User, UserRole
from backend.app.models.branch import Branch
from backend.app.models.staff_branch import StaffBranch
from backend.app.models.assignment import PatientAssignment
from backend.app.models.patient import Patient, PatientStatus

router = APIRouter(
    prefix="/admin",
    tags=["Administration"],
    dependencies=[Depends(require_role(UserRole.ADMIN))],
)


class BranchCreate(BaseModel):
    code: str = Field(min_length=2, max_length=30)
    name: str = Field(min_length=2, max_length=120)
    location: str | None = Field(default=None, max_length=255)


class BranchUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=2, max_length=120)
    location: str | None = Field(default=None, max_length=255)


class StaffCreate(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    branch_ids: list[int] = Field(default_factory=list)


class StaffBranchAssignment(BaseModel):
    branch_ids: list[int] = Field(default_factory=list)


class PasswordReset(BaseModel):
    password: str = Field(min_length=8, max_length=128)


def role_from_email(email: str) -> UserRole:
    domain = email.rsplit("@", 1)[-1].lower()

    if domain == settings.nurse_email_domain.lower():
        return UserRole.NURSE

    if domain == settings.doctor_email_domain.lower():
        return UserRole.DOCTOR

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Staff email must use the nurse.vitalis.com or doctor.vitalis.com domain.",
    )


def staff_response(user: User) -> dict:
    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "role": user.role.value,
        "is_active": user.is_active,
        "created_at": user.created_at,
    }


@router.get("/staff")
def list_staff(db: Session = Depends(get_db)):
    staff = db.scalars(
        select(User)
        .where(User.role.in_([UserRole.NURSE, UserRole.DOCTOR]))
        .order_by(User.name, User.id)
    ).all()

    return [staff_response(user) for user in staff]


@router.post("/staff", status_code=status.HTTP_201_CREATED)
def create_staff(data: StaffCreate, db: Session = Depends(get_db)):
    email = str(data.email).strip().lower()
    role = role_from_email(email)

    existing = db.scalar(select(User).where(User.email == email))
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    branch_ids = list(dict.fromkeys(data.branch_ids))
    if not branch_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Assign at least one active branch when creating staff.",
        )

    active_branches = db.scalars(
        select(Branch).where(
            Branch.id.in_(branch_ids),
            Branch.is_active.is_(True),
        )
    ).all()
    if len(active_branches) != len(branch_ids):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Every assigned branch must exist and be active.",
        )

    user = User(
        name=data.name.strip(),
        email=email,
        password_hash=hash_password(data.password),
        role=role,
        is_active=True,
    )

    try:
        db.add(user)
        db.flush()
        for branch_id in branch_ids:
            db.add(StaffBranch(staff_id=user.id, branch_id=branch_id))
        db.commit()
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Could not create the account. Check whether the email is already registered.",
        )

    db.refresh(user)
    return staff_response(user)


@router.put("/staff/{staff_id}/password")
def reset_staff_password(
    staff_id: int,
    data: PasswordReset,
    db: Session = Depends(get_db),
):
    user = db.get(User, staff_id)

    if user is None or user.role not in [UserRole.NURSE, UserRole.DOCTOR]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Staff account not found.",
        )

    user.password_hash = hash_password(data.password)
    db.commit()

    return {"message": "Staff password updated successfully."}


@router.patch("/staff/{staff_id}/deactivate")
def deactivate_staff(
    staff_id: int,
    db: Session = Depends(get_db),
):
    user = db.get(User, staff_id)

    if user is None or user.role not in [UserRole.NURSE, UserRole.DOCTOR]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Staff account not found.",
        )

    if not user.is_active:
        return {"message": "Staff account is already inactive."}

    if user.role == UserRole.DOCTOR:
        active_assignment_id = db.scalar(
            select(PatientAssignment.id).where(
                PatientAssignment.doctor_id == user.id,
                PatientAssignment.unassigned_at.is_(None),
            ).limit(1)
        )
        if active_assignment_id is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "This doctor still has active patient assignments. "
                    "Reassign those patients before deactivating the account."
                ),
            )

    user.is_active = False
    db.commit()

    return {
        "message": "Staff account deactivated.",
        "note": "Patient reassignment checks will be added before this operation is enabled in the interface.",
    }



@router.get("/branches")
def list_branches(
    include_inactive: bool = False,
    db: Session = Depends(get_db),
):
    query = select(Branch).order_by(Branch.name, Branch.id)
    if not include_inactive:
        query = query.where(Branch.is_active.is_(True))

    branches = db.scalars(query).all()
    return [
        {
            "id": branch.id,
            "code": branch.code,
            "name": branch.name,
            "location": branch.location,
            "is_active": branch.is_active,
            "created_at": branch.created_at,
        }
        for branch in branches
    ]


@router.post("/branches", status_code=status.HTTP_201_CREATED)
def create_branch(
    data: BranchCreate,
    db: Session = Depends(get_db),
):
    code = data.code.strip().upper()
    existing = db.scalar(select(Branch).where(Branch.code == code))

    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A branch with this code already exists.",
        )

    branch = Branch(
        code=code,
        name=data.name.strip(),
        location=data.location.strip() if data.location else None,
    )
    db.add(branch)

    try:
        db.commit()
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Could not create branch. The branch code may already exist.",
        )

    db.refresh(branch)
    return {
        "id": branch.id,
        "code": branch.code,
        "name": branch.name,
        "location": branch.location,
        "is_active": branch.is_active,
        "created_at": branch.created_at,
    }


@router.patch("/branches/{branch_id}")
def update_branch(
    branch_id: int,
    data: BranchUpdate,
    db: Session = Depends(get_db),
):
    branch = db.get(Branch, branch_id)
    if branch is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Branch not found.",
        )

    updates = data.model_dump(exclude_unset=True)
    for field, value in updates.items():
        if isinstance(value, str):
            value = value.strip()
        setattr(branch, field, value)

    db.commit()
    db.refresh(branch)

    return {
        "id": branch.id,
        "code": branch.code,
        "name": branch.name,
        "location": branch.location,
        "is_active": branch.is_active,
        "created_at": branch.created_at,
    }


@router.patch("/branches/{branch_id}/deactivate")
def deactivate_branch(
    branch_id: int,
    db: Session = Depends(get_db),
):
    branch = db.get(Branch, branch_id)
    if branch is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Branch not found.",
        )

    if not branch.is_active:
        return {"message": "Branch is already inactive.", "branch_id": branch.id}

    active_patient_id = db.scalar(
        select(Patient.id)
        .where(
            Patient.branch_id == branch_id,
            Patient.status == PatientStatus.ACTIVE,
        )
        .limit(1)
    )
    if active_patient_id is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "This branch still has active patients. Transfer or resolve "
                "their care through an explicit workflow before deactivating it."
            ),
        )

    branch.is_active = False
    db.commit()

    return {"message": "Branch deactivated.", "branch_id": branch.id}


@router.patch("/branches/{branch_id}/reactivate")
def reactivate_branch(
    branch_id: int,
    db: Session = Depends(get_db),
):
    branch = db.get(Branch, branch_id)
    if branch is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Branch not found.",
        )

    if branch.is_active:
        return {"message": "Branch is already active.", "branch_id": branch.id}

    branch.is_active = True
    db.commit()

    return {"message": "Branch reactivated.", "branch_id": branch.id}



@router.get("/staff/{staff_id}/branches")
def get_staff_branches(
    staff_id: int,
    db: Session = Depends(get_db),
):
    staff = db.get(User, staff_id)
    if staff is None or staff.role not in [UserRole.NURSE, UserRole.DOCTOR]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Staff account not found.",
        )

    memberships = db.scalars(
        select(StaffBranch)
        .where(StaffBranch.staff_id == staff_id)
        .order_by(StaffBranch.branch_id)
    ).all()

    return [
        {
            "branch_id": membership.branch.id,
            "code": membership.branch.code,
            "name": membership.branch.name,
            "is_active": membership.branch.is_active,
            "assigned_at": membership.assigned_at,
        }
        for membership in memberships
    ]


@router.put("/staff/{staff_id}/branches")
def assign_staff_branches(
    staff_id: int,
    data: StaffBranchAssignment,
    db: Session = Depends(get_db),
):
    staff = db.get(User, staff_id)
    if staff is None or staff.role not in [UserRole.NURSE, UserRole.DOCTOR]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Staff account not found.",
        )

    branch_ids = list(dict.fromkeys(data.branch_ids))
    if branch_ids:
        branches = db.scalars(
            select(Branch).where(
                Branch.id.in_(branch_ids),
                Branch.is_active.is_(True),
            )
        ).all()

        if len(branches) != len(branch_ids):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Every branch must exist and be active.",
            )

    existing = db.scalars(
        select(StaffBranch).where(StaffBranch.staff_id == staff_id)
    ).all()
    existing_by_branch = {
        membership.branch_id: membership for membership in existing
    }
    requested_ids = set(branch_ids)

    for branch_id, membership in existing_by_branch.items():
        if branch_id not in requested_ids:
            db.delete(membership)

    for branch_id in requested_ids - set(existing_by_branch):
        db.add(StaffBranch(staff_id=staff_id, branch_id=branch_id))

    db.commit()

    return {
        "staff_id": staff_id,
        "branch_ids": sorted(requested_ids),
        "message": "Branch assignments saved.",
    }
