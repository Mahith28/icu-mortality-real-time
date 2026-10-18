from backend.app.db.database import Base, engine
from backend.app.models.branch import Branch
from backend.app.models.staff_branch import StaffBranch

Base.metadata.create_all(
    bind=engine,
    tables=[
        Branch.__table__,
        StaffBranch.__table__,
    ],
)

print("Branch tables are ready.")
