from backend.app.db.database import Base, engine
from backend.app.models.nurse_batch import NurseBatch
from backend.app.models.nurse_batch_membership import NurseBatchMembership

Base.metadata.create_all(
    bind=engine,
    tables=[
        NurseBatch.__table__,
        NurseBatchMembership.__table__,
    ],
)

print("Nurse batch tables are ready.")
