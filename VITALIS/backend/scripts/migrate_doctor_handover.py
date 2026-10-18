from backend.app.db.database import Base, engine

# Import all referenced models so SQLAlchemy registers their foreign keys.
from backend.app.models.doctor_shift import DoctorShift
from backend.app.models.doctor_departure import DoctorDeparture
from backend.app.models.handover_alert import HandoverAlert

Base.metadata.create_all(
    bind=engine,
    tables=[
        DoctorShift.__table__,
        DoctorDeparture.__table__,
        HandoverAlert.__table__,
    ],
)

print("Doctor shift, departure, and handover tables are ready.")
