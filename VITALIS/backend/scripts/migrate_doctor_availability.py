from sqlalchemy import inspect

from backend.app.db.database import Base, engine
import backend.app.models  # Register all models


def main():
    inspector = inspect(engine)

    if "doctor_availability" in inspector.get_table_names():
        print("doctor_availability already exists; no migration needed.")
        return

    Base.metadata.create_all(
        bind=engine,
        tables=[Base.metadata.tables["doctor_availability"]],
    )

    print("Created doctor_availability.")


if __name__ == "__main__":
    main()
