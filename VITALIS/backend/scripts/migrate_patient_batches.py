from sqlalchemy import inspect, text

from backend.app.db.database import engine


with engine.begin() as conn:
    inspector = inspect(conn)
    columns = {
        column["name"]
        for column in inspector.get_columns("patients")
    }

    if "batch_id" not in columns:
        conn.execute(text("""
            ALTER TABLE patients
            ADD COLUMN batch_id INTEGER
            REFERENCES nurse_batches(id) ON DELETE RESTRICT
        """))
        conn.execute(text("""
            CREATE INDEX ix_patients_batch_id
            ON patients(batch_id)
        """))

print("Patient batch relationship migration complete.")
