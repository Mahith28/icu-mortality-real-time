"""Remove the retired nurse-batch schema after strict safety checks."""

from sqlalchemy import inspect, text

from backend.app.db.database import engine


def remove_nurse_batches() -> None:
    with engine.begin() as conn:
        inspector = inspect(conn)
        tables = set(inspector.get_table_names())

        required = {"patients", "nurse_batches", "nurse_batch_memberships"}
        missing = required - tables
        if missing:
            raise RuntimeError(
                f"Migration aborted; required tables are missing: {sorted(missing)}"
            )

        patient_count = conn.execute(
            text("SELECT COUNT(*) FROM patients")
        ).scalar_one()
        assigned_count = conn.execute(
            text("SELECT COUNT(*) FROM patients WHERE batch_id IS NOT NULL")
        ).scalar_one()
        batch_count = conn.execute(
            text("SELECT COUNT(*) FROM nurse_batches")
        ).scalar_one()
        membership_count = conn.execute(
            text("SELECT COUNT(*) FROM nurse_batch_memberships")
        ).scalar_one()

        print(f"Patient records: {patient_count}")
        print(f"Patients with batch assignment: {assigned_count}")
        print(f"Nurse batches: {batch_count}")
        print(f"Batch memberships: {membership_count}")

        if assigned_count or batch_count or membership_count:
            raise RuntimeError(
                "Migration aborted: batch-related data exists. "
                "No schema changes were made."
            )

        # Permit only the two known foreign keys that depend on nurse_batches.
        references = conn.execute(
            text("""
                SELECT tc.table_name, tc.constraint_name, ccu.table_name
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.constraint_column_usage AS ccu
                  ON tc.constraint_catalog = ccu.constraint_catalog
                 AND tc.constraint_schema = ccu.constraint_schema
                 AND tc.constraint_name = ccu.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY'
                  AND tc.constraint_schema = current_schema()
                  AND ccu.table_name IN (
                      'nurse_batches', 'nurse_batch_memberships'
                  )
            """)
        ).all()

        expected = {
            ("patients", "patients_batch_id_fkey", "nurse_batches"),
            (
                "nurse_batch_memberships",
                "nurse_batch_memberships_batch_id_fkey",
                "nurse_batches",
            ),
        }
        actual = {(row[0], row[1], row[2]) for row in references}

        if actual != expected:
            raise RuntimeError(
                "Migration aborted; foreign-key dependencies differ from "
                f"the reviewed schema. Expected {sorted(expected)}, "
                f"found {sorted(actual)}. No schema changes were made."
            )

        # Confirm the only patient index involving batch_id is the expected index.
        patient_indexes = inspect(conn).get_indexes("patients")
        batch_indexes = [
            index["name"]
            for index in patient_indexes
            if "batch_id" in (index.get("column_names") or [])
        ]
        if batch_indexes != ["ix_patients_batch_id"]:
            raise RuntimeError(
                "Migration aborted; unexpected patient batch indexes: "
                f"{batch_indexes}. No schema changes were made."
            )

        # Preserve the patient-to-branch foreign key and all patient rows.
        conn.execute(text("""
            ALTER TABLE patients
            DROP CONSTRAINT patients_batch_id_fkey
        """))
        conn.execute(text("""
            DROP INDEX ix_patients_batch_id
        """))
        conn.execute(text("""
            ALTER TABLE patients
            DROP COLUMN batch_id
        """))
        conn.execute(text("DROP TABLE nurse_batch_memberships"))
        conn.execute(text("DROP TABLE nurse_batches"))

        # Verify the patient table and its branch relationship still exist.
        remaining_columns = {
            column["name"] for column in inspect(conn).get_columns("patients")
        }
        if "batch_id" in remaining_columns or "branch_id" not in remaining_columns:
            raise RuntimeError(
                "Post-migration schema verification failed; transaction will roll back."
            )

        branch_fk_names = {
            fk["name"]
            for fk in inspect(conn).get_foreign_keys("patients")
            if "branches" in fk.get("referred_table", "")
        }
        if "patients_branch_id_fkey" not in branch_fk_names:
            raise RuntimeError(
                "Patient-to-branch foreign key missing; transaction will roll back."
            )

        final_patient_count = conn.execute(
            text("SELECT COUNT(*) FROM patients")
        ).scalar_one()
        if final_patient_count != patient_count:
            raise RuntimeError(
                "Patient count changed; transaction will roll back."
            )

        print("Migration completed successfully.")
        print(f"Patient records preserved: {final_patient_count}")
        print("Patient-to-branch foreign key preserved.")


if __name__ == "__main__":
    remove_nurse_batches()
