from backend.app.db.database import Base, engine
from backend.app.models import User
Base.metadata.create_all(bind=engine)
print("Database tables created successfully.")
