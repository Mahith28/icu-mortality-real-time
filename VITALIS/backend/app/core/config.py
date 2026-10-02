from pydantic_settings import BaseSettings, SettingsConfigDict
class Settings(BaseSettings):
    app_name: str = "VITALIS"
    app_version: str = "1.0.0"
    database_url: str
    secret_key: str
    access_token_expire_minutes: int = 60
    kafka_bootstrap_servers: str = "localhost:9092"
    risk_topic: str = "risk_scores"
    explanation_topic: str = "risk_explanations"
    realtime_static_topic: str = "icu_static"
    realtime_lifecycle_topic: str = "icu_lifecycle"
    model_version: str = "vitalis-1.0"
    doctor_email_domain: str = "doctor.vitalis.com"
    nurse_email_domain: str = "nurse.vitalis.com"
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )
settings = Settings()