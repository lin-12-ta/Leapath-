from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str = "mysql+pymysql://career_app:lintaisthebest@localhost:3306/career_navigator"
    jwt_secret: str = "dev-only-change-me"
    access_token_minutes: int = 15
    refresh_token_days: int = 14
    # Credentials belong in the local environment, never in source control.
    openrouter_api_key: str = ""
    openrouter_model: str = "openrouter/free"
    openrouter_embedding_model: str = "nvidia/nemotron-3-embed-1b:free"
    frontend_origin: str = "http://localhost:5173"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
