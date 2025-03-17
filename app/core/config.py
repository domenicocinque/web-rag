from pathlib import Path
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ROOT_PATH: Path = Path(__file__).parent.parent.parent
    APP_PATH: Path = ROOT_PATH / "app"

    model_config = SettingsConfigDict(env_file=APP_PATH / "core" / ".env")

    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "web-rag"

    OPENAI_API_KEY: SecretStr


settings = Settings()
