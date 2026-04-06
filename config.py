from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    DB_USER: str
    DB_PASSWORD: str
    DB_NAME: str
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    
    MODEL_PATH: str = "ml_model.pkl"
    LEAGUES: List[str] = ["La_liga", "EPL", "Bundesliga", "Serie_A", "Ligue_1", "RFPL"]
    
    class Config:
        env_file = ".env"

settings = Settings()