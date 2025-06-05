from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables from .env if available (for local dev)
load_dotenv()

class Settings(BaseSettings):
    # API Info
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Clinical Transcript Processor"
    
    # Server Settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "False").lower() == "true"
    
    # Security / API Key Authentication
    API_KEY_NAME: str = "X-API-Key"
    API_KEY: str = os.getenv("API_KEY")

    # OpenAI Settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4")

    # Future-proofing
    DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    CPT_LCD_DICT_PATH: str = os.getenv("CPT_LCD_DICT_PATH", "app/data/cpt_lcd_dictionary.json")

    class Config:
        case_sensitive = True

# Instantiate global settings object
settings = Settings()

# ✅ Validate required environment variables
missing = []
if not settings.API_KEY:
    missing.append("API_KEY")
if not settings.OPENAI_API_KEY:
    missing.append("OPENAI_API_KEY")

if missing:
    raise ValueError(f"Missing required environment variable(s): {', '.join(missing)}")
