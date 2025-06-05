from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Clinical Transcript Processor"
    
    # Server Settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "False").lower() == "true"
    
    # Security
    API_KEY_NAME: str = "X-API-Key"
    API_KEY: str = os.getenv("API_KEY", "your-secret-key-here")  # Change in production
    
    # OpenAI Settings
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4")
    
    # Database Settings (for future use)
    DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # CPT/LCD Dictionary Path
    CPT_LCD_DICT_PATH: str = os.getenv(
        "CPT_LCD_DICT_PATH",
        "app/data/cpt_lcd_dictionary.json"
    )
    
    class Config:
        case_sensitive = True

# Create global settings object
settings = Settings()

# Validate required settings
if not settings.OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required") 