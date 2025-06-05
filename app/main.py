from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime

# Import our custom modules
from app.services.transcript_processor import TranscriptProcessor
from app.services.cpt_lcd_matcher import CPTLCDMatcher
from app.core.config import settings
from app.core.security import verify_api_key

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Clinical Transcript Processor",
    description="API for processing clinical transcripts and generating insurance-compliant documentation",
    version="1.0.0"
)

# Configure CORS for iOS app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific iOS app domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key security
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

class TranscriptRequest(BaseModel):
    transcript: str
    patient_id: Optional[str] = None
    visit_date: Optional[datetime] = None

class TranscriptResponse(BaseModel):
    date: str
    diagnosis: str
    pain_rating: str
    prior_treatment: List[str]
    subjective_complaints: str
    objective_findings: Dict[str, Any]
    assessment: str
    plan: List[str]
    functional_limitations: str
    symptom_duration: str
    procedures_mentioned: List[str]
    cpt_suggestions: List[str]
    lcd_codes: List[str]
    lcd_warnings: List[str]

@app.post("/process_transcript", response_model=TranscriptResponse)
async def process_transcript(
    request: TranscriptRequest,
    api_key: str = Security(API_KEY_HEADER)
) -> TranscriptResponse:
    """
    Process a clinical transcript and return structured data with CPT/LCD suggestions.
    
    Args:
        request: TranscriptRequest containing the transcript text and optional metadata
        api_key: API key for authentication
        
    Returns:
        TranscriptResponse containing structured clinical data and coding suggestions
    """
    try:
        # Verify API key
        if not verify_api_key(api_key):
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        # Initialize processors
        transcript_processor = TranscriptProcessor()
        cpt_lcd_matcher = CPTLCDMatcher()
        
        # Process transcript
        logger.info(f"Processing transcript for patient {request.patient_id}")
        extracted_data = await transcript_processor.process(request.transcript)
        
        # Match CPT/LCD codes
        coding_data = await cpt_lcd_matcher.match(extracted_data)
        
        # Combine results
        response_data = {**extracted_data, **coding_data}
        
        return TranscriptResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Error processing transcript: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG_MODE
    ) 