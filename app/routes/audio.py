from fastapi import APIRouter, File, UploadFile, HTTPException, Security
from fastapi.security import APIKeyHeader
from openai import AsyncOpenAI
import logging
from typing import Dict, Any
import json

from app.core.config import settings
from app.core.security import verify_api_key
from app.services.transcript_processor import TranscriptProcessor
from app.services.cpt_lcd_matcher import CPTLCDMatcher
from app.models.transcript_response import TranscriptResponse

# Configure logging
logger = logging.getLogger(__name__)

# API Key security
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

router = APIRouter()
openai = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

@router.post("/transcribe_audio", response_model=TranscriptResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    api_key: str = Security(API_KEY_HEADER)
) -> TranscriptResponse:
    """
    Transcribe audio file to text and process it into structured clinical data.
    
    Args:
        file: Audio file to transcribe
        api_key: API key for authentication
        
    Returns:
        TranscriptResponse containing structured clinical data
    """
    try:
        # Verify API key
        if not verify_api_key(api_key):
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        # Read audio file
        audio_bytes = await file.read()
        
        # Transcribe audio using Whisper
        logger.info(f"Transcribing audio file: {file.filename}")
        response = await openai.audio.transcriptions.create(
            model="whisper-1",
            file=(file.filename, audio_bytes),
            response_format="text"
        )
        transcript = response.strip()
        
        # Log transcript in debug mode
        if settings.DEBUG_MODE:
            logger.debug("Transcribed text: %s", transcript)
        
        # Initialize processors
        transcript_processor = TranscriptProcessor()
        cpt_lcd_matcher = CPTLCDMatcher()
        
        # Process transcript
        logger.info("Processing transcript")
        extracted_data = await transcript_processor.process(transcript)
        
        # Match CPT/LCD codes
        coding_data = await cpt_lcd_matcher.match(extracted_data)
        
        # Convert coding data to new format
        cpt_codes = [
            {
                "code": code,
                "description": coding_data.get("descriptions", {}).get(code, ""),
                "requires_lcd": code in coding_data.get("lcd_codes", []),
                "lcd_code": coding_data.get("lcd_codes", {}).get(code)
            }
            for code in coding_data.get("cpt_suggestions", [])
        ]
        
        # Combine results into new format
        response_data = {
            "patient_info": extracted_data.get("patient_info"),
            "chief_complaint": extracted_data.get("chief_complaint"),
            "history_of_present_illness": extracted_data.get("history_of_present_illness"),
            "assessment": extracted_data.get("assessment"),
            "plan": extracted_data.get("plan"),
            "pain_rating": extracted_data.get("pain_rating"),
            "prior_treatments": extracted_data.get("prior_treatments"),
            "vital_signs": extracted_data.get("vital_signs"),
            "past_medical_history": extracted_data.get("past_medical_history"),
            "social_history": extracted_data.get("social_history"),
            "family_history": extracted_data.get("family_history"),
            "review_of_systems": extracted_data.get("review_of_systems"),
            "exam_findings": extracted_data.get("exam_findings"),
            "imaging_summary": extracted_data.get("imaging_summary"),
            "recommended_cpt_codes": cpt_codes,
            "qpp_measures": extracted_data.get("qpp_measures", []),
            "follow_up_instructions": extracted_data.get("follow_up_instructions"),
            "date": extracted_data.get("date"),
            "prompt": transcript  # Include original transcript
        }
        
        return TranscriptResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 