from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
import openai
import shutil
import os
import logging
from typing import Dict, Any
from app.core.config import settings
from app.core.auth import verify_api_key
from app.services.transcript_processor import TranscriptProcessor
from app.services.cpt_lcd_matcher import CPTLCDMatcher
from app.models.transcript_response import TranscriptResponse

# Configure logging
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

# Initialize transcript processor
transcript_processor = TranscriptProcessor()

router = APIRouter()

@router.post("/transcribe", response_model=Dict[str, Any])
async def transcribe_audio(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Transcribe an audio file using OpenAI's Whisper API and process the transcript.
    
    Args:
        file: Audio file uploaded via multipart/form-data
        api_key: API key for authentication
        
    Returns:
        Dict containing:
        - transcript: Raw transcript text
        - processed_data: Structured clinical data
        - error: Any error message (if applicable)
        
    Raises:
        HTTPException: For various error conditions
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Validate file type
    allowed_types = ["audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/mp4"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(allowed_types)}"
        )
    
    try:
        # Create temp directory if it doesn't exist
        temp_dir = "temp_audio"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save uploaded file to temp directory
        file_location = os.path.join(temp_dir, f"temp_{file.filename}")
        try:
            with open(file_location, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Transcribe using Whisper API
            logger.info(f"Transcribing audio file: {file.filename}")
            with open(file_location, "rb") as audio_file:
                response = await client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="json"
                )
            
            transcript_text = response.text
            
            # Process the transcript
            logger.info("Processing transcript with GPT")
            processed_data = await transcript_processor.process(transcript_text)
            
            return {
                "transcript": transcript_text,
                "processed_data": processed_data,
                "error": None
            }
            
        finally:
            # Clean up temp file
            if os.path.exists(file_location):
                os.remove(file_location)
                
    except openai.APIError as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error transcribing audio: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error processing audio file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio file: {str(e)}"
        )

@router.post("/transcribe-only", response_model=Dict[str, str])
async def transcribe_only(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
) -> Dict[str, str]:
    """
    Transcribe an audio file using OpenAI's Whisper API without additional processing.
    This endpoint is useful for testing or when only the raw transcript is needed.
    
    Args:
        file: Audio file uploaded via multipart/form-data
        api_key: API key for authentication
        
    Returns:
        Dict containing:
        - text: Raw transcript text
        - error: Any error message (if applicable)
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Validate file type
    allowed_types = ["audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/mp4"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(allowed_types)}"
        )
    
    try:
        # Create temp directory if it doesn't exist
        temp_dir = "temp_audio"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save uploaded file to temp directory
        file_location = os.path.join(temp_dir, f"temp_{file.filename}")
        try:
            with open(file_location, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Transcribe using Whisper API
            logger.info(f"Transcribing audio file: {file.filename}")
            with open(file_location, "rb") as audio_file:
                response = await client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="json"
                )
            
            return {
                "text": response.text,
                "error": None
            }
            
        finally:
            # Clean up temp file
            if os.path.exists(file_location):
                os.remove(file_location)
                
    except openai.APIError as e:
        logger.error(f"OpenAI API error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"text": None, "error": f"Error transcribing audio: {str(e)}"}
        )
    except Exception as e:
        logger.error(f"Error processing audio file: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"text": None, "error": f"Error processing audio file: {str(e)}"}
        )

@router.post("/transcribe_audio", response_model=TranscriptResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
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
        response = await client.audio.transcriptions.create(
            model="whisper-1",
            file=(file.filename, audio_bytes),
            response_format="text"
        )
        transcript = response.strip()
        
        # Log transcript in debug mode
        if settings.DEBUG_MODE:
            logger.debug("Transcribed text: %s", transcript)
        
        # Initialize processors
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