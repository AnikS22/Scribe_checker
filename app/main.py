from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime
import re

# Import our custom modules
from app.services.transcript_processor import TranscriptProcessor
from app.services.cpt_lcd_matcher import CPTLCDMatcher
from app.core.config import settings
from app.core.security import verify_api_key
from app.models.transcript_response import (
    TranscriptResponse, PatientInfo, PainRating,
    QPPMeasure, CPTCode
)
from app.routes.audio import router as audio_router

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

# Include routers
app.include_router(audio_router, prefix="/api/v1", tags=["audio"])

# API Key security
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

"""
PII Redaction Module

This module provides basic PII (Personally Identifiable Information) redaction
for clinical transcripts to support HIPAA compliance. The current implementation
focuses on basic name detection and redaction, but should be expanded to include:

1. More comprehensive name detection (including nicknames, aliases)
2. Address detection and redaction
3. Phone number detection and redaction
4. Email address detection and redaction
5. Medical record number detection and redaction
6. Insurance information redaction
7. Date of birth redaction (while preserving age)
8. Social security number detection and redaction

Note: This is a basic implementation and should be enhanced with:
- More sophisticated NLP for better name detection
- Regular expression patterns for other PII types
- Machine learning models for improved accuracy
- Regular updates to patterns and rules
- Audit logging of redacted information
- Validation of redaction effectiveness
"""

def anonymize_transcript(transcript: str) -> str:
    """
    Redact PII from clinical transcripts to support HIPAA compliance.
    
    Args:
        transcript: Raw transcript text containing potential PII
        
    Returns:
        str: Transcript with PII redacted
        
    Note:
        This is a basic implementation that should be expanded for
        production use. Current limitations:
        - Only handles basic name patterns
        - May miss some name variations
        - Does not handle all PII types
        - No validation of redaction effectiveness
    """
    # Store original for debug logging
    original = transcript
    
    # Specific terms to always redact
    specific_terms = [
        r'\bSahai\b',
        r'\bSOC\b',
        r'\bSpine Orthopedic Center\b',
        r'\bAsh\b',
        r'\bAshish\b'
    ]
    
    # Common medical titles and prefixes
    titles = r'\b(?:Dr\.|Mr\.|Mrs\.|Ms\.|Miss|Prof\.|Doctor|Nurse|PA|NP|RN|MD|DO)\b'
    
    # Pattern to match names after titles
    # This will match: "Dr. Smith", "Mr. John Smith", etc.
    name_pattern = f"{titles}\\s+[A-Z][a-z]+(?:\\s+[A-Z][a-z]+)?"
    
    # Pattern to match standalone names (capitalized words that might be names)
    # This is more aggressive and might need tuning
    standalone_names = r'\b[A-Z][a-z]+(?:-[A-Z][a-z]+)?\b'
    
    # First pass: Replace specific terms
    for term in specific_terms:
        transcript = re.sub(term, '[REDACTED]', transcript, flags=re.IGNORECASE)
    
    # Second pass: Replace names after titles
    transcript = re.sub(name_pattern, '[REDACTED]', transcript)
    
    # Third pass: Replace potential standalone names
    # We're more conservative here to avoid over-redaction
    # Only replace if the word is capitalized and not at the start of a sentence
    words = transcript.split()
    for i, word in enumerate(words):
        if (i > 0 and  # Not at start of sentence
            word[0].isupper() and  # Starts with capital
            len(word) > 2 and  # Not a short word
            not any(c.isdigit() for c in word) and  # Not a number
            word not in ['I', 'A', 'The', 'This', 'That', 'These', 'Those']):  # Common words
            words[i] = '[REDACTED]'
    transcript = ' '.join(words)
    
    # Log redaction in debug mode
    if settings.DEBUG_MODE:
        logger.debug("Original transcript: %s", original)
        logger.debug("Redacted transcript: %s", transcript)
        # Log specific term redactions
        for term in specific_terms:
            if re.search(term, original, re.IGNORECASE):
                logger.debug(f"Redacted specific term: {term}")
    
    return transcript

class TranscriptRequest(BaseModel):
    transcript: str
    patient_id: Optional[str] = None
    visit_date: Optional[datetime] = None

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
        
        # Anonymize transcript before processing
        sanitized_transcript = anonymize_transcript(request.transcript)
        
        # Process transcript
        logger.info(f"Processing transcript for patient {request.patient_id}")
        extracted_data = await transcript_processor.process(sanitized_transcript)
        
        # Match CPT/LCD codes
        coding_data = await cpt_lcd_matcher.match(extracted_data)
        
        # Convert coding data to new format
        cpt_codes = [
            CPTCode(
                code=code,
                description=coding_data.get("descriptions", {}).get(code, ""),
                requires_lcd=code in coding_data.get("lcd_codes", []),
                lcd_code=coding_data.get("lcd_codes", {}).get(code)
            )
            for code in coding_data.get("cpt_suggestions", [])
        ]
        
        # Create patient info from request
        patient_info = PatientInfo(
            visit_date=request.visit_date.isoformat() if request.visit_date else None,
            visit_location=None  # Could be extracted from transcript if available
        )
        
        # Combine results into new format
        response_data = {
            "patient_info": patient_info,
            "chief_complaint": extracted_data.get("chief_complaint"),
            "history_of_present_illness": extracted_data.get("history_of_present_illness"),
            "assessment": extracted_data.get("assessment"),
            "plan": extracted_data.get("plan"),
            "pain_rating": PainRating(
                level=str(extracted_data.get("pain_rating", {}).get("level")),
                location=extracted_data.get("pain_rating", {}).get("location")
            ) if extracted_data.get("pain_rating") else None,
            "prior_treatments": ", ".join(extracted_data.get("prior_treatment", [])),
            "exam_findings": extracted_data.get("objective_findings", {}).get("range_of_motion"),
            "recommended_cpt_codes": cpt_codes,
            "qpp_measures": [],  # To be implemented
            "date": extracted_data.get("date")
        }
        
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