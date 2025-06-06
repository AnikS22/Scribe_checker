from pydantic import BaseModel
from typing import Optional, Any
from app.models.transcript_response import PatientInfo, PainRating

class TranscriptProcessorResponse(BaseModel):
    patient_info: Optional[PatientInfo] = None
    chief_complaint: Optional[str] = None
    history_of_present_illness: Optional[str] = None
    assessment: Optional[str] = None
    plan: Optional[str] = None
    pain_rating: Optional[PainRating] = None
    prior_treatments: Optional[str] = None
    vital_signs: Optional[str] = None
    past_medical_history: Optional[str] = None
    social_history: Optional[str] = None
    family_history: Optional[str] = None
    review_of_systems: Optional[str] = None
    exam_findings: Optional[str] = None
    imaging_summary: Optional[str] = None
    follow_up_instructions: Optional[str] = None
    date: Optional[str] = None
    prompt: Optional[str] = None
    # Add any other fields returned by TranscriptProcessor (Agent 1) here

    # Ensure patient_info is included if TranscriptProcessor returns it
    patient_info: Optional[Any] = None # Use Any or define PatientInfo locally/import if needed 