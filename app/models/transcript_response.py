from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class PatientInfo(BaseModel):
    age: Optional[str] = None
    sex: Optional[str] = None
    visit_date: Optional[str] = None
    visit_location: Optional[str] = None

class PainRating(BaseModel):
    level: Optional[str] = None
    location: Optional[str] = None

class QPPMeasure(BaseModel):
    measure_id: str
    title: str
    status: str

class CPTCode(BaseModel):
    code: str
    description: str
    requires_lcd: bool
    lcd_code: Optional[str] = None

class TranscriptResponse(BaseModel):
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
    qpp_measures: Optional[List[QPPMeasure]] = []
    recommended_cpt_codes: Optional[List[CPTCode]] = []
    follow_up_instructions: Optional[str] = None
    date: Optional[str] = None
    prompt: Optional[str] = None  # Original transcript/prompt 