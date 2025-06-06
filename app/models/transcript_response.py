from pydantic import BaseModel, Field
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
    measure_id: Optional[str] = None
    title: Optional[str] = None
    status: Optional[str] = None

class CPTCode(BaseModel):
    code: Optional[str] = None
    description: Optional[str] = None
    requires_lcd: Optional[bool] = None
    lcd_code: Optional[str] = None

class LCDValidationResult(BaseModel):
    cpt_code: Optional[str] = None
    lcd_code: Optional[str] = None
    requirements: Optional[List[str]] = None
    status: Optional[str] = None # e.g., "Meets" | "Partially Meets" | "Does Not Meet"

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
    qpp_measures: Optional[List[QPPMeasure]] = None
    recommended_cpt_codes: Optional[List[CPTCode]] = None
    follow_up_instructions: Optional[str] = None
    date: Optional[str] = None
    prompt: Optional[str] = None  # Original transcript/prompt

    # New fields for agent orchestration results
    icd_codes: Optional[List[str]] = Field(default_factory=list)
    lcd_validation: Optional[List[LCDValidationResult]] = Field(default_factory=list) 