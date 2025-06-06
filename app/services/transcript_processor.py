from typing import Dict, Any, List
import json
import logging
from openai import AsyncOpenAI
from app.core.config import settings

logger = logging.getLogger(__name__)

class TranscriptProcessor:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL
        
    async def process(self, transcript: str) -> Dict[str, Any]:
        """
        Process a clinical transcript using GPT-4 to extract structured data.
        
        Args:
            transcript: Raw transcript text
            
        Returns:
            Dict containing structured clinical data
        """
        try:
            # Prepare the prompt for GPT-4
            prompt = self._create_extraction_prompt(transcript)
            
            # Call OpenAI API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent extraction
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            extracted_data = json.loads(response.choices[0].message.content)
            
            # Validate and clean the extracted data
            cleaned_data = self._clean_extracted_data(extracted_data)
            
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Error processing transcript: {str(e)}")
            raise
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for GPT-4."""
        return """You are a medical transcription expert. Extract structured clinical data from the provided transcript.
        Follow these rules:
        1. Extract only factual information present in the transcript
        2. Use standardized medical terminology
        3. Format dates as YYYY-MM-DD
        4. Include all relevant clinical findings
        5. Return data in the specified JSON format
        6. If a field is not mentioned, use null or empty values
        7. Maintain patient privacy by not including PHI"""
    
    def _create_extraction_prompt(self, transcript: str) -> str:
        """Create the extraction prompt for GPT-4."""
        return f"""Extract the following information from this clinical transcript into JSON format:

        {transcript}

        Required fields:
        - patient_info: Object containing:
          - age: Patient's age
          - sex: Patient's sex
          - visit_date: Visit date (YYYY-MM-DD)
          - visit_location: Location of visit
        - chief_complaint: Primary reason for visit
        - history_of_present_illness: Detailed history
        - assessment: Clinical assessment
        - plan: Treatment plan
        - pain_rating: Object containing:
          - level: Pain level (0-10)
          - location: Location of pain
        - prior_treatments: Previous treatments
        - vital_signs: Vital signs
        - past_medical_history: PMH
        - social_history: Social history
        - family_history: Family history
        - review_of_systems: ROS
        - exam_findings: Physical exam findings
        - imaging_summary: Imaging results
        - follow_up_instructions: Follow-up plan
        - date: Visit date (YYYY-MM-DD)

        Return the data in valid JSON format."""
    
    def _clean_extracted_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean and validate the extracted data.
        
        Args:
            data: Raw extracted data from GPT-4
            
        Returns:
            Cleaned and validated data
        """
        # Define default values for all fields
        default_values = {
            "patient_info": {
                "age": None,
                "sex": None,
                "visit_date": None,
                "visit_location": None
            },
            "chief_complaint": None,
            "history_of_present_illness": None,
            "assessment": None,
            "plan": None,
            "pain_rating": {
                "level": None,
                "location": None
            },
            "prior_treatments": None,
            "vital_signs": None,
            "past_medical_history": None,
            "social_history": None,
            "family_history": None,
            "review_of_systems": None,
            "exam_findings": None,
            "imaging_summary": None,
            "follow_up_instructions": None,
            "date": None
        }
        
        # Add missing fields with default values
        for field, default_value in default_values.items():
            if field not in data:
                data[field] = default_value
        
        # Ensure patient_info is properly structured
        if not isinstance(data.get("patient_info"), dict):
            data["patient_info"] = default_values["patient_info"]
        else:
            for key in ["age", "sex", "visit_date", "visit_location"]:
                if key not in data["patient_info"]:
                    data["patient_info"][key] = None
        
        # Ensure pain_rating is properly structured
        if isinstance(data.get("pain_rating"), str):
            # Try to parse pain rating from string (e.g., "7/10 in lower back")
            try:
                parts = data["pain_rating"].split(" in ", 1)
                level = parts[0].split("/")[0] if "/" in parts[0] else None
                location = parts[1] if len(parts) > 1 else None
                data["pain_rating"] = {"level": level, "location": location}
            except:
                data["pain_rating"] = {"level": None, "location": data["pain_rating"]}
        elif not isinstance(data.get("pain_rating"), dict):
            data["pain_rating"] = default_values["pain_rating"]
        
        # Convert prior_treatments to string if it's a list
        if isinstance(data.get("prior_treatments"), list):
            data["prior_treatments"] = ", ".join(data["prior_treatments"])
        
        return data 