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
        - date: Visit date (YYYY-MM-DD)
        - diagnosis: Primary diagnosis
        - pain_rating: Pain level (0-10) with location
        - prior_treatment: Array of previous treatments
        - subjective_complaints: Patient's reported symptoms
        - objective_findings: Object containing:
          - range_of_motion: ROM findings
          - tenderness: Areas of tenderness
          - neurological_deficits: Any neurological findings
        - assessment: Clinical assessment
        - plan: Array of treatment plans
        - functional_limitations: Impact on daily activities
        - symptom_duration: How long symptoms present
        - procedures_mentioned: Array of mentioned procedures

        Return the data in valid JSON format."""
    
    def _clean_extracted_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean and validate the extracted data.
        
        Args:
            data: Raw extracted data from GPT-4
            
        Returns:
            Cleaned and validated data
        """
        # Ensure all required fields exist
        required_fields = {
            "date", "diagnosis", "pain_rating", "prior_treatment",
            "subjective_complaints", "objective_findings", "assessment",
            "plan", "functional_limitations", "symptom_duration",
            "procedures_mentioned"
        }
        
        # Add missing fields with default values
        for field in required_fields:
            if field not in data:
                if field in ["prior_treatment", "plan", "procedures_mentioned"]:
                    data[field] = []
                elif field == "objective_findings":
                    data[field] = {
                        "range_of_motion": "",
                        "tenderness": "",
                        "neurological_deficits": ""
                    }
                else:
                    data[field] = ""
        
        # Clean and standardize data
        if isinstance(data["prior_treatment"], str):
            data["prior_treatment"] = [data["prior_treatment"]]
        if isinstance(data["plan"], str):
            data["plan"] = [data["plan"]]
        if isinstance(data["procedures_mentioned"], str):
            data["procedures_mentioned"] = [data["procedures_mentioned"]]
        
        return data 