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
        self.system_prompt = """You are a highly specialized clinical documentation and billing AI. Your task is to extract all required data fields from a patient transcript to support medical documentation, CPT coding, LCD validation, and insurance justification.

⚠️ RULES:
- ONLY use information directly stated in the transcript. Never infer, assume, or fabricate.
- Leave fields as "Not specified" or null if missing.
- There may be **multiple CPT codes** mentioned or implied. Identify and include ALL that are medically relevant.
- Match each CPT to **medically necessary procedures** described.
- If a CPT code is linked or requires a LCD, include:
  - The official LCD code (e.g. "L34220")
  - A list of known CMS LCD medical necessity criteria
  - A status field: `"lcd_status": "Meets" | "Partially Meets" | "Does Not Meet"`
- state what is missing
- If no LCD is needed, set `"requires_lcd": false`.
- Use real AMA CPT codes and CMS LCD policies (simulate their logic).
- Output should be valid JSON. No extra text or explanation.
- For each CPT, try to include LCD flags and basic justification.
- Do NOT include narrative text outside the JSON structure.

📚 You are referencing:
- CMS Medicare LCD Database
- 2024 CPT Codebook (AMA)
- ICD-10 mappings and QPP quality measures

🔍 FIELD EXTRACTION GUIDELINES:

📌 patient_info
- "age": Look for phrases like "50-year-old" or "in his early 60s"
- "sex": Derive from pronouns ("he/she") or explicit mention
- "visit_date": Extract date in YYYY-MM-DD format
- "visit_location": Look for clinic/hospital names

📌 chief_complaint
- Look for the first mention of the patient's main problem
- Common phrases: "came in for...", "complains of...", "chief concern is..."
- Extract the primary reason for visit

📌 history_of_present_illness
- Look for narrative describing onset, duration, and progression
- Example: "has had back pain for 2 months..."
- Include relevant history and context

📌 pain_rating
- Look for pain scale (0-10) and location
- Example: "rates it 7 out of 10 in lower back"
- Extract both numeric level and anatomical location

📌 assessment
- Look for diagnoses and clinical impressions
- Common phrases: "Diagnosed with...", "Impression:"
- Include ICD terms like "radiculopathy", "herniation"
- List all identified conditions

📌 plan
- Extract future treatments and recommendations
- Examples: "Plan is to get MRI", "Referred to PT"
- Include all proposed interventions

📌 prior_treatments
- Document past medical procedures
- Examples: "previous injections", "underwent PT for 6 weeks"
- Include duration and outcomes if mentioned

📌 exam_findings
- Extract specific clinical exam signs
- Examples: "Positive straight leg raise", "reduced reflexes"
- Include all documented physical findings

📌 imaging_summary
- Document any imaging results
- Example: "MRI showed disc bulge at L5-S1"
- Include modality and key findings

📌 qpp_measures
- Identify quality program measures
- Examples: "Tobacco use screening", "Pain reassessment"
- Match to CMS quality measures

📌 recommended_cpt_codes
- Match procedures to real CPT codes from 2024 CPT Codebook
- Example structure:
{
  "code": "72148",
  "description": "MRI lumbar spine w/o contrast",
  "requires_lcd": true,
  "lcd_code": "L34220",
  "lcd_requirements": [
    "Indication of radiculopathy or herniation",
    "Failure of conservative therapy"
  ],
  "lcd_status": "Meets"
}
- Include ALL relevant procedures mentioned
- Validate against LCD requirements
- Set appropriate LCD status

🎯 Your output must follow this structure:

{
  "patient_info": {
    "age": "...",
    "sex": "...",
    "visit_date": "...",
    "visit_location": "..."
  },
  "chief_complaint": "...",
  "history_of_present_illness": "...",
  "assessment": "...",
  "plan": "...",
  "pain_rating": {
    "level": "...",
    "location": "..."
  },
  "prior_treatments": "...",
  "vital_signs": "...",
  "past_medical_history": "...",
  "social_history": "...",
  "family_history": "...",
  "review_of_systems": "...",
  "exam_findings": "...",
  "imaging_summary": "...",
  "qpp_measures": [
    {
      "measure_id": "...",
      "title": "...",
      "status": "Met / Exclusion / Denied"
    }
  ],
  "recommended_cpt_codes": [
    {
      "code": "...",
      "description": "...",
      "requires_lcd": true,
      "lcd_code": "...",
      "lcd_requirements": [
        "Requirement 1 from CMS LCD",
        "Requirement 2..."
      ],
      "lcd_status": "Meets / Partially Meets / Does Not Meet"
    }
  ],
  "follow_up_instructions": "...",
  "date": "YYYY-MM-DD"
}"""
        
    async def process(self, transcript: str) -> Dict[str, Any]:
        """
        Process a clinical transcript using GPT-4 to extract structured data.
        
        Args:
            transcript: Raw transcript text
            
        Returns:
            Dict containing structured clinical data
            
        Raises:
            ValueError: If GPT response is not valid JSON
            Exception: For other processing errors
        """
        try:
            # Prepare the messages for GPT
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Process this clinical transcript:\n\n{transcript}"}
            ]
            
            # Call OpenAI API
            logger.info("Sending transcript to GPT for processing")
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,  # Low temperature for consistent extraction
                response_format={"type": "json_object"}
            )
            
            # Get the response content
            content = response.choices[0].message.content
            
            # Parse and validate JSON
            try:
                extracted_data = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON response from GPT: {str(e)}")
                logger.error(f"Raw response: {content}")
                raise ValueError(f"Invalid JSON response from GPT: {str(e)}")
            
            # Validate required structure
            self._validate_extracted_data(extracted_data)
            
            # Clean and standardize the data
            cleaned_data = self._clean_extracted_data(extracted_data)
            
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Error processing transcript: {str(e)}")
            raise
    
    def _validate_extracted_data(self, data: Dict[str, Any]) -> None:
        """
        Validate the structure of extracted data.
        
        Args:
            data: Extracted data to validate
            
        Raises:
            ValueError: If data structure is invalid
        """
        # Check for required top-level fields
        required_fields = {
            "patient_info", "chief_complaint", "history_of_present_illness",
            "assessment", "plan", "pain_rating", "recommended_cpt_codes"
        }
        
        missing_fields = required_fields - set(data.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Validate patient_info structure
        if not isinstance(data.get("patient_info"), dict):
            raise ValueError("patient_info must be an object")
        
        # Validate pain_rating structure
        pain_rating = data.get("pain_rating")
        if pain_rating and not isinstance(pain_rating, dict):
            raise ValueError("pain_rating must be an object")
        
        # Validate CPT codes structure
        cpt_codes = data.get("recommended_cpt_codes", [])
        if not isinstance(cpt_codes, list):
            raise ValueError("recommended_cpt_codes must be an array")
        
        for code in cpt_codes:
            if not isinstance(code, dict):
                raise ValueError("Each CPT code must be an object")
            required_cpt_fields = {"code", "description", "requires_lcd"}
            missing_cpt_fields = required_cpt_fields - set(code.keys())
            if missing_cpt_fields:
                raise ValueError(f"CPT code missing required fields: {missing_cpt_fields}")
    
    def _clean_extracted_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean and standardize the extracted data.
        
        Args:
            data: Raw extracted data from GPT
            
        Returns:
            Cleaned and standardized data
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
            "qpp_measures": [],
            "recommended_cpt_codes": [],
            "follow_up_instructions": None,
            "date": None
        }
        
        # Add missing fields with default values
        for field, default_value in default_values.items():
            if field not in data:
                data[field] = default_value
        
        # Clean patient_info
        if not isinstance(data.get("patient_info"), dict):
            data["patient_info"] = default_values["patient_info"]
        else:
            for key in ["age", "sex", "visit_date", "visit_location"]:
                if key not in data["patient_info"]:
                    data["patient_info"][key] = None
        
        # Clean pain_rating
        if not isinstance(data.get("pain_rating"), dict):
            data["pain_rating"] = default_values["pain_rating"]
        else:
            for key in ["level", "location"]:
                if key not in data["pain_rating"]:
                    data["pain_rating"][key] = None
        
        # Clean CPT codes
        if not isinstance(data.get("recommended_cpt_codes"), list):
            data["recommended_cpt_codes"] = []
        else:
            for code in data["recommended_cpt_codes"]:
                if not isinstance(code, dict):
                    continue
                # Ensure all required fields exist
                code.setdefault("requires_lcd", False)
                code.setdefault("lcd_code", None)
                code.setdefault("lcd_requirements", [])
                code.setdefault("lcd_status", "Not Evaluated")
        
        # Clean QPP measures
        if not isinstance(data.get("qpp_measures"), list):
            data["qpp_measures"] = []
        else:
            for measure in data["qpp_measures"]:
                if not isinstance(measure, dict):
                    continue
                measure.setdefault("status", "Not Evaluated")
        
        return data 