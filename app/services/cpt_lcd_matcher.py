from typing import Dict, Any, List, Set
import json
import logging
from pathlib import Path
from app.core.config import settings

logger = logging.getLogger(__name__)

class CPTLCDMatcher:
    def __init__(self):
        self.cpt_lcd_dict = self._load_cpt_lcd_dictionary()
        
    def _load_cpt_lcd_dictionary(self) -> Dict[str, Dict[str, Any]]:
        """
        Load the CPT/LCD dictionary from JSON file.
        
        Returns:
            Dict containing CPT/LCD mappings and requirements
        """
        try:
            dict_path = Path(settings.CPT_LCD_DICT_PATH)
            if not dict_path.exists():
                logger.warning(f"CPT/LCD dictionary not found at {dict_path}, using default")
                return self._get_default_dictionary()
                
            with open(dict_path, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Error loading CPT/LCD dictionary: {str(e)}")
            return self._get_default_dictionary()
    
    def _get_default_dictionary(self) -> Dict[str, Dict[str, Any]]:
        """Get a default CPT/LCD dictionary."""
        return {
            "lumbar mri": {
                "cpt": "72148",
                "lcd": "L34220",
                "required_fields": [
                    "severity",
                    "duration",
                    "functional_limitations",
                    "neurologic_deficits"
                ]
            },
            "facet joint injection": {
                "cpt": "64493",
                "lcd": "L34993",
                "required_fields": [
                    "pain_rating",
                    "prior_treatment",
                    "trigger_point_location"
                ]
            },
            "physical therapy": {
                "cpt": "97110",
                "lcd": "L33611",
                "required_fields": [
                    "functional_limitations",
                    "prior_treatment",
                    "assessment"
                ]
            }
            # Add more procedures as needed
        }
    
    async def match(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Match procedures to CPT codes and check LCD requirements.
        
        Args:
            extracted_data: Structured data from transcript processor
            
        Returns:
            Dict containing CPT suggestions, LCD codes, and warnings
        """
        try:
            # Initialize result containers
            cpt_suggestions: Set[str] = set()
            lcd_codes: Set[str] = set()
            lcd_warnings: List[str] = []
            
            # Get mentioned procedures
            procedures = extracted_data.get("procedures_mentioned", [])
            plan_items = extracted_data.get("plan", [])
            
            # Combine procedures and plan items for matching
            all_items = set(
                [p.lower() for p in procedures] +
                [p.lower() for p in plan_items]
            )
            
            # Match each item to CPT/LCD codes
            for item in all_items:
                for procedure, codes in self.cpt_lcd_dict.items():
                    if procedure in item:
                        # Add CPT code
                        cpt_suggestions.add(codes["cpt"])
                        # Add LCD code
                        lcd_codes.add(codes["lcd"])
                        
                        # Check required fields
                        missing_fields = self._check_required_fields(
                            codes["required_fields"],
                            extracted_data
                        )
                        
                        if missing_fields:
                            warning = (
                                f"Missing required fields for {procedure} "
                                f"(CPT {codes['cpt']}, LCD {codes['lcd']}): "
                                f"{', '.join(missing_fields)}"
                            )
                            lcd_warnings.append(warning)
            
            return {
                "cpt_suggestions": sorted(list(cpt_suggestions)),
                "lcd_codes": sorted(list(lcd_codes)),
                "lcd_warnings": lcd_warnings
            }
            
        except Exception as e:
            logger.error(f"Error matching CPT/LCD codes: {str(e)}")
            raise
    
    def _check_required_fields(
        self,
        required_fields: List[str],
        extracted_data: Dict[str, Any]
    ) -> List[str]:
        """
        Check if all required fields are present in the extracted data.
        
        Args:
            required_fields: List of required field names
            extracted_data: Structured data from transcript processor
            
        Returns:
            List of missing required fields
        """
        missing_fields = []
        
        for field in required_fields:
            # Handle nested fields (e.g., objective_findings.neurological_deficits)
            if "." in field:
                parent, child = field.split(".")
                if (
                    parent not in extracted_data or
                    not extracted_data[parent].get(child)
                ):
                    missing_fields.append(field)
            # Handle array fields
            elif field in ["prior_treatment", "plan", "procedures_mentioned"]:
                if not extracted_data.get(field):
                    missing_fields.append(field)
            # Handle regular fields
            elif not extracted_data.get(field):
                missing_fields.append(field)
        
        return missing_fields 