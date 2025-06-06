# agents_config.py

# Agent IDs for our pipeline
AGENT_IDS = {
    "clinical_extractor": "asst_6wQa9NCLIIMMnYD1dUB8SZ9T",  # Agent1: Clinical_Extractor (runs locally via TranscriptProcessor)
    "json_to_icd": "asst_pf5Vgar3flb1ZsUQWV4B0NKm",       # Agent2: Json_to_icd
    "lcd_validator": "asst_UqHVm42UQsg7zQDJxxkMCS5U",      # Agent5: LCD_Validator_Agentv1
    "cpt_codes": "asst_bRpc7ot4kRu6asq7CTcoKK0I",          # Agent6: CPTcodes
    # Agent3 LcdChecker is not needed here — LCD Validator replaces it.
} 