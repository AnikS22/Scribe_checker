# Clinical Transcript Processor Backend

A FastAPI-based backend service that processes clinical transcripts and generates insurance-compliant documentation. The service uses GPT-4 for intelligent extraction of clinical data and matches procedures to CPT/LCD codes.

## Features

- 📝 Transcript processing using GPT-4
- 🏷️ CPT code suggestion
- 📋 LCD requirement checking
- 🔒 API key authentication
- 🏗️ Modular, extensible architecture
- 📊 Structured JSON output

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file:
```bash
OPENAI_API_KEY=your_openai_api_key
API_KEY=your_custom_api_key
DEBUG_MODE=True
```

4. Run the server:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Example Usage

### Process a Transcript

```python
import requests
import json

API_URL = "http://localhost:8000/process_transcript"
API_KEY = "your_custom_api_key"

# Example transcript
transcript = """
Patient presents with 7/10 lower back pain radiating down left leg for 2 months.
Previous treatment includes NSAIDs and physical therapy with minimal relief.
On exam, limited lumbar ROM, positive straight leg raise on left, and decreased
sensation in L5 distribution. Plan: Order lumbar MRI, continue PT, consider
epidural if no improvement.
"""

# Make API request
response = requests.post(
    API_URL,
    headers={"X-API-Key": API_KEY},
    json={
        "transcript": transcript,
        "patient_id": "12345",
        "visit_date": "2024-03-15"
    }
)

# Print structured response
print(json.dumps(response.json(), indent=2))
```

### Example Response

```json
{
  "date": "2024-03-15",
  "diagnosis": "Lumbar radiculopathy",
  "pain_rating": "7/10 in lower back radiating down left leg",
  "prior_treatment": ["NSAIDs", "physical therapy"],
  "subjective_complaints": "Lower back pain radiating down left leg for 2 months",
  "objective_findings": {
    "range_of_motion": "Limited lumbar ROM",
    "tenderness": "",
    "neurological_deficits": "Positive straight leg raise on left, decreased sensation in L5 distribution"
  },
  "assessment": "Lumbar radiculopathy with left L5 distribution symptoms",
  "plan": ["Order lumbar MRI", "Continue PT", "Consider epidural if no improvement"],
  "functional_limitations": "",
  "symptom_duration": "2 months",
  "procedures_mentioned": ["MRI", "epidural"],
  "cpt_suggestions": ["72148", "62323"],
  "lcd_codes": ["L34220", "L34993"],
  "lcd_warnings": [
    "Missing required fields for lumbar mri (CPT 72148, LCD L34220): functional_limitations",
    "Missing required fields for epidural injection (CPT 62323, LCD L34993): functional_limitations"
  ]
}
```

## Project Structure

```
backend/
├── app/
│   ├── core/
│   │   ├── config.py      # Configuration settings
│   │   └── security.py    # Authentication and security
│   ├── services/
│   │   ├── transcript_processor.py  # GPT-4 processing
│   │   └── cpt_lcd_matcher.py      # Code matching
│   └── data/
│       └── cpt_lcd_dictionary.json  # CPT/LCD mappings
├── main.py                # FastAPI application
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Future Enhancements

- [ ] PDF generation
- [ ] FHIR integration
- [ ] CMS lookup integration
- [ ] User authentication
- [ ] Database integration
- [ ] Audit logging
- [ ] Rate limiting
- [ ] Webhook notifications

## Security Notes

- Always use HTTPS in production
- Rotate API keys regularly
- Implement proper user authentication
- Add rate limiting
- Enable audit logging
- Consider HIPAA compliance requirements

## License

MIT License 