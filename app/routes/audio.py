from fastapi import APIRouter, File, UploadFile, HTTPException, Security
from fastapi.security import APIKeyHeader
from openai import AsyncOpenAI
import logging
from typing import Dict, Any, List
import json
import asyncio
import time # Import time for polling

from app.core.config import settings
from app.core.security import verify_api_key
from app.services.transcript_processor import TranscriptProcessor
# Import both response models
from app.models.transcript_response import TranscriptResponse, LCDValidationResult, CPTCode, PatientInfo, PainRating, QPPMeasure
from app.models.transcript_processor_response import TranscriptProcessorResponse # Import the new model
from app.core.agents_config import AGENT_IDS # Import AGENT_IDS

# Configure logging
logger = logging.getLogger(__name__)

# API Key security
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

router = APIRouter()
# Initialize OpenAI client with your API key
openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

# --- Helper Function to Call Assistant API ---
async def call_assistant_agent(agent_id: str, input_data: Dict[str, Any]) -> str:
    """
    Calls an OpenAI Assistant with the given input data and returns the response.
    Input data is sent as a JSON string.
    """
    logger.info(f"Calling Assistant: {agent_id}")
    thread = await openai_client.beta.threads.create()
    
    # Add message to the thread (send input_data as a JSON string)
    await openai_client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=json.dumps(input_data)
    )
    
    # Run the assistant
    run = await openai_client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=agent_id,
    )
    
    # Poll for run completion
    while run.status not in ["completed", "failed", "cancelled", "expired"]:
        await asyncio.sleep(1)
        run = await openai_client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
        logger.info(f"Assistant {agent_id} run status: {run.status}")
    
    if run.status != "completed":
        raise Exception(f"Assistant run failed with status: {run.status}")
    
    # Retrieve messages after completion
    messages = await openai_client.beta.threads.messages.list(
        thread_id=thread.id,
        order="asc"
    )
    
    # Find the last message from the assistant
    assistant_messages = [m for m in messages.data if m.role == "assistant"]
    if not assistant_messages:
        raise Exception(f"No response message from Assistant {agent_id}")
        
    # Assuming the assistant's response is in the last message's content
    # and is a single text block.
    last_assistant_message = assistant_messages[-1]
    if not last_assistant_message.content:
         raise Exception(f"Assistant {agent_id} returned empty content")

    # Extract text from the message content block(s)
    response_text = ""
    for content_block in last_assistant_message.content:
        if content_block.type == 'text':
            response_text += content_block.text.value + "\n"
    
    if not response_text.strip():
         raise Exception(f"Assistant {agent_id} returned empty text content")

    logger.info(f"Received response from Assistant: {agent_id}")
    return response_text.strip() # Return the text response

# --- Router Endpoint ---

@router.post("/transcribe_audio", response_model=TranscriptResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    api_key: str = Security(API_KEY_HEADER)
) -> TranscriptResponse:
    """
    Transcribe audio file to text and process it into structured clinical data
    by orchestrating multiple AI agents.

    Args:
        file: Audio file to transcribe
        api_key: API key for authentication

    Returns:
        TranscriptResponse containing structured clinical data, including
        ICD codes, CPT codes, and LCD validation results.
    """
    try:
        # Verify API key
        if not verify_api_key(api_key):
            raise HTTPException(status_code=401, detail="Invalid API key")

        # Read audio file
        audio_bytes = await file.read()

        # Transcribe audio using Whisper
        logger.info(f"Transcribing audio file: {file.filename}")
        whisper_response = await openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=(file.filename, audio_bytes),
            response_format="text"
        )
        transcript = whisper_response.strip()

        # Log transcript in debug mode
        if settings.DEBUG_MODE:
            logger.debug("Transcribed text: %s", transcript)

        # --- Agent Orchestration ---

        # Agent1: Clinical_Extractor (TranscriptProcessor)
        logger.info("Calling Agent1: Clinical_Extractor (TranscriptProcessor)")
        transcript_processor = TranscriptProcessor()
        # Get raw data from TranscriptProcessor
        extracted_data_raw = await transcript_processor.process(transcript)
        
        # Validate and structure the initial extracted data using TranscriptProcessorResponse
        # This handles potential missing fields from the first agent gracefully
        extracted_data = TranscriptProcessorResponse(**extracted_data_raw)

        # Agent 2: Json_to_icd
        logger.info("Calling Agent2: Json_to_icd")
        try:
            icd_codes_raw_response = await call_assistant_agent(
                agent_id=AGENT_IDS["json_to_icd"],
                input_data=extracted_data.dict() # Pass dictionary representation
            )
            icd_parsed_data = json.loads(icd_codes_raw_response) if icd_codes_raw_response else {}
        except Exception as e:
             logger.error(f"Error during ICD agent call: {e}")
             icd_parsed_data = {}
             raise HTTPException(status_code=500, detail=f"Error processing data with ICD agent: {e}")

        # Agent 6: CPTcodes (Input includes extracted_data fields + ICD codes)
        logger.info("Calling Agent6: CPTcodes")
        # Prepare input for CPT agent including relevant fields from extracted_data and icd_codes
        cpt_agent_input = {
            "chief_complaint": extracted_data.chief_complaint,
            "assessment": extracted_data.assessment,
            "plan": extracted_data.plan,
            "exam_findings": extracted_data.exam_findings,
            "imaging_summary": extracted_data.imaging_summary,
            "history_of_present_illness": extracted_data.history_of_present_illness,
            "prior_treatments": extracted_data.prior_treatments,
            "icd_codes": icd_parsed_data.get("icd_codes", []) # Include ICD codes from Agent 2
        }
        # Ensure None values are not included in the input JSON for the agent unless strictly necessary
        cpt_agent_input_cleaned = {k: v for k, v in cpt_agent_input.items() if v is not None}
        try:
            cpt_codes_raw_response = await call_assistant_agent(
                agent_id=AGENT_IDS["cpt_codes"],
                input_data=cpt_agent_input_cleaned
            )
            cpt_parsed_data = json.loads(cpt_codes_raw_response) if cpt_codes_raw_response else {}
        except Exception as e:
            logger.error(f"Error during CPT agent call: {e}")
            cpt_parsed_data = {}
            raise HTTPException(status_code=500, detail=f"Error processing data with CPT agent: {e}")

        # Agent 5: LCD_Validator_Agentv1 (Depends on CPT codes response)
        logger.info("Calling Agent5: LCD_Validator_Agentv1")
        # Pass the parsed CPT response data (specifically the list of CPT codes) to the LCD validator agent
        lcd_agent_input = {
            "recommended_cpt_codes": cpt_parsed_data.get("recommended_cpt_codes", [])
        }
        try:
            lcd_validation_raw_response = await call_assistant_agent(
                agent_id=AGENT_IDS["lcd_validator"],
                input_data=lcd_agent_input
            )
            lcd_parsed_data = json.loads(lcd_validation_raw_response) if lcd_validation_raw_response else {}
        except Exception as e:
             logger.error(f"Error during LCD validator agent call: {e}")
             lcd_parsed_data = {}
             raise HTTPException(status_code=500, detail=f"Error processing data with LCD validator agent: {e}")

        # --- Merge Results ---

        logger.info("Merging agent results")
        # Assemble the final TranscriptResponse from validated extracted data and agent results
        # Use .dict() to get a dictionary representation of the validated first-stage data
        final_response = TranscriptResponse(
            **extracted_data.dict(), # Include all fields from TranscriptProcessorResponse
            icd_codes=icd_parsed_data.get("icd_codes", []), # Get from parsed ICD agent data, default to []
            recommended_cpt_codes=cpt_parsed_data.get("recommended_cpt_codes", []), # Get from parsed CPT agent data, default to []
            lcd_validation=lcd_parsed_data.get("lcd_validation", []) # Get from parsed LCD agent data, default to []
            # Note: evidence_suggestions is also in TranscriptResponse and would need to be populated if you add that agent
        )

        # Return the merged results using the Pydantic model for final validation/serialization
        return final_response

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions directly
        raise http_exc
    except Exception as e:
        # Log and raise other exceptions
        logger.error(f"Error processing audio: {str(e)}")
        # Use a generic 500 error for unhandled exceptions
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}") 