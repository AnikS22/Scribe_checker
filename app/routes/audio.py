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
from app.models.transcript_response import TranscriptResponse, LCDValidationResult, CPTCode, PatientInfo, PainRating
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
        # Use openai_client instead of openai if both were imported
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
        extracted_data = await transcript_processor.process(transcript)

        # Agents 2 & 6: Json_to_icd and CPTcodes (Run in parallel)
        logger.info("Calling Agents 2 & 6 in parallel")
        # Replace hardcoded agent_ids with references from agents_config.py
        icd_codes_task = asyncio.create_task(
            call_assistant_agent(agent_id=AGENT_IDS["json_to_icd"], input_data=extracted_data)
        )
        cpt_codes_task = asyncio.create_task(
            call_assistant_agent(agent_id=AGENT_IDS["cpt_codes"], input_data=extracted_data)
        )

        # Wait for parallel tasks to complete and handle potential errors
        try:
            icd_codes_raw_response, cpt_codes_raw_response = await asyncio.gather(
                icd_codes_task,
                cpt_codes_task
            )
        except Exception as e:
            logger.error(f"Error during parallel agent calls (ICD/CPT): {e}")
            raise HTTPException(status_code=500, detail=f"Error processing data with ICD/CPT agents: {e}")

        # Parse raw JSON string responses from agents
        try:
            # Assuming ICD agent returns a JSON list of strings, e.g., '["M54.5", "G89.1"]'.
            # Need to handle potential empty or invalid responses.
            icd_codes = json.loads(icd_codes_raw_response) if icd_codes_raw_response else []
            if not isinstance(icd_codes, list):
                 raise ValueError("ICD agent did not return a list")

            # Assuming CPT agent returns a JSON list of CPT code dictionaries.
            cpt_codes_response = json.loads(cpt_codes_raw_response) if cpt_codes_raw_response else []
            if not isinstance(cpt_codes_response, list):
                 raise ValueError("CPT agent did not return a list")

        except (json.JSONDecodeError, ValueError) as e:
             logger.error(f"Error parsing agent responses (ICD/CPT): {e}")
             raise HTTPException(status_code=500, detail=f"Error parsing data from ICD/CPT agents: {e}")

        # Agent 5: LCD_Validator_Agentv1 (Depends on CPT codes response)
        logger.info("Calling Agent5: LCD_Validator_Agentv1")
        # Replace hardcoded agent_id with reference from agents_config.py
        try:
            lcd_validation_raw_response = await call_assistant_agent(
                agent_id=AGENT_IDS["lcd_validator"],
                input_data=cpt_codes_response # Pass the parsed CPT response
            )
        except Exception as e:
            logger.error(f"Error during LCD validator agent call: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing data with LCD validator agent: {e}")

        # Parse raw JSON string response from LCD agent
        try:
            # Assuming LCD agent returns a JSON list of LCD validation result dictionaries.
            lcd_validation_results = json.loads(lcd_validation_raw_response) if lcd_validation_raw_response else []
            if not isinstance(lcd_validation_results, list):
                 raise ValueError("LCD agent did not return a list")
        except (json.JSONDecodeError, ValueError) as e:
             logger.error(f"Error parsing LCD agent response: {e}")
             raise HTTPException(status_code=500, detail=f"Error parsing data from LCD validator agent: {e}")

        # --- Merge Results ---

        logger.info("Merging agent results")
        # Start with the data from the Clinical Extractor
        response_data = extracted_data.copy()

        # Add/Overwrite fields from other agents
        # Ensure the merged data matches the TranscriptResponse Pydantic model
        response_data["icd_codes"] = icd_codes
        # Ensure recommended_cpt_codes matches the structure expected by the Pydantic model
        response_data["recommended_cpt_codes"] = cpt_codes_response
        response_data["lcd_validation"] = lcd_validation_results
        response_data["prompt"] = transcript # Ensure original prompt is included

        # Return the merged results using the Pydantic model for validation/serialization
        return TranscriptResponse(**response_data)

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions directly
        raise http_exc
    except Exception as e:
        # Log and raise other exceptions
        logger.error(f"Error processing audio: {str(e)}")
        # Use a generic 500 error for unhandled exceptions
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}") 