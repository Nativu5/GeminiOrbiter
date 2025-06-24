import uuid
from datetime import datetime, timezone
from pathlib import Path

import orjson
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from gemini_webapi.constants import Model
from loguru import logger

# Constants for message splitting
MAX_MSG_CHAR_LENGTH = 990000  # Max characters per message to Gemini (slightly less than 1M for safety)
SPLIT_THRESHOLD_CHAR_LENGTH = 500000  # Split message if longer than this
CONTINUATION_PROMPT = "\n\n(System note: The previous message was part of a longer text. Please reply with only 'OK' to acknowledge and receive the next part. Do not add any other commentary or analysis yet.)"
ACKNOWLEDGEMENT_PHRASES = ["ok", "ok.", "okay", "okay.", "got it", "got it.", "understood", "understood."]


from ..models import (
    ChatCompletionRequest,
    ConversationInStore,
    Message,
    ModelData,
    ModelListResponse,
)
from ..services import LMDBConversationStore, SingletonGeminiClient
from ..utils.helper import estimate_tokens
from .middleware import get_temp_dir, verify_api_key

router = APIRouter()


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models(api_key: str = Depends(verify_api_key)):
    now = int(datetime.now(tz=timezone.utc).timestamp())

    models = []
    for model in Model:
        m_name = model.model_name
        if not m_name or m_name == "unspecified":
            continue

        models.append(
            ModelData(
                id=m_name,
                created=now,
                owned_by="gemini-web",
            )
        )

    return ModelListResponse(data=models)


@router.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key),
    tmp_dir: Path = Depends(get_temp_dir),
):
    client = SingletonGeminiClient()
    db = LMDBConversationStore()
    model = Model.from_name(request.model)

    if len(request.messages) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one message is required in the conversation.",
        )

    # Check if conversation is reusable
    session = None
    if _check_reusable(request.messages):
        try:
            # Exclude the last message from user
            if old_conv := db.find(model.model_name, request.messages[:-1]):
                session = client.start_chat(metadata=old_conv.metadata, model=model)
        except Exception as e:
            session = None
            logger.warning(f"Error checking LMDB for reusable session: {e}")

    if session:
        # Just send the last message to the existing session
        model_input, files = await client.process_message(
            request.messages[-1], tmp_dir, tagged=False
        )
        logger.debug(f"Found reusable session: {session.metadata}")
    else:
        # Start a new session and concat messages into a single string
        session = client.start_chat(model=model)
        try:
            model_input, files = await client.process_conversation(request.messages, tmp_dir)
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        except Exception as e:
            logger.exception(f"Error in preparing conversation: {e}")
            raise
        logger.debug("New session started.")

    # Generate response
    final_model_output_parts = []
    original_model_input_for_storage = model_input # Keep the original for storage

    try:
        logger.debug(f"Input length: {len(model_input)}, files count: {len(files)}")
        if len(model_input) > SPLIT_THRESHOLD_CHAR_LENGTH:
            logger.info(f"Message length {len(model_input)} exceeds SPLIT_THRESHOLD_CHAR_LENGTH {SPLIT_THRESHOLD_CHAR_LENGTH}. Splitting message.")
            input_chunks = _split_text(model_input, SPLIT_THRESHOLD_CHAR_LENGTH, MAX_MSG_CHAR_LENGTH - len(CONTINUATION_PROMPT) - 100) # -100 for safety margin

            current_files = files # Send files only with the first chunk
            for i, chunk in enumerate(input_chunks):
                is_last_chunk = (i == len(input_chunks) - 1)
                message_to_send = chunk

                if not is_last_chunk:
                    message_to_send += CONTINUATION_PROMPT

                logger.debug(f"Sending chunk {i+1}/{len(input_chunks)} of length {len(message_to_send)}")
                response_chunk = await session.send_message(message_to_send, files=current_files)
                current_files = [] # Clear files after first chunk

                chunk_output = client.extract_output(response_chunk, include_thoughts=False) # Don't include thoughts for intermediate chunks
                logger.debug(f"Received response for chunk {i+1}: '{chunk_output[:100]}...'")

                if not is_last_chunk:
                    if not any(ack.lower() == chunk_output.strip().lower() for ack in ACKNOWLEDGEMENT_PHRASES):
                        logger.warning(f"Did not receive expected acknowledgement for chunk {i+1}. Received: '{chunk_output}'. Proceeding, but this might cause issues.")
                        # Potentially, we could raise an error here or retry. For now, we log and proceed.
                        # If the AI gives a substantive response instead of 'OK', we might lose it here.
                        # However, the prompt explicitly asks for 'OK'.
                else:
                    # This is the last chunk, so its response is the final one (or part of it)
                    final_model_output_parts.append(client.extract_output(response_chunk, include_thoughts=True)) # Include thoughts for the final response

            model_output = "\n".join(final_model_output_parts)
            # Stored output should be based on the final response only
            stored_output = model_output
        else:
            response = await session.send_message(model_input, files=files)
            model_output = client.extract_output(response)
            stored_output = client.extract_output(response, include_thoughts=False)

    except Exception as e:
        logger.exception(f"Error generating content from Gemini API: {e}")
        raise

    # After cleaning, persist the conversation using the original full input
    try:
        last_message = Message(role="assistant", content=stored_output)
        conv = ConversationInStore(
            model=model.model_name,
            metadata=session.metadata,
            messages=[*request.messages, last_message],
        )
        key = db.store(conv)
        logger.debug(f"Conversation saved to LMDB with key: {key}")
    except Exception as e:
        # We can still return the response even if saving fails
        logger.warning(f"Failed to save conversation to LMDB: {e}")

    # Return with streaming or standard response
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    timestamp = int(datetime.now(tz=timezone.utc).timestamp())

    # Use original_model_input_for_storage for token calculation if splitting happened
    input_for_token_calc = original_model_input_for_storage if len(original_model_input_for_storage) > SPLIT_THRESHOLD_CHAR_LENGTH and len(model_input) != len(original_model_input_for_storage) else model_input

    if request.stream:
        # Streaming response currently doesn't calculate input tokens, so it's less critical here.
        # However, if it did, we'd need to consider how to handle it.
        # For now, model_output is the full output.
        return _create_streaming_response(model_output, completion_id, timestamp, request.model)
    else:
        return _create_standard_response(
            model_output, completion_id, timestamp, request.model, input_for_token_calc
        )


def _check_reusable(messages: list[Message]) -> bool:
    """
    Check if the conversation is reusable based on the message history.
    """
    if not messages or len(messages) < 2:
        return False

    # Last message must from the user
    if messages[-1].role != "user" or not messages[-1].content:
        return False

    # The second last message must be from the assistant or system
    if messages[-2].role not in ["assistant", "system"]:
        return False

    return True


def _create_streaming_response(
    model_output: str, completion_id: str, created_time: int, model: str
) -> StreamingResponse:
    """Create streaming response"""

    async def generate_stream():
        # Send start event
        data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {orjson.dumps(data).decode('utf-8')}\n\n"

        # Stream output text in chunks for efficiency
        chunk_size = 32
        for i in range(0, len(model_output), chunk_size):
            chunk = model_output[i : i + chunk_size]
            data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
            }
            yield f"data: {orjson.dumps(data).decode('utf-8')}\n\n"

        # Send end event
        data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {orjson.dumps(data).decode('utf-8')}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


def _create_standard_response(
    model_output: str, completion_id: str, created_time: int, model: str, model_input: str
) -> dict:
    """Create standard response"""
    # Calculate token usage
    prompt_tokens = estimate_tokens(model_input)
    completion_tokens = estimate_tokens(model_output)
    total_tokens = prompt_tokens + completion_tokens

    result = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_time,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": model_output},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }

    logger.debug(f"Response created with {total_tokens} total tokens")
    return result


def _split_text(text: str, preferred_chunk_size: int, max_chunk_size: int) -> list[str]:
    """
    Splits a long text into chunks, preferring splits at newlines.
    Ensures no chunk exceeds max_chunk_size.
    """
    chunks = []
    current_pos = 0
    text_len = len(text)

    while current_pos < text_len:
        end_pos = min(current_pos + preferred_chunk_size, text_len)

        # If the remaining text is smaller than preferred_chunk_size, take it all
        if end_pos >= text_len:
            actual_end_pos = text_len
        else:
            # Try to find a newline to split at, searching backwards from preferred end_pos
            # then forwards if no suitable newline is found backwards.
            newline_pos = text.rfind('\n', current_pos, end_pos)

            if newline_pos != -1 and (end_pos - newline_pos) < (preferred_chunk_size / 2): # Prefer splitting if newline is in the latter half
                actual_end_pos = newline_pos + 1
            else:
                # If no good newline found by rfind, try searching forward up to max_chunk_size
                # This helps avoid very small chunks if a newline is just before preferred_chunk_size
                search_forward_limit = min(current_pos + max_chunk_size, text_len)
                newline_pos_forward = text.find('\n', end_pos, search_forward_limit)
                if newline_pos_forward != -1:
                    actual_end_pos = newline_pos_forward + 1
                else:
                    # If no newline found at all, split at max_chunk_size or end of text
                    actual_end_pos = min(current_pos + max_chunk_size, text_len)

        # Ensure the chunk does not exceed max_chunk_size
        if actual_end_pos > current_pos + max_chunk_size:
             actual_end_pos = current_pos + max_chunk_size

        # If a split results in an empty chunk (e.g. multiple newlines), advance current_pos
        if actual_end_pos == current_pos:
            if current_pos < text_len and text[current_pos] == '\n':
                 chunks.append('\n') # Keep the newline if it was the split point
            current_pos +=1
            continue

        chunks.append(text[current_pos:actual_end_pos])
        current_pos = actual_end_pos

        # Safety break if something goes wrong, though theoretically current_pos should always advance.
        if current_pos == text_len and not chunks[-1]: # Avoid infinite loop if last chunk is empty and text_len not reached
            break
        if len(chunks) > 1000: # Safety break for extremely long texts / bad splitting
             logger.error("Splitting resulted in too many chunks, aborting split.")
             return [text] # Fallback to sending the original text if splitting goes wrong

    # Filter out potential empty strings that might result from splitting, unless it's a single newline chunk
    return [chunk for chunk in chunks if chunk or chunk =='\n']
