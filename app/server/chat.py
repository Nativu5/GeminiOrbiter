import uuid
from datetime import datetime, timezone
from pathlib import Path

import orjson
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from gemini_webapi.constants import Model
from loguru import logger

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
    try:
        logger.debug(f"Input length: {len(model_input)}, files count: {len(files)}")
        response = await session.send_message(model_input, files=files)
    except Exception as e:
        logger.exception(f"Error generating content from Gemini API: {e}")
        raise

    # Format and clean the output
    model_output = client.extract_output(response)
    stored_output = client.extract_output(response, include_thoughts=False)

    # After cleaning, persist the conversation
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
    if request.stream:
        return _create_streaming_response(model_output, completion_id, timestamp, request.model)
    else:
        return _create_standard_response(
            model_output, completion_id, timestamp, request.model, model_input
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
