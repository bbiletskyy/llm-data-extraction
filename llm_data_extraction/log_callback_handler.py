from typing import Dict, Any, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from loguru import logger


class LogCallbackHandler(BaseCallbackHandler):
    """
    Simple callback handler for logging LLM outputs.
    """

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], *, run_id: UUID,
                            parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None,
                            metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        logger.info(f"Chat model started, tags: {tags}, run_id: {run_id}, messages: {messages}")

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], *, run_id: UUID,
                     parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None,
                     metadata: Optional[Dict[str, Any]] = None, **kwargs: Any,) -> Any:
        logger.info(f"LLM started, tags: {tags}, run_id: {run_id}, prompts: {prompts}")

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any,):
        logger.info(f"LLM ended, tags: {kwargs.get('tags')}, run_id: {run_id}, response: {response}")
