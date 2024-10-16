import json
from typing import Any, Dict, List, Optional, Tuple, Union

import boto3

from ...utils.logging import logger
from ..messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, _message_type_lookups
from .base import LLM


def _format_anthropic_messages(messages: List[BaseMessage]):
    # TODO: Handle tool messages, image messages

    system: Optional[str] = None
    formatted_messages: List[Dict] = list()

    for idx, message in enumerate(messages):
        if message.type == "system":
            if idx != 0:
                raise ValueError("System message must be at beginning of message list.")
            if not isinstance(message.content, str):
                raise ValueError(
                    "System message must be a string, " f"instead was: {type(message.content)}"
                )
            system = message.content
            continue

        role = _message_type_lookups[message.type]
        if not isinstance(message.content, str):
            # parse as dict
            assert isinstance(
                message.content, list
            ), "Anthropic message content must be str or list of dicts"

            # populate content
            content = []
            # TODO: Handle tools and AIMessage
            for item in message.content:
                if isinstance(item, str):
                    content.append({"type": "text", "text": item})
                elif isinstance(item, dict):
                    if "type" not in item:
                        raise ValueError("Dict content item must have a type key")
                    elif item["type"] == "text":
                        text = item.get("text", "")
                        if text.strip():
                            content.append({"type": "text", "text": text})
                else:
                    raise ValueError(
                        f"Content items must be str or dict, instead was: {type(item)}"
                    )
        else:
            content = message.content

        formatted_messages.append({"role": role, "content": content})
    return system, formatted_messages


class LLMInputOutputAdapter:
    @classmethod
    def prepare_input(
        cls,
        provider: str,
        model_kwargs: Dict[str, Any],
        system: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        input_body = {**model_kwargs}
        if provider == "anthropic":
            if messages:
                input_body["anthropic_version"] = "bedrock-2023-05-31"
                input_body["messages"] = messages
                if system:
                    input_body["system"] = system

                input_body["max_tokens"] = model_kwargs.get("max_tokens", 1024)

            if "temperature" in model_kwargs:
                input_body["temperature"] = model_kwargs["temperature"]

        return input_body

    @classmethod
    def prepare_output(cls, provider: str, response: Any) -> dict:
        text = ""
        response_body = json.loads(response.get("body").read().decode())

        if provider == "anthropic":
            if "content" in response_body:
                content = response_body.get("content")
                if len(content) == 1 and content[0]["type"] == "text":
                    text = content[0].get("text")

        headers = response.get("ResponseMetadata", {}).get("HTTPHeaders", {})
        prompt_tokens = int(headers.get("x-amzn-bedrock-input-token-count", 0))
        completion_tokens = int(headers.get("x-amzn-bedrock-output-token-count", 0))
        return {
            "text": text,
            "body": response_body,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "stop_reason": response_body.get("stop_reason"),
        }


class PromptAdaptor:
    @classmethod
    def format_messages(
        cls, provider: str, messages: List[BaseMessage]
    ) -> Tuple[Optional[str], List[Dict]]:
        if provider == "anthropic":
            return _format_anthropic_messages(messages)

        raise NotImplementedError(f"Provider {provider} not supported for format messages")


class BedrockLLM(LLM):
    def __init__(
        self,
        model_id: str,
        model_kwargs: Optional[Dict[str, Any]],
        region_name: str,
        client: Any = None,
    ) -> None:
        self.model_id = model_id
        self.model_kwargs = model_kwargs
        self.client = (
            client
            if client is not None
            else boto3.client("bedrock-runtime", region_name=region_name)
        )

    def _standardize_message_format(self, messages: Union[str, List[Tuple[str]]]) -> List[Dict]:
        standardized_messages = list()

        if isinstance(messages, str):
            standardized_messages.append(HumanMessage(content=messages))

        if isinstance(messages, list):
            for type, text in messages:
                if type == "system":
                    standardized_messages.append(SystemMessage(content=text))
                elif type == "human":
                    standardized_messages.append(HumanMessage(content=text))
                elif type == "ai":
                    standardized_messages.append(AIMessage(content=text))
                else:
                    raise ValueError(
                        f"Message type should be 'system', 'human', or 'ai', instead was {type}"
                    )
        return standardized_messages

    def _get_provider(self) -> str:
        return self.model_id.split(".")[0]

    def _prepare_input_and_invoke(
        self,
        provider: str = None,
        system: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        input_body = LLMInputOutputAdapter.prepare_input(
            provider=provider, system=system, messages=messages, model_kwargs=self.model_kwargs
        )
        body = json.dumps(input_body)
        accept = "application/json"
        contentType = "application/json"

        request_options = {
            "body": body,
            "modelId": self.model_id,
            "accept": accept,
            "contentType": contentType,
        }

        try:
            response = self.client.invoke_model(**request_options)
            (text, body, usage_info, stop_reason) = LLMInputOutputAdapter.prepare_output(
                provider, response
            ).values()

        except Exception as e:
            logger.error(f"Error raised by bedrock service: {e}")
            raise e

        llm_output = {"usage": usage_info, "stop_reason": stop_reason}
        return text, body, llm_output

    def invoke(self, messages: List[Tuple[str]]):
        messages = self._standardize_message_format(messages)
        provider = self._get_provider()
        system, formatted_messages = PromptAdaptor.format_messages(
            provider=provider, messages=messages
        )

        text, body, llm_output = self._prepare_input_and_invoke(
            provider=provider, system=system, messages=formatted_messages
        )

        return text, body, llm_output
