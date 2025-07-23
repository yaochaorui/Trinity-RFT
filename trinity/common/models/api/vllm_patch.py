"""Patch for vllm OpenAI API server.

1. Mocks the `add_signal_handler` method to do nothing.
2. Adds `token_ids` and `prompt_token_ids` to the `ChatCompletionResponse`.
"""
import asyncio
import functools
import json
import time
from typing import Optional, Union

from pydantic import Field, TypeAdapter
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.api_server import (
    build_app,
    create_server_socket,
    init_app_state,
)
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    ErrorResponse,
    FunctionCall,
    FunctionDefinition,
    PromptTokenUsageInfo,
    ToolCall,
    UsageInfo,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_engine import clamp_prompt_logprobs
from vllm.entrypoints.openai.tool_parsers.mistral_tool_parser import MistralToolCall
from vllm.outputs import RequestOutput
from vllm.transformers_utils.tokenizer import MistralTokenizer
from vllm.utils import FlexibleArgumentParser, set_ulimit

from trinity.utils.log import get_logger


class PatchedChatCompletionResponseChoice(ChatCompletionResponseChoice):
    token_ids: list[int] = Field(default_factory=list)


class PatchedChatCompletionResponse(ChatCompletionResponse):
    prompt_token_ids: list[int] = Field(default_factory=list)
    choices: list[PatchedChatCompletionResponseChoice] = list[ChatCompletionResponseChoice]


# TODO: add patch to stream generator
async def chat_completion_full_generator(  # noqa C901
    self,
    request,
    result_generator,
    request_id,
    model_name,
    conversation,
    tokenizer,
    request_metadata,
) -> Union[ErrorResponse, ChatCompletionResponse]:
    created_time = int(time.time())
    final_res: Optional[RequestOutput] = None
    logger = get_logger(__name__)

    try:
        async for res in result_generator:
            final_res = res
    except asyncio.CancelledError:
        return self.create_error_response("Client disconnected")
    except ValueError as e:
        # TODO: Use a vllm-specific Validation Error
        return self.create_error_response(str(e))

    assert final_res is not None

    choices: list[ChatCompletionResponseChoice] = []

    role = self.get_chat_request_role(request)
    for output in final_res.outputs:
        token_ids = output.token_ids
        out_logprobs = output.logprobs

        if request.logprobs and request.top_logprobs is not None:
            assert out_logprobs is not None, "Did not output logprobs"
            logprobs = self._create_chat_logprobs(
                token_ids=token_ids,
                top_logprobs=out_logprobs,
                num_output_top_logprobs=request.top_logprobs,
                tokenizer=tokenizer,
                return_as_token_id=request.return_tokens_as_token_ids,
            )
        else:
            logprobs = None
        auto_tools_called = False

        if self.reasoning_parser:
            try:
                reasoning_parser = self.reasoning_parser(tokenizer)
            except RuntimeError as e:
                logger.exception("Error in reasoning parser creation.")
                return self.create_error_response(str(e))
            # If the reasoning parser is enabled,
            # tool calls are extracted exclusively from the content.
            reasoning_content, content = reasoning_parser.extract_reasoning_content(
                output.text, request=request
            )
        else:
            reasoning_content = None
            content = output.text

        # if auto tools are not enabled, and a named tool choice using
        #   outlines is not being used
        if (not self.enable_auto_tools or not self.tool_parser) and (
            not isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam)
            and request.tool_choice != "required"
        ):
            message = ChatMessage(role=role, reasoning_content=reasoning_content, content=content)

        # if the request uses tools and specified a tool choice
        elif (
            request.tool_choice and type(request.tool_choice) is ChatCompletionNamedToolChoiceParam
        ):
            tool_call_class = (
                MistralToolCall if isinstance(tokenizer, MistralTokenizer) else ToolCall
            )
            message = ChatMessage(
                role=role,
                reasoning_content=reasoning_content,
                content="",
                tool_calls=[
                    tool_call_class(
                        function=FunctionCall(
                            name=request.tool_choice.function.name, arguments=content
                        )
                    )
                ],
            )

        elif request.tool_choice and request.tool_choice == "required":
            tool_call_class = (
                MistralToolCall if isinstance(tokenizer, MistralTokenizer) else ToolCall
            )

            # the fields of FunctionDefinition are a superset of the
            # tool call outputs and can be used for parsing
            assert content is not None
            tool_calls = TypeAdapter(list[FunctionDefinition]).validate_json(content)
            message = ChatMessage(
                role=role,
                content="",
                tool_calls=[
                    tool_call_class(
                        function=FunctionCall(
                            name=tool_call.name,
                            arguments=json.dumps(tool_call.parameters, ensure_ascii=False),
                        )
                    )
                    for tool_call in tool_calls
                ],
            )

        # if the request doesn't use tool choice
        # OR specifies to not use a tool
        elif not request.tool_choice or request.tool_choice == "none":
            message = ChatMessage(role=role, reasoning_content=reasoning_content, content=content)

        # handle when there are tools and tool choice is auto
        elif (
            request.tools
            and (request.tool_choice == "auto" or request.tool_choice is None)
            and self.enable_auto_tools
            and self.tool_parser
        ):
            try:
                tool_parser = self.tool_parser(tokenizer)
            except RuntimeError as e:
                logger.exception("Error in tool parser creation.")
                return self.create_error_response(str(e))

            tool_call_info = tool_parser.extract_tool_calls(
                content if content is not None else "", request=request
            )
            # In the OpenAI API the finish_reason is "tools_called"
            # if the tool choice is auto and the model produced a tool
            # call. The same is not true for named function calls
            auto_tools_called = tool_call_info.tools_called
            if tool_call_info.tools_called:
                message = ChatMessage(
                    role=role,
                    reasoning_content=reasoning_content,
                    content=tool_call_info.content,
                    tool_calls=tool_call_info.tool_calls,
                )

            else:
                # FOR NOW make it a chat message; we will have to detect
                # the type to make it later.
                message = ChatMessage(
                    role=role, reasoning_content=reasoning_content, content=content
                )

        # undetermined case that is still important to handle
        else:
            logger.error(
                "Error in chat_completion_full_generator - cannot determine"
                " if tools should be extracted. Returning a standard chat "
                "completion."
            )
            message = ChatMessage(role=role, reasoning_content=reasoning_content, content=content)

        choice_data = PatchedChatCompletionResponseChoice(
            index=output.index,
            message=message,
            logprobs=logprobs,
            finish_reason="tool_calls"
            if auto_tools_called
            else output.finish_reason
            if output.finish_reason
            else "stop",
            stop_reason=output.stop_reason,
            token_ids=output.token_ids,
        )
        choices.append(choice_data)

    if request.echo:
        last_msg_content: Union[str, list[dict[str, str]]] = ""
        if conversation and "content" in conversation[-1] and conversation[-1].get("role") == role:
            last_msg_content = conversation[-1]["content"] or ""
        if isinstance(last_msg_content, list):
            last_msg_content = "\n".join(msg["text"] for msg in last_msg_content)

        for choice in choices:
            full_message = last_msg_content + (choice.message.content or "")
            choice.message.content = full_message

    assert final_res.prompt_token_ids is not None
    num_prompt_tokens = len(final_res.prompt_token_ids)
    if final_res.encoder_prompt_token_ids is not None:
        num_prompt_tokens += len(final_res.encoder_prompt_token_ids)
    num_generated_tokens = sum(len(output.token_ids) for output in final_res.outputs)
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )
    if self.enable_prompt_tokens_details and final_res.num_cached_tokens:
        usage.prompt_tokens_details = PromptTokenUsageInfo(
            cached_tokens=final_res.num_cached_tokens
        )

    request_metadata.final_usage_info = usage

    response = PatchedChatCompletionResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=choices,
        usage=usage,
        prompt_logprobs=clamp_prompt_logprobs(final_res.prompt_logprobs),
        kv_transfer_params=final_res.kv_transfer_params,
        prompt_token_ids=final_res.prompt_token_ids,
    )

    return response


async def run_server_in_ray(args, engine_client):
    # workaround to make sure that we bind the port before the engine is set up.
    # This avoids race conditions with ray.
    # see https://github.com/vllm-project/vllm/issues/8204
    sock_addr = (args.host, args.port)
    sock = create_server_socket(sock_addr)

    # workaround to avoid footguns where uvicorn drops requests with too
    # many concurrent requests active
    set_ulimit()
    app = build_app(args)

    vllm_config = await engine_client.get_vllm_config()
    await init_app_state(engine_client, vllm_config, app.state, args)

    await patch_and_serve_http(app, sock, args)

    # # NB: Await server shutdown only after the backend context is exited
    # try:
    #     await shutdown_task
    # finally:
    #     sock.close()


def dummy_add_signal_handler(self, *args, **kwargs):
    # DO NOTHING HERE
    pass


async def patch_and_serve_http(app, sock, args):
    """Patch the add_signal_handler method and serve the app."""
    loop = asyncio.get_event_loop()
    original_add_signal_handler = loop.add_signal_handler
    loop.add_signal_handler = functools.partial(dummy_add_signal_handler, loop)
    OpenAIServingChat.chat_completion_full_generator = chat_completion_full_generator

    try:
        shutdown_task = await serve_http(
            app,
            sock=sock,
            enable_ssl_refresh=args.enable_ssl_refresh,
            host=args.host,
            port=args.port,
            log_level="info",
            access_log=True,
            timeout_keep_alive=10,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
        )
        await shutdown_task
    finally:
        loop.add_signal_handler = original_add_signal_handler
        sock.close()


async def run_api_server_in_ray_actor(async_llm, host: str, port: int, model_path: str):
    parser = FlexibleArgumentParser(description="Run the OpenAI API server.")
    args = make_arg_parser(parser)
    args = parser.parse_args(["--host", str(host), "--port", str(port), "--model", model_path])
    print(args)
    await run_server_in_ray(args, async_llm)
