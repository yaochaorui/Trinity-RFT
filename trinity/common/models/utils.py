# -*- coding: utf-8 -*-
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.distributed._tensor import DTensor, Placement, Shard

from trinity.utils.log import get_logger


def tokenize_and_mask_messages_hf(
    tokenizer: Any,
    messages: List[dict],
    tools: Optional[List[dict]] = None,
    chat_template: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Calculate the assistant token mask with `chat_template`.

    Args:
        tokenizer (Any): The tokenizer.
        messages (List[dict]): Messages with `role` and `content` fields.
        tools (Optional[List[dict]]): The list of tool dictionaries.
        chat_template (str): The chat template with `{% generation %}` symbol.

    Returns:
        `torch.Tensor`: The token_ids (sequence_length)
        `torch.Tensor`: Assistant_masks (sequence_length).
        `int`: Prompt length.
    """
    token_dict = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        chat_template=chat_template,
        add_generation_prompt=False,
        padding=False,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False,
        return_assistant_tokens_mask=True,
        return_dict=True,
    )
    # find the first assistant token, the tokens before are prompt tokens
    prompt_length = torch.argmax(token_dict["assistant_masks"][0]).item()
    return token_dict["input_ids"][0], token_dict["assistant_masks"][0], prompt_length


def tokenize_and_mask_messages_default(
    tokenizer: Any,
    messages: List[dict],
    tools: Optional[List[dict]] = None,
    chat_template: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Calculate the assistant token mask.

    Args:
        tokenizer (Any): The tokenizer.
        messages (List[dict]): Messages with `role` and `content` fields.
        tools (Optional[List[dict]]): The list of tool dictionaries.
        chat_template (str): The chat template with `{% generation %}` symbol.

    Returns:
        `torch.Tensor`: The token_ids (sequence_length)
        `torch.Tensor`: Assistant_masks (sequence_length).
        `int`: Prompt length.

    Note:
        This method is based on the assumption that as the number of chat rounds increases,
        the tokens of the previous round are exactly the prefix tokens of the next round.
        If the assumption is not met, the function may produce incorrect results.
        Please check the chat template before using this method.
    """

    tokens = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        chat_template=chat_template,
        add_generation_prompt=False,
        padding=False,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    assistant_token_mask = torch.zeros(tokens.shape[1], dtype=torch.int)
    for idx, message in enumerate(messages):
        if message["role"] == "assistant":
            prompt_token_ids = tokenizer.apply_chat_template(
                messages[:idx],
                tools=tools,
                chat_template=chat_template,
                add_generation_prompt=True,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_length = prompt_token_ids.shape[1]
            prompt_response_token_ids = tokenizer.apply_chat_template(
                messages[: idx + 1],
                tools=tools,
                chat_template=chat_template,
                add_generation_prompt=False,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_response_length = prompt_response_token_ids.shape[1]
            assistant_token_mask[prompt_length:prompt_response_length] = 1
    prompt_length = torch.argmax(assistant_token_mask).item()
    return tokens[0], assistant_token_mask, prompt_length


def get_action_mask_method(chat_template: Optional[str] = None) -> Callable:
    """Get the action mask method according to the chat template.

    Args:
        chat_template (str): The chat template. If { % generation % } is present, use HF tokenizer's `return_assistant_tokens_mask`.

    Returns:
        The action mask method.
    """
    if chat_template is None:
        return tokenize_and_mask_messages_default
    # check if the chat template contains `{% generation %}` symbol
    elif re.search(r"\{\%-?\s*generation\s*-?\%\}", chat_template):
        return tokenize_and_mask_messages_hf
    else:
        return tokenize_and_mask_messages_default


def get_checkpoint_dir_with_step_num(
    checkpoint_root_path: str,
    trainer_type: str = "verl",
    step_num: Optional[int] = None,
) -> Tuple[str, int]:
    """Get the checkpoint directory from a root checkpoint directory.

    Args:
        checkpoint_root_path (str): The root checkpoint directory.
        trainer_type (str): The trainer type. Only support "verl" for now.
        step_num (Optional[int], optional): The step number. If specified,
            load the checkpoint with the specified step number. If None,
            load the latest checkpoint. Defaults to None.

    Returns:
        Tuple[str, int]: The checkpoint directory and the step number of the checkpoint.
    """
    if trainer_type == "verl":
        return get_verl_checkpoint_info(checkpoint_path=checkpoint_root_path, step_num=step_num)
    else:
        raise NotImplementedError(f"Unsupported trainer type {trainer_type}")


def load_state_dict(checkpoint_dir: str, trainer_type: str = "verl") -> dict:
    """Load state dict from a checkpoint dir.

    Args:
        checkpoint_dir (str): The checkpoint directory.
        trainer_type (str): The trainer type. Only support "verl" for now.
    """
    if trainer_type == "verl":
        return load_state_dict_from_verl_checkpoint(checkpoint_dir)
    else:
        raise NotImplementedError(f"Unsupported trainer type {trainer_type}")


# copy from verl/scripts/model_merger.py
def merge_by_placement(tensors: List[torch.Tensor], placement: Placement):
    if placement.is_replicate():
        return tensors[0]
    elif placement.is_partial():
        raise NotImplementedError("Partial placement is not supported yet")
    elif placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    else:
        raise ValueError(f"Unsupported placement: {placement}")


def get_verl_checkpoint_info(
    checkpoint_path: str, step_num: Optional[int] = None
) -> Tuple[str, int]:
    """Get the checkpoint directory from a Verl root checkpoint directory.

    Args:
        checkpoint_path (str): The root checkpoint directory.
        step_num (Optional[int], optional): The step number. If specified,
            load the checkpoint with the specified step number. If None,
            load the latest checkpoint. Defaults to None.

    Returns:
        Tuple[str, int]: The checkpoint directory and the step number of the checkpoint.
    """
    if step_num is None:
        # load latest checkpoint
        iteration_file = os.path.join(checkpoint_path, "latest_checkpointed_iteration.txt")
        if os.path.exists(iteration_file):
            with open(
                iteration_file, "r", encoding="utf-8"
            ) as f:  # TODO: this file may be modified simultaneously
                iteration = f.read().strip()
                return os.path.join(checkpoint_path, f"global_step_{iteration}"), int(iteration)
        else:
            raise FileNotFoundError(f"No iteration file found in {checkpoint_path}")
    else:
        # load specific iteration checkpoint
        return os.path.join(checkpoint_path, f"global_step_{step_num}"), step_num


# copy from verl/scripts/model_merger.py
def load_state_dict_from_verl_checkpoint(checkpoint_path: str) -> dict:  # noqa: C901
    """Load state dict from a Verl checkpoint."""
    logger = get_logger(__name__)
    logger.info(f"Loading state dict from {checkpoint_path}")
    assert not checkpoint_path.endswith(
        "huggingface"
    ), "The local_dir should not end with huggingface"

    # copy rank zero to find the shape of (dp, fsdp)
    rank = 0
    world_size = 0
    for filename in os.listdir(checkpoint_path):
        match = re.match(r"model_world_size_(\d+)_rank_0\.pt", filename)
        if match:
            world_size = match.group(1)
            break
    assert world_size, "No model file with the proper format"

    state_dict = torch.load(
        os.path.join(checkpoint_path, f"model_world_size_{world_size}_rank_{rank}.pt"),
        map_location="cpu",
    )
    pivot_key = sorted(list(state_dict.keys()))[0]
    weight = state_dict[pivot_key]
    assert isinstance(weight, torch.distributed._tensor.DTensor)
    # get sharding info
    device_mesh = weight.device_mesh
    mesh = device_mesh.mesh
    mesh_dim_names = device_mesh.mesh_dim_names

    logger.info(f"Got device mesh {mesh}, mesh_dim_names {mesh_dim_names}")

    assert mesh_dim_names in (("fsdp",),), f"Unsupported mesh_dim_names {mesh_dim_names}"

    if "tp" in mesh_dim_names:
        # fsdp * tp
        total_shards = mesh.shape[-1] * mesh.shape[-2]
        mesh_shape = (mesh.shape[-2], mesh.shape[-1])
    else:
        # fsdp
        total_shards = mesh.shape[-1]
        mesh_shape = (mesh.shape[-1],)

    logger.info(f"Processing model shards with {total_shards} {mesh_shape} in total")

    model_state_dict_lst = []
    model_state_dict_lst.append(state_dict)
    model_state_dict_lst.extend([""] * (total_shards - 1))

    def process_one_shard(rank):
        model_path = os.path.join(checkpoint_path, f"model_world_size_{world_size}_rank_{rank}.pt")
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        model_state_dict_lst[rank] = state_dict  # noqa: F821
        return state_dict

    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count())) as executor:  # type: ignore
        for rank in range(1, total_shards):
            executor.submit(process_one_shard, rank)
    state_dict = {}
    param_placements: Dict[str, List[Placement]] = {}
    keys = set(model_state_dict_lst[0].keys())
    for key in keys:
        state_dict[key] = []
        for model_state_dict in model_state_dict_lst:
            try:
                tensor = model_state_dict.pop(key)
            except:  # noqa: E722
                logger.info("-" * 30)
                logger.info(model_state_dict.keys())
            if isinstance(tensor, DTensor):
                state_dict[key].append(tensor._local_tensor.bfloat16())
                placements = tuple(tensor.placements)
                # replicated placement at dp dimension can be discarded
                if mesh_dim_names[0] == "dp":
                    placements = placements[1:]
                if key not in param_placements:
                    param_placements[key] = placements
                else:
                    assert param_placements[key] == placements
            else:
                state_dict[key] = tensor.bfloat16()

    del model_state_dict_lst

    for key in sorted(state_dict):
        if not isinstance(state_dict[key], list):
            logger.info(f"No need to merge key {key}")
            continue
        # merge shards
        placements: Tuple[Shard] = param_placements[key]  # type: ignore
        if len(mesh_shape) == 1:
            # 1-D list, FSDP without TP
            assert len(placements) == 1
            shards = state_dict[key]
            state_dict[key] = merge_by_placement(shards, placements[0])
        else:
            # 2-D list, FSDP + TP
            raise NotImplementedError("FSDP + TP is not supported yet")

    return state_dict
