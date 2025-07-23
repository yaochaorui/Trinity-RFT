import copy
import os
import subprocess
import tempfile
from typing import List

import streamlit as st
import yaml

from trinity.algorithm.advantage_fn.advantage_fn import ADVANTAGE_FN
from trinity.algorithm.algorithm import ALGORITHM_TYPE
from trinity.algorithm.entropy_loss_fn.entropy_loss_fn import ENTROPY_LOSS_FN
from trinity.algorithm.kl_fn.kl_fn import KL_FN
from trinity.algorithm.policy_loss_fn.policy_loss_fn import POLICY_LOSS_FN
from trinity.algorithm.sample_strategy.sample_strategy import SAMPLE_STRATEGY
from trinity.common.constants import StorageType
from trinity.manager.config_registry.config_registry import CONFIG_GENERATORS
from trinity.manager.config_registry.trainer_config_manager import use_critic

register_map = {
    "sample_strategy": SAMPLE_STRATEGY,
    "policy_loss_fn": POLICY_LOSS_FN,
    "advantage_fn": ADVANTAGE_FN,
    "kl_loss_fn": KL_FN,
    "kl_penalty_fn": KL_FN,
    "entropy_loss_fn": ENTROPY_LOSS_FN,
}


class ConfigManager:
    def __init__(self):
        self.unfinished_fields = set()
        CONFIG_GENERATORS.set_unfinished_fields(self.unfinished_fields)
        st.set_page_config(page_title="Trinity-RFT Config Generator", page_icon=":robot:")
        st.title("Trinity-RFT Config Generator")
        if "_init_config_manager" not in st.session_state:
            self.reset_session_state()
        self.maintain_session_state()
        mode = st.pills(
            "Select Mode",
            options=["Beginner Mode", "Expert Mode"],
            default="Beginner Mode",
            label_visibility="collapsed",
        )
        if mode == "Beginner Mode":
            self.beginner_mode()
        else:
            self.expert_mode()
        if "config_generated" not in st.session_state:
            st.session_state.config_generated = False
        if "is_running" not in st.session_state:
            st.session_state.is_running = False
        self.generate_config()

    def reset_session_state(self):
        st.session_state["_init_config_manager"] = True
        for key, value in CONFIG_GENERATORS.default_config.items():
            st.session_state[key] = value

    def maintain_session_state(self):
        st.session_state["_init_config_manager"] = True
        for key in CONFIG_GENERATORS.default_config:
            st.session_state[key] = st.session_state[key]

        def maintain_list_state(prefix, key_list):
            last_idx, del_num = 0, 0
            for idx in range(st.session_state[f"_{prefix}s_num"]):
                if st.session_state.get(f"{prefix}_{idx}_del_flag", False):
                    del_num += 1
                    continue
                for key in key_list:
                    full_key = f"{prefix}_{idx}_{key}"
                    last_full_key = f"{prefix}_{last_idx}_{key}"
                    st.session_state[last_full_key] = st.session_state[full_key]
                last_idx += 1
            st.session_state[f"_{prefix}s_num"] -= del_num

        self.eval_dataset_keys = [
            "name",
            "path",
            "split",
            "subset_name",
            "prompt_key",
            "response_key",
            "temperature",
            "logprobs",
            "n",
        ]
        maintain_list_state("eval_taskset", self.eval_dataset_keys)

        self.inference_model_keys = [
            "model_path",
            "engine_type",
            "engine_num",
            "tensor_parallel_size",
            "use_v1",
            "enforce_eager",
            "enable_prefix_caching",
            "enable_chunked_prefill",
            "gpu_memory_utilization",
            "dtype",
            "seed",
            "enable_thinking",
            "enable_openai_api",
        ]
        maintain_list_state("auxiliary_model", self.inference_model_keys)

    def get_configs(self, *config_names: str, columns_spec: List[int] = None):
        CONFIG_GENERATORS.get_configs(*config_names, columns_spec=columns_spec)

    def beginner_mode(self):
        st.header("Essential Configs")
        self.get_configs("project", "exp_name", columns_spec=[1, 2])

        self.get_configs("model_path")

        self.get_configs("checkpoint_root_dir")

        if st.session_state["algorithm_type"] != "dpo":
            self.get_configs("taskset_path")
        else:
            self.get_configs("experience_buffer_path")

        self.get_configs("algorithm_type", "sft_warmup_steps", "monitor_type")
        if st.session_state["sft_warmup_steps"] > 0:
            self.get_configs("sft_warmup_dataset_path")

        st.header("Important Configs")
        self.get_configs("node_num", "gpu_per_node", "engine_num", "tensor_parallel_size")

        self.get_configs("total_epochs", "train_batch_size", "ppo_epochs", "repeat_times")

        self.get_configs("storage_type", "max_prompt_tokens", "max_response_tokens")

        self.get_configs("sync_interval", "eval_interval", "save_interval")

        if st.session_state["algorithm_type"] != "dpo":
            self.get_configs("taskset_args")
        else:
            self.get_configs("dpo_dataset_kwargs")

        if st.session_state["sft_warmup_steps"] > 0:
            self.get_configs("sft_warmup_dataset_args")

        self.get_configs("default_workflow_type", "default_reward_fn_type")

        self.get_configs(
            "actor_ppo_micro_batch_size_per_gpu",
            "actor_lr",
            "ref_log_prob_micro_batch_size_per_gpu",
        )

        self.get_configs("critic_ppo_micro_batch_size_per_gpu", "critic_lr")

    def _expert_model_part(self):
        self.get_configs("project", "exp_name", columns_spec=[1, 2])

        self.get_configs("model_path")
        self.get_configs("critic_model_path")

        self.get_configs("checkpoint_root_dir")

        self.get_configs("monitor_type", "node_num", "gpu_per_node")
        self.get_configs("max_prompt_tokens", "max_response_tokens")

    def _expert_buffer_part(self):
        self.get_configs("total_epochs", "train_batch_size")

        self.get_configs("default_workflow_type", "default_reward_fn_type")
        self.get_configs("system_prompt")
        self.get_configs("reply_prefix")

        if st.session_state["algorithm_type"] != "dpo":
            with st.expander("Taskset Configs", expanded=True):
                self.get_configs("taskset_path")
                self.get_configs("taskset_args")
        else:
            with st.expander("DPO Dataset Configs", expanded=True):
                self.get_configs("experience_buffer_path")
                self.get_configs("storage_type")
                self.get_configs("dpo_dataset_kwargs")

        with st.expander("Eval Tasksets Configs", expanded=True):
            self.get_configs("eval_tasksets")

        with st.expander("SFT Dataset Configs"):
            self.get_configs("sft_warmup_dataset_path")
            self.get_configs("sft_warmup_dataset_args")

        if st.session_state["algorithm_type"] != "dpo":
            with st.expander("Experiences Buffer Configs", expanded=True):
                self.get_configs("storage_type")
                self.get_configs("experience_buffer_path")
                self.get_configs("use_priority_queue")
                self.get_configs("reuse_cooldown_time", "priority_fn", "priority_decay")

        self.buffer_advanced_tab = st.expander("Advanced Config")
        with self.buffer_advanced_tab:
            self.get_configs("buffer_max_retry_times", "max_retry_interval")

    def _expert_explorer_part(self):
        self.get_configs("sync_method", "sync_interval", "sync_timeout")

        self.get_configs(
            "runner_per_model", "max_timeout", "explorer_max_retry_times", "eval_interval"
        )

        self.get_configs("bench_on_latest_checkpoint")

        with st.expander("Rollout Model Config", expanded=True):
            self.get_configs("engine_type", "engine_num", "tensor_parallel_size")

            self.get_configs("gpu_memory_utilization", "dtype", "seed")

            self.get_configs(
                "use_v1", "enforce_eager", "enable_prefix_caching", "enable_chunked_prefill"
            )

            self.get_configs("enable_thinking", "enable_openai_api")

        with st.expander("Auxiliary Models", expanded=True):
            self.get_configs("auxiliary_models")

    def _expert_trainer_part(self):
        self.get_configs("algorithm_type", "repeat_times", "save_interval")
        self.get_configs("sample_strategy", "advantage_fn", "entropy_loss_fn")
        self.get_configs("policy_loss_fn", "kl_penalty_fn", "kl_loss_fn")

        with st.expander("Advanced Algorithm Config"):
            algorithm = ALGORITHM_TYPE.get(st.session_state["algorithm_type"])
            default_config = algorithm.default_config()
            config_key_list = []
            for key in default_config.keys():
                value = st.session_state[key]
                if key == "repeat_times":
                    continue
                default_args = register_map[key].get(value).default_args()
                for sub_key in default_args.keys():
                    full_key = sub_key + "_in_" + key
                    config_key_list.append(full_key)

            idx = 0
            while idx < len(config_key_list):
                delta = 3 if len(config_key_list) - idx != 4 else 2
                key_list = config_key_list[idx : idx + delta]
                idx += delta
                self.get_configs(*key_list)

        self.get_configs("enable_preview")

        if st.session_state["trainer_type"] == "verl":
            self._expert_verl_trainer_part()

    def _expert_verl_training_part(self):
        st.subheader("RL Training Config")
        self.get_configs("training_args")

        self.get_configs("ppo_epochs", "training_strategy", "resume_mode")

        self.get_configs("param_offload", "optimizer_offload", "forward_prefetch")
        self.get_configs("resume_from_path")

        with st.expander("Advanced Config"):
            self.get_configs("critic_warmup", "total_training_steps")

            self.get_configs("default_hdfs_dir")

            self.get_configs("remove_previous_ckpt_in_save", "del_local_ckpt_after_load")

            self.get_configs("max_actor_ckpt_to_keep", "max_critic_ckpt_to_keep")

    def _expert_verl_actor_part(self):
        st.subheader("Actor Model Config")
        self.get_configs(
            "actor_ppo_micro_batch_size_per_gpu",
            "ref_log_prob_micro_batch_size_per_gpu",
            "actor_ulysses_sequence_parallel_size",
            "actor_entropy_from_logits_with_chunking",
            "actor_entropy_checkpointing",
        )

        self.get_configs("actor_lr", "actor_warmup_style", "actor_lr_warmup_steps_ratio")

        self.get_configs("actor_grad_clip")

        self.get_configs("actor_checkpoint")

    def _expert_verl_critic_part(self):
        st.subheader("Critic Model Config")
        self.get_configs(
            "critic_ppo_micro_batch_size_per_gpu", "critic_ulysses_sequence_parallel_size"
        )

        self.get_configs("critic_lr", "critic_warmup_style", "critic_lr_warmup_steps_ratio")

        self.get_configs("critic_grad_clip", "critic_cliprange_value")
        self.get_configs("critic_checkpoint")

    def _expert_verl_trainer_part(self):
        name2func = {
            "RL Training Config": self._expert_verl_training_part,
            "Actor and Ref Config": self._expert_verl_actor_part,
        }
        if use_critic():
            name2func["Critic Config"] = self._expert_verl_critic_part

        tabs = st.tabs([name for name in name2func])
        for tab, func in zip(tabs, name2func.values()):
            with tab:
                func()

    def expert_mode(self):
        tab2func = {
            "Model": self._expert_model_part,
            "Buffer": self._expert_buffer_part,
            "Explorer and Synchronizer": self._expert_explorer_part,
            "Trainer": self._expert_trainer_part,
        }
        if st.session_state["mode"] == "train":
            del tab2func["Explorer and Synchronizer"]
        tabs = st.tabs(list(tab2func.keys()))
        for tab, func in zip(tabs, tab2func.values()):
            with tab:
                func()

    def _generate_verl_config(self):
        balance_batch = "balance_batch" in st.session_state["training_args"]
        enable_gradient_checkpointing = (
            "gradient_checkpointing" in st.session_state["training_args"]
        )
        use_remove_padding = "remove_padding" in st.session_state["training_args"]
        use_dynamic_bsz = "dynamic_bsz" in st.session_state["training_args"]

        if st.session_state["training_strategy"] == "fsdp":
            fsdp_config = {
                "wrap_policy": {"min_num_params": 0},
                "param_offload": st.session_state["param_offload"],
                "optimizer_offload": st.session_state["optimizer_offload"],
                "fsdp_size": -1,
                "forward_prefetch": st.session_state["forward_prefetch"],
            }
        else:
            fsdp_config = {}

        ppo_max_token_len_per_gpu = st.session_state["repeat_times"] * (
            st.session_state["max_prompt_tokens"] + st.session_state["max_response_tokens"]
        )

        trainer_config = {
            "actor_rollout_ref": {
                "model": {
                    "external_lib": None,
                    "override_config": {},
                    "enable_gradient_checkpointing": enable_gradient_checkpointing,
                    "use_remove_padding": use_remove_padding,
                },
                "actor": {
                    "strategy": st.session_state["training_strategy"],
                    "ppo_micro_batch_size_per_gpu": st.session_state[
                        "actor_ppo_micro_batch_size_per_gpu"
                    ],
                    "use_dynamic_bsz": use_dynamic_bsz,
                    "ppo_max_token_len_per_gpu": ppo_max_token_len_per_gpu,
                    "ppo_epochs": st.session_state["ppo_epochs"],
                    "shuffle": False,
                    "ulysses_sequence_parallel_size": st.session_state[
                        "actor_ulysses_sequence_parallel_size"
                    ],
                    "entropy_from_logits_with_chunking": st.session_state[
                        "actor_entropy_from_logits_with_chunking"
                    ],
                    "entropy_checkpointing": st.session_state["actor_entropy_checkpointing"],
                    "checkpoint": {"contents": st.session_state["actor_checkpoint"]},
                    "optim": {
                        "lr": st.session_state["actor_lr"],
                        "lr_warmup_steps_ratio": st.session_state["actor_lr_warmup_steps_ratio"],
                        "warmup_style": st.session_state["actor_warmup_style"],
                        "total_training_steps": (
                            -1
                            if st.session_state["total_training_steps"] is None
                            else st.session_state["total_training_steps"]
                        ),
                    },
                    "fsdp_config": copy.deepcopy(fsdp_config),
                },
                "ref": {
                    "fsdp_config": copy.deepcopy(fsdp_config),
                    "log_prob_micro_batch_size_per_gpu": st.session_state[
                        "ref_log_prob_micro_batch_size_per_gpu"
                    ],
                    "log_prob_use_dynamic_bsz": use_dynamic_bsz,
                    "log_prob_max_token_len_per_gpu": ppo_max_token_len_per_gpu,
                    "ulysses_sequence_parallel_size": st.session_state[
                        "actor_ulysses_sequence_parallel_size"
                    ],
                    "entropy_from_logits_with_chunking": st.session_state[
                        "actor_entropy_from_logits_with_chunking"
                    ],
                    "entropy_checkpointing": st.session_state["actor_entropy_checkpointing"],
                },
            },
            "critic": {},
            "trainer": {
                "balance_batch": balance_batch,
                "resume_mode": st.session_state["resume_mode"],
                "resume_from_path": st.session_state["resume_from_path"],
                "default_hdfs_dir": st.session_state["default_hdfs_dir"],
                "remove_previous_ckpt_in_save": st.session_state["remove_previous_ckpt_in_save"],
                "del_local_ckpt_after_load": st.session_state["del_local_ckpt_after_load"],
                "val_before_train": False,
                "max_actor_ckpt_to_keep": st.session_state["max_actor_ckpt_to_keep"],
                "max_critic_ckpt_to_keep": st.session_state["max_critic_ckpt_to_keep"],
            },
        }

        if use_critic():
            trainer_config["trainer"]["critic_warmup"] = st.session_state["critic_warmup"]
            trainer_config["critic"] = {
                "strategy": st.session_state["training_strategy"],
                "optim": {
                    "lr": st.session_state["critic_lr"],
                    "lr_warmup_steps_ratio": st.session_state["critic_lr_warmup_steps_ratio"],
                    "warmup_style": st.session_state["critic_warmup_style"],
                    "total_training_steps": (
                        -1
                        if st.session_state["total_training_steps"] is None
                        else st.session_state["total_training_steps"]
                    ),
                },
                "model": {
                    "override_config": {},
                    "external_lib": None,
                    "enable_gradient_checkpointing": enable_gradient_checkpointing,
                    "use_remove_padding": use_remove_padding,
                    "fsdp_config": copy.deepcopy(fsdp_config),
                },
                "ppo_mini_batch_size": st.session_state["train_batch_size"],
                "ppo_micro_batch_size_per_gpu": st.session_state[
                    "critic_ppo_micro_batch_size_per_gpu"
                ],
                "forward_micro_batch_size_per_gpu": st.session_state[
                    "critic_ppo_micro_batch_size_per_gpu"
                ],
                "use_dynamic_bsz": use_dynamic_bsz,
                "ppo_max_token_len_per_gpu": ppo_max_token_len_per_gpu * 2,
                "forward_max_token_len_per_gpu": ppo_max_token_len_per_gpu * 2,
                "ulysses_sequence_parallel_size": st.session_state[
                    "critic_ulysses_sequence_parallel_size"
                ],
                "ppo_epochs": st.session_state["ppo_epochs"],
                "shuffle": False,
                "grad_clip": st.session_state["critic_grad_clip"],
                "cliprange_value": st.session_state["critic_cliprange_value"],
                "checkpoint": {"contents": st.session_state["critic_checkpoint"]},
            }
        else:
            del trainer_config["critic"]
        return trainer_config

    def _gen_algorithm_config(self):
        algorithm_config = {
            "algorithm_type": st.session_state["algorithm_type"],
        }
        algorithm = ALGORITHM_TYPE.get(st.session_state["algorithm_type"])
        default_config = algorithm.default_config()
        current_config = {}
        for key in default_config.keys():
            current_config[key] = value = st.session_state[key]
            if key == "repeat_times":
                continue
            default_args = register_map[key].get(value).default_args()
            args = {}
            for sub_key in default_args.keys():
                full_key = sub_key + "_in_" + key
                args[sub_key] = st.session_state.get(full_key, default_args[sub_key])
            if default_args != args:
                current_config[key + "_args"] = args
        if default_config != current_config:
            algorithm_config.update(current_config)
        return algorithm_config

    def _gen_buffer_config(self):
        experience_buffer_path = st.session_state["experience_buffer_path"].strip()
        if st.session_state["algorithm_type"] != "dpo":
            if (
                not experience_buffer_path
                and st.session_state["storage_type"] == StorageType.SQL.value
            ):
                experience_buffer_path = f"sqlite:///{os.path.join(st.session_state['checkpoint_root_dir'], '.cache', st.session_state['project'], st.session_state['exp_name'])}/data.db"

        sft_storage_type = (
            StorageType.SQL.value
            if "://" in st.session_state["sft_warmup_dataset_path"]
            else StorageType.FILE.value
        )  # TODO

        buffer_config = {
            "batch_size": st.session_state["train_batch_size"],
            "total_epochs": st.session_state["total_epochs"],
            "explorer_input": {},
            "trainer_input": {
                "experience_buffer": {
                    "name": "experience_buffer",
                    "storage_type": st.session_state["storage_type"],
                    "path": experience_buffer_path,
                },
                "sft_warmup_steps": st.session_state["sft_warmup_steps"],
            },
            "max_retry_times": st.session_state["buffer_max_retry_times"],
            "max_retry_interval": st.session_state["max_retry_interval"],
        }
        if st.session_state["algorithm_type"] != "dpo":
            experience_buffer = buffer_config["trainer_input"]["experience_buffer"]
            experience_buffer["use_priority_queue"] = st.session_state["use_priority_queue"]
            experience_buffer["reuse_cooldown_time"] = st.session_state["reuse_cooldown_time"]
            experience_buffer["replay_buffer_kwargs"] = {
                "priority_fn": st.session_state["priority_fn"],
                "decay": st.session_state["priority_decay"],
            }

        if st.session_state["mode"] != "train":
            buffer_config["explorer_input"] = {
                "taskset": {
                    "name": "taskset",
                    "storage_type": StorageType.FILE.value,
                    "path": st.session_state["taskset_path"],
                    "split": st.session_state["taskset_split"],
                    "subset_name": st.session_state["taskset_subset_name"],
                    "format": {
                        "prompt_key": st.session_state["taskset_prompt_key"],
                        "response_key": st.session_state["taskset_response_key"],
                    },
                    "rollout_args": {
                        "temperature": st.session_state["temperature"],
                        "logprobs": st.session_state["logprobs"],
                    },
                },
                "eval_tasksets": [],
                "default_workflow_type": st.session_state["default_workflow_type"],
                "default_reward_fn_type": st.session_state["default_reward_fn_type"],
                "system_prompt": st.session_state["system_prompt"],
                "reply_prefix": st.session_state["reply_prefix"],
            }
            for idx in range(st.session_state["_eval_tasksets_num"]):
                if st.session_state[f"eval_taskset_{idx}_path"].strip():
                    buffer_config["explorer_input"]["eval_tasksets"].append(
                        {
                            "name": st.session_state[f"eval_taskset_{idx}_name"],
                            "path": st.session_state[f"eval_taskset_{idx}_path"],
                            "split": st.session_state[f"eval_taskset_{idx}_split"],
                            "subset_name": st.session_state[f"eval_taskset_{idx}_subset_name"],
                            "format": {
                                "prompt_key": st.session_state[f"eval_taskset_{idx}_prompt_key"],
                                "response_key": st.session_state[
                                    f"eval_taskset_{idx}_response_key"
                                ],
                            },
                            "rollout_args": {
                                "temperature": st.session_state[f"eval_taskset_{idx}_temperature"],
                                "logprobs": st.session_state[f"eval_taskset_{idx}_logprobs"],
                                "n": st.session_state[f"eval_taskset_{idx}_n"],
                            },
                        }
                    )
        else:
            del buffer_config["explorer_input"]

        if st.session_state["algorithm_type"] == "dpo":
            experience_buffer = buffer_config["trainer_input"]["experience_buffer"]
            experience_buffer["split"] = st.session_state["dpo_dataset_train_split"]
            experience_buffer["format"] = {
                "prompt_type": st.session_state["dpo_dataset_prompt_type"],
                "prompt_key": st.session_state["dpo_dataset_prompt_key"],
                "chosen_key": st.session_state["dpo_dataset_chosen_key"],
                "rejected_key": st.session_state["dpo_dataset_rejected_key"],
            }
        if st.session_state["sft_warmup_dataset_path"].strip():
            buffer_config["trainer_input"]["sft_warmup_dataset"] = {
                "name": "sft_warmup_dataset",
                "storage_type": sft_storage_type,
                "path": st.session_state["sft_warmup_dataset_path"],
                "split": st.session_state["sft_warmup_train_split"],
                "format": {
                    "prompt_type": st.session_state["sft_warmup_prompt_type"],
                    "messages_key": st.session_state["sft_warmup_messages_key"],
                    "prompt_key": st.session_state["sft_warmup_prompt_key"],
                    "response_key": st.session_state["sft_warmup_response_key"],
                },
            }

        return buffer_config

    def _gen_explorer_config(self):
        explorer_config = {
            "runner_per_model": st.session_state["runner_per_model"],
            "max_timeout": st.session_state["max_timeout"],
            "max_retry_times": st.session_state["explorer_max_retry_times"],
            "rollout_model": {
                key: st.session_state[key]
                for key in self.inference_model_keys
                if key != "model_path"
                # "max_prompt_tokens": None,  # TODO
                # "max_response_tokens": None,  # TODO
                # "chat_template": None,  # TODO: add chat template
            },
            "auxiliary_models": [],
            "eval_interval": st.session_state["eval_interval"],
            "bench_on_latest_checkpoint": st.session_state["bench_on_latest_checkpoint"],
        }
        for i in range(st.session_state["_auxiliary_models_num"]):
            auxiliary_model_config = {
                key: st.session_state[f"auxiliary_model_{i}_{key}"]
                for key in self.inference_model_keys
            }
            explorer_config["auxiliary_models"].append(auxiliary_model_config)
        return explorer_config

    def generate_config(self):
        if st.session_state["trainer_type"] == "verl":
            trainer_config = self._generate_verl_config()
        else:
            raise ValueError(f"Invalid trainer type: {st.session_state['trainer_type']}")

        if len(self.unfinished_fields) > 0:
            disable_generate = True
            help_messages = (
                f"Please check following fields: `{'`, `'.join(self.unfinished_fields)}`"
            )
        else:
            disable_generate = False
            help_messages = None
        if st.button(
            "Generate Config",
            disabled=disable_generate,
            help=help_messages,
            use_container_width=True,
            icon=":material/create_new_folder:",
        ):
            st.session_state.config_generated = True
            st.session_state.is_running = False
        if st.session_state.config_generated:
            config = {
                "mode": st.session_state["mode"],
                "project": st.session_state["project"],
                "name": st.session_state["exp_name"],
                "checkpoint_root_dir": st.session_state["checkpoint_root_dir"],
                "algorithm": self._gen_algorithm_config(),
                "data_processor": {},  # TODO: Add data processor config
                "model": {
                    "model_path": st.session_state["model_path"],
                    "max_prompt_tokens": st.session_state["max_prompt_tokens"],
                    "max_response_tokens": st.session_state["max_response_tokens"],
                },
                "cluster": {
                    "node_num": st.session_state["node_num"],
                    "gpu_per_node": st.session_state["gpu_per_node"],
                },
                "buffer": self._gen_buffer_config(),
                "explorer": self._gen_explorer_config(),
                "trainer": {
                    "trainer_type": st.session_state["trainer_type"],
                    "save_interval": st.session_state["save_interval"],
                    "enable_preview": st.session_state["enable_preview"],
                    "actor_grad_clip": st.session_state["actor_grad_clip"],
                    "trainer_config": trainer_config,
                },
                "monitor": {
                    "monitor_type": st.session_state["monitor_type"],
                },
                "synchronizer": {
                    "sync_method": st.session_state["sync_method"],
                    "sync_interval": st.session_state["sync_interval"],
                    "sync_timeout": st.session_state["sync_timeout"],
                },
            }

            if use_critic():
                config["model"]["critic_model_path"] = (
                    st.session_state["critic_model_path"].strip()
                    if st.session_state["critic_model_path"].strip()
                    else st.session_state["model_path"]
                )

            st.session_state.config_generated = True
            st.header("Generated Config File")
            buttons = st.container()
            save_btn, run_btn = buttons.columns(2, vertical_alignment="bottom")
            yaml_config = yaml.dump(config, allow_unicode=True, sort_keys=False)
            save_btn.download_button(
                "Save",
                data=yaml_config,
                file_name=f"{config['project']}-{config['name']}.yaml",
                mime="text/plain",
                icon=":material/download:",
                use_container_width=True,
            )
            run_btn.button(
                "Run",
                on_click=self.run_config,
                args=(
                    buttons,
                    yaml_config,
                ),
                icon=":material/terminal:",
                use_container_width=True,
                disabled=st.session_state.is_running,
            )
            st.code(yaml_config, language="yaml")

    def run_config(self, parent, yaml_config: str) -> None:
        st.session_state.is_running = True

        import ray

        # first check if ray is running
        ray_status = subprocess.run(
            ["ray", "status"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if ray_status.returncode != 0:
            parent.warning(
                "Ray cluster is not running. Please start Ray first using `ray start --head`."
            )
            return
        context = ray.init(ignore_reinit_error=True)
        dashboard_url = context.dashboard_url
        # save config to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmpfile:
            tmpfile.write(yaml_config)
            tmpfile_path = tmpfile.name

        # submit ray job
        try:
            subprocess.run(
                [
                    "ray",
                    "job",
                    "submit",
                    "--no-wait",
                    "--",
                    "python",
                    "-m",
                    "trinity.cli.launcher",
                    "run",
                    "--config",
                    tmpfile_path,
                ],
                text=True,
                capture_output=True,
                check=True,
            )
            parent.success(
                f"Job submitted successfully!\n\n"
                f"View progress in the Ray Dashboard: http://{dashboard_url}",
                icon="✅",
            )
        except subprocess.CalledProcessError as e:
            parent.error(f"Failed to submit job:\n\n{e.stderr}", icon="❌")
            st.session_state.is_running = False


if __name__ == "__main__":
    config_manager = ConfigManager()
