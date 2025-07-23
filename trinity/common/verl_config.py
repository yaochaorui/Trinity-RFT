import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf

from trinity.common.config import BufferConfig, Config, SynchronizerConfig
from trinity.common.constants import EXPLORER_NAME
from trinity.utils.log import get_logger

logger = get_logger(__name__)


@dataclass
class Data:
    train_batch_size: int = 1024


@dataclass
class ActorModel:
    path: str = ""
    external_lib: Optional[str] = None
    override_config: Dict[str, Any] = field(default_factory=dict)
    enable_gradient_checkpointing: bool = True
    use_remove_padding: bool = False
    use_fused_kernels: bool = False
    custom_chat_template: Optional[str] = None


@dataclass
class Optim:
    lr: float = 1e-6
    lr_warmup_steps: int = -1
    lr_warmup_steps_ratio: float = 0.0
    min_lr_ratio: Optional[float] = 0.0
    warmup_style: str = "constant"
    total_training_steps: int = -1
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])


@dataclass
class WrapPolicy:
    min_num_params: int = 0


@dataclass
class FSDPConfig:
    wrap_policy: WrapPolicy = field(default_factory=WrapPolicy)
    min_num_params: int = 0
    param_offload: bool = False
    optimizer_offload: bool = False
    fsdp_size: int = -1
    forward_prefetch: bool = False


@dataclass
class Checkpoint:
    load_contents: List[str] = field(default_factory=lambda: ["model", "optimizer", "extra"])
    save_contents: List[str] = field(default_factory=lambda: ["model", "optimizer", "extra"])


@dataclass
class Actor:
    strategy: str = "fsdp"
    ppo_mini_batch_size: int = 256
    ppo_micro_batch_size: Optional[int] = None
    ppo_micro_batch_size_per_gpu: int = 1
    use_dynamic_bsz: bool = False
    ppo_max_token_len_per_gpu: int = (
        16384  # n * ${data.max_prompt_length} + ${data.max_response_length}
    )
    grad_clip: float = 1.0
    ppo_epochs: int = 1
    shuffle: bool = False
    ulysses_sequence_parallel_size: int = 1
    entropy_from_logits_with_chunking: bool = False
    entropy_checkpointing: bool = False
    checkpoint: Checkpoint = field(default_factory=Checkpoint)
    optim: Optim = field(default_factory=Optim)
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)
    # do not set
    loss_agg_mode: str = "token-mean"
    clip_ratio: float = 0.2
    entropy_coeff: float = 0.001
    use_kl_loss: bool = False
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"


@dataclass
class Ref:
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)
    log_prob_micro_batch_size: Optional[int] = None
    log_prob_micro_batch_size_per_gpu: int = 1
    log_prob_use_dynamic_bsz: bool = False
    log_prob_max_token_len_per_gpu: int = 0
    ulysses_sequence_parallel_size: int = 1
    entropy_from_logits_with_chunking: bool = False
    entropy_checkpointing: bool = False
    checkpoint: Checkpoint = field(
        default_factory=lambda: Checkpoint(load_contents=["model"], save_contents=["model"])
    )


@dataclass
class _ValKwargs:
    do_sample: bool = False


@dataclass
class _MultiTurn:
    enable: bool = False


@dataclass
class Rollout:
    # do not set
    val_kwargs: _ValKwargs = field(default_factory=_ValKwargs)
    multi_turn: _MultiTurn = field(default_factory=_MultiTurn)
    temperature: float = 1.0
    n: int = 1  # > 1 for grpo
    log_prob_micro_batch_size: Optional[int] = None
    log_prob_micro_batch_size_per_gpu: int = 1


@dataclass
class ActorRolloutRef:
    hybrid_engine: bool = True
    model: ActorModel = field(default_factory=ActorModel)
    actor: Actor = field(default_factory=Actor)
    ref: Ref = field(default_factory=Ref)
    rollout: Rollout = field(default_factory=Rollout)
    synchronizer: Optional[SynchronizerConfig] = None
    explorer_name: str = EXPLORER_NAME


@dataclass
class CriticModel:
    path: str = ""
    tokenizer_path: str = ""
    override_config: Dict[str, str] = field(default_factory=dict)
    external_lib: Optional[str] = None
    enable_gradient_checkpointing: bool = True
    use_remove_padding: bool = False
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)


@dataclass
class Critic:
    strategy: str = "fsdp"
    optim: Optim = field(default_factory=Optim)
    model: CriticModel = field(default_factory=CriticModel)
    ppo_mini_batch_size: int = 0
    ppo_micro_batch_size: Optional[int] = None
    ppo_micro_batch_size_per_gpu: int = 1
    forward_micro_batch_size: Optional[int] = None
    forward_micro_batch_size_per_gpu: Optional[int] = None
    use_dynamic_bsz: bool = False
    ppo_max_token_len_per_gpu: int = 0
    forward_max_token_len_per_gpu: int = 0
    ulysses_sequence_parallel_size: int = 1
    ppo_epochs: int = 0
    shuffle: bool = False
    grad_clip: float = 0.0
    cliprange_value: float = 0.0
    checkpoint: Checkpoint = field(default_factory=Checkpoint)
    rollout_n: int = 1
    loss_agg_mode: str = "token-mean"


@dataclass
class _RewardModel:
    input_tokenizer: Optional[str] = None
    path: str = ""
    external_lib: Optional[str] = None
    use_remove_padding: bool = False
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)


@dataclass
class RewardModel:
    enable: bool = False
    strategy: str = "fsdp"
    model: _RewardModel = field(default_factory=_RewardModel)
    micro_batch_size_per_gpu: int = 1
    max_length: Optional[int] = None
    ulysses_sequence_parallel_size: int = 1
    use_dynamic_bsz: bool = False
    forward_max_token_len_per_gpu: int = 0
    reward_manager: str = "naive"


@dataclass
class CustomRewardFunction:
    path: Optional[str] = None
    name: str = "compute_score"


@dataclass
class KL_Ctrl:
    type: str = "fixed"
    kl_coef: float = 0.001
    horizon: float = 10000
    target_kl: float = 0.1


@dataclass
class Algorithm:
    # ! DO NOT SET gamma or lam below; they are kept here merely for compatibility with verl,
    # and their values will be overwritten by those in AlgorithmConfig.advantage_fn_args
    # if they are really needed (e.g., for GAE advantage/returns computation)
    gamma: float = 1.0
    lam: float = 1.0
    adv_estimator: str = "gae"
    norm_adv_by_std_in_grpo: bool = True
    use_kl_in_reward: bool = False
    kl_penalty: str = "kl"
    kl_ctrl: KL_Ctrl = field(default_factory=KL_Ctrl)


@dataclass
class Trainer:
    balance_batch: bool = True
    total_epochs: int = 30
    total_training_steps: Optional[int] = None
    project_name: str = ""
    experiment_name: str = ""
    logger: List[str] = field(default_factory=list)
    val_generations_to_log_to_wandb: int = 0
    nnodes: int = 0
    n_gpus_per_node: int = 0
    save_freq: int = 0
    resume_mode: str = "auto"
    resume_from_path: str = ""
    test_freq: int = 0
    critic_warmup: int = 0
    default_hdfs_dir: Optional[str] = None
    remove_previous_ckpt_in_save: bool = False  # deprecated
    del_local_ckpt_after_load: bool = False
    default_local_dir: str = ""
    val_before_train: bool = False
    training_rollout_mode: str = "parallel"
    enable_exp_buffer: bool = True
    sync_freq: int = 0
    sft_warmup_steps: int = 0
    max_actor_ckpt_to_keep: Optional[int] = None
    max_critic_ckpt_to_keep: Optional[int] = None


@dataclass
class veRLConfig:
    data: Data = field(default_factory=Data)
    actor_rollout_ref: ActorRolloutRef = field(default_factory=ActorRolloutRef)
    critic: Critic = field(default_factory=Critic)
    reward_model: RewardModel = field(default_factory=RewardModel)
    custom_reward_function: CustomRewardFunction = field(default_factory=CustomRewardFunction)
    algorithm: Algorithm = field(default_factory=Algorithm)
    trainer: Trainer = field(default_factory=Trainer)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    synchronizer: Optional[SynchronizerConfig] = None
    enable_preview: bool = True

    def synchronize_config(self, config: Config) -> None:  # noqa: C901
        """Synchronize config."""
        if config.mode != "train":
            rollout_gpu_num = (
                config.explorer.rollout_model.tensor_parallel_size
                * config.explorer.rollout_model.engine_num
                + sum(
                    [
                        model.tensor_parallel_size * model.engine_num
                        for model in config.explorer.auxiliary_models
                    ]
                )
            )
        else:
            rollout_gpu_num = 0

        if config.cluster.node_num == 1:
            # for single node scenarios, rollout and training are on the same node
            self.trainer.nnodes = config.cluster.node_num
            self.trainer.n_gpus_per_node = config.cluster.gpu_per_node - rollout_gpu_num
        else:
            # for multi-node scenarios, some nodes for rollout, others for training
            assert (
                rollout_gpu_num % config.cluster.gpu_per_node == 0
            ), "rollout_gpu_num must be divisible by `gpu_per_node`"
            rollout_node_num = math.ceil(rollout_gpu_num / config.cluster.gpu_per_node)
            self.trainer.nnodes = config.cluster.node_num - rollout_node_num
            if self.trainer.nnodes < 1:
                raise ValueError("The number of training nodes must be greater than 0")
            self.trainer.n_gpus_per_node = config.cluster.gpu_per_node

        world_size = self.trainer.nnodes * self.trainer.n_gpus_per_node
        if config.buffer.batch_size % world_size != 0:
            raise ValueError(
                f"batch_size ({config.buffer.batch_size}) must be divisible by ({world_size})"
            )

        self.trainer.sync_freq = config.synchronizer.sync_interval
        self.trainer.save_freq = config.trainer.save_interval
        self.trainer.project_name = config.project
        self.trainer.experiment_name = config.name
        self.trainer.default_local_dir = config.checkpoint_job_dir
        self.trainer.sft_warmup_steps = config.buffer.trainer_input.sft_warmup_steps
        if not config.continue_from_checkpoint:
            self.trainer.resume_mode = "disable"
        else:
            self.trainer.resume_mode = "auto"

        self.buffer = config.buffer
        # TODO: use dynamic read_batch_size to support multi-round scenarios
        # Get the experiences of one explore step
        self.data.train_batch_size = config.buffer.batch_size

        self.synchronizer = config.synchronizer
        self.actor_rollout_ref.synchronizer = config.synchronizer
        self.actor_rollout_ref.explorer_name = config.explorer.name

        # Actor / Critic config
        self.actor_rollout_ref.model.path = config.model.model_path
        self.actor_rollout_ref.model.custom_chat_template = config.model.custom_chat_template
        self.critic.model.path = config.model.critic_model_path
        self.critic.model.tokenizer_path = config.model.critic_model_path
        self.actor_rollout_ref.actor.ppo_mini_batch_size = (
            config.buffer.batch_size
        )  # TODO: may allow user to change
        self.actor_rollout_ref.rollout.temperature = (
            config.buffer.explorer_input.taskset.rollout_args.temperature
        )
        self.actor_rollout_ref.rollout.n = config.algorithm.repeat_times
        self.critic.ppo_mini_batch_size = config.buffer.batch_size
        self.critic.rollout_n = self.actor_rollout_ref.rollout.n
        self.critic.synchronizer = config.synchronizer

        if config.trainer.actor_grad_clip is not None:
            self.actor_rollout_ref.actor.grad_clip = config.trainer.actor_grad_clip

        # Algorithm related config
        self.actor_rollout_ref.actor.use_kl_loss = config.algorithm.kl_loss_fn != "none"
        self.algorithm.use_kl_in_reward = config.algorithm.kl_penalty_fn != "none"
        # TODO (yanxi): it seems that adv_estimator now only affects whether use_critic is set to
        # True or False in RayPPOTrainer.__init__() (and hence in VerlPPOTrainerWrapper).
        # Need to double check whether this is indeed the case,
        # and see if adv_estimator can be removed completely.

        if config.algorithm.algorithm_type == "dpo":  # for DPO
            logger.warning("DPO micro batch size is doubled for computing loss.")
            self.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu *= 2
            self.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu *= 2
            if self.actor_rollout_ref.rollout.n != 2:
                self.actor_rollout_ref.rollout.n = 2
        # TODO: check other fields
        self.enable_preview = config.trainer.enable_preview


def load_config(config_path: str) -> veRLConfig:
    schema = OmegaConf.structured(veRLConfig)
    yaml_config = OmegaConf.load(config_path)
    try:
        config = OmegaConf.merge(schema, yaml_config)
        return OmegaConf.to_object(config)
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}") from e
