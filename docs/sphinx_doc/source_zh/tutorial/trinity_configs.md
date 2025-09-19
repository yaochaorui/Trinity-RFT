(Configuration Guide)=
# 参数配置指南

本节详细描述了 **Trinity-RFT** 中使用的参数。

## 概述

**Trinity-RFT** 的配置通过一个 `YAML` 文件定义，并根据不同的模块划分为多个部分。以下是一个基础配置文件的示例：

```yaml
project: Trinity-RFT
name: example
mode: both
checkpoint_root_dir: ${oc.env:TRINITY_CHECKPOINT_ROOT_DIR,./checkpoints}
continue_from_checkpoint: true

algorithm:
  # 算法相关参数
  ...
model:
  # 模型相关配置
  ...
cluster:
  # 集群节点和 GPU 设置
  ...
buffer:
  # 数据相关配置
  ...
explorer:
  # explorer 相关设置（rollout 模型、workflow runners）
  ...
trainer:
  # trainer 特定参数
  ...
synchronizer:
  # 模型权重同步设置
  ...
monitor:
  # 监控配置（例如 WandB、TensorBoard 或 MLFlow）
  ...
service:
  # 使用的服务
  ...
data_processor:
  # 数据预处理设置
  ...
log:
  # Ray actor 日志
  ...

stages:
  # 多阶段训练
  ...
```

每个部分将在下文详细说明。关于此处未涵盖的具体参数的更多细节，请参考[源码](https://github.com/modelscope/Trinity-RFT/blob/main/trinity/common/config.py)。

```{tip}
Trinity-RFT 使用[OmegaConf](https://omegaconf.readthedocs.io/en/latest/) 来加载 YAML 配置文件。
它支持一些高级特性，如[变量插值](https://omegaconf.readthedocs.io/en/latest/usage.html#variable-interpolation)和[环境变量替换](https://omegaconf.readthedocs.io/en/latest/custom_resolvers.html#oc-env)。
用户可利用这些特性简化配置。
```

---

## 全局配置

这些是适用于整个实验的一般性设置。

```yaml
project: Trinity-RFT
name: example
mode: both
checkpoint_root_dir: ${oc.env:TRINITY_CHECKPOINT_ROOT_DIR,./checkpoints}   # TRINITY_CHECKPOINT_ROOT_DIR 是预先设置的环境变量
```

- `project`: 项目名称。
- `name`: 当前实验的名称。
- `mode`: Trinity-RFT 的运行模式。选项包括：
  - `both`: 同时启动 trainer 和 explorer（默认）。
  - `train`: 仅启动 trainer。
  - `explore`: 仅启动 explorer。
  - `bench`: 用于 benchmark 测试。
- `checkpoint_root_dir`: 所有检查点和日志的根目录。该实验的检查点将存储在 `<checkpoint_root_dir>/<project>/<name>/` 路径下。
- `continue_from_checkpoint`: 若设置为 `true`，实验将从检查点路径中的最新检查点继续；否则，会将当前实验重命名为 `<name>_<timestamp>` 并启动新实验。
- `ray_namespace`: 当前实验中启动模块的命名空间。若未指定，则默认为 `<project>/<name>`。

---

## 算法配置

指定 RFT 算法类型及其相关超参数。

```yaml
algorithm:
  algorithm_type: grpo
  repeat_times: 8

  # 以下参数为可选
  # 若未指定，将根据 `algorithm_type` 自动设置
  sample_strategy: "default"
  advantage_fn: "ppo"
  kl_penalty_fn: "none"
  kl_loss_fn: "k2"
  entropy_loss_fn: "default"
```

- `algorithm_type`: 强化学习算法类型。支持类型：`ppo`、`grpo`、`opmd`、`dpo`、`sft`、`mix`。
- `repeat_times`: 每个任务重复的次数。默认为 `1`。在 `dpo` 中自动设为 `2`。某些算法如 GRPO 和 OPMD 要求 `repeat_times` > 1。
- `sample_strategy`: 从 experience buffer 加载 experience 时使用的采样策略。
- `advantage_fn`: 用于计算优势值的函数。
- `kl_penalty_fn`: 用于在奖励中计算 KL 惩罚的函数。
- `kl_loss_fn`: 用于计算 KL 损失的函数。
- `entropy_loss_fn`: 用于计算熵损失的函数。

---

## 监控配置

用于在执行过程中记录指标。

```yaml
monitor:
  monitor_type: wandb
  monitor_args:
    base_url: http://localhost:8080
    api_key: your_api_key
  enable_ray_timeline: False
```

- `monitor_type`: 监控系统类型。选项：
  - `wandb`: 记录到 [Weights & Biases](https://docs.wandb.ai/quickstart/)。需要登录并设置 `WANDB_API_KEY`。项目和运行名称与全局配置中的 `project` 和 `name` 字段一致。
  - `tensorboard`: 记录到 [TensorBoard](https://www.tensorflow.org/tensorboard)。文件保存在 `<checkpoint_root_dir>/<project>/<name>/monitor/tensorboard` 下。
  - `mlflow`: 记录到 [MLFlow](https://mlflow.org/)。如果设置了 [MLFlow 认证](https://mlflow.org/docs/latest/ml/auth/)，请在运行前将 `MLFLOW_TRACKING_USERNAME` 和 `MLFLOW_TRACKING_PASSWORD` 设置为环境变量。
- `monitor_args`: 初始化监控器的参数字典。
  - 对于 `wandb`：
    - `base_url`: 若设置，将覆盖 `WANDB_BASE_URL`。
    - `api_key`: 若设置，将覆盖 `WANDB_API_KEY`。
  - 对于 `mlflow`：
    - `uri`: MLFlow 实例的 URI。强烈建议设置；默认为 `http://localhost:5000`。
    - `username`: 若设置，将覆盖 `MLFLOW_TRACKING_USERNAME`。
    - `password`: 若设置，将覆盖 `MLFLOW_TRACKING_PASSWORD`。
- `enable_ray_timeline`: 若为 `True`，将导出一个 `timeline.json` 文件到 `<checkpoint_root_dir>/<project>/<name>/monitor`。可在 Chrome 浏览器中访问 [chrome://tracing](chrome://tracing) 查看。

---

## 模型配置

定义模型路径和 token 限制。

```yaml
model:
  model_path: ${oc.env:MODEL_PATH}  # MODEL_PATH 是预先设置的环境变量
  critic_model_path: ${model.model_path}  # 使用 model.model_path 的值
  max_response_tokens: 16384
  max_model_len: 20480
```

- `model_path`: 被训练模型的路径。
- `critic_model_path`: 可选的独立 critic 模型路径。若为空，则默认为 `model_path`。
- `max_response_tokens`: 模型生成的回复中允许的最大 token 数。
- `max_model_len`: 序列中最大 token 数。

---

## 集群配置

定义使用的节点数和每节点的 GPU 数。

```yaml
cluster:
  node_num: 1
  gpu_per_node: 8
```

- `node_num`: 计算节点总数。
- `gpu_per_node`: 每节点可用的 GPU 数量。

---

## 缓冲区配置

配置 explorer 和 trainer 使用的数据缓冲区（Buffer）。

```yaml
buffer:
  batch_size: 32
  train_batch_size: 256
  total_epochs: 100

  explorer_input:
    taskset:
      ...
    eval_tasksets:
      ...
  trainer_input:
    experience_buffer:
      ...
    auxiliary_buffers:
      buffer_1:
        ...
      buffer_2:
        ...

  default_workflow_type: 'math_workflow'
  default_reward_fn_type: 'countdown_reward'
```

- `batch_size`: 每个训练步骤使用的任务数。*请勿手动将此值乘以 `algorithm.repeat_times`*。
- `train_batch_size`: 每个训练步骤使用的 experience 数量。默认为 `batch_size` * `algorithm.repeat_times`。
- `total_epochs`: 总训练轮数。
- `total_steps`: 总训练步数（可选）。若指定，则 `total_epochs` 不生效。

### Explorer 输入

定义 explorer 用于训练和评估的数据集。

```yaml
buffer:
  explorer_input:
    taskset:
      name: countdown_train
      storage_type: file
      path: ${oc.env:TRINITY_TASKSET_PATH}
      split: train
      format:
        prompt_key: 'question'
        response_key: 'answer'
      rollout_args:
        temperature: 1.0
      default_workflow_type: 'math_workflow'
      default_reward_fn_type: 'countdown_reward'

    eval_tasksets:
    - name: countdown_eval
      storage_type: file
      path: ${oc.env:TRINITY_TASKSET_PATH}
      split: test
      repeat_times: 1
      format:
        prompt_type: `plaintext`
        prompt_key: 'question'
        response_key: 'answer'
      rollout_args:
        temperature: 0.1
      default_workflow_type: 'math_workflow'
      default_reward_fn_type: 'countdown_reward'
    ...
```

- `buffer.explorer_input.taskset`: 用于训练探索策略的任务数据集。
- `buffer.explorer_input.eval_taskset`: 用于评估的任务数据集列表。

每个任务数据集的配置定义如下：

- `name`: 数据集名称。该名称将用作 Ray actor 的名称，因此必须唯一。
- `storage_type`: 数据集的存储方式。选项：`file`、`queue`、`sql`。
  - `file`: 数据集存储在 `jsonl`/`parquet` 文件中。数据文件组织需符合 HuggingFace 标准。*建议大多数情况下使用此存储类型。*
  - `queue`: 数据集存储在队列中。队列是一个简单的 FIFO 队列，用于存储任务数据集。*除非你明确了解其用途，否则不要为此类数据集使用此类型。*
  - `sql`: 数据集存储在 SQL 数据库中。*此类型尚不稳定，将在未来版本中优化。*
- `path`: 任务数据集的路径。
  - 对于 `file` 类型，路径指向包含任务数据集文件的目录。
  - 对于 `queue` 类型，路径为可选。可通过在此指定 sqlite 数据库路径来备份队列数据。
  - 对于 `sql` 类型，路径指向 sqlite 数据库文件。
- `subset_name`: 任务数据集的子集名称。默认为 `None`。
- `split`: 任务数据集的划分。默认为 `train`。
- `repeat_times`: 为一个任务生成的 rollout 数量。若未设置，则自动设为 `algorithm.repeat_times`（`taskset`）或 `1`（`eval_tasksets`）。
- `rollout_args`: rollout 参数。
  - `temperature`: 采样温度。
- `default_workflow_type`: 应用于该数据集的工作流逻辑类型。若未指定，则使用 `buffer.default_workflow_type`。
- `default_reward_fn_type`: 探索过程中使用的奖励函数。若未指定，则使用 `buffer.default_reward_fn_type`。
- `workflow_args`: 用于补充数据集级别参数的字典。

### Trainer 输入

定义 trainer 使用的 experience buffer 和可选的辅助数据集。

```yaml
buffer:
  ...
  trainer_input:
    experience_buffer:
      name: countdown_buffer
      storage_type: queue
      path: sqlite:///countdown_buffer.db
      max_read_timeout: 1800

    auxiliary_buffers:
      sft_dataset:
        name: sft_dataset
        storage_type: file
        path: ${oc.env:TRINITY_SFT_DATASET_PATH}
        format:
          prompt_key: 'question'
          response_key: 'answer'
      other_buffer:
        ...
```

- `experience_buffer`: 它是 Trainer 的输入，也是 Explorer 的输出。即使在 explore 模式下也必须定义。
  - `name`: experience buffer 的名称。该名称将用作 Ray actor 的名称，因此必须唯一。
  - `storage_type`: experience buffer 的存储类型。
    - `queue`: experience 数据存储在队列中。大多数使用场景推荐此类型。
    - `sql`: experience 数据存储在 SQL 数据库中。
    - `file`: experience 数据存储在 JSON 文件中。仅建议在 `explore` 模式下用于调试目的。
  - `path`: experience buffer 的路径。
    - 对于 `queue` 类型，此字段可选。可在此指定 SQLite 数据库或 JSON 文件路径以备份队列数据。
    - 对于 `file` 类型，路径指向包含数据集文件的目录。
    - 对于 `sql` 类型，路径指向 SQLite 数据库文件。
  - `format`: 定义数据集中 prompt 和 response 的键。
    - `prompt_type`: 指定数据集中 prompt 的类型。目前支持 `plaintext`、`messages`。
      - `plaintext`: prompt 为 string 格式。
      - `messages`: prompt 为消息列表。
    - `prompt_key`: 指定数据集中包含用户 prompt 的列。仅适用于 `plaintext`。
    - `response_key`: 指定数据集中包含 response 的列。仅适用于 `plaintext`。
    - `system_prompt_key`: 指定数据集中包含 system prompt 的列。仅适用于 `plaintext`。
    - `system_prompt`: 以字符串形式指定 system prompt。优先级低于 `system_prompt_key`。仅适用于 `plaintext`。
    - `messages_key`: 指定数据集中包含 messages 的列。仅适用于 `messages`。
    - `tools_key`: 指定数据集中包含 tools 的列。支持 `plaintext` 和 `messages`，但 tool 数据应组织为 dict 列表。
    - `chosen_key`: 指定数据集中包含 DPO chosen 数据的列。支持 `plaintext` 和 `messages`，数据类型应与 prompt 类型一致。
    - `rejected_key`: 类似于 `chosen_key`，但指定 DPO rejected 数据的列。
    - `enable_concatenated_multi_turn`: 启用拼接的多轮 SFT 数据预处理。仅适用于 `messages`，且仅在 SFT 算法中生效。
    - `chat_template`: 以字符串形式指定 chat template。若未提供，则使用 `model.custom_chat_template`。
  - `max_read_timeout`: 读取新 experience 数据的最大等待时间（秒）。若超时，则直接返回不完整批次。仅当 `storage_type` 为 `queue` 时生效。默认为 1800 秒（30 分钟）。
  - `use_priority_queue`: 仅当 `storage_type` 为 `queue` 时生效。若设为 `True`，队列为优先级队列，允许优先处理某些 experience。默认为 `False`。
  - `reuse_cooldown_time`: 仅当 `storage_type` 为 `queue` 且 `use_priority_queue` 为 `True` 时生效。若设置，指定 experience 重用的冷却时间（秒）。若未指定，默认为 `None`，表示 experience 不可被重复使用。
- `auxiliary_buffers`: trainer 使用的可选缓冲区。为字典结构，每个键为 buffer 名称，值为 buffer 配置。每个 buffer 配置与 `experience_buffer` 类似。

---

## Explorer 配置

控制 rollout 模型和工作流执行。

```yaml
explorer:
  name: explorer
  runner_per_model: 8
  max_timeout: 900
  max_retry_times: 2
  env_vars: {}
  rollout_model:
    engine_type: vllm
    engine_num: 1
    tensor_parallel_size: 1
    enable_history: False
  auxiliary_models:
  - model_path: Qwen/Qwen2.5-7B-Instruct
    tensor_parallel_size: 1
  eval_interval: 100
  eval_on_startup: True
```

- `name`: explorer 的名称。该名称将用作 Ray actor 的名称，因此必须唯一。
- `runner_per_model`: 每个 rollout 模型的并行工作流执行器数量。
- `max_timeout`: 工作流完成的最大时间（秒）。
- `max_retry_times`: 工作流的最大重试次数。
- `env_vars`: 为每个工作流执行器设置的环境变量。
- `rollout_model.engine_type`: 推理引擎类型。支持 `vllm_async` 和 `vllm`，二者的含义相同，都使用了异步引擎。后续版本会只保留 `vllm`。
- `rollout_model.engine_num`: 推理引擎数量。
- `rollout_model.tensor_parallel_size`: 张量并行度。
- `rollout_model.enable_history`: 是否启用模型调用历史记录。若设为 `True`，模型包装器会自动记录模型调用返回的 experience。请定期通过 `extract_experience_from_history` 提取历史，以避免内存溢出。默认为 `False`。
- `auxiliary_models`: 用于自定义工作流的额外模型。
- `eval_interval`: 模型评估的间隔（以步为单位）。
- `eval_on_startup`: 是否在启动时评估模型。更准确地说，是在第 0 步使用原始模型评估，因此重启时不会触发。

---

## Synchronizer 配置

控制 trainer 和 explorer 之间的模型权重同步方式。详细介绍可以参考 {ref}`Synchronizer 介绍 <Synchronizer>`。

```yaml
synchronizer:
  sync_method: 'nccl'
  sync_interval: 10
  sync_offset: 0
  sync_timeout: 1200
```

- `sync_method`: 同步方法。选项：
  - `nccl`: 使用 NCCL 进行快速同步。仅 `both` 模式支持。
  - `checkpoint`: 从磁盘加载最新模型。`train`、`explore` 或 `bench` 模式均支持。
- `sync_interval`: trainer 和 explorer 之间模型权重同步的间隔（步）。
- `sync_offset`: trainer 和 explorer 之间模型权重同步的偏移量（步）。explorer 可在 trainer 开始训练前运行 `sync_offset` 步。
- `sync_timeout`: 同步超时时间。

---

## Trainer 配置

指定 trainer 的后端和行为。

```yaml
trainer:
  name: trainer
  trainer_type: 'verl'
  save_interval: 100
  total_steps: 1000
  trainer_config: null
  trainer_config_path: ''
```

- `name`: trainer 的名称。该名称将用作 Ray actor 的名称，因此必须唯一。
- `trainer_type`: trainer 后端实现。目前仅支持 `verl`。
- `save_interval`: 保存模型检查点的频率（步）。
- `total_steps`: 总训练步数。
- `trainer_config`: 内联提供的 trainer 配置。
- `trainer_config_path`: trainer 配置文件的路径。`trainer_config_path` 和 `trainer_config` 只能指定其一。

---

## 服务配置

配置 Trinity-RFT 使用的服务。目前仅支持 Data Juicer 服务。

```yaml
service:
  data_juicer:
    server_url: 'http://127.0.0.1:5005'
    auto_start: true
    port: 5005
```

- `server_url`: data juicer 服务的 URL。
- `auto_start`: 是否自动启动 data juicer 服务。
- `port`: 当 `auto_start` 为 true 时，Data Juicer 服务使用的端口。

---

## DataProcessor 配置

配置 task / experience 处理流水线，详情请参考 {ref}`数据处理 <Data Processing>` 部分。

```yaml
data_processor:
  task_pipeline:
    num_process: 32
    operators:
      - name: "llm_difficulty_score_filter"
        args:
          api_or_hf_model: "qwen2.5-7b-instruct"
          min_score: 0.0
          input_keys: ["question", "answer"]
          field_names: ["Question", "Answer"]
    inputs:  # 输出将自动设置为 explorer 输入
      - ${oc.env:TRINITY_TASKSET_PATH}
    target_fields: ["question", "answer"]
  experience_pipeline:
    operators:
      - name: data_juicer
        args:
          config_path: 'examples/grpo_gsm8k_experience_pipeline/dj_scoring_exp.yaml'
      - name: reward_shaping_mapper
        args:
          reward_shaping_configs:
            - stats_key: 'llm_quality_score'
              op_type: ADD
              weight: 1.0
```

--

## 日志配置

Ray actor 日志配置。

```yaml
log:
  level: INFO
  group_by_node: False
```

- `level`: 日志级别（支持 `DEBUG`、`INFO`、`WARNING`、`ERROR`）。
- `group_by_node`: 是否按节点 IP 分组日志。若设为 `True`，actor 的日志将保存到 `<checkpoint_root_dir>/<project>/<name>/log/<node_ip>/<actor_name>.log`，否则保存到 `<checkpoint_root_dir>/<project>/<name>/log/<actor_name>.log`。

---

## 阶段配置

对于多阶段训练，可以在 `stages` 字段中定义多个阶段。每个阶段可以有自己的 `algorithm`、`buffer` 和其他配置。如果某个参数未在阶段中指定，则继承自全局配置。多个阶段将按定义顺序依次执行。

```yaml
stages:
  - stage_name: sft_warmup
    mode: train
    algorithm:
      algorithm_type: sft
    buffer:
      train_batch_size: 64
      total_steps: 100
      trainer_input:
        experience_buffer:
          name: sft_buffer
          path: ${oc.env:TRINITY_DATASET_PATH}
  - stage_name: rft
```

- `stage_name`: 阶段名称。应唯一，并将用作实验名称的后缀。
- `mode`: 该阶段 Trinity-RFT 的运行模式。若未指定，则继承自全局 `mode`。
- `algorithm`: 该阶段的算法配置。若未指定，则继承自全局 `algorithm`。
- `buffer`: 该阶段的缓冲区配置。若未指定，则继承自全局 `buffer`。
- `explorer`: 该阶段的 explorer 配置。若未指定，则继承自全局 `explorer`。
- `trainer`: 该阶段的 trainer 配置。若未指定，则继承自全局 `trainer`。

---

## veRL Trainer 配置（高级）

针对使用 `verl` trainer 后端的高级用户。包含 actor/critic 模型、优化器参数和训练循环的细粒度设置。

> 有关完整参数含义，请参考 [veRL 文档](https://verl.readthedocs.io/en/latest/examples/config.html)。

```yaml
actor_rollout_ref:
  hybrid_engine: True
  model:
    external_lib: null
    override_config: { }
    enable_gradient_checkpointing: True
    use_remove_padding: True
    use_fused_kernels: False
    fused_kernel_options:
      impl_backend: None
  actor:
    strategy: fsdp  # 这是为了向后兼容
    # ppo_micro_batch_size: 8 # 将被弃用，使用 ppo_micro_batch_size_per_gpu
    ppo_micro_batch_size_per_gpu: 4
    use_dynamic_bsz: True
    ppo_max_token_len_per_gpu: 16384 # n * ${data.max_model_len}
    grad_clip: 1.0
    ppo_epochs: 1
    shuffle: False
    ulysses_sequence_parallel_size: 1 # sp size
    entropy_from_logits_with_chunking: false
    entropy_checkpointing: false
    checkpoint:
      load_contents: ['model', 'optimizer', 'extra']
      save_contents: ['model', 'optimizer', 'extra']
    optim:
      lr: 1e-6
      lr_warmup_steps_ratio: 0.  # 总步数将在运行时注入
      # min_lr_ratio: null   # 仅在余弦预热时有用
      warmup_style: constant  # 从 constant/cosine 中选择
      total_training_steps: -1  # 必须由程序覆盖
    fsdp_config:
      wrap_policy:
        # transformer_layer_cls_to_wrap: None
        min_num_params: 0
      param_offload: False
      optimizer_offload: False
      fsdp_size: -1
      forward_prefetch: False
  ref:
    fsdp_config:
      param_offload: False
      wrap_policy:
        # transformer_layer_cls_to_wrap: None
        min_num_params: 0
      fsdp_size: -1
      forward_prefetch: False
    # log_prob_micro_batch_size: 4 # 将被弃用，使用 log_prob_micro_batch_size_per_gpu
    log_prob_micro_batch_size_per_gpu: 4
    log_prob_use_dynamic_bsz: ${actor_rollout_ref.actor.use_dynamic_bsz}
    log_prob_max_token_len_per_gpu: ${actor_rollout_ref.actor.ppo_max_token_len_per_gpu}
    ulysses_sequence_parallel_size: ${actor_rollout_ref.actor.ulysses_sequence_parallel_size} # sp size
    entropy_from_logits_with_chunking: ${actor_rollout_ref.actor.entropy_from_logits_with_chunking}
    entropy_checkpointing: ${actor_rollout_ref.actor.entropy_checkpointing}

critic:
  strategy: fsdp
  optim:
    lr: 1e-5
    lr_warmup_steps_ratio: 0.  # 总步数将在运行时注入
    # min_lr_ratio: null   # 仅在余弦预热时有用
    warmup_style: constant  # 从 constant/cosine 中选择
    total_training_steps: -1  # 必须由程序覆盖
  model:
    override_config: { }
    external_lib: ${actor_rollout_ref.model.external_lib}
    enable_gradient_checkpointing: True
    use_remove_padding: False
    fsdp_config:
      param_offload: False
      optimizer_offload: False
      wrap_policy:
        # transformer_layer_cls_to_wrap: None
        min_num_params: 0
      fsdp_size: -1
      forward_prefetch: False
  ppo_micro_batch_size_per_gpu: 8
  forward_micro_batch_size_per_gpu: ${critic.ppo_micro_batch_size_per_gpu}
  use_dynamic_bsz: ${actor_rollout_ref.actor.use_dynamic_bsz}
  ppo_max_token_len_per_gpu: 32768 # (${actor_rollout_ref.actor.ppo_max_token_len_per_gpu}) * 2
  forward_max_token_len_per_gpu: ${critic.ppo_max_token_len_per_gpu}
  ulysses_sequence_parallel_size: 1 # sp size
  ppo_epochs: ${actor_rollout_ref.actor.ppo_epochs}
  shuffle: ${actor_rollout_ref.actor.shuffle}
  grad_clip: 1.0
  cliprange_value: 0.5

trainer:
  balance_batch: True
  # total_training_steps: null
  resume_mode: auto
  resume_from_path: ""
  critic_warmup: 0
  default_hdfs_dir: null
  remove_previous_ckpt_in_save: False
  del_local_ckpt_after_load: False
  val_before_train: False
  max_actor_ckpt_to_keep: 5
  max_critic_ckpt_to_keep: 5
```


- `actor_rollout_ref.model.enable_gradient_checkpointing`: 是否启用梯度检查点，以减少 GPU 内存使用。
- `actor_rollout_ref.model.use_remove_padding`: 是否移除填充 token，以减少训练时间。
- `actor_rollout_ref.model.use_fused_kernels`: 是否使用自定义融合内核（如 FlashAttention、融合 MLP）。
- `actor_rollout_ref.model.fused_kernel_options.impl_backend`: 融合内核的实现后端。若 `use_fused_kernels` 为 true，则使用此选项。可选值："triton" 或 "torch"。
- `actor_rollout_ref.actor.use_dynamic_bsz`: 是否重新组织批数据，具体是拼接较短数据以减少实际训练过程中的批大小。
- `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`: 单个 GPU 一次前向传播的批大小。
- `actor_rollout_ref.actor.ulysses_sequence_parallel_size`: Ulysses 序列并行大小。
- `actor_rollout_ref.actor.entropy_from_logits_with_chunking`: 使用分块计算熵以降低内存峰值。
- `actor_rollout_ref.actor.entropy_checkpointing`: 重新计算熵。
- `actor_rollout_ref.actor.checkpoint`: 要加载和保存的内容。使用 'hf_model' 可将整个模型保存为 hf 格式；目前仅使用分片检查点以节省空间。
- `actor_rollout_ref.actor.optim.lr`: actor 模型的学习率。
- `actor_rollout_ref.actor.optim.lr_warmup_steps_ratio`: 学习率预热步数比例。
- `actor_rollout_ref.actor.optim.warmup_style`: 学习率预热方式。
- `actor_rollout_ref.actor.optim.total_training_steps`: actor 模型的总训练步数。
- `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu`: 单个 GPU 在一次参考模型前向传播中的批大小。

- `critic.model.enable_gradient_checkpointing`: 是否启用梯度检查点，以减少 GPU 内存使用。
- `critic.model.use_remove_padding`: 是否移除填充 token，以减少训练时间。
- `critic.optim.lr`: critic 模型的学习率。
- `critic.optim.lr_warmup_steps_ratio`: 学习率预热步数比例。
- `critic.optim.warmup_style`: 学习率预热风格。
- `critic.optim.total_training_steps`: critic 模型的总训练步数。
- `critic.ppo_micro_batch_size_per_gpu`: 单个 GPU 在一次 critic 模型前向传播中的批大小。
- `critic.ulysses_sequence_parallel_size`: Ulysses 序列并行大小。
- `critic.grad_clip`: critic 模型训练的梯度裁剪。
- `critic.cliprange_value`: 用于计算值损失。

- `trainer.balance_batch`: 训练期间是否在 GPU 间平衡批大小。
- `trainer.resume_mode`: 训练的恢复模式。支持 `disable`、`auto` 和 `resume_path`。默认是`auto`，即查找最后一个检查点以恢复；若找不到，则从头开始训练。
- `trainer.resume_from_path`: 恢复路径。
- `trainer.critic_warmup`: 在实际策略学习前训练 critic 模型的步数。
- `trainer.default_hdfs_dir`: 保存检查点的默认 HDFS 目录。
- `trainer.remove_previous_ckpt_in_save`: 保存时是否删除之前的检查点。
- `trainer.del_local_ckpt_after_load`: 加载后是否删除本地检查点。
- `trainer.max_actor_ckpt_to_keep`: 最多保留的 actor 检查点数量。
- `trainer.max_critic_ckpt_to_keep`: 最多保留的 critic 检查点数量。
