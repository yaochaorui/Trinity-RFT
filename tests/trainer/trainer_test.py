"""Tests for trainer."""

import multiprocessing
import os
import shutil
import time
import unittest
from copy import deepcopy
from datetime import datetime

import ray
from parameterized import parameterized_class

from tests.tools import (
    RayUnittestBase,
    TensorBoardParser,
    get_checkpoint_path,
    get_model_path,
    get_template_config,
    get_unittest_dataset_config,
)
from trinity.cli.launcher import bench, both, explore, train
from trinity.common.config import Config, StorageConfig
from trinity.common.constants import (
    LOG_DIR_ENV_VAR,
    LOG_LEVEL_ENV_VAR,
    StorageType,
    SyncMethod,
    SyncStyle,
)
from trinity.common.models.utils import get_checkpoint_dir_with_step_num
from trinity.manager.manager import CacheManager


class BaseTrainerCase(RayUnittestBase):
    def setUp(self):
        ray.init(ignore_reinit_error=True)
        self.config = get_template_config()
        self.config.buffer.total_epochs = 2
        self.config.buffer.batch_size = 4
        self.config.model.model_path = get_model_path()
        self.config.explorer.rollout_model.engine_type = "vllm_async"
        self.config.algorithm.repeat_times = 3
        self.config.project = "Trainer-unittest"
        self.config.name = f"trainer-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.config.monitor.monitor_type = "tensorboard"
        self.config.checkpoint_root_dir = get_checkpoint_path()
        self.config.synchronizer.sync_interval = 2
        self.config.synchronizer.sync_method = SyncMethod.NCCL
        self.config.explorer.eval_interval = 4


@parameterized_class(
    ("strategy",),
    [
        ("fsdp",),
        ("megatron",),
    ],
)
class TestTrainerCountdown(BaseTrainerCase):
    def test_trainer(self):
        """Test the both and bench mode."""
        # test both mode
        self.config.explorer.rollout_model.use_v1 = False
        self.config.buffer.explorer_input.taskset = get_unittest_dataset_config("countdown")
        self.config.buffer.explorer_input.eval_tasksets.append(
            get_unittest_dataset_config("countdown", "test")
        )
        self.config.buffer.explorer_input.eval_tasksets.append(
            get_unittest_dataset_config("copy_countdown", "test")
        )
        self.config.trainer.save_interval = 4
        self.config.check_and_update()
        _trainer_config = self.config.trainer.trainer_config
        if self.strategy == "megatron":
            _trainer_config.actor_rollout_ref.actor.strategy = "megatron"
            _trainer_config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size = 2
            _trainer_config.actor_rollout_ref.ref.megatron.tensor_model_parallel_size = 2
            _trainer_config.critic.strategy = "megatron"
            _trainer_config.critic.megatron.tensor_model_parallel_size = 2
        _trainer_config.trainer.max_actor_ckpt_to_keep = 2
        _trainer_config.trainer.max_critic_ckpt_to_keep = 2
        both(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        rollout_metrics = parser.metric_list("rollout")
        self.assertTrue(len(rollout_metrics) > 0)
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 8)
        eval_metrics = parser.metric_list("eval")
        self.assertTrue(len(eval_metrics) > 0)
        self.assertEqual(parser.metric_max_step(eval_metrics[0]), 8)
        actor_metrics = parser.metric_list("actor")
        self.assertTrue(len(actor_metrics) > 0)
        self.assertEqual(parser.metric_max_step(actor_metrics[0]), 8)
        actor_kl_metrics = parser.metric_list("actor/kl")
        self.assertTrue(len(actor_kl_metrics) > 0)
        critic_kl_metrics = parser.metric_list("critic/kl")
        self.assertTrue(len(critic_kl_metrics) > 0)
        response_metrics = parser.metric_list("response_length")
        self.assertTrue(len(response_metrics) > 0)
        self.assertEqual(parser.metric_max_step(response_metrics[0]), 8)
        ray.shutdown(_exiting_interpreter=True)
        # check checkpoint
        checkpoint_step_4, _ = get_checkpoint_dir_with_step_num(
            checkpoint_root_path=self.config.checkpoint_job_dir,
            trainer_type=self.config.trainer.trainer_type,
            step_num=4,
        )
        # check save lastest checkpoint
        checkpoint_step_8, step_num = get_checkpoint_dir_with_step_num(
            checkpoint_root_path=self.config.checkpoint_job_dir,
            trainer_type=self.config.trainer.trainer_type,
        )
        self.assertTrue(len(os.listdir(os.path.join(checkpoint_step_4, "actor"))) > 0)
        self.assertTrue(len(os.listdir(os.path.join(checkpoint_step_8, "actor"))) > 0)
        self.assertEqual(step_num, 8)
        # TODO: Reinit will fail when using v1 engine, find a way to fix it
        ray.init(ignore_reinit_error=True, namespace=self.config.ray_namespace)
        # test bench mode
        self.config.mode = "bench"
        self.config.synchronizer.sync_method = SyncMethod.CHECKPOINT
        self.config.explorer.bench_on_latest_checkpoint = False
        self.config.check_and_update()
        bench(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        for prefix in ["eval", "bench"]:
            countdown_metrics = parser.metric_list(f"{prefix}/countdown")
            copy_countdown_metrics = parser.metric_list(f"{prefix}/copy_countdown")
            self.assertTrue(len(countdown_metrics) > 0)
            self.assertTrue(len(copy_countdown_metrics) > 0)
            countdown_metric_steps = parser.metric_steps(countdown_metrics[0])
            countdown_copy_metric_steps = parser.metric_steps(copy_countdown_metrics[0])
            self.assertEqual([0, 4, 8], countdown_metric_steps)
            self.assertEqual([0, 4, 8], countdown_copy_metric_steps)

    def tearDown(self):
        # remove dir only when the test passed
        shutil.rmtree(self.config.checkpoint_job_dir)


class TestStepAheadAsyncRL(BaseTrainerCase):
    def test_trainer(self):
        """Test the explore step ahead trainer."""
        # train 4 step, sync_offset=1, sync_interval=2
        # Explorer:
        # | 1 | 2 | 3 |sync| 4 |
        # |---|---|---|sync|---|
        # Trainer:
        #     | 1 | 2 |sync| 3 | 4 |
        #     |---|---|sync|---|---|
        self.config.buffer.total_epochs = 1
        self.config.buffer.explorer_input.taskset = get_unittest_dataset_config("countdown")
        self.config.trainer.save_interval = 4
        self.config.synchronizer.sync_interval = 2
        self.config.synchronizer.sync_offset = 1
        self.config.check_and_update()
        self.config.trainer.trainer_config.trainer.max_actor_ckpt_to_keep = 1
        self.config.trainer.trainer_config.trainer.max_critic_ckpt_to_keep = 1

        both(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        rollout_metrics = parser.metric_list("rollout")
        self.assertTrue(len(rollout_metrics) > 0)
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 4)
        actor_metrics = parser.metric_list("actor")
        self.assertTrue(len(actor_metrics) > 0)
        self.assertEqual(parser.metric_max_step(actor_metrics[0]), 4)
        actor_kl_metrics = parser.metric_list("actor/kl")
        self.assertTrue(len(actor_kl_metrics) > 0)
        critic_kl_metrics = parser.metric_list("critic/kl")
        self.assertTrue(len(critic_kl_metrics) > 0)
        response_metrics = parser.metric_list("response_length")
        self.assertTrue(len(response_metrics) > 0)
        self.assertEqual(parser.metric_max_step(response_metrics[0]), 4)
        ray.shutdown(_exiting_interpreter=True)
        # check checkpoint

        checkpoint_step_4, step_num = get_checkpoint_dir_with_step_num(
            checkpoint_root_path=self.config.checkpoint_job_dir,
            trainer_type=self.config.trainer.trainer_type,
        )
        self.assertEqual(step_num, 4)
        self.assertTrue(os.path.exists(checkpoint_step_4))

    def tearDown(self):
        # remove dir only when the test passed
        shutil.rmtree(self.config.checkpoint_job_dir)


@parameterized_class(
    ("fsdp_strategy", "offloading"),
    [
        ("fsdp", False),
        ("fsdp2", False),
        ("fsdp", True),
        ("fsdp2", True),
    ],
)
class TestTrainerGSM8K(BaseTrainerCase):
    def test_trainer(self):
        """Test GSM8K."""
        # test both mode
        self.config.algorithm.algorithm_type = "grpo"
        self.config.algorithm.repeat_times = 4
        self.config.algorithm.advantage_fn = "grpo"
        self.config.algorithm.advantage_fn_args = {
            "epsilon": 1e-6,
        }
        # self.config.algorithm.repeat_times = 8  # TODO: used for real testing
        # self.config.buffer.batch_size = 96  # TODO: used for real testing
        self.config.buffer.total_epochs = 1
        self.config.buffer.explorer_input.taskset = get_unittest_dataset_config("gsm8k")
        self.config.check_and_update()
        self.config.trainer.trainer_config.trainer.max_actor_ckpt_to_keep = 2
        actor_rollout_ref = self.config.trainer.trainer_config.actor_rollout_ref
        actor_rollout_ref.actor.strategy = self.fsdp_strategy
        actor_rollout_ref.actor.optim.lr = 1e-5
        if self.fsdp_strategy == "fsdp":
            actor_rollout_ref.actor.fsdp_config.param_offload = self.offloading
            actor_rollout_ref.actor.fsdp_config.optimizer_offload = self.offloading
            actor_rollout_ref.ref.fsdp_config.param_offload = self.offloading
            actor_rollout_ref.ref.fsdp_config.optimizer_offload = self.offloading
        else:  # fsdp2
            actor_rollout_ref.actor.fsdp_config.offload_policy = self.offloading
            actor_rollout_ref.ref.fsdp_config.offload_policy = self.offloading
        both(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        rollout_metrics = parser.metric_list("rollout")
        self.assertTrue(len(rollout_metrics) > 0)
        pipeline_metrics = parser.metric_list("pipeline")
        self.assertTrue(len(pipeline_metrics) > 0)
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 4)
        actor_metrics = parser.metric_list("actor")
        self.assertTrue(len(actor_metrics) > 0)
        self.assertEqual(parser.metric_max_step(actor_metrics[0]), 4)
        response_metrics = parser.metric_list("response_length")
        self.assertTrue(len(response_metrics) > 0)
        self.assertEqual(parser.metric_max_step(response_metrics[0]), 4)
        # TODO: used for real testing
        # rewards = parser.metric_values("critic/rewards/mean")
        # self.assertTrue(0.4 < rewards[0] < 0.55)
        # self.assertTrue(0.4 < rewards[1] < 0.55)
        # self.assertTrue(0.6 < rewards[2] < 0.7)
        # self.assertTrue(0.6 < rewards[3] < 0.7)

    def tearDown(self):
        # remove dir only when the test passed
        shutil.rmtree(self.config.checkpoint_job_dir)


class TestTrainerSFTWarmupGSM8K(BaseTrainerCase):
    def test_trainer(self):
        """Test GSM8K With SFT."""
        # test both mode
        self.config.algorithm.algorithm_type = "grpo"
        self.config.algorithm.repeat_times = 4
        self.config.algorithm.advantage_fn = "grpo"
        self.config.algorithm.advantage_fn_args = {}
        self.config.buffer.total_epochs = 1
        self.config.buffer.explorer_input.taskset = get_unittest_dataset_config("gsm8k")
        self.config.synchronizer.sync_interval = 1
        self.config.trainer.save_interval = 8
        # sft data is only enough for 2 steps
        self.config.buffer.trainer_input.sft_warmup_dataset = get_unittest_dataset_config(
            "sft_for_gsm8k"
        )
        self.config.buffer.trainer_input.sft_warmup_steps = 3
        self.config.buffer.trainer_input.experience_buffer = StorageConfig(
            name="test_sql_storage",
            max_read_timeout=20,
            storage_type=StorageType.SQL,
            max_retry_times=10,
        )
        self.config.check_and_update()
        self.config.trainer.trainer_config.trainer.max_actor_ckpt_to_keep = 2
        self.config.trainer.trainer_config.actor_rollout_ref.actor.optim.lr = 1e-5
        both(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        rollout_metrics = parser.metric_list("rollout")
        self.assertTrue(len(rollout_metrics) > 0)
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 7)
        actor_metrics = parser.metric_list("actor")
        self.assertTrue(len(actor_metrics) > 0)
        sft_metrics = parser.metric_list("actor/sft")
        self.assertEqual(parser.metric_max_step(sft_metrics[0]), 3)  # SFT
        self.assertEqual(parser.metric_max_step(actor_metrics[-1]), 7)  # RFT
        response_metrics = parser.metric_list("response_length")
        self.assertTrue(len(response_metrics) > 0)
        self.assertEqual(parser.metric_min_step(response_metrics[0]), 1)
        self.assertEqual(parser.metric_max_step(response_metrics[0]), 7)
        # test save checkpoint when sft finish
        self.assertEqual(
            get_checkpoint_dir_with_step_num(
                checkpoint_root_path=self.config.checkpoint_job_dir, trainer_type="verl", step_num=2
            )[1],
            2,
        )
        # test save checkpoint at last step
        checkpoint_dir, step_num = get_checkpoint_dir_with_step_num(
            checkpoint_root_path=self.config.checkpoint_job_dir,
            trainer_type="verl",
        )
        self.assertEqual(step_num, 7)
        self.assertTrue(len(os.listdir(os.path.join(checkpoint_dir, "actor"))) > 0)

    def tearDown(self):
        # TODO: remove dir only when the test passed
        shutil.rmtree(self.config.checkpoint_job_dir)


class TestTrainerDPO(BaseTrainerCase):
    def test_trainer(self):
        """Test DPO."""
        # test both mode
        self.config.mode = "train"
        self.config.algorithm.algorithm_type = "dpo"
        self.config.algorithm.policy_loss_fn = "dpo"
        self.config.algorithm.policy_loss_fn_args = {}
        self.config.buffer.total_epochs = 2
        self.config.buffer.total_steps = 4  # step has higher priority than epoch
        self.config.synchronizer.sync_interval = 4
        self.config.buffer.train_batch_size = 8
        self.config.buffer.trainer_input.experience_buffer = get_unittest_dataset_config("dpo")
        self.config.check_and_update()
        self.config.trainer.trainer_config.trainer.max_actor_ckpt_to_keep = 2
        self.config.trainer.trainer_config.actor_rollout_ref.actor.optim.lr = 5e-7
        train(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        actor_metrics = parser.metric_list("actor")
        self.assertTrue(len(actor_metrics) > 0)
        self.assertEqual(parser.metric_max_step(actor_metrics[0]), 4)

    def tearDown(self):
        # remove dir only when the test passed
        shutil.rmtree(self.config.checkpoint_job_dir)


class TestTrainerSFT(BaseTrainerCase):
    def test_trainer(self):
        """Test SFT."""
        # test both mode
        self.config.mode = "train"
        self.config.algorithm.algorithm_type = "sft"
        self.config.algorithm.policy_loss_fn = "sft"
        self.config.algorithm.policy_loss_fn_args = {}
        self.config.algorithm.kl_loss_fn = "none"
        self.config.algorithm.entropy_loss_fn = "none"
        self.config.synchronizer.sync_interval = 4
        self.config.buffer.train_batch_size = 4
        self.config.buffer.total_epochs = 2
        self.config.buffer.trainer_input.experience_buffer = get_unittest_dataset_config(
            "sft_for_gsm8k"
        )
        self.config.check_and_update()
        train(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        actor_metrics = parser.metric_list("actor")
        self.assertTrue(len(actor_metrics) > 0)
        self.assertEqual(parser.metric_max_step(actor_metrics[0]), 4)

    def tearDown(self):
        # remove dir only when the test passed
        shutil.rmtree(self.config.checkpoint_job_dir)


class TestTrainerToolsSFT(BaseTrainerCase):
    def test_trainer_tools(self):
        """Test SFT with tools."""
        # test both mode
        self.config.mode = "train"
        self.config.algorithm.algorithm_type = "sft"
        self.config.algorithm.policy_loss_fn = "sft"
        self.config.algorithm.policy_loss_fn_args = {}
        self.config.algorithm.kl_loss_fn = "none"
        self.config.algorithm.entropy_loss_fn = "none"
        self.config.synchronizer.sync_interval = 4
        self.config.buffer.train_batch_size = 4
        self.config.buffer.total_epochs = 4
        self.config.buffer.trainer_input.experience_buffer = get_unittest_dataset_config(
            "sft_with_tools"
        )
        self.config.check_and_update()
        train(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))
        actor_metrics = parser.metric_list("actor")
        self.assertTrue(len(actor_metrics) > 0)
        self.assertEqual(parser.metric_max_step(actor_metrics[0]), 4)

    def tearDown(self):
        # remove dir only when the test passed
        shutil.rmtree(self.config.checkpoint_job_dir)


def run_trainer(config: Config) -> None:
    ray.init(
        namespace=config.ray_namespace,
        runtime_env={
            "env_vars": {
                LOG_DIR_ENV_VAR: config.log.save_dir,
                LOG_LEVEL_ENV_VAR: "INFO",
            }
        },
    )
    train(config)


def run_explorer(config: Config) -> None:
    ray.init(
        namespace=config.ray_namespace,
        runtime_env={
            "env_vars": {
                LOG_DIR_ENV_VAR: config.log.save_dir,
                LOG_LEVEL_ENV_VAR: "INFO",
            }
        },
    )
    explore(config)


@parameterized_class(
    ("use_priority_queue", "strategy"),
    [(False, "fsdp"), (True, "fsdp"), (True, "megatron")],
)
class TestFullyAsyncMode(unittest.TestCase):
    def setUp(self):
        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            multiprocessing.set_start_method("spawn", force=True)

    def test_fully_async_mode(self):
        config = get_template_config()
        config.project = "unittest"
        config.name = f"fully_async_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        config.checkpoint_root_dir = get_checkpoint_path()
        config.buffer.total_epochs = 1
        config.buffer.batch_size = 4
        config.cluster.gpu_per_node = 2
        config.cluster.node_num = 1
        config.model.model_path = get_model_path()
        config.buffer.explorer_input.taskset = get_unittest_dataset_config("countdown")
        config.buffer.trainer_input.experience_buffer = StorageConfig(
            name="exp_buffer",
            storage_type=StorageType.QUEUE,
            use_priority_queue=self.use_priority_queue,
        )
        config.synchronizer.sync_method = SyncMethod.CHECKPOINT
        config.synchronizer.sync_style = SyncStyle.DYNAMIC_BY_EXPLORER
        config.synchronizer.sync_interval = 8
        config.monitor.monitor_type = "tensorboard"
        trainer_config = deepcopy(config)
        trainer_config.mode = "train"
        trainer_config.buffer.train_batch_size = 4
        trainer_config.check_and_update()
        if self.strategy == "megatron":
            _trainer_config = trainer_config.trainer.trainer_config
            _trainer_config.actor_rollout_ref.actor.strategy = "megatron"
            _trainer_config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size = 2
            _trainer_config.actor_rollout_ref.ref.megatron.tensor_model_parallel_size = 2
            _trainer_config.critic.strategy = "megatron"
            _trainer_config.critic.megatron.tensor_model_parallel_size = 2

        explorer1_config = deepcopy(config)
        explorer1_config.trainer = deepcopy(trainer_config.trainer)
        explorer1_config.mode = "explore"
        explorer1_config.explorer.name = "explorer1"
        config.cluster.gpu_per_node = 1
        config.cluster.node_num = 1
        explorer1_config.explorer.rollout_model.engine_num = 1
        explorer1_config.explorer.rollout_model.tensor_parallel_size = 1
        explorer1_config.buffer.trainer_input.experience_buffer = StorageConfig(
            name="exp_buffer",
            storage_type=StorageType.QUEUE,
        )
        explorer2_config = deepcopy(explorer1_config)
        explorer2_config.trainer = deepcopy(trainer_config.trainer)
        explorer1_config.check_and_update()

        trainer_process = multiprocessing.Process(target=run_trainer, args=(trainer_config,))
        trainer_process.start()

        ray.init(ignore_reinit_error=True)
        while True:
            try:
                ray.get_actor("queue-exp_buffer", namespace=trainer_config.ray_namespace)
                break
            except ValueError:
                print("waiting for trainer to start.")
                time.sleep(5)

        explorer_process_1 = multiprocessing.Process(target=run_explorer, args=(explorer1_config,))
        explorer_process_1.start()

        time.sleep(5)
        explorer2_config.explorer.name = "explorer2"
        explorer2_config.check_and_update()
        explorer_process_2 = multiprocessing.Process(target=run_explorer, args=(explorer2_config,))
        explorer_process_2.start()

        explorer_process_1.join()
        explorer_process_2.join()

        # wait for trainer process to finish.
        trainer_process.join(timeout=200)

        # check the tensorboard
        parser = TensorBoardParser(
            os.path.join(trainer_config.monitor.cache_dir, "tensorboard", "trainer")
        )
        actor_metrics = parser.metric_list("actor")
        self.assertEqual(parser.metric_max_step(actor_metrics[0]), 8)
        parser = TensorBoardParser(
            os.path.join(explorer1_config.monitor.cache_dir, "tensorboard", "explorer1")
        )
        rollout_metrics = parser.metric_list("rollout")
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 4)
        parser = TensorBoardParser(
            os.path.join(explorer2_config.monitor.cache_dir, "tensorboard", "explorer2")
        )
        rollout_metrics = parser.metric_list("rollout")
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 4)
        # check the checkpoint
        explorer1_cache = CacheManager(explorer1_config)
        cache = explorer1_cache.load_explorer()
        self.assertEqual(cache["latest_iteration"], 4)
        explorer2_cache = CacheManager(explorer2_config)
        cache = explorer2_cache.load_explorer()
        self.assertEqual(cache["latest_iteration"], 4)
        # check the lastest checkpoint
        self.assertEqual(
            get_checkpoint_dir_with_step_num(
                checkpoint_root_path=explorer1_config.checkpoint_job_dir,
                trainer_type="verl",
            )[1],
            8,
        )
        self.assertEqual(
            get_checkpoint_dir_with_step_num(
                checkpoint_root_path=explorer2_config.checkpoint_job_dir,
                trainer_type="verl",
            )[1],
            8,
        )
        log_files = os.listdir(os.path.join(explorer1_config.checkpoint_job_dir, "log"))
        self.assertTrue("trainer.log" in log_files)
        self.assertTrue("synchronizer.log" in log_files)
        self.assertTrue("explorer1.log" in log_files)
        self.assertTrue("explorer2.log" in log_files)
        self.assertTrue("explorer1_runner_0.log" in log_files)
        self.assertTrue("explorer1_runner_7.log" in log_files)
        self.assertTrue("explorer2_runner_0.log" in log_files)
        self.assertTrue("explorer2_runner_7.log" in log_files)
        self.assertTrue("explorer1_experience_pipeline.log" in log_files)
        self.assertTrue("explorer2_experience_pipeline.log" in log_files)
        files_to_check = ["trainer.log", "synchronizer.log", "explorer1.log", "explorer2.log"]
        for file_name in files_to_check:
            with open(os.path.join(explorer1_config.checkpoint_job_dir, "log", file_name)) as f:
                lines = f.readlines()
                self.assertTrue(len(lines) > 0)
        ray.shutdown()

    def tearDown(self):
        checkpoint_path = get_checkpoint_path()
        shutil.rmtree(os.path.join(checkpoint_path, "unittest"))


class TestTrainerMIX(BaseTrainerCase):
    def test_trainer(self):
        """Test MIX algorithm."""
        # gsm8k has 16 tasks, sft_for_gsm8k has 8 tasks
        # total 4 steps, each step: read 4 tasks from gsm8k, 16 tasks from sft_for_gsm8k
        self.config.algorithm.algorithm_type = "mix"
        self.config.algorithm.repeat_times = 4
        self.config.algorithm.sample_strategy = "mix"
        self.config.algorithm.advantage_fn = "grpo"
        self.config.algorithm.sample_strategy_args = {"expert_data_ratio": 0.5}  # rft=4*4 : sft=16
        self.config.algorithm.policy_loss_fn = "mix"
        self.config.buffer.batch_size = 4
        self.config.buffer.train_batch_size = 32
        self.config.buffer.total_epochs = 1
        self.config.buffer.explorer_input.taskset = get_unittest_dataset_config("gsm8k")
        self.config.synchronizer.sync_interval = 1
        self.config.trainer.save_interval = 1
        self.config.buffer.trainer_input.sft_warmup_dataset = get_unittest_dataset_config(
            "sft_for_gsm8k"
        )
        self.config.buffer.trainer_input.sft_warmup_dataset.total_epochs = 8  # test this works
        self.config.check_and_update()
        self.config.buffer.trainer_input.experience_buffer.max_read_timeout = 20
        self.config.trainer.trainer_config.trainer.max_actor_ckpt_to_keep = 2
        both(self.config)
        parser = TensorBoardParser(os.path.join(self.config.monitor.cache_dir, "tensorboard"))

        # test rollout metrics
        rollout_metrics = parser.metric_list("rollout")
        self.assertTrue(len(rollout_metrics) > 0)
        self.assertEqual(parser.metric_max_step(rollout_metrics[0]), 4)
        self.assertEqual(
            parser.metric_values("pipeline/experience_count")[1], 16
        )  # 16 rft experiences
        # test actor metrics
        actor_metrics = parser.metric_list("actor")
        self.assertTrue(len(actor_metrics) > 0)
        expert_metrics = parser.metric_list("actor/expert/")
        self.assertEqual(parser.metric_max_step(expert_metrics[0]), 4)  # SFT
        usual_metrics = parser.metric_list("actor/usual/")
        self.assertEqual(parser.metric_max_step(usual_metrics[0]), 4)  # RFT
        response_metrics = parser.metric_list("response_length")
        self.assertTrue(len(response_metrics) > 0)
        self.assertEqual(parser.metric_min_step(response_metrics[0]), 1)
        self.assertEqual(parser.metric_max_step(response_metrics[0]), 4)
        # test save checkpoint at last step
        checkpoint_dir, step_num = get_checkpoint_dir_with_step_num(
            checkpoint_root_path=self.config.checkpoint_job_dir,
            trainer_type="verl",
        )
        self.assertEqual(step_num, 4)
        self.assertTrue(len(os.listdir(os.path.join(checkpoint_dir, "actor"))) > 0)

    def tearDown(self):
        shutil.rmtree(self.config.checkpoint_job_dir)
