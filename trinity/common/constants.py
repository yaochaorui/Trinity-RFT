# -*- coding: utf-8 -*-
"""Constants."""
from enum import Enum, EnumMeta

# names

EXPLORER_NAME = "explorer"
TRAINER_NAME = "trainer"

ROLLOUT_WEIGHT_SYNC_GROUP_NAME = "rollout_weight_sync"

# trinity env var names
PLUGIN_DIRS_ENV_VAR = "TRINITY_PLUGIN_DIRS"
LOG_DIR_ENV_VAR = "TRINITY_LOG_DIR"  # log dir
LOG_LEVEL_ENV_VAR = "TRINITY_LOG_LEVEL"  # global log level
LOG_NODE_IP_ENV_VAR = "TRINITY_LOG_NODE_IP"  # whether to organize logs by node IP


# constants

MAX_MODEL_LEN = 4096


# enumerate types


class CaseInsensitiveEnumMeta(EnumMeta):
    def __getitem__(cls, name):
        return super().__getitem__(name.upper())

    def __getattr__(cls, name):
        if not name.startswith("_"):
            return cls[name.upper()]
        return super().__getattr__(name)

    def __call__(cls, value, *args, **kwargs):
        return super().__call__(value.lower(), *args, **kwargs)


class CaseInsensitiveEnum(Enum, metaclass=CaseInsensitiveEnumMeta):
    pass


class PromptType(CaseInsensitiveEnum):
    """Prompt Type."""

    MESSAGES = "messages"  # a list of message dict
    PLAINTEXT = "plaintext"  # user prompt text and assistant response text


class StorageType(CaseInsensitiveEnum):
    """Storage Type."""

    SQL = "sql"
    QUEUE = "queue"
    FILE = "file"


class SyncMethodEnumMeta(CaseInsensitiveEnumMeta):
    def __call__(cls, value, *args, **kwargs):
        if value == "online":
            value = "nccl"
        elif value == "offline":
            value = "checkpoint"
        try:
            return super().__call__(value, *args, **kwargs)
        except Exception:
            raise ValueError(f"Invalid SyncMethod: {value}")


class SyncMethod(CaseInsensitiveEnum, metaclass=SyncMethodEnumMeta):
    """Sync Method."""

    NCCL = "nccl"
    CHECKPOINT = "checkpoint"
    MEMORY = "memory"


class RunningStatus(Enum):
    """Running status of explorer and trainer."""

    RUNNING = "running"
    REQUIRE_SYNC = "require_sync"
    WAITING_SYNC = "waiting_sync"
    STOPPED = "stopped"


class OpType(Enum):
    """Operator type for reward shaping."""

    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"


class SyncStyle(CaseInsensitiveEnum):
    FIXED = "fixed"
    DYNAMIC_BY_TRAINER = "dynamic_by_trainer"
    DYNAMIC_BY_EXPLORER = "dynamic_by_explorer"
