"""SQLAlchemy models for different data."""

from typing import Dict, Optional, Tuple

from sqlalchemy import JSON, Column, Float, Integer, LargeBinary, Text, create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool

from trinity.common.experience import Experience
from trinity.utils.log import get_logger
from trinity.utils.registry import Registry

SQL_SCHEMA = Registry("sql_schema")

Base = declarative_base()


@SQL_SCHEMA.register_module("task")
class TaskModel(Base):  # type: ignore
    """Model for storing tasks in SQLAlchemy."""

    __abstract__ = True

    id = Column(Integer, primary_key=True, autoincrement=True)
    raw_task = Column(JSON, nullable=False)

    @classmethod
    def from_dict(cls, dict: Dict):
        return cls(raw_task=dict)


@SQL_SCHEMA.register_module("experience")
class ExperienceModel(Base):  # type: ignore
    """SQLAlchemy model for Experience."""

    __abstract__ = True

    id = Column(Integer, primary_key=True, autoincrement=True)
    # for single turn
    prompt = Column(Text, nullable=True)
    response = Column(Text, nullable=True)
    # for multi turn
    message_list = Column(JSON, nullable=True)
    reward = Column(Float, nullable=True)
    # serialized experience object
    experience_bytes = Column(LargeBinary, nullable=True)
    consumed = Column(Integer, default=0, index=True)

    def to_experience(self) -> Experience:
        """Load the experience from the database."""
        return Experience.deserialize(self.experience_bytes)

    @classmethod
    def from_experience(cls, experience: Experience):
        """Save the experience to database."""
        return cls(
            experience_bytes=experience.serialize(),
            reward=experience.reward,
            prompt=experience.prompt_text,
            response=experience.response_text,
            message_list=experience.messages,
        )


@SQL_SCHEMA.register_module("sft")
class SFTDataModel(Base):  # type: ignore
    """SQLAlchemy model for SFT data."""

    __abstract__ = True

    id = Column(Integer, primary_key=True, autoincrement=True)
    message_list = Column(JSON, nullable=True)
    experience_bytes = Column(LargeBinary, nullable=True)

    def to_experience(self) -> Experience:
        """Load the experience from the database."""
        return Experience.deserialize(self.experience_bytes)

    @classmethod
    def from_experience(cls, experience: Experience):
        """Save the experience to database."""
        return cls(
            experience_bytes=experience.serialize(),
            message_list=experience.messages,
        )


@SQL_SCHEMA.register_module("dpo")
class DPODataModel(Base):  # type: ignore
    """SQLAlchemy model for DPO data."""

    __abstract__ = True

    id = Column(Integer, primary_key=True, autoincrement=True)
    chosen_message_list = Column(JSON, nullable=True)
    rejected_message_list = Column(JSON, nullable=True)
    experience_bytes = Column(LargeBinary, nullable=True)

    def to_experience(self) -> Experience:
        """Load the experience from the database."""
        return Experience.deserialize(self.experience_bytes)

    @classmethod
    def from_experience(cls, experience: Experience):
        """Save the experience to database."""
        return cls(
            experience_bytes=experience.serialize(),
            chosen_message_list=experience.chosen_messages,
            rejected_message_list=experience.rejected_messages,
        )


def init_engine(db_url: str, table_name, schema_type: Optional[str]) -> Tuple:
    """Get the sqlalchemy engine."""
    logger = get_logger(__name__)
    engine = create_engine(db_url, poolclass=NullPool)

    if schema_type is None:
        schema_type = "task"

    base_class = SQL_SCHEMA.get(schema_type)

    table_attrs = {
        "__tablename__": table_name,
        "__abstract__": False,
        "__table_args__": {"keep_existing": True},
    }
    table_cls = type(table_name, (base_class,), table_attrs)

    try:
        Base.metadata.create_all(engine, checkfirst=True)
    except OperationalError:
        logger.warning(f"Failed to create table {table_name}, assuming it already exists.")

    return engine, table_cls
