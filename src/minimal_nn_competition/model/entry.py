"""Submission metadata and checkpoint helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)

ARCHITECTURES = ("linear", "mlp")
ACTIVATIONS = ("relu", "identity")


class SubmissionError(ValueError):
    """Raised when a submission is missing or malformed."""


class SubmissionConfig(BaseModel):
    """Validated metadata required to evaluate a state-dict submission."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    benchmark: str = Field(min_length=1)
    name: str = Field(min_length=1)
    author: str = Field(min_length=1)
    architecture: str
    input_size: StrictInt = Field(gt=0)
    output_size: StrictInt = Field(gt=0)
    hidden_sizes: tuple[StrictInt, ...] = ()
    activation: str = "relu"
    normalisation: StrictBool = False
    input_indices: tuple[StrictInt, ...] | StrictBool | None = None

    @field_validator("architecture")
    @classmethod
    def _validate_architecture(cls, value: str) -> str:
        if value not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {', '.join(ARCHITECTURES)}")
        return value

    @field_validator("hidden_sizes")
    @classmethod
    def _validate_hidden_sizes(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(size <= 0 for size in value):
            raise ValueError("hidden_sizes must be a list of positive integers")
        return value

    @field_validator("activation")
    @classmethod
    def _validate_activation(cls, value: str) -> str:
        if value not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {', '.join(ACTIVATIONS)}")
        return value

    @field_validator("input_indices")
    @classmethod
    def _validate_input_indices(
        cls, value: tuple[int, ...] | bool | None
    ) -> tuple[int, ...] | bool | None:
        if value is None or value is True:
            return value
        if value is False:
            raise ValueError("input_indices must be true or a list of feature indices")
        if not value:
            raise ValueError("input_indices must not be empty")
        if any(index < 0 for index in value):
            raise ValueError("input_indices must contain non-negative integers")
        if len(set(value)) != len(value):
            raise ValueError("input_indices must not contain duplicates")
        return value

    @model_validator(mode="after")
    def _validate_linear(self) -> SubmissionConfig:
        if self.architecture == "linear" and self.hidden_sizes:
            raise ValueError("linear submissions must not declare hidden_sizes")
        return self

    @classmethod
    def from_file(cls, path: Path) -> SubmissionConfig:
        try:
            raw: Any = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise SubmissionError(f"Could not read config {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise SubmissionError("Submission config must be a JSON object")
        try:
            return cls.model_validate(raw)
        except ValidationError as exc:
            raise SubmissionError(f"Invalid submission config {path}: {exc}") from exc


@dataclass(frozen=True)
class Submission:
    """A config and checkpoint pair located under a submission directory."""

    config: SubmissionConfig
    checkpoint: Path

    @classmethod
    def from_directory(cls, directory: Path) -> Submission:
        config_path = directory / "config.json"
        checkpoint = directory / "model.pt"
        if not checkpoint.is_file():
            raise SubmissionError(f"Missing checkpoint: {checkpoint}")
        config = SubmissionConfig.from_file(config_path)
        return cls(config, checkpoint)


@dataclass(frozen=True)
class LoadedSubmission:
    """A submission with its config, raw state dict, and executable model."""

    config: SubmissionConfig
    state_dict: dict[str, Any]
    model: Any


def load_state_dict(submission: Submission) -> dict[str, Any]:
    """Load a CPU state dict without enabling arbitrary pickle execution."""
    try:
        import torch
    except ImportError as exc:
        raise SubmissionError("PyTorch is required to load model.pt submissions") from exc
    try:
        state = torch.load(submission.checkpoint, map_location="cpu", weights_only=True)
    except (EOFError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SubmissionError(f"Could not load checkpoint {submission.checkpoint}: {exc}") from exc
    if not isinstance(state, dict):
        raise SubmissionError("Checkpoint must contain a state-dict mapping")
    return state


def build_model(config: SubmissionConfig) -> Any:
    """Build the configured feed-forward network."""
    try:
        from torch import nn
    except ImportError as exc:
        raise SubmissionError("PyTorch is required to build submission models") from exc

    if config.architecture == "linear":
        return nn.Linear(config.input_size, config.output_size)

    layers: list[Any] = []
    input_size = config.input_size
    for hidden_size in config.hidden_sizes:
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU() if config.activation == "relu" else nn.Identity())
        input_size = hidden_size
    layers.append(nn.Linear(input_size, config.output_size))
    return nn.Sequential(*layers)


def materialize_submission(submission: Submission) -> LoadedSubmission:
    """Validate and materialize a submission for evaluation."""
    state_dict = load_state_dict(submission)
    model = build_model(submission.config)
    try:
        model.load_state_dict(state_dict, strict=True)
    except (KeyError, RuntimeError, TypeError) as exc:
        raise SubmissionError(
            f"Checkpoint does not match architecture {submission.config.architecture!r}: {exc}"
        ) from exc
    model.eval()
    return LoadedSubmission(submission.config, state_dict, model)


def parameter_count(state_dict: dict[str, Any]) -> int:
    """Return the total number of scalar values in a state dict."""
    total = 0
    for name, value in state_dict.items():
        numel = getattr(value, "numel", None)
        if not callable(numel):
            raise SubmissionError(f"State-dict entry {name!r} is not a tensor")
        total += int(numel())
    return total
