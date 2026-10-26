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

from minimal_nn_competition.benchmarks.snake.names import SNAKE_BENCHMARKS

ARCHITECTURES = ("linear", "mlp")
ACTIVATIONS = ("relu", "identity")

MAX_PARAMETERS = 10_000_000
MAX_HIDDEN_LAYERS = 10
MAX_CHECKPOINT_BYTES = 100 * 1024 * 1024


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
    description: str = ""
    input_indices: tuple[StrictInt, ...] | StrictBool | None = None

    @field_validator("architecture")
    @classmethod
    def _validate_architecture(cls, value):
        if value not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {', '.join(ARCHITECTURES)}")
        return value

    @field_validator("hidden_sizes")
    @classmethod
    def _validate_hidden_sizes(cls, value):
        if len(value) > MAX_HIDDEN_LAYERS:
            raise ValueError(f"hidden_sizes may contain at most {MAX_HIDDEN_LAYERS} layers")
        if min(value, default=1) <= 0:
            raise ValueError("hidden_sizes must be a list of positive integers")
        return value

    @field_validator("activation")
    @classmethod
    def _validate_activation(cls, value):
        if value not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {', '.join(ACTIVATIONS)}")
        return value

    @field_validator("input_indices")
    @classmethod
    def _validate_input_indices(cls, value):
        if value is None or value is True:
            return value
        if value is False:
            raise ValueError("input_indices must be true or a list of feature indices")
        if not value:
            raise ValueError("input_indices must not be empty")
        if min(value) < 0:
            raise ValueError("input_indices must contain non-negative integers")
        if len(set(value)) != len(value):
            raise ValueError("input_indices must not contain duplicates")
        return value

    @model_validator(mode="after")
    def _validate_parameter_count(self):
        total = declared_parameter_count(self)
        if total > MAX_PARAMETERS:
            raise ValueError(
                f"submission would require {total} parameters, above the {MAX_PARAMETERS} maximum"
            )
        return self

    @model_validator(mode="after")
    def _validate_cross_fields(self):
        if self.architecture == "linear" and self.hidden_sizes:
            raise ValueError("linear submissions must not declare hidden_sizes")
        if self.benchmark not in SNAKE_BENCHMARKS and self.input_indices is not None:
            raise ValueError("input_indices is only supported for snake benchmarks")
        return self

    @classmethod
    def from_file(cls, path):
        try:
            raw = json.loads(path.read_text())
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
    def from_directory(cls, directory):
        checkpoint = directory / "model.pt"
        if not checkpoint.is_file():
            raise SubmissionError(f"Missing checkpoint: {checkpoint}")
        return cls(SubmissionConfig.from_file(directory / "config.json"), checkpoint)


@dataclass(frozen=True)
class LoadedSubmission:
    """A submission with its config, raw state dict, and executable model."""

    config: SubmissionConfig
    state_dict: dict[str, Any]
    model: Any


def declared_parameter_count(config):
    """Return the number of parameters implied by a submission's config."""
    total = 0
    previous = config.input_size
    for hidden_size in (*config.hidden_sizes, config.output_size):
        total += previous * hidden_size + hidden_size
        previous = hidden_size
    return total


def load_state_dict(submission):
    """Load a CPU state dict without enabling arbitrary pickle execution."""
    checkpoint = submission.checkpoint
    try:
        size = checkpoint.stat().st_size
    except OSError as exc:
        raise SubmissionError(f"Could not stat checkpoint {checkpoint}: {exc}") from exc
    if size == 0:
        raise SubmissionError(f"Checkpoint must not be empty: {checkpoint}")
    if size > MAX_CHECKPOINT_BYTES:
        raise SubmissionError(
            f"Checkpoint is too large ({size} bytes); maximum is {MAX_CHECKPOINT_BYTES} bytes"
        )
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


def build_model(config):
    """Build the configured feed-forward network."""
    try:
        from torch import nn
    except ImportError as exc:
        raise SubmissionError("PyTorch is required to build submission models") from exc

    if config.architecture == "linear":
        return nn.Linear(config.input_size, config.output_size)

    layers = []
    input_size = config.input_size
    for hidden_size in config.hidden_sizes:
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU() if config.activation == "relu" else nn.Identity())
        input_size = hidden_size
    layers.append(nn.Linear(input_size, config.output_size))
    return nn.Sequential(*layers)


def _finite_state_dict(state_dict):
    try:
        import torch
    except ImportError as exc:
        raise SubmissionError("PyTorch is required to validate model.pt submissions") from exc
    for name, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            raise SubmissionError(f"State-dict entry {name!r} is not a tensor")
        if value.is_floating_point() and not torch.isfinite(value).all().item():
            raise SubmissionError(f"State-dict entry {name!r} contains non-finite values")


def materialize_submission(submission):
    """Validate and materialize a submission for evaluation."""
    state_dict = load_state_dict(submission)
    total = parameter_count(state_dict)
    if total > MAX_PARAMETERS:
        raise SubmissionError(
            f"Checkpoint contains {total} parameters, above the {MAX_PARAMETERS} maximum"
        )
    _finite_state_dict(state_dict)
    model = build_model(submission.config)
    try:
        model.load_state_dict(state_dict, strict=True)
    except (KeyError, RuntimeError, TypeError) as exc:
        raise SubmissionError(
            f"Checkpoint does not match architecture {submission.config.architecture!r}: {exc}"
        ) from exc
    model.eval()
    return LoadedSubmission(submission.config, state_dict, model)


def parameter_count(state_dict):
    """Return the total number of scalar values in a state dict."""
    total = 0
    for name, value in state_dict.items():
        numel = getattr(value, "numel", None)
        if not callable(numel):
            raise SubmissionError(f"State-dict entry {name!r} is not a tensor")
        total += int(numel())
    return total
