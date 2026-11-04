import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


def now():
    return datetime.now(timezone.utc).isoformat()


class APTRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    robust: bool = Field(default=False, strict=True, description="false 使用 APT，true 使用 SPT")
    dataset: str = "eurosat"
    config: str = "vit_b32"
    data_root: str = "./DATA"
    n_ctx: int = Field(default=16, ge=1, le=77)
    shots: Literal[1, 2, 4, 8, 16] = 16
    eps: float = Field(default=4.0, ge=0, le=255)
    alpha: float = Field(default=0.67, gt=0, le=255)
    steps: int = Field(default=3, ge=1)
    seed: int = Field(default=1, ge=0, le=4294967295)
    prompt: Literal["onfly", "perturbed"] = "onfly"
    palpha: float = Field(default=0.001, gt=0)
    class_token_position: Literal["end", "middle", "front"] = "end"
    csc: bool = False
    robust_test: bool = True
    gpu: str = "0"
    max_epoch: Optional[int] = Field(default=None, ge=1)

    @field_validator("dataset", "config")
    @classmethod
    def safe_name(cls, value):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            raise ValueError("Use a config name without a path or extension")
        return value

    @field_validator("gpu")
    @classmethod
    def valid_gpu(cls, value):
        if not re.fullmatch(r"\d+", value):
            raise ValueError("Specify one GPU index, e.g. '0'")
        return value

    @field_validator("eps", "alpha", "palpha")
    @classmethod
    def finite_number(cls, value):
        if not math.isfinite(value):
            raise ValueError("Value must be finite")
        return value


def build_command(request, root, output):
    trainer = "SPT" if request.robust else "APT"
    dataset_config = root / "configs" / "datasets" / (request.dataset + ".yaml")
    trainer_config = root / "configs" / "trainers" / trainer / (request.config + ".yaml")
    for path in (dataset_config, trainer_config):
        if not path.is_file():
            raise ValueError("Config does not exist: {}".format(path))
    data = Path(request.data_root).expanduser()
    if not data.is_absolute():
        data = root / data
    if not data.is_dir():
        raise ValueError("Dataset root does not exist: {}".format(data))
    command = [
        sys.executable, "-u", str(root / "train_api.py"),
        "--root", str(data.resolve()), "--trainer", trainer,
        "--dataset_config_file", str(dataset_config),
        "--config_file", str(trainer_config), "--output_dir", str(output),
        "--eps", str(request.eps), "--alpha", str(request.alpha),
        "--steps", str(request.steps), "--seed", str(request.seed),
        "TRAINER.COOP.N_CTX", str(request.n_ctx),
        "TRAINER.COOP.CSC", str(request.csc),
        "TRAINER.COOP.CLASS_TOKEN_POSITION", request.class_token_position,
        "TRAINER.COOP.ATK_TEST", str(request.robust_test),
        "DATASET.NUM_SHOTS", str(request.shots),
    ]
    if trainer == "APT":
        command.extend([
            "TRAINER.APT.PROMPT", request.prompt,
            "TRAINER.APT.PALPHA", str(request.palpha),
        ])
    if request.max_epoch is not None:
        command.extend(["OPTIM.MAX_EPOCH", str(request.max_epoch)])
    return command


