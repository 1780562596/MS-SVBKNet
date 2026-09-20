from __future__ import annotations

import ast
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import yaml


class Config(dict):
    """Dictionary with recursive attribute access."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    __setattr__ = dict.__setitem__


def _convert(value: Any) -> Any:
    if isinstance(value, dict):
        return Config({k: _convert(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_convert(v) for v in value]
    return value


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw = deepcopy(raw)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        dotted, text = item.split("=", 1)
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            value = yaml.safe_load(text)
        node = raw
        keys = dotted.split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value
    cfg = _convert(raw)
    validate_config(cfg)
    return cfg


def to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain(item) for item in value]
    return value


def _choice(value: Any, choices: set[str], name: str) -> None:
    if value not in choices:
        options = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of {{{options}}}, got {value!r}")


def validate_config(cfg: Config) -> None:
    """Fail early on configurations that cannot represent a valid experiment."""
    required = {"experiment", "data", "model", "loss", "train", "eval"}
    missing = required.difference(cfg)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")

    if int(cfg.data.crop_size) <= 0:
        raise ValueError("data.crop_size must be positive")
    if int(cfg.data.crop_size) % 4:
        raise ValueError("data.crop_size must be divisible by 4 for the encoder-decoder")
    if int(cfg.model.base_channels) <= 0 or int(cfg.model.degradation_channels) <= 0:
        raise ValueError("model channel counts must be positive")
    if int(cfg.model.kernel_size) <= 0 or int(cfg.model.kernel_size) % 2 != 1:
        raise ValueError("model.kernel_size must be a positive odd integer")
    if int(cfg.model.num_kernels) <= 0:
        raise ValueError("model.num_kernels must be positive")
    blur_variant = str(getattr(cfg.data, "blur_variant", "")).strip()
    if not blur_variant:
        raise ValueError(
            "data.blur_variant must be explicit (for GoPro use 'blur' or 'blur_gamma'); "
            "the loader never mixes variants"
        )
    if bool(getattr(cfg.data, "persistent_workers", False)) and bool(cfg.experiment.deterministic):
        raise ValueError(
            "data.persistent_workers must be false when experiment.deterministic=true "
            "so worker RNG can be reconstructed after resume"
        )
    for name in (
        "expected_trainval_pairs",
        "expected_test_pairs",
        "expected_trainval_scenes",
        "expected_train_scenes",
        "expected_val_scenes",
        "expected_test_scenes",
    ):
        if int(getattr(cfg.data, name, 0)) < 0:
            raise ValueError(f"data.{name} must be non-negative")

    dictionary_mode = getattr(
        cfg.model,
        "dictionary_mode",
        "learned" if bool(getattr(cfg.model, "use_dictionary", False)) else "dense",
    )
    _choice(dictionary_mode, {"learned", "fixed", "dense"}, "model.dictionary_mode")
    if bool(cfg.model.use_svbkm) and not bool(cfg.model.use_mdrm):
        raise ValueError("SVBKM requires model.use_mdrm=true")
    if bool(cfg.model.use_crcm) and not bool(cfg.model.use_svbkm):
        raise ValueError("CRCM requires model.use_svbkm=true")

    if float(cfg.loss.reliability_alpha) < 0:
        raise ValueError("loss.reliability_alpha must be non-negative")
    if float(cfg.loss.cred_initial) < 0 or float(cfg.loss.cred_maximum) < float(cfg.loss.cred_initial):
        raise ValueError("credible-loss weights must satisfy 0 <= initial <= maximum")
    if not 0 <= float(cfg.loss.cred_momentum) < 1:
        raise ValueError("loss.cred_momentum must be in [0, 1)")
    _choice(str(getattr(cfg.loss, "gan_mode", "wgan_gp")), {"wgan_gp", "hinge", "lsgan"}, "loss.gan_mode")

    if int(cfg.train.epochs) <= 0 or int(cfg.train.batch_size) <= 0:
        raise ValueError("train.epochs and train.batch_size must be positive")
    if len(cfg.train.betas) != 2:
        raise ValueError("train.betas must contain exactly two values")
    if int(getattr(cfg.train, "preview_interval", 0)) < 0:
        raise ValueError("train.preview_interval must be non-negative")
    if int(getattr(cfg.train, "max_steps_per_epoch", 0)) < 0:
        raise ValueError("train.max_steps_per_epoch must be non-negative")

    tile = int(cfg.eval.tile)
    overlap = int(cfg.eval.overlap)
    if tile > 0 and not 0 <= overlap < tile:
        raise ValueError("eval.overlap must satisfy 0 <= overlap < eval.tile")
    _choice(str(getattr(cfg.eval, "color_space", "rgb")), {"rgb", "y"}, "eval.color_space")
    protocol = str(getattr(cfg.eval, "metric_protocol", "float_rgb"))
    _choice(protocol, {"float_rgb", "gopro_matlab"}, "eval.metric_protocol")
    if protocol == "gopro_matlab" and str(getattr(cfg.eval, "color_space", "rgb")) != "rgb":
        raise ValueError("eval.metric_protocol=gopro_matlab requires eval.color_space=rgb")
    backend = str(getattr(cfg.eval, "lpips_backend", "official_vgg16"))
    _choice(backend, {"official_vgg16"}, "eval.lpips_backend")


def config_fingerprint(cfg: Config) -> str:
    payload = json.dumps(to_plain(cfg), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def architecture_fingerprint(cfg: Config | dict[str, Any]) -> str:
    """Hash only architecture-defining values for safe checkpoint loading."""
    model = cfg["model"]
    payload = json.dumps(to_plain(model), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def training_fingerprint(cfg: Config | dict[str, Any]) -> str:
    """Hash state-affecting settings while allowing a longer epoch budget."""
    plain = to_plain(cfg)
    train = dict(plain["train"])
    train.pop("epochs", None)
    train.pop("preview_interval", None)
    payload = {
        "experiment": {
            "seed": plain["experiment"]["seed"],
            "deterministic": plain["experiment"]["deterministic"],
        },
        "data": plain["data"],
        "model": plain["model"],
        "loss": plain["loss"],
        "train": train,
        "eval": plain["eval"],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def verify_checkpoint_architecture(
    checkpoint: dict[str, Any], cfg: Config, *, require_metadata: bool = True
) -> str:
    """Reject a checkpoint that was built for a different network graph."""
    current = architecture_fingerprint(cfg)
    saved = checkpoint.get("architecture_sha256")
    if not saved and isinstance(checkpoint.get("config"), dict) and "model" in checkpoint["config"]:
        saved = architecture_fingerprint(checkpoint["config"])
    if not saved:
        if require_metadata:
            raise RuntimeError(
                "Checkpoint has no architecture metadata. Use a checkpoint produced by this "
                "code version; architecture-safe loading cannot be guaranteed."
            )
        return current
    if saved != current:
        raise RuntimeError(
            "Checkpoint architecture does not match the current model configuration "
            f"(checkpoint={saved}, current={current})."
        )
    return current


def verify_checkpoint_training(checkpoint: dict[str, Any], cfg: Config) -> str:
    """Require identical state-affecting settings for an exact epoch-boundary resume."""
    current = training_fingerprint(cfg)
    saved = checkpoint.get("training_sha256")
    if not saved and isinstance(checkpoint.get("config"), dict):
        saved = training_fingerprint(checkpoint["config"])
    if not saved:
        raise RuntimeError("Checkpoint lacks the metadata required for exact resume")
    if saved != current:
        raise RuntimeError(
            "Resume configuration changes state-affecting settings "
            f"(checkpoint={saved}, current={current}). Only epochs, output/name/device and "
            "preview_interval may change for exact resume."
        )
    if checkpoint.get("train_loader_generator_state") is None:
        raise RuntimeError("Checkpoint lacks the DataLoader RNG state required for exact resume")
    return current
