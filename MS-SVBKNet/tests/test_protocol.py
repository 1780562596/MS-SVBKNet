from __future__ import annotations

import csv
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from skimage.metrics import structural_similarity

from ms_svbknet.config import (
    architecture_fingerprint,
    load_config,
    training_fingerprint,
    verify_checkpoint_architecture,
    verify_checkpoint_training,
)
from ms_svbknet.data import Pair, audit_pair_splits, discover_pairs, make_loader
from ms_svbknet.metrics import MetricBundle, psnr, ssim


def _save(path: Path, value: int, size: int = 24) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), (value, value, value)).save(path)


def test_gopro_variants_are_explicit_and_never_mixed(tmp_path: Path):
    for variant, value in (("blur", 10), ("blur_gamma", 20)):
        _save(tmp_path / "train" / "scene" / variant / "0001.png", value)
    _save(tmp_path / "train" / "scene" / "sharp" / "0001.png", 30)
    linear = discover_pairs(tmp_path, "train", blur_variant="blur")
    gamma = discover_pairs(tmp_path, "train", blur_variant="blur_gamma")
    assert len(linear) == len(gamma) == 1
    assert linear[0].blur.parent.name == "blur"
    assert gamma[0].blur.parent.name == "blur_gamma"


def test_missing_sharp_is_a_hard_error(tmp_path: Path):
    _save(tmp_path / "test" / "scene" / "blur_gamma" / "0001.png", 10)
    with pytest.raises(FileNotFoundError, match="no matching sharp"):
        discover_pairs(tmp_path, "test", blur_variant="blur_gamma")


def test_csv_duplicate_pair_is_rejected(tmp_path: Path):
    blur = tmp_path / "train" / "scene" / "blur" / "0001.png"
    sharp = tmp_path / "train" / "scene" / "sharp" / "0001.png"
    _save(blur, 10)
    _save(sharp, 20)
    index = tmp_path / "pairs.csv"
    with index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["blur", "sharp"])
        writer.writeheader()
        for _ in range(2):
            writer.writerow({"blur": blur.relative_to(tmp_path), "sharp": sharp.relative_to(tmp_path)})
    with pytest.raises(ValueError, match="Duplicate pair"):
        discover_pairs(tmp_path, index_file="pairs.csv")


def test_scene_leakage_is_rejected(tmp_path: Path):
    train = Pair(
        tmp_path / "train" / "scene_a" / "blur" / "0001.png",
        tmp_path / "train" / "scene_a" / "sharp" / "0001.png",
    )
    val = Pair(
        tmp_path / "train" / "scene_a" / "blur" / "0002.png",
        tmp_path / "train" / "scene_a" / "sharp" / "0002.png",
    )
    with pytest.raises(ValueError, match="Scene leakage"):
        audit_pair_splits({"train": [train], "val": [val]}, blur_variant="blur")


def test_validation_test_scene_leakage_is_rejected(tmp_path: Path):
    val = Pair(tmp_path / "shared" / "blur" / "1.png", tmp_path / "shared" / "sharp" / "1.png")
    test = Pair(tmp_path / "shared" / "blur" / "2.png", tmp_path / "shared" / "sharp" / "2.png")
    with pytest.raises(ValueError, match="Scene leakage: val/test"):
        audit_pair_splits({"val": [val], "test": [test]})


def test_paper_scene_counts_reject_old_twenty_two_split():
    from ms_svbknet.data import validate_expected_scene_counts

    cfg = load_config("configs/full.yaml")
    report = {"counts": {"train": 2000, "val": 103}, "scene_counts": {"train": 20, "val": 2}}
    with pytest.raises(ValueError, match="train scene count mismatch"):
        validate_expected_scene_counts(report, cfg.data)
    report["scene_counts"] = {"train": 18, "val": 4}
    validate_expected_scene_counts(report, cfg.data)


def test_ssim_matches_skimage_gaussian_valid_protocol():
    rng = np.random.default_rng(42)
    sharp_np = rng.random((37, 41, 3), dtype=np.float32)
    restored_np = np.clip(sharp_np + rng.normal(0, 0.04, sharp_np.shape), 0, 1).astype(np.float32)
    sharp = torch.from_numpy(sharp_np).permute(2, 0, 1).unsqueeze(0).mul(2).sub(1)
    restored = torch.from_numpy(restored_np).permute(2, 0, 1).unsqueeze(0).mul(2).sub(1)
    expected = structural_similarity(
        restored_np,
        sharp_np,
        data_range=1.0,
        channel_axis=2,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
    )
    assert ssim(restored, sharp) == pytest.approx(expected, abs=2e-5)


def test_gopro_metric_protocol_quantizes_before_psnr():
    sharp = torch.full((1, 3, 16, 16), -1.0)
    restored = sharp.clone()
    restored[:, :, 0, 0] = -1.0 + 0.49 * 2.0 / 255.0
    assert psnr(restored, sharp, metric_protocol="gopro_matlab") == 100.0
    assert psnr(restored, sharp, metric_protocol="float_rgb") < 100.0


def test_checkpoint_model_binding_and_resume_signature():
    cfg = load_config("configs/smoke.yaml")
    checkpoint = {
        "architecture_sha256": architecture_fingerprint(cfg),
        "training_sha256": training_fingerprint(cfg),
        "train_loader_generator_state": torch.Generator().get_state(),
    }
    assert verify_checkpoint_architecture(checkpoint, cfg) == architecture_fingerprint(cfg)
    extended = load_config("configs/smoke.yaml", ["train.epochs=2"])
    assert verify_checkpoint_training(checkpoint, extended) == training_fingerprint(cfg)
    changed = load_config("configs/smoke.yaml", ["model.base_channels=8"])
    with pytest.raises(RuntimeError, match="architecture"):
        verify_checkpoint_architecture(checkpoint, changed)
    changed_batch = load_config("configs/smoke.yaml", ["train.batch_size=2"])
    with pytest.raises(RuntimeError, match="state-affecting"):
        verify_checkpoint_training(checkpoint, changed_batch)


def test_loader_generator_resume_reconstructs_next_epoch(tmp_path: Path):
    for index in range(4):
        _save(tmp_path / "train" / f"scene_{index}" / "blur" / f"{index}.png", 10 + index, 32)
        _save(tmp_path / "train" / f"scene_{index}" / "sharp" / f"{index}.png", 20 + index, 32)
    overrides = [
        f"data.root={str(tmp_path)!r}",
        "data.train_split=train",
        "data.crop_size=16",
        "data.num_workers=0",
        "train.batch_size=1",
    ]
    cfg = load_config("configs/smoke.yaml", overrides)
    first = make_loader(cfg, "train", True, seed=123)
    first.dataset.set_epoch(1)
    list(first)
    saved = first.repro_generator.get_state()
    first.dataset.set_epoch(2)
    expected = [(batch["path"][0], batch["blur"].clone()) for batch in first]

    resumed = make_loader(cfg, "train", True, seed=123)
    resumed.repro_generator.set_state(saved)
    resumed.dataset.set_epoch(2)
    actual = [(batch["path"][0], batch["blur"].clone()) for batch in resumed]
    assert [item[0] for item in actual] == [item[0] for item in expected]
    assert all(torch.equal(left[1], right[1]) for left, right in zip(actual, expected))


def test_standard_lpips_vgg16_backend_contract(monkeypatch):
    captured = {}

    class FakeOfficialLPIPS(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            captured.update(kwargs)
            self.scale = torch.nn.Parameter(torch.ones(()))

        def forward(self, restored, sharp):
            return (restored - sharp).square().mean((1, 2, 3), keepdim=True) * self.scale

    monkeypatch.setitem(sys.modules, "lpips", types.SimpleNamespace(LPIPS=FakeOfficialLPIPS))
    bundle = MetricBundle(
        torch.device("cpu"),
        use_lpips=True,
        metric_protocol="float_rgb",
        lpips_backend="official_vgg16",
    )
    assert captured == {
        "pretrained": True,
        "net": "vgg",
        "version": "0.1",
        "lpips": True,
        "pnet_rand": False,
        "pnet_tune": False,
        "eval_mode": True,
        "verbose": False,
    }
    assert bundle.perceptual_backbone == "vgg16"
    assert bundle.lpips_version == "0.1"
    assert bundle.lpips_calibrated is True
    assert all(not parameter.requires_grad for parameter in bundle.lpips_model.parameters())
    image = torch.rand(1, 3, 16, 16).mul(2).sub(1)
    assert bundle(image, image)["lpips"] == pytest.approx(0.0, abs=1e-8)
    changed = image.clone()
    changed[:, 0] = changed[:, 0].mul(0.5)
    assert bundle(changed, image)["lpips"] > 0


def test_all_release_configs_use_standard_lpips_vgg16():
    config_paths = sorted(Path("configs").glob("*.yaml"))
    assert config_paths
    for path in config_paths:
        assert load_config(path).eval.lpips_backend == "official_vgg16"
    with pytest.raises(ValueError, match="official_vgg16"):
        load_config("configs/smoke.yaml", ["eval.lpips_backend=vgg19"])
