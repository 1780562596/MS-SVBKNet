from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from ms_svbknet.config import load_config
from ms_svbknet.data import discover_pairs
from ms_svbknet.losses import DynamicCredWeight, credible_reblur_loss
from ms_svbknet.metrics import psnr, ssim
from ms_svbknet.models import MSSVBKNet
from ms_svbknet.models.discriminator import ImprovedPatchDiscriminator
from ms_svbknet.models.svbkm import SVBKM, basis_mixture_blur, direct_spatial_blur
from ms_svbknet.utils import tiled_restore


def test_model_shapes_and_kernel_normalization():
    cfg = load_config("configs/full.yaml", ["model.base_channels=8", "model.degradation_channels=8", "model.num_kernels=4", "model.kernel_size=5"])
    model = MSSVBKNet(cfg).eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        outputs = model(x)
        reblurred = model.reblur(outputs["restored"], outputs)
    assert outputs["restored"].shape == x.shape
    assert reblurred.shape == x.shape
    assert torch.allclose(outputs["kernel_weights"].sum(1), torch.ones(1, 64, 64), atol=1e-5)
    assert torch.allclose(outputs["basis_kernels"].sum((1, 2, 3)), torch.ones(4), atol=1e-5)


def test_constant_image_is_preserved_by_normalized_basis():
    image = torch.ones(1, 3, 32, 32)
    basis = torch.ones(2, 1, 5, 5) / 25.0
    weights = torch.zeros(1, 2, 32, 32)
    weights[:, 0] = 0.25
    weights[:, 1] = 0.75
    result = basis_mixture_blur(image, weights, basis)
    assert torch.allclose(result, image, atol=1e-5)


def test_grouped_basis_blur_matches_explicit_reference():
    torch.manual_seed(2)
    image = torch.randn(1, 3, 13, 15)
    basis = torch.rand(4, 1, 5, 5)
    basis = basis / basis.sum((1, 2, 3), keepdim=True)
    weights = torch.softmax(torch.randn(1, 4, 13, 15), dim=1)
    actual = basis_mixture_blur(image, weights, basis)
    padded = F.pad(image, (2, 2, 2, 2), mode="reflect")
    responses = []
    for index in range(4):
        kernel = basis[index : index + 1].expand(3, 1, 5, 5)
        responses.append(F.conv2d(padded, kernel, groups=3))
    expected = (torch.stack(responses, dim=1) * weights.unsqueeze(2)).sum(dim=1)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_fixed_dictionary_is_normalized_and_frozen():
    module = SVBKM(8, num_kernels=6, kernel_size=7, dictionary_mode="fixed")
    weights, basis = module(torch.randn(1, 8, 12, 10))
    assert module.basis_logits is None
    assert "fixed_basis" in dict(module.named_buffers())
    assert torch.allclose(weights.sum(1), torch.ones(1, 12, 10), atol=1e-6)
    assert torch.allclose(basis.sum((1, 2, 3)), torch.ones(6), atol=1e-6)


def test_dense_kernel_blur_preserves_constants():
    image = torch.ones(1, 3, 9, 11)
    logits = torch.randn(1, 25, 9, 11)
    kernels = torch.softmax(logits, dim=1)
    actual = direct_spatial_blur(image, kernels, 5, chunk_rows=3)
    assert torch.allclose(actual, image, atol=1e-5)


def test_improved_patchgan_matches_revised_paper_table():
    discriminator = ImprovedPatchDiscriminator(global_pool=False)
    patches = discriminator(torch.randn(1, 3, 256, 256))
    assert patches.shape == (1, 1, 14, 14)
    assert sum(parameter.numel() for parameter in discriminator.parameters()) == 2_830_337


def test_credible_reblur_formula_and_scheduler_resume():
    blurred = torch.zeros(1, 3, 4, 4)
    reblurred = torch.ones_like(blurred) * 0.25
    loss, reliability, error = credible_reblur_loss(reblurred, blurred, alpha=2.0, detach_reliability=True)
    assert torch.allclose(error, torch.full_like(error, 0.25))
    assert torch.allclose(reliability, torch.full_like(reliability, torch.exp(torch.tensor(-0.5))))
    assert torch.allclose(loss, torch.tensor(0.25) * torch.exp(torch.tensor(-0.5)))
    scheduler = DynamicCredWeight(initial=0.1, maximum=3.0, momentum=0.5)
    scheduler.update(torch.tensor(2.0))
    scheduler.update(torch.tensor(1.0))
    state = scheduler.state_dict()
    restored = DynamicCredWeight(initial=0.1, maximum=3.0, momentum=0.5)
    restored.load_state_dict(state)
    assert restored.state_dict() == state


def test_metrics_apply_same_crop_and_support_y_channel():
    sharp = torch.zeros(1, 3, 16, 16)
    restored = sharp.clone()
    restored[..., 0, :] = 1.0
    assert psnr(restored, sharp, crop_border=1) == 100.0
    assert psnr(restored, sharp, crop_border=0) < 100.0
    assert ssim(restored, sharp, crop_border=1) == pytest.approx(1.0, abs=1e-6)
    assert psnr(sharp, sharp, color_space="y") == 100.0


def test_invalid_module_dependency_is_rejected():
    with pytest.raises(ValueError, match="SVBKM requires"):
        load_config("configs/full.yaml", ["model.use_mdrm=false", "model.use_svbkm=true"])


def test_tiled_restore_preserves_identity_and_shape():
    class IdentityModel(torch.nn.Module):
        def forward(self, image):
            return {"restored": image}

    image = torch.randn(1, 3, 37, 45)
    restored = tiled_restore(IdentityModel(), image, tile=24, overlap=8)
    assert restored.shape == image.shape
    assert torch.allclose(restored, image, atol=1e-5)


def test_pair_discovery_keeps_scene_identity(tmp_path: Path):
    for scene in ("scene_a", "scene_b"):
        for kind, value in (("blur", 0), ("sharp", 255)):
            directory = tmp_path / "train" / scene / kind
            directory.mkdir(parents=True)
            Image.new("RGB", (16, 16), color=(value, value, value)).save(directory / "000001.png")
    pairs = discover_pairs(tmp_path, "train")
    assert len(pairs) == 2
    assert all(pair.blur.parent.parent == pair.sharp.parent.parent for pair in pairs)
