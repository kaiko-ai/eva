"""Tests for the forward passes of the pathology backbones in the model registry."""

import pytest
import torch

from eva.vision.models import wrappers

BATCH_SIZE = 1
N_COMPARE = 5


@pytest.mark.skip(reason="Not possible to run in CI due to disk space and memory restrictions")
@pytest.mark.parametrize(
    "model_name, expected_values, expected_shape, concat_mean_patch_tokens",
    sum(
        [
            [
                (
                    "pathology/kaiko_vits16",
                    torch.Tensor([-4.8422, -2.8297, 2.8580, 0.4756, -0.0418]),
                    (BATCH_SIZE, 384),
                    concat_mean_patch_tokens,
                ),
                (
                    "pathology/kaiko_vits8",
                    torch.Tensor([6.0802, -0.3768, -2.0065, 0.0425, -0.5345]),
                    (BATCH_SIZE, 384),
                    concat_mean_patch_tokens,
                ),
                (
                    "pathology/kaiko_vitb16",
                    torch.Tensor([-0.6163, -5.5932, 0.9185, -3.3077, 0.1800]),
                    (BATCH_SIZE, 768),
                    concat_mean_patch_tokens,
                ),
                (
                    "pathology/kaiko_vitb8",
                    torch.Tensor([-0.9519, 2.1435, -1.3973, 1.8827, -0.0044]),
                    (BATCH_SIZE, 768),
                    concat_mean_patch_tokens,
                ),
                (
                    "pathology/kaiko_vitl14",
                    torch.Tensor([-1.1789, 0.5188, -4.7079, 0.2481, -2.1786]),
                    (BATCH_SIZE, 1024),
                    concat_mean_patch_tokens,
                ),
                (
                    "pathology/bioptimus_h_optimus_0",
                    torch.Tensor([0.5110, -0.1243, 0.3575, -0.8805, -0.2732]),
                    (BATCH_SIZE, 1536),
                    concat_mean_patch_tokens,
                ),
                (
                    "pathology/prov_gigapath",
                    torch.Tensor([0.7893, -0.6509, -0.2221, 0.5051, 0.4274]),
                    (BATCH_SIZE, 1536),
                    concat_mean_patch_tokens,
                ),
                (
                    "pathology/histai_hibou_b",
                    torch.Tensor([-0.8283, 1.0077, -2.4298, 1.6838, -0.4240]),
                    (BATCH_SIZE, 768),
                    concat_mean_patch_tokens,
                ),
                (
                    "pathology/histai_hibou_l",
                    torch.Tensor([-0.5136, -1.2975, 0.9338, 1.3067, -0.4138]),
                    (BATCH_SIZE, 1024),
                    concat_mean_patch_tokens,
                ),
                (
                    "pathology/lunit_vits16",
                    torch.Tensor([2.3240, -3.5531, 0.6269, -0.1491, -1.7312]),
                    (BATCH_SIZE, 384),
                    concat_mean_patch_tokens,
                ),
                (
                    "pathology/lunit_vits8",
                    torch.Tensor([-1.8391, 6.4864, -1.0888, -0.8426, 0.2072]),
                    (BATCH_SIZE, 384),
                    concat_mean_patch_tokens,
                ),
                (
                    "pathology/mahmood_uni",
                    torch.Tensor([-0.5453, -0.3392, 2.0546, -1.6987, 0.8996]),
                    (BATCH_SIZE, 1024),
                    concat_mean_patch_tokens,
                ),
                (
                    "pathology/owkin_phikon",
                    torch.Tensor([2.9856, 0.6126, 0.6679, 1.3643, -1.1573]),
                    (BATCH_SIZE, 768),
                    concat_mean_patch_tokens,
                ),
                (
                    "pathology/owkin_phikon_v2",
                    torch.Tensor([0.2607, -0.0367, -0.3205, 0.5113, 0.0222]),
                    (BATCH_SIZE, 1024),
                    concat_mean_patch_tokens,
                ),
                (
                    "pathology/paige_virchow2",
                    torch.Tensor([1.4649, 0.7918, -1.2957, -1.9513, -0.0634]),
                    (BATCH_SIZE, 1280),
                    concat_mean_patch_tokens,
                ),
            ]
            for concat_mean_patch_tokens in [False, True]
        ],
        [],
    ),
)
def test_forward_pass_expected_values(
    model_name: str,
    expected_values: torch.Tensor,
    expected_shape: tuple[int, int],
    concat_mean_patch_tokens: bool,
    random_generator: torch.Generator,
):
    """Test if forward pass returns the expected values."""
    model = wrappers.ModelFromRegistry(
        model_name=model_name,
        model_extra_kwargs={"concat_mean_patch_tokens": concat_mean_patch_tokens},
    )

    input_tensor = torch.randn(BATCH_SIZE, 3, 224, 224, generator=random_generator)
    output = model(input_tensor)

    assert torch.allclose(output[0, :N_COMPARE], expected_values, atol=1e-4)
    if concat_mean_patch_tokens:
        expected_shape = (expected_shape[0], 2 * expected_shape[1])
    assert output.shape == expected_shape


@pytest.fixture
def random_generator(seed: int = 42):
    """Fixture for seeded torch random generator."""
    generator = torch.Generator()
    generator.manual_seed(42)
    return generator
