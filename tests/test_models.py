"""Faithfulness tests for the 1989 reconstruction.

Le Cun et al. (1989, p. 402) report that their network "has 4635 units,
98442 connections, and 2578 independent parameters". Reproducing all three
numbers from the layer definitions alone checks the layer sizes, kernels,
weight and bias sharing and the *number* of H2-H3 connections. The totals do
not depend on *which* 20 of the 48 possible connections are present, so the
pattern of Table 1 is checked separately against an independent transcription.
"""

from __future__ import annotations

import keras
import numpy as np
import pytest
from keras import layers

from lecun1989.layers import (
    LECUN_1989_TABLE1,
    ConnectionTableConv2D,
    TrainableSubsampling2D,
    scaled_tanh,
)
from lecun1989.models import build_chollet_2021, build_lecun_1989, network_accounting


@pytest.fixture(scope="module")
def lecun():
    keras.utils.set_random_seed(0)
    return build_lecun_1989()


# --------------------------------------------------------------------------- #
# The published totals (p. 402)
# --------------------------------------------------------------------------- #
def test_independent_parameters_match_paper(lecun):
    assert lecun.count_params() == 2578


def test_connections_match_paper(lecun):
    assert network_accounting(lecun)["connections"] == 98442


def test_units_match_paper(lecun):
    # 784 input + 2304 + 576 + 768 + 192 + 10 = 4634 units, plus the bias unit.
    accounting = network_accounting(lecun)
    assert accounting["units"] == 4634
    assert accounting["units_including_bias_unit"] == 4635


def test_layer_shapes_match_paper(lecun):
    shapes = {layer.name: tuple(layer.output.shape[1:]) for layer in lecun.layers}
    assert shapes["H1"] == (24, 24, 4)
    assert shapes["H2"] == (12, 12, 4)
    assert shapes["H3"] == (8, 8, 12)
    assert shapes["H4"] == (4, 4, 12)
    assert shapes["output"] == (10,)


def test_per_layer_parameters(lecun):
    params = {layer.name: layer.count_params() for layer in lecun.layers}
    assert params["H1"] == 4 * 26          # 4 maps x (25 weights + bias), p. 400
    assert params["H2"] == 4 * 2           # one shared weight + bias per map
    assert params["H3"] == 20 * 25 + 12    # 20 connections in Table 1
    assert params["H4"] == 12 * 2
    assert params["output"] == 10 * (192 + 1)


def test_chollet_parameters():
    # 320 + 18,496 + 16,010, as printed by model.summary() in Chollet (2021).
    assert build_chollet_2021().count_params() == 34826


# --------------------------------------------------------------------------- #
# Table 1
# --------------------------------------------------------------------------- #
# Table 1 (p. 400) transcribed a second time, column by column, in a different
# form from the matrix in layers.py: H3 map -> the H2 maps it reads from.
TABLE1_BY_H3_MAP = {
    1: {1}, 2: {1, 2}, 3: {1, 2}, 4: {2}, 5: {1, 2}, 6: {1, 2},
    7: {3}, 8: {3, 4}, 9: {3, 4}, 10: {4}, 11: {3, 4}, 12: {3, 4},
}


def test_table1_matches_paper_exactly():
    for h3, h2_maps in TABLE1_BY_H3_MAP.items():
        column = LECUN_1989_TABLE1[:, h3 - 1]
        assert {i + 1 for i in np.flatnonzero(column)} == h2_maps, f"H3.{h3}"


def test_model_uses_table1(lecun):
    np.testing.assert_array_equal(lecun.get_layer("H3").table, LECUN_1989_TABLE1)


def test_table1_has_twenty_connections():
    assert LECUN_1989_TABLE1.shape == (4, 12)
    assert LECUN_1989_TABLE1.sum() == 20


def test_table1_forms_two_independent_modules():
    """'The network is composed of two almost independent modules' (p. 402)."""
    assert LECUN_1989_TABLE1[2:, :6].sum() == 0
    assert LECUN_1989_TABLE1[:2, 6:].sum() == 0


def test_table1_one_or_two_inputs_per_map():
    """'Each unit receptive field is composed of one or two 5 by 5
    neighborhoods' (pp. 400-402)."""
    per_map = LECUN_1989_TABLE1.sum(axis=0)
    assert set(per_map) == {1, 2}
    assert list(np.flatnonzero(per_map == 1) + 1) == [1, 4, 7, 10]


# --------------------------------------------------------------------------- #
# ConnectionTableConv2D behaves like a masked convolution
# --------------------------------------------------------------------------- #
def _identity_layer():
    keras.utils.set_random_seed(1)
    layer = ConnectionTableConv2D(LECUN_1989_TABLE1, kernel_size=5, activation="linear")
    layer.build((None, 12, 12, 4))
    return layer


def test_equivalent_to_masked_dense_convolution():
    layer = _identity_layer()
    x = np.random.default_rng(0).normal(size=(2, 12, 12, 4)).astype("float32")

    dense = layers.Conv2D(12, 5, activation="linear")
    dense.build((None, 12, 12, 4))
    kernel = keras.ops.convert_to_numpy(layer.full_kernel())
    bias = keras.ops.convert_to_numpy(layer.bias)
    dense.set_weights([kernel, bias])

    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(layer(x)),
        keras.ops.convert_to_numpy(dense(x)),
        rtol=1e-5, atol=1e-5,
    )


def test_absent_connections_are_structurally_zero():
    kernel = keras.ops.convert_to_numpy(_identity_layer().full_kernel())
    absent = LECUN_1989_TABLE1 == 0
    assert np.all(kernel[:, :, absent] == 0)
    assert np.all(np.abs(kernel[:, :, ~absent]).sum(axis=(0, 1)) > 0)


def test_module_independence_in_forward_pass():
    """Changing H2.3 and H2.4 must leave H3.1-H3.6 untouched."""
    layer = _identity_layer()
    rng = np.random.default_rng(2)
    x = rng.normal(size=(1, 12, 12, 4)).astype("float32")
    x2 = x.copy()
    x2[..., 2:] = rng.normal(size=x2[..., 2:].shape)
    y1 = keras.ops.convert_to_numpy(layer(x))
    y2 = keras.ops.convert_to_numpy(layer(x2))
    np.testing.assert_allclose(y1[..., :6], y2[..., :6], atol=1e-6)
    assert not np.allclose(y1[..., 6:], y2[..., 6:])


def test_initialisation_scales_with_per_map_fan_in():
    """Maps with two inputs (fan-in 50) get a narrower init than maps with one."""
    layer = _identity_layer()
    kernel = keras.ops.convert_to_numpy(layer.kernel)
    pairs = np.argwhere(LECUN_1989_TABLE1 == 1)
    single = LECUN_1989_TABLE1.sum(axis=0)[pairs[:, 1]] == 1
    assert np.abs(kernel[..., single]).max() <= np.sqrt(3 / 25) + 1e-6
    assert np.abs(kernel[..., ~single]).max() <= np.sqrt(3 / 50) + 1e-6


def test_rejects_malformed_tables():
    with pytest.raises(ValueError):
        ConnectionTableConv2D(np.array([[1, 2]]))
    with pytest.raises(ValueError):
        ConnectionTableConv2D(np.array([[1, 0], [1, 0]]))  # map 2 has no input


# --------------------------------------------------------------------------- #
# TrainableSubsampling2D
# --------------------------------------------------------------------------- #
def test_subsampling_reduces_to_average_pooling_at_init():
    layer = TrainableSubsampling2D(activation="linear")
    x = np.random.default_rng(3).normal(size=(2, 8, 8, 3)).astype("float32")
    expected = x.reshape(2, 4, 2, 4, 2, 3).mean(axis=(2, 4))
    np.testing.assert_allclose(keras.ops.convert_to_numpy(layer(x)), expected, atol=1e-6)


def test_subsampling_has_two_parameters_per_map():
    layer = TrainableSubsampling2D()
    layer.build((None, 8, 8, 5))
    assert layer.count_params() == 10


# --------------------------------------------------------------------------- #
# Squashing function and variants
# --------------------------------------------------------------------------- #
def test_scaled_tanh_maps_targets_to_themselves():
    """f(+/-1) = +/-1, so +/-1 targets are reachable without saturating."""
    out = keras.ops.convert_to_numpy(scaled_tanh(np.array([-1.0, 0.0, 1.0], dtype="float32")))
    np.testing.assert_allclose(out, [-1.0, 0.0, 1.0], atol=1e-3)


def test_dense_variant_adds_the_missing_connections():
    keras.utils.set_random_seed(0)
    dense = build_lecun_1989(h3="dense", h3_dropout=0.5)
    # 28 extra H2->H3 connections of 25 weights each.
    assert dense.count_params() == 2578 + 28 * 25
    assert any(isinstance(l, layers.SpatialDropout2D) for l in dense.layers)


def test_softmax_head_keeps_parameter_count():
    keras.utils.set_random_seed(0)
    assert build_lecun_1989(head="softmax").count_params() == 2578


def test_invalid_options_rejected():
    with pytest.raises(ValueError):
        build_lecun_1989(h3="sparse")
    with pytest.raises(ValueError):
        build_lecun_1989(head="sigmoid")
    with pytest.raises(ValueError):
        build_lecun_1989(h3_dropout=1.0)


def test_model_round_trips_through_keras_serialisation(tmp_path, lecun):
    path = tmp_path / "lecun.keras"
    lecun.save(path)
    restored = keras.models.load_model(path)
    x = np.random.default_rng(4).uniform(-1, 1, size=(2, 28, 28, 1)).astype("float32")
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(lecun(x)),
        keras.ops.convert_to_numpy(restored(x)),
        atol=1e-6,
    )
    assert restored.count_params() == 2578
