"""Tests for preprocessing, partitioning and the statistical helpers."""

from __future__ import annotations

import numpy as np
import pytest

from lecun1989.data import encode_targets, load_mnist, scale_images
from lecun1989.stats import confusion_matrix, mcnemar_exact, mean_ci
from lecun1989.training import CHOLLET_RECIPE, LECUN_RECIPE, Recipe, make_optimizer


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def test_unit_scaling_range():
    x = scale_images(np.array([[[0, 255]]], dtype="uint8"), "unit")
    assert x.shape == (1, 1, 2, 1)
    assert x.min() == 0.0 and x.max() == 1.0


def test_symmetric_scaling_maps_background_and_ink():
    x = scale_images(np.array([[[0, 255]]], dtype="uint8"), "symmetric")
    np.testing.assert_allclose(x.ravel(), [-1.0, 1.0])


def test_pm1_targets():
    t = encode_targets(np.array([3]), "pm1")
    assert t[0, 3] == 1.0
    assert (np.delete(t[0], 3) == -1.0).all()


def test_bad_options_rejected():
    with pytest.raises(ValueError):
        scale_images(np.zeros((1, 2, 2), dtype="uint8"), "zscore")
    with pytest.raises(ValueError):
        encode_targets(np.array([0]), "smooth")


# --------------------------------------------------------------------------- #
# Partition (downloads MNIST once, ~11 MB)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def mnist():
    return load_mnist("unit", "onehot")


def test_partition_sizes(mnist):
    assert (len(mnist.train), len(mnist.val), len(mnist.test)) == (54000, 6000, 10000)


def test_validation_is_the_last_ten_percent(mnist):
    """Matches Keras' validation_split, which takes the final samples."""
    from lecun1989.data import load_raw

    (x_train, y_train), _ = load_raw()
    np.testing.assert_array_equal(mnist.val.labels, y_train[-6000:])
    np.testing.assert_array_equal(mnist.train.labels, y_train[:54000])


def test_all_classes_present_in_every_split(mnist):
    for split in (mnist.train, mnist.val, mnist.test):
        assert set(np.unique(split.labels)) == set(range(10))


# --------------------------------------------------------------------------- #
# Recipes
# --------------------------------------------------------------------------- #
def test_chollet_recipe_matches_published_example():
    assert (CHOLLET_RECIPE.batch_size, CHOLLET_RECIPE.epochs) == (128, 15)
    assert CHOLLET_RECIPE.optimizer == "adam"
    assert CHOLLET_RECIPE.loss == "categorical_crossentropy"


def test_lecun_recipe_follows_paper():
    assert LECUN_RECIPE.epochs == 30          # "30 training passes", p. 402
    assert LECUN_RECIPE.input_scaling == "symmetric"
    assert LECUN_RECIPE.targets == "pm1"
    assert LECUN_RECIPE.loss == "mean_squared_error"


def test_make_optimizer_honours_learning_rate():
    opt = make_optimizer(LECUN_RECIPE.with_(optimizer="sgd", learning_rate=0.05))
    assert float(opt.learning_rate) == pytest.approx(0.05)
    with pytest.raises(ValueError):
        make_optimizer(Recipe("x", "unit", "onehot", "mse", "lbfgs", None, 1, 1))


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def test_mcnemar_identical_predictions():
    y = np.arange(10) % 10
    assert mcnemar_exact(y, y, y) == {"a_only": 0, "b_only": 0, "p_value": 1.0}


def test_mcnemar_detects_clear_difference():
    y = np.zeros(200, dtype=int)
    a = y.copy()
    b = y.copy()
    b[:40] = 1
    result = mcnemar_exact(y, a, b)
    assert result["a_only"] == 40 and result["b_only"] == 0
    assert result["p_value"] < 1e-6


def test_mean_ci_brackets_mean():
    mean, lo, hi = mean_ci([0.98, 0.99, 0.985, 0.987, 0.983])
    assert lo < mean < hi


def test_confusion_matrix_counts():
    cm = confusion_matrix([0, 1, 1, 2], [0, 1, 2, 2], n_classes=3)
    assert cm.tolist() == [[1, 0, 0], [0, 1, 1], [0, 0, 1]]
