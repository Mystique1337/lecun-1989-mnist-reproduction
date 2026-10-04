"""MNIST loading, partitioning and preprocessing.

The partition reproduces Chollet (2021) exactly. Keras' ``validation_split``
"is selected from the last samples in the x and y data provided, before
shuffling", so ``validation_split=0.1`` trains on the first 54,000 training
images and validates on the last 6,000. Every model in this project uses that
identical 54,000 / 6,000 / 10,000 partition, so differences between them cannot
come from different data.
"""

from __future__ import annotations

from dataclasses import dataclass

import keras
import numpy as np

NUM_CLASSES = 10
VALIDATION_SPLIT = 0.1


@dataclass(frozen=True)
class Split:
    x: np.ndarray
    y: np.ndarray        # model targets (one-hot or +/-1)
    labels: np.ndarray   # integer class labels, for analysis

    def __len__(self) -> int:
        return len(self.labels)


@dataclass(frozen=True)
class MNIST:
    train: Split
    val: Split
    test: Split


def load_raw():
    """The 60,000 / 10,000 MNIST split as distributed by Keras (uint8)."""
    return keras.datasets.mnist.load_data()


def scale_images(images: np.ndarray, scaling: str) -> np.ndarray:
    """Map uint8 pixels to floats and add the channel axis.

    ``"unit"`` gives [0, 1], as in Chollet (2021).
    ``"symmetric"`` gives [-1, 1], as in Le Cun et al. (1989, p. 398): the
    background maps to -1 and full ink to +1, the same polarity as the paper's
    feature-map illustrations (p. 399).
    """
    x = images.astype("float32")
    if scaling == "unit":
        x = x / 255.0
    elif scaling == "symmetric":
        x = x / 127.5 - 1.0
    else:
        raise ValueError("scaling must be 'unit' or 'symmetric'")
    return np.expand_dims(x, -1)


def encode_targets(labels: np.ndarray, targets: str) -> np.ndarray:
    """``"onehot"`` for softmax/cross-entropy; ``"pm1"`` for +1 on the true
    class and -1 elsewhere, the targets of Le Cun et al. (1989, p. 399)."""
    onehot = keras.utils.to_categorical(labels, NUM_CLASSES).astype("float32")
    if targets == "onehot":
        return onehot
    if targets == "pm1":
        return 2.0 * onehot - 1.0
    raise ValueError("targets must be 'onehot' or 'pm1'")


def load_mnist(scaling: str, targets: str, validation_split: float = VALIDATION_SPLIT) -> MNIST:
    """Load MNIST and partition it exactly as Keras' ``validation_split`` does."""
    (x_train, y_train), (x_test, y_test) = load_raw()
    n_val = int(round(len(x_train) * validation_split))
    cut = len(x_train) - n_val

    def make(images, labels):
        return Split(scale_images(images, scaling), encode_targets(labels, targets), labels.astype("int64"))

    return MNIST(
        train=make(x_train[:cut], y_train[:cut]),
        val=make(x_train[cut:], y_train[cut:]),
        test=make(x_test, y_test),
    )


def class_counts(labels: np.ndarray) -> np.ndarray:
    return np.bincount(labels, minlength=NUM_CLASSES)
