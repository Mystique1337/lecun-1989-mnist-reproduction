"""Training recipes and a single, timed, seeded training run.

A *recipe* is everything about training that is not the network: input scaling,
target encoding, loss, optimiser, batch size and number of epochs. Keeping the
two apart is what lets the analysis separate "the 1989 network is smaller" from
"the 1989 network was trained differently".
"""

from __future__ import annotations

import os
import platform
import time
import warnings
from dataclasses import asdict, dataclass, field, replace
from typing import Callable

import keras
import numpy as np

from .data import load_mnist


@dataclass(frozen=True)
class Recipe:
    name: str
    input_scaling: str      # "unit" [0,1] or "symmetric" [-1,1]
    targets: str            # "onehot" or "pm1"
    loss: str               # Keras loss identifier
    optimizer: str          # "adam", "sgd" or "rmsprop"
    learning_rate: float | None
    batch_size: int
    epochs: int
    momentum: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def with_(self, **changes) -> "Recipe":
        return replace(self, **changes)


# Chollet (2021): Adam with Keras defaults, categorical cross-entropy,
# batch 128, 15 epochs, inputs in [0, 1], one-hot targets.
CHOLLET_RECIPE = Recipe(
    name="chollet_2021",
    input_scaling="unit",
    targets="onehot",
    loss="categorical_crossentropy",
    optimizer="adam",
    learning_rate=None,
    batch_size=128,
    epochs=15,
)

# Le Cun et al. (1989): inputs in [-1, 1] (p. 398), +/-1 targets (p. 399),
# 30 passes through the training set (p. 402). The paper reports its results
# as mean squared error (p. 402), which is taken as the training loss.
# The optimiser and learning rate are placeholders: the paper used a
# second-order variant of back-propagation (p. 402) that Keras does not
# provide, so the substitute is chosen on the validation set in the notebook.
LECUN_RECIPE = Recipe(
    name="lecun_1989",
    input_scaling="symmetric",
    targets="pm1",
    loss="mean_squared_error",
    optimizer="sgd",
    learning_rate=0.01,
    batch_size=32,
    epochs=30,
)


def make_optimizer(recipe: Recipe) -> keras.optimizers.Optimizer:
    kwargs = {} if recipe.learning_rate is None else {"learning_rate": recipe.learning_rate}
    if recipe.optimizer == "adam":
        return keras.optimizers.Adam(**kwargs)
    if recipe.optimizer == "rmsprop":
        return keras.optimizers.RMSprop(**kwargs)
    if recipe.optimizer == "sgd":
        return keras.optimizers.SGD(momentum=recipe.momentum, **kwargs)
    raise ValueError(f"unknown optimizer {recipe.optimizer!r}")


class EpochTimer(keras.callbacks.Callback):
    """Wall-clock time of every epoch, including its validation pass."""

    def on_train_begin(self, logs=None):
        self.epoch_seconds: list[float] = []

    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_seconds.append(time.perf_counter() - self._t0)


def configure_determinism() -> None:
    """Make TensorFlow kernels deterministic so a seed fixes the result."""
    if keras.backend.backend() == "tensorflow":
        import tensorflow as tf

        try:
            tf.config.experimental.enable_op_determinism()
        except Exception as exc:  # pragma: no cover - older TF
            warnings.warn(f"deterministic ops unavailable, runs may vary: {exc}")


_DATA_CACHE: dict = {}


def get_data(recipe: Recipe):
    key = (recipe.input_scaling, recipe.targets)
    if key not in _DATA_CACHE:
        _DATA_CACHE[key] = load_mnist(recipe.input_scaling, recipe.targets)
    return _DATA_CACHE[key]


@dataclass
class RunResult:
    model_name: str
    recipe: dict
    seed: int
    parameters: int
    train_seconds: float
    epoch_seconds: list[float]
    history: dict
    val_accuracy: float
    test_loss: float
    test_accuracy: float
    test_predictions: np.ndarray = field(repr=False)
    test_scores: np.ndarray = field(repr=False)
    inference_seconds: float = float("nan")
    # The trained model, kept for inspection (filters, misclassifications).
    model: keras.Model | None = field(default=None, repr=False, compare=False)

    @property
    def test_error(self) -> float:
        return 1.0 - self.test_accuracy

    def summary(self) -> dict:
        return {
            "model": self.model_name,
            "recipe": self.recipe["name"],
            "seed": self.seed,
            "parameters": self.parameters,
            "train_seconds": self.train_seconds,
            "seconds_per_epoch": float(np.mean(self.epoch_seconds)),
            "val_accuracy": self.val_accuracy,
            "test_loss": self.test_loss,
            "test_accuracy": self.test_accuracy,
            "test_error_pct": 100.0 * self.test_error,
            "inference_ms_per_image": 1000.0 * self.inference_seconds / max(len(self.test_predictions), 1),
        }


def train_and_evaluate(
    build_model: Callable[[], keras.Model],
    recipe: Recipe,
    seed: int,
    evaluate_test: bool = True,
    verbose: int = 0,
) -> RunResult:
    """Build, train and evaluate one model under one recipe with one seed.

    The test set is only touched when ``evaluate_test`` is true. Model
    selection in the notebook calls this with ``evaluate_test=False`` so that
    every design decision is made on the validation set.
    """
    configure_determinism()
    keras.utils.set_random_seed(seed)
    data = get_data(recipe)

    model = build_model()
    model.compile(
        optimizer=make_optimizer(recipe),
        loss=recipe.loss,
        # Compares argmax(target) with argmax(output), so it is correct for
        # both one-hot and +/-1 targets. The string "accuracy" is not: with an
        # MSE loss Keras would resolve it to exact element-wise equality.
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
    )

    timer = EpochTimer()
    t0 = time.perf_counter()
    history = model.fit(
        data.train.x, data.train.y,
        batch_size=recipe.batch_size,
        epochs=recipe.epochs,
        validation_data=(data.val.x, data.val.y),
        callbacks=[timer],
        verbose=verbose,
    )
    train_seconds = time.perf_counter() - t0

    val_accuracy = float(history.history["val_accuracy"][-1])
    test_loss = test_accuracy = inference_seconds = float("nan")
    predictions = scores = np.array([])
    if evaluate_test:
        test_loss, test_accuracy = (float(v) for v in model.evaluate(
            data.test.x, data.test.y, batch_size=1024, verbose=0))
        # Warm up once so graph tracing is not billed to inference time.
        model.predict(data.test.x[:1024], batch_size=1024, verbose=0)
        t1 = time.perf_counter()
        scores = model.predict(data.test.x, batch_size=1024, verbose=0)
        inference_seconds = time.perf_counter() - t1
        predictions = scores.argmax(axis=1)

    result = RunResult(
        model_name=model.name,
        recipe=recipe.to_dict(),
        seed=seed,
        parameters=int(model.count_params()),
        train_seconds=train_seconds,
        epoch_seconds=timer.epoch_seconds,
        history={k: [float(v) for v in vals] for k, vals in history.history.items()},
        val_accuracy=val_accuracy,
        test_loss=test_loss,
        test_accuracy=test_accuracy,
        test_predictions=predictions,
        test_scores=scores,
        inference_seconds=inference_seconds,
        model=model,
    )
    return result


def environment() -> dict:
    """Record what produced a result, so timings can be interpreted."""
    info = {
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "processor": platform.processor() or platform.machine(),
        "logical_cpus": os.cpu_count(),
        "keras": keras.__version__,
        "backend": keras.backend.backend(),
        "numpy": np.__version__,
    }
    if keras.backend.backend() == "tensorflow":
        import tensorflow as tf

        info["tensorflow"] = tf.__version__
        info["devices"] = [d.device_type for d in tf.config.list_physical_devices()]
    return info
