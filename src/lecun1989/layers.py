"""Keras layers needed to rebuild the 1989 network that Keras does not provide.

Le Cun et al. (1989, pp. 400-402) describe two layer types with no direct
equivalent in a modern framework:

* **Averaging/subsampling layers (H2, H4).** Each unit averages a
  non-overlapping 2x2 patch of one feature map, and "all the weights are
  constrained to be equal, even within a single unit" (p. 400). A unit
  therefore has one shared weight and one bias per map. The paper does not
  say whether these units are squashed; the squashing function is applied
  here, as in the subsampling layers of LeCun et al. (1998, pp. 2284-2285).
  ``keras.layers.AveragePooling2D`` has no trainable parameters, so it alone
  would not reproduce the published parameter count.

* **A partially connected convolution between H2 and H3.** Each H3 map reads
  from one or two H2 maps according to Table 1 (p. 400), not from all of them.
  ``keras.layers.Conv2D`` always connects every input map to every output map.

Both layers below are written with backend-agnostic ``keras.ops`` calls. They
are tested on the TensorFlow backend, which produced every reported result.
"""

from __future__ import annotations

import keras
import numpy as np
from keras import ops

# --------------------------------------------------------------------------- #
# Table 1 of Le Cun et al. (1989, p. 400): which H2 maps feed which H3 maps.
# Rows are the 4 H2 maps, columns the 12 H3 maps; 1 marks a connection.
# The two blocks of six columns are the "two almost independent modules"
# described on p. 402: H3.1-H3.6 read only from H2.1/H2.2, and H3.7-H3.12
# only from H2.3/H2.4.
# --------------------------------------------------------------------------- #
LECUN_1989_TABLE1 = np.array(
    [
        # 1  2  3  4  5  6  7  8  9 10 11 12
        [1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # H2.1
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # H2.2
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1],  # H2.3
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  # H2.4
    ],
    dtype="int32",
)


# --------------------------------------------------------------------------- #
# Squashing function
# --------------------------------------------------------------------------- #
@keras.saving.register_keras_serializable(package="lecun1989")
def scaled_tanh(x):
    """The scaled hyperbolic tangent f(a) = 1.7159 tanh(2a/3).

    The 1989 paper only says each convolution is "followed by a squashing
    function" (p. 399). This particular scaling is the one recommended by
    LeCun et al. (1998, pp. 2285, 2318) for targets of +/-1: it gives
    f(+/-1) = +/-1, so the targets sit inside the function's range rather than
    at its asymptotes, where gradients vanish.
    """
    return 1.7159 * ops.tanh(x * (2.0 / 3.0))


# --------------------------------------------------------------------------- #
# H2 / H4: trainable averaging and subsampling
# --------------------------------------------------------------------------- #
@keras.saving.register_keras_serializable(package="lecun1989")
class TrainableSubsampling2D(keras.layers.Layer):
    """2x2 averaging with one trainable coefficient and one bias per map.

    Each output unit computes ``f(w_c * mean(2x2 patch) + b_c)`` for map ``c``.
    Because averaging and summing differ only by a constant factor absorbed
    into ``w_c``, this is the layer of Le Cun et al. (1989, p. 400) with
    exactly two free parameters per map.
    """

    def __init__(self, activation=scaled_tanh, **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        channels = int(input_shape[-1])
        self.coefficient = self.add_weight(
            name="coefficient", shape=(channels,), initializer="ones", trainable=True
        )
        self.bias = self.add_weight(
            name="bias", shape=(channels,), initializer="zeros", trainable=True
        )

    def call(self, inputs):
        pooled = ops.average_pool(
            inputs, pool_size=2, strides=2, padding="valid", data_format="channels_last"
        )
        return self.activation(pooled * self.coefficient + self.bias)

    def compute_output_shape(self, input_shape):
        batch, height, width, channels = input_shape
        half = lambda n: None if n is None else n // 2  # noqa: E731
        return (batch, half(height), half(width), channels)

    def get_config(self):
        config = super().get_config()
        config["activation"] = keras.activations.serialize(self.activation)
        return config


# --------------------------------------------------------------------------- #
# H3: convolution restricted to the connections of Table 1
# --------------------------------------------------------------------------- #
@keras.saving.register_keras_serializable(package="lecun1989")
class ConnectionTableConv2D(keras.layers.Layer):
    """A 'valid' convolution where output map j reads only the input maps
    marked in column j of a binary connection table.

    Only the kernels for existing connections are stored as weights, so the
    layer's parameter count is genuine: Keras reports 20 x 25 + 12 = 512 for
    Table 1, not the 4 x 12 x 25 + 12 = 1,212 of a dense ``Conv2D``. In the
    forward pass those kernels are scattered into a full (k, k, in, out) kernel
    whose absent connections are structurally zero, so they never receive a
    gradient and can never become non-zero.

    Weights are initialised uniformly in +/- sqrt(3 / fan_in), giving a standard
    deviation of 1/sqrt(fan_in), where fan_in is counted per output map (25 for a
    map with one input, 50 for a map with two). A dense initialiser would use the
    same fan-in for every map and mis-scale half of them.
    """

    def __init__(self, table, kernel_size=5, activation=scaled_tanh, **kwargs):
        super().__init__(**kwargs)
        self.table = np.asarray(table, dtype="int32")
        if self.table.ndim != 2 or not np.isin(self.table, (0, 1)).all():
            raise ValueError("table must be a 2-D binary array of shape (in_maps, out_maps)")
        if (self.table.sum(axis=0) == 0).any():
            raise ValueError("every output map needs at least one input connection")
        self.kernel_size = int(kernel_size)
        self.activation = keras.activations.get(activation)

    @property
    def filters(self) -> int:
        return int(self.table.shape[1])

    @property
    def n_connections(self) -> int:
        return int(self.table.sum())

    def build(self, input_shape):
        in_maps, out_maps = self.table.shape
        if int(input_shape[-1]) != in_maps:
            raise ValueError(
                f"connection table expects {in_maps} input maps, got {input_shape[-1]}"
            )
        k = self.kernel_size
        pairs = np.argwhere(self.table == 1)  # (n_connections, 2) as (input, output)

        # One-hot scatter matrix: connection index -> flattened (input, output) slot.
        scatter = np.zeros((len(pairs), in_maps * out_maps), dtype="float32")
        for idx, (i, j) in enumerate(pairs):
            scatter[idx, i * out_maps + j] = 1.0
        self._scatter = scatter

        self.kernel = self.add_weight(
            name="kernel", shape=(k, k, len(pairs)), initializer="zeros", trainable=True
        )
        self.bias = self.add_weight(
            name="bias", shape=(out_maps,), initializer="zeros", trainable=True
        )

        # Per-output-map fan-in initialisation. Drawn from NumPy's global
        # generator, which keras.utils.set_random_seed() also seeds.
        fan_in = k * k * self.table.sum(axis=0)
        limits = np.sqrt(3.0 / fan_in[pairs[:, 1]]).astype("float32")
        init = np.random.uniform(-1.0, 1.0, size=(k, k, len(pairs))).astype("float32")
        self.kernel.assign(init * limits)

    def full_kernel(self):
        """Assemble the dense (k, k, in, out) kernel with absent links zeroed."""
        k = self.kernel_size
        in_maps, out_maps = self.table.shape
        scatter = ops.convert_to_tensor(self._scatter, dtype=self.kernel.dtype)
        dense = ops.matmul(self.kernel, scatter)
        return ops.reshape(dense, (k, k, in_maps, out_maps))

    def call(self, inputs):
        outputs = ops.conv(
            inputs, self.full_kernel(), strides=1, padding="valid",
            data_format="channels_last",
        )
        return self.activation(outputs + self.bias)

    def compute_output_shape(self, input_shape):
        batch, height, width, _ = input_shape
        shrink = lambda n: None if n is None else n - self.kernel_size + 1  # noqa: E731
        return (batch, shrink(height), shrink(width), self.filters)

    def get_config(self):
        config = super().get_config()
        config.update(
            table=self.table.tolist(),
            kernel_size=self.kernel_size,
            activation=keras.activations.serialize(self.activation),
        )
        return config
