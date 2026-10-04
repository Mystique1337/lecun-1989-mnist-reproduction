"""Model definitions.

``build_chollet_2021`` reproduces the Keras example "Simple MNIST convnet"
(Chollet, 2021) layer for layer.

``build_lecun_1989`` is that code modified to follow Le Cun et al. (1989,
pp. 399-402): four hidden layers H1-H4 of 4, 4, 12 and 12 feature maps,
5x5 kernels, trainable 2x2 averaging instead of max pooling, the H2-H3
connection scheme of Table 1, a squashing function after every layer and a
10-unit output layer. Its options exist so that each departure from the
original can be switched on separately and measured:

* ``h3="dense"`` with ``h3_dropout`` replaces Table 1 by full connections plus
  dropout, a commonly suggested modern substitute;
* ``head="softmax"`` swaps the 1989 output (squashing units trained towards
  +/-1 targets with mean squared error) for Chollet's softmax output, which is
  how the 1989 topology is trained under Chollet's recipe.
"""

from __future__ import annotations

import keras
from keras import layers

from .layers import (
    LECUN_1989_TABLE1,
    ConnectionTableConv2D,
    TrainableSubsampling2D,
    scaled_tanh,
)

NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)


# --------------------------------------------------------------------------- #
# Chollet (2021)
# --------------------------------------------------------------------------- #
def build_chollet_2021(num_classes: int = NUM_CLASSES, input_shape=INPUT_SHAPE) -> keras.Model:
    """Verbatim architecture of Chollet (2021), "Simple MNIST convnet"."""
    # Adapted from Chollet, F. (2021). Simple MNIST convnet. Keras.
    # Accessed: 04.10.2026 from https://keras.io/examples/vision/mnist_convnet/
    # The layer stack below is unchanged from that example; only the model name
    # is added so results can be told apart.
    return keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="chollet_2021",
    )


# --------------------------------------------------------------------------- #
# Le Cun et al. (1989)
# --------------------------------------------------------------------------- #
# Chollet's Sequential model above is the starting point; each layer is then
# replaced to follow Le Cun, Y., Boser, B., Denker, J. S., Henderson, D.,
# Howard, R. E., Hubbard, W., & Jackel, L. D. (1989). Handwritten digit
# recognition with a back-propagation network. Advances in Neural Information
# Processing Systems 2, 396-404. Page references are given beside each layer.
def build_lecun_1989(
    h3: str = "table1",
    h3_dropout: float = 0.0,
    dropout_kind: str = "spatial",
    head: str = "tanh",
    num_classes: int = NUM_CLASSES,
    input_shape=INPUT_SHAPE,
    name: str | None = None,
) -> keras.Model:
    """The 1989 network rebuilt in Keras.

    Parameters
    ----------
    h3:
        ``"table1"`` connects H2 to H3 exactly as in Table 1 (p. 400);
        ``"dense"`` connects every H2 map to every H3 map.
    h3_dropout:
        Dropout rate applied to the H2 maps before H3 (0 disables it).
    dropout_kind:
        ``"spatial"`` drops whole feature maps (``SpatialDropout2D``), the same
        granularity at which Table 1 removes connections; ``"standard"`` drops
        individual units (``Dropout``).
    head:
        ``"tanh"`` gives the 1989 output of 10 squashing units for +/-1 targets;
        ``"softmax"`` gives Chollet's probabilistic output.
    """
    if h3 not in {"table1", "dense"}:
        raise ValueError("h3 must be 'table1' or 'dense'")
    if head not in {"tanh", "softmax"}:
        raise ValueError("head must be 'tanh' or 'softmax'")
    if dropout_kind not in {"spatial", "standard"}:
        raise ValueError("dropout_kind must be 'spatial' or 'standard'")
    if not 0.0 <= h3_dropout < 1.0:
        raise ValueError("h3_dropout must be in [0, 1)")

    init = "lecun_uniform"  # std = 1/sqrt(fan_in), as for ConnectionTableConv2D
    model_layers = [
        keras.Input(shape=input_shape),
        # H1: 4 shared-weight feature maps of 24x24, 5x5 receptive fields (p. 400).
        layers.Conv2D(4, kernel_size=(5, 5), activation=scaled_tanh,
                      kernel_initializer=init, name="H1"),
        # H2: averaging and 2:1 subsampling to 4 maps of 12x12 (p. 400).
        TrainableSubsampling2D(name="H2"),
    ]

    if h3_dropout > 0:
        dropout_cls = layers.SpatialDropout2D if dropout_kind == "spatial" else layers.Dropout
        model_layers.append(dropout_cls(h3_dropout, name="H2_dropout"))

    # H3: 12 feature maps of 8x8, 5x5 receptive fields (pp. 400-402).
    if h3 == "table1":
        model_layers.append(ConnectionTableConv2D(LECUN_1989_TABLE1, kernel_size=5, name="H3"))
    else:
        model_layers.append(
            layers.Conv2D(12, kernel_size=(5, 5), activation=scaled_tanh,
                          kernel_initializer=init, name="H3")
        )

    model_layers += [
        # H4: averaging and 2:1 subsampling to 12 maps of 4x4 (p. 402).
        TrainableSubsampling2D(name="H4"),
        layers.Flatten(name="flatten"),
        # Output: 10 units fully connected to H4 (p. 402).
        layers.Dense(
            num_classes,
            activation=scaled_tanh if head == "tanh" else "softmax",
            kernel_initializer=init,
            name="output",
        ),
    ]

    if name is None:
        name = "lecun_1989" + ("" if h3 == "table1" else "_dense") \
            + (f"_drop{h3_dropout:g}" if h3_dropout else "") \
            + ("" if head == "tanh" else "_softmax")
    return keras.Sequential(model_layers, name=name)


# --------------------------------------------------------------------------- #
# Network accounting in the terms Le Cun et al. (1989, p. 402) report
# --------------------------------------------------------------------------- #
def network_accounting(model: keras.Model) -> dict:
    """Count units, connections and free parameters the way the paper does.

    A *connection* is one weighted input to one unit, with each unit's bias
    counted as a connection from a constant bias unit. Weight sharing means
    many connections use the same *free parameter*, which is why the paper's
    98,442 connections need only 2,578 parameters.

    Dropout and Flatten layers are transparent: they add no units.
    """
    input_shape = model.inputs[0].shape
    units = {"input": int(input_shape[1] * input_shape[2] * input_shape[3])}
    connections = {}
    in_channels = int(input_shape[3])

    for layer in model.layers:
        if isinstance(layer, (layers.Dropout, layers.SpatialDropout2D, layers.Flatten)):
            continue
        out_shape = layer.output.shape
        n_units = 1
        for dim in out_shape[1:]:
            n_units *= int(dim)

        if isinstance(layer, ConnectionTableConv2D):
            area = layer.kernel_size ** 2
            per_map_inputs = layer.table.sum(axis=0)  # inputs feeding each output map
            positions = int(out_shape[1] * out_shape[2])
            conn = int(sum(positions * (area * n + 1) for n in per_map_inputs))
            in_channels = layer.filters
        elif isinstance(layer, layers.Conv2D):
            kh, kw = layer.kernel_size
            conn = n_units * (kh * kw * in_channels + 1)
            in_channels = layer.filters
        elif isinstance(layer, TrainableSubsampling2D):
            conn = n_units * (2 * 2 + 1)
        elif isinstance(layer, layers.MaxPooling2D):
            conn = 0  # no weighted connections
        elif isinstance(layer, layers.Dense):
            fan_in = int(layer.input.shape[-1])
            conn = n_units * (fan_in + 1)
        else:
            raise TypeError(f"no accounting rule for layer {layer.name} ({type(layer).__name__})")

        units[layer.name] = n_units
        connections[layer.name] = conn

    return {
        "units_per_layer": units,
        "connections_per_layer": connections,
        "units": sum(units.values()),
        "units_including_bias_unit": sum(units.values()) + 1,
        "connections": sum(connections.values()),
        "parameters": int(model.count_params()),
    }
