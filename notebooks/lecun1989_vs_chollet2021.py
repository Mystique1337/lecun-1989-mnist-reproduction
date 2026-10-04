# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Le Cun et al. (1989) rebuilt in Keras, compared with Chollet (2021)
#
# This notebook runs every experiment, in order, from raw data to the final tables and figures.
# Every reported number is computed here.
#
# **Sources of the code.** The baseline network is the Keras example *Simple MNIST convnet* by
# Chollet (2021). The reconstruction modifies that code to follow Le Cun et al. (1989), "Handwritten
# digit recognition with a back-propagation network", *Advances in Neural Information Processing
# Systems 2*, pp. 396-404. Both sources are cited at the point of use. The reusable code (custom
# layers, models, training loop, statistics, figures) lives in the package `src/lecun1989/`.
#
# **Protocol.** Every design decision for the reconstruction (optimiser, learning rate, dropout
# substitute) is made on the validation set. The test set is used only in Section 4, once per run,
# after all choices are fixed.
#
# | Section | Content |
# |---|---|
# | 1 | The MNIST dataset |
# | 2 | Both models, and a check of the reconstruction against the paper's published totals |
# | 3 | Model selection on the validation set |
# | 4 | Final experiments: four configurations, five seeds each |
# | 5 | Statistical comparison |
# | 6 | Learning behaviour and error analysis |
# | 7 | Export of tables, figures and the graphical abstract |

# %%
# Setup. The TensorFlow backend is the one Chollet (2021) was written for.
import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import warnings
from pathlib import Path

# Keep the executed notebook readable: silence library deprecation notices.
warnings.filterwarnings("ignore")

import keras
import numpy as np
import pandas as pd
from IPython.display import Image, display

import lecun1989
from lecun1989 import figures as F
from lecun1989.data import class_counts, load_mnist, load_raw
from lecun1989.layers import LECUN_1989_TABLE1
from lecun1989.models import build_chollet_2021, build_lecun_1989, network_accounting
from lecun1989.stats import confusion_matrix, mcnemar_exact, mean_ci
from lecun1989.training import CHOLLET_RECIPE, LECUN_RECIPE, environment, train_and_evaluate

# The project root is located from the installed package, so the notebook runs
# from any working directory (install with `pip install -e .`).
ROOT = Path(lecun1989.__file__).resolve().parents[2]
assert (ROOT / "pyproject.toml").exists(), "install the package with `pip install -e .`"

# LECUN1989_SMOKE=1 runs every cell with 1 epoch and 2 seeds into a scratch
# folder, to check the notebook end to end in minutes. It is off for real runs.
SMOKE = os.environ.get("LECUN1989_SMOKE") == "1"
OUT = Path(os.environ.get("LECUN1989_OUT", ROOT / "results"))
FIG, TAB, MET = OUT / "figures", OUT / "tables", OUT / "metrics"
for d in (FIG, TAB, MET):
    d.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
if SMOKE:
    SEEDS = [0, 1]
    CHOLLET_RECIPE = CHOLLET_RECIPE.with_(epochs=1)
    LECUN_RECIPE = LECUN_RECIPE.with_(epochs=1)
F.set_style()
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

ENV = environment()
(MET / "environment.json").write_text(json.dumps(ENV, indent=2))
for k, v in ENV.items():
    print(f"{k:>14}: {v}")

# %% [markdown]
# ## 1 The MNIST dataset

# %%
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = load_raw()
print("training images:", x_train_raw.shape, x_train_raw.dtype)
print("test images:    ", x_test_raw.shape, x_test_raw.dtype)
print("pixel range:    ", x_train_raw.min(), "to", x_train_raw.max())
print("share of pixels that are exactly 0 (background): "
      f"{100 * np.mean(x_train_raw == 0):.1f}%")

# Where does the ink sit? MNIST centres each digit by centre of mass in a
# 28x28 frame, so the outer rows and columns should be almost always blank.
ink = (x_train_raw > 0).any(axis=0)
rows, cols = np.flatnonzero(ink.any(axis=1)), np.flatnonzero(ink.any(axis=0))
print(f"rows ever inked: {rows.min()}-{rows.max()}, columns ever inked: {cols.min()}-{cols.max()}")
bbox = np.array([(np.ptp(np.flatnonzero(im.any(axis=1))) + 1, np.ptp(np.flatnonzero(im.any(axis=0))) + 1)
                 for im in (x_train_raw[:10000] > 0)])
print(f"per-image ink bounding box (first 10,000): median height {np.median(bbox[:, 0]):.0f}px, "
      f"median width {np.median(bbox[:, 1]):.0f}px, max {bbox.max()}px")

# %%
# The partition used by every model: Keras' validation_split=0.1 takes the
# LAST 10% of the training arrays, so Chollet (2021) trains on images 0-53,999.
data = load_mnist("unit", "onehot")
split_table = pd.DataFrame(
    {
        "split": ["train", "validation", "test"],
        "images": [len(data.train), len(data.val), len(data.test)],
        "source": ["MNIST training set, images 0-53,999", "MNIST training set, images 54,000-59,999",
                   "MNIST test set"],
    }
)
counts = pd.DataFrame(
    {"digit": range(10), "train": class_counts(data.train.labels),
     "validation": class_counts(data.val.labels), "test": class_counts(data.test.labels)}
)
counts["test_share_%"] = 100 * counts["test"] / counts["test"].sum()
split_table.to_csv(TAB / "t01_partition.csv", index=False)
counts.to_csv(TAB / "t01_class_counts.csv", index=False)
display(split_table)
display(counts.round(2))
print(f"largest / smallest class in training split: "
      f"{counts['train'].max() / counts['train'].min():.2f}x")

# %%
F.save(F.mnist_samples(x_test_raw, y_test_raw, per_class=10), FIG / "fig01_mnist_samples.png")
F.save(F.class_distribution(counts["train"], counts["validation"], counts["test"]),
       FIG / "fig01b_class_distribution.png")
display(Image(FIG / "fig01_mnist_samples.png", width=330))
display(Image(FIG / "fig01b_class_distribution.png", width=520))

# %% [markdown]
# ## 2 The two models
#
# ### 2.1 Chollet (2021)
# The architecture below is the Keras example unchanged (`build_chollet_2021` in `src/lecun1989/models.py`).

# %%
keras.utils.set_random_seed(0)
chollet = build_chollet_2021()
chollet.summary()

# %% [markdown]
# ### 2.2 Le Cun et al. (1989), reconstructed
#
# Chollet's `Sequential` model is modified layer by layer to follow pp. 399-402 of the paper. The two
# layers Keras lacks are implemented in `src/lecun1989/layers.py`: `TrainableSubsampling2D` (H2, H4)
# and `ConnectionTableConv2D` (H3, with the connection scheme of Table 1, p. 400).

# %%
keras.utils.set_random_seed(0)
lecun = build_lecun_1989()
lecun.summary()

print("Table 1 (rows: H2 maps, columns: H3 maps):")
print(pd.DataFrame(LECUN_1989_TABLE1, index=[f"H2.{i + 1}" for i in range(4)],
                   columns=[f"H3.{j + 1}" for j in range(12)]).replace({1: "X", 0: "."}).to_string())

# %% [markdown]
# **Check against the paper.** Le Cun et al. (1989, p. 402) state that the network "has 4635 units,
# 98442 connections, and 2578 independent parameters". The accounting below derives all three numbers
# from the Keras model itself. Matching all three confirms the layer sizes, kernels, weight and bias
# sharing and the number of H2-H3 connections. The totals do not depend on which 20 of the 48 possible
# connections are present, so the pattern of Table 1 is checked against the paper by a separate unit test.

# %%
acc = network_accounting(lecun)
check = pd.DataFrame(
    {
        "quantity": ["units (incl. bias unit)", "connections", "independent parameters"],
        "Le Cun et al. (1989, p. 402)": [4635, 98442, 2578],
        "this reconstruction": [acc["units_including_bias_unit"], acc["connections"], acc["parameters"]],
    }
)
check["match"] = check.iloc[:, 1] == check.iloc[:, 2]
per_layer = pd.DataFrame(
    {
        "layer": list(acc["connections_per_layer"]),
        "units": [acc["units_per_layer"][k] for k in acc["connections_per_layer"]],
        "connections": list(acc["connections_per_layer"].values()),
        "parameters": [lecun.get_layer(k).count_params() for k in acc["connections_per_layer"]],
    }
)
check.to_csv(TAB / "t02_paper_check.csv", index=False)
per_layer.to_csv(TAB / "t02_per_layer.csv", index=False)
display(check)
display(per_layer)
assert check["match"].all(), "reconstruction does not reproduce the published totals"

# %%
F.save(F.architecture_diagram(), FIG / "fig02_architectures.png")
display(Image(FIG / "fig02_architectures.png", width=620))

# %% [markdown]
# ## 3 Model selection on the validation set
#
# Two things about the reconstruction cannot be read off the paper and are chosen here, on the 6,000
# validation images only.
#
# **Optimiser.** The paper used "a second-order version of back-propagation" (p. 402), which Keras does
# not provide. Two substitutes are compared: plain stochastic gradient descent, the first-order
# algorithm the 1989 method extends, at four learning rates; and RMSprop, which, like a diagonal
# second-order method, gives every weight its own step size. The selection rule, fixed in advance,
# is the highest validation accuracy after the 30 epochs the paper used.
#
# **A dropout substitute for Table 1.** A natural modern alternative to the sparse H2-H3 connections
# is to connect everything and regularise with dropout. H3 is therefore connected to all four H2 maps, and dropout is applied to
# the H2 maps. `SpatialDropout2D` removes whole maps, the same granularity at which Table 1 removes
# connections; ordinary `Dropout` removes single units. Both are tried at two rates, alongside a fully
# connected H3 without dropout as a reference.

# %%
selection_rows = []

def select(label, build, recipe, group):
    r = train_and_evaluate(build, recipe, seed=0, evaluate_test=False)
    best = int(np.argmax(r.history["val_accuracy"])) + 1
    row = {"group": group, "configuration": label, "parameters": r.parameters,
           "val_accuracy_final": r.val_accuracy,
           "val_accuracy_best": max(r.history["val_accuracy"]), "best_epoch": best,
           "train_loss_final": r.history["loss"][-1], "train_seconds": r.train_seconds}
    selection_rows.append(row)
    print(f"{label:<42} final val acc {r.val_accuracy:.4f}  "
          f"(best {row['val_accuracy_best']:.4f} @ epoch {best}, {r.train_seconds:.0f}s)")
    return r

optimiser_grid = [
    ("SGD, learning rate 0.01", LECUN_RECIPE.with_(optimizer="sgd", learning_rate=0.01)),
    ("SGD, learning rate 0.03", LECUN_RECIPE.with_(optimizer="sgd", learning_rate=0.03)),
    ("SGD, learning rate 0.1", LECUN_RECIPE.with_(optimizer="sgd", learning_rate=0.1)),
    ("SGD, learning rate 0.3", LECUN_RECIPE.with_(optimizer="sgd", learning_rate=0.3)),
    ("RMSprop, learning rate 0.001", LECUN_RECIPE.with_(optimizer="rmsprop", learning_rate=0.001)),
]
opt_results = {label: select(label, build_lecun_1989, rec, "optimiser") for label, rec in optimiser_grid}

best_opt_label = max(opt_results, key=lambda k: opt_results[k].val_accuracy)
LECUN_FINAL = dict(optimiser_grid)[best_opt_label]
print(f"\nselected: {best_opt_label}")

# %%
dropout_grid = [
    ("Full H2-H3 connections, no dropout", dict(h3="dense", h3_dropout=0.0)),
    ("Full + SpatialDropout2D 0.25", dict(h3="dense", h3_dropout=0.25, dropout_kind="spatial")),
    ("Full + SpatialDropout2D 0.5", dict(h3="dense", h3_dropout=0.5, dropout_kind="spatial")),
    ("Full + Dropout 0.25", dict(h3="dense", h3_dropout=0.25, dropout_kind="standard")),
    ("Full + Dropout 0.5", dict(h3="dense", h3_dropout=0.5, dropout_kind="standard")),
]
drop_results = {
    label: select(label, lambda kw=kw: build_lecun_1989(**kw), LECUN_FINAL, "dropout")
    for label, kw in dropout_grid
}
candidates = {label: drop_results[label] for label, kw in dropout_grid if kw.get("h3_dropout", 0) > 0}
best_drop_label = max(candidates, key=lambda k: candidates[k].val_accuracy)
DROPOUT_KW = dict(dropout_grid)[best_drop_label]
print(f"\nselected dropout substitute: {best_drop_label}")
print(f"Table 1 reconstruction at the same settings (seed 0): "
      f"{opt_results[best_opt_label].val_accuracy:.4f}")

selection = pd.DataFrame(selection_rows)
selection.to_csv(TAB / "t03_model_selection.csv", index=False)
display(selection.round(4))

# %% [markdown]
# ### Parameters of the three networks
#
# Chollet (2021), the paper, and this implementation side by side.

# %%
opt_name = {"sgd": "SGD", "rmsprop": "RMSprop", "adam": "Adam"}[LECUN_FINAL.optimizer]
params_table = pd.DataFrame(
    [
        ("Data set", "MNIST", "U.S. Postal Service zip codes + printed digits (p. 397)", "MNIST"),
        ("Training / validation / test", "54,000 / 6,000 / 10,000", "9,840 / none reported / 2,707 (p. 397)",
         "54,000 / 6,000 / 10,000"),
        ("Input", "28×28, scaled to [0, 1]", "16×16 digit in a 28×28 plane, [−1, 1] (pp. 398, 400)",
         "28×28, scaled to [−1, 1]"),
        ("Hidden layers", "2 conv + 2 max pool", "4: H1-H4 (p. 400)", "4: H1-H4"),
        ("Feature maps per layer", "32, 64", "4, 4, 12, 12 (pp. 400-402)", "4, 4, 12, 12"),
        ("Filter size", "3×3", "5×5 conv, 2×2 averaging (p. 400)", "5×5 conv, 2×2 averaging"),
        ("Pooling", "Max, 2×2", "Trainable averaging, 2×2 (p. 400)", "Trainable averaging, 2×2"),
        ("H2-H3 connections", "n/a", "Table 1, 20 of 48 (p. 400)", "Table 1, exact"),
        ("Activation", "ReLU; softmax output", "Squashing function (p. 399)", "1.7159 tanh(2a/3)"),
        ("Targets and loss", "One-hot, cross-entropy", "±1 targets (p. 399), MSE (p. 402)", "±1 targets, MSE"),
        ("Regularisation", "Dropout 0.5", "Weight sharing, sparse connections", "Weight sharing, Table 1"),
        ("Free parameters", "34,826", "2,578 (p. 402)", f"{lecun.count_params():,}"),
        ("Optimiser", "Adam, learning rate 0.001", "Second-order back-propagation (p. 402)",
         f"{opt_name}, learning rate {LECUN_FINAL.learning_rate:g}"),
        ("Batch size", "128", "Not stated", str(LECUN_FINAL.batch_size)),
        ("Epochs", "15", "30 passes (p. 402)", str(LECUN_FINAL.epochs)),
    ],
    columns=["Aspect", "Chollet (2021)", "Le Cun et al. (1989)", "This implementation"],
)
params_table.to_csv(TAB / "t04_parameters.csv", index=False)
display(params_table)

# %% [markdown]
# ## 4 Final experiments
#
# Four configurations, each trained with five seeds and evaluated once on the test set:
#
# | Key | Network | Training recipe | Why it is here |
# |---|---|---|---|
# | `chollet` | Chollet (2021) | Chollet (2021) | the modern baseline |
# | `lecun` | 1989 network with Table 1 | 1989 recipe (selected optimiser) | the reconstruction |
# | `lecun_dropout` | 1989 network, full H2-H3 + selected dropout | 1989 recipe | tests dropout as a substitute for Table 1 |
# | `lecun_chollet_recipe` | 1989 network with Table 1, softmax output | Chollet (2021) | separates the effect of the network from the effect of the training recipe |
#
# Runs are interleaved by seed (all four configurations for seed 0, then seed 1, ...) so that any drift in
# machine load affects every configuration equally. Training time is the wall-clock duration of
# `model.fit`, which, as in Chollet's example, includes the validation pass after each epoch.

# %%
CONFIGS = {
    "chollet": (build_chollet_2021, CHOLLET_RECIPE),
    "lecun": (build_lecun_1989, LECUN_FINAL),
    "lecun_dropout": (lambda: build_lecun_1989(**DROPOUT_KW), LECUN_FINAL),
    "lecun_chollet_recipe": (lambda: build_lecun_1989(head="softmax"), CHOLLET_RECIPE),
}

runs = {key: [] for key in CONFIGS}
for seed in SEEDS:
    for key, (build, recipe) in CONFIGS.items():
        verbose = 2 if (seed == 0 and key in {"chollet", "lecun"}) else 0
        if verbose:
            print(f"\n--- {key}, seed {seed}: full training log ---")
        r = train_and_evaluate(build, recipe, seed=seed, evaluate_test=True, verbose=verbose)
        runs[key].append(r)
        print(f"seed {seed}  {key:<22} test acc {r.test_accuracy:.4f}  test loss {r.test_loss:.4f}  "
              f"train {r.train_seconds:6.1f}s  ({np.mean(r.epoch_seconds):.2f}s/epoch)")
        if seed == 0:
            r.model.save(MET / f"model_{key}_seed0.keras")

# %%
per_run = pd.DataFrame([r.summary() | {"config": key} for key, rs in runs.items() for r in rs])
per_run.to_csv(TAB / "t05_all_runs.csv", index=False)
for key, rs in runs.items():
    for r in rs:
        (MET / f"history_{key}_seed{r.seed}.json").write_text(json.dumps(
            {"history": r.history, "epoch_seconds": r.epoch_seconds, "recipe": r.recipe}, indent=1))
        np.save(MET / f"pred_{key}_seed{r.seed}.npy", r.test_predictions)
display(per_run.drop(columns=["recipe"]).round(4))

# %%
def summarise(key):
    rs = runs[key]
    row = {"config": key, "label": F.SERIES[key]["label"], "parameters": rs[0].parameters,
           "loss_function": rs[0].recipe["loss"]}
    for name, values in [
        ("test_accuracy_pct", [100 * r.test_accuracy for r in rs]),
        ("test_error_pct", [100 * r.test_error for r in rs]),
        ("test_loss", [r.test_loss for r in rs]),
        ("train_seconds", [r.train_seconds for r in rs]),
        ("seconds_per_epoch", [np.mean(r.epoch_seconds) for r in rs]),
        ("inference_ms_per_image", [1000 * r.inference_seconds / len(r.test_predictions) for r in rs]),
    ]:
        m, lo, hi = mean_ci(values)
        row |= {f"{name}_mean": m, f"{name}_sd": float(np.std(values, ddof=1)),
                f"{name}_lo": lo, f"{name}_hi": hi}
    return row

summary = pd.DataFrame([summarise(k) for k in CONFIGS]).set_index("config")
summary.to_csv(TAB / "t06_summary.csv")

headline = pd.DataFrame(
    {
        "Model": [F.SERIES[k]["label"] for k in CONFIGS],
        "Parameters": [f"{summary.loc[k, 'parameters']:,}" for k in CONFIGS],
        "Test loss (function)": [f"{summary.loc[k, 'test_loss_mean']:.4f} ± {summary.loc[k, 'test_loss_sd']:.4f} "
                                 f"({'CE' if 'cross' in summary.loc[k, 'loss_function'] else 'MSE'})"
                                 for k in CONFIGS],
        "Test accuracy (%)": [f"{summary.loc[k, 'test_accuracy_pct_mean']:.2f} ± {summary.loc[k, 'test_accuracy_pct_sd']:.2f}"
                              for k in CONFIGS],
        "Training time (s)": [f"{summary.loc[k, 'train_seconds_mean']:.1f} ± {summary.loc[k, 'train_seconds_sd']:.1f}"
                              for k in CONFIGS],
        "Time per epoch (s)": [f"{summary.loc[k, 'seconds_per_epoch_mean']:.2f}" for k in CONFIGS],
    }
)
headline.to_csv(TAB / "t07_headline.csv", index=False)
print("Mean ± standard deviation over 5 seeds; test set of 10,000 images")
display(headline)

# %% [markdown]
# Note that the two test losses are **not** on a common scale: Chollet's model is trained and evaluated with
# categorical cross-entropy, the 1989 recipe with mean squared error against ±1 targets. Accuracy is the
# comparable measure. The fourth configuration trains the 1989 network with cross-entropy, which does give
# a loss directly comparable with Chollet's.

# %% [markdown]
# ## 5 Statistical comparison
#
# Accuracy differences of a few tenths of a percent on 10,000 images need a test. Two complementary
# views are reported: a confidence interval across seeds (does the gap survive retraining?) and an exact
# McNemar test on each seed's paired predictions (does the gap survive the choice of test images?).

# %%
y_test = data.test.labels
pairs = [("chollet", "lecun"), ("lecun", "lecun_dropout"), ("lecun", "lecun_chollet_recipe"),
         ("chollet", "lecun_chollet_recipe")]
rows = []
for a, b in pairs:
    diffs = [100 * (ra.test_accuracy - rb.test_accuracy) for ra, rb in zip(runs[a], runs[b])]
    m, lo, hi = mean_ci(diffs)
    tests = [mcnemar_exact(y_test, ra.test_predictions, rb.test_predictions) for ra, rb in zip(runs[a], runs[b])]
    rows.append({
        "comparison": f"{a} vs {b}",
        "mean_difference_pp": m, "ci95_low_pp": lo, "ci95_high_pp": hi,
        "seeds_a_better": sum(d > 0 for d in diffs),
        "mcnemar_significant_seeds": sum(t["p_value"] < 0.05 for t in tests),
        "mcnemar_max_p": max(t["p_value"] for t in tests),
        "discordant_a_only_mean": np.mean([t["a_only"] for t in tests]),
        "discordant_b_only_mean": np.mean([t["b_only"] for t in tests]),
    })
stats = pd.DataFrame(rows)
stats.to_csv(TAB / "t08_statistics.csv", index=False)
display(stats.round(4))

# %% [markdown]
# ## 6 Learning behaviour and errors

# %%
curves = {k: np.array([r.history["val_accuracy"] for r in runs[k]]) for k in CONFIGS}
F.save(F.learning_curves(curves, ylim=(0.955, 0.995)), FIG / "fig03_learning_curves.png")
display(Image(FIG / "fig03_learning_curves.png", width=560))

final_train_gap = pd.DataFrame({
    "config": list(CONFIGS),
    "final_train_accuracy": [np.mean([r.history["accuracy"][-1] for r in runs[k]]) for k in CONFIGS],
    "final_val_accuracy": [np.mean([r.history["val_accuracy"][-1] for r in runs[k]]) for k in CONFIGS],
})
final_train_gap["gap_pp"] = 100 * (final_train_gap["final_train_accuracy"] - final_train_gap["final_val_accuracy"])
final_train_gap.to_csv(TAB / "t09_generalisation_gap.csv", index=False)
display(final_train_gap.round(4))

# %%
F.save(
    F.metric_dots(summary, list(CONFIGS), [
        ("test_error_pct", "Test error (%)", False),
        ("train_seconds", "Training time (s)", False),
    ]),
    FIG / "fig04_comparison.png",
)
display(Image(FIG / "fig04_comparison.png", width=560))

# %%
# Error analysis on the seed-0 models (the seed shown in the full training logs).
cm = {k: confusion_matrix(y_test, runs[k][0].test_predictions) for k in ("chollet", "lecun")}
F.save(F.confusion_pair(cm, {"chollet": "Chollet (2021)", "lecun": "Le Cun et al. (1989), rebuilt"}),
       FIG / "fig05_confusion.png")
display(Image(FIG / "fig05_confusion.png", width=600))

per_class = pd.DataFrame({
    "digit": range(10),
    "test_images": np.bincount(y_test, minlength=10),
    **{f"{k}_error_pct": [100 * np.mean(np.array([r.test_predictions for r in runs[k]])[:, y_test == d] != d)
                          for d in range(10)] for k in CONFIGS},
})
per_class.to_csv(TAB / "t10_per_class_error.csv", index=False)
display(per_class.round(2))

off = {k: (c - np.diag(np.diag(c))) for k, c in cm.items()}
for k, c in off.items():
    i, j = np.unravel_index(np.argsort(c, axis=None)[::-1][:3], c.shape)
    print(f"{k:<8} most frequent confusions (seed 0): " +
          ", ".join(f"{a}→{b} ({c[a, b]})" for a, b in zip(i, j)))

# %%
# Which images does the 1989 network get wrong that Chollet's gets right?
lecun_wrong = runs["lecun"][0].test_predictions != y_test
chollet_wrong = runs["chollet"][0].test_predictions != y_test
print(f"seed 0: both wrong {np.sum(lecun_wrong & chollet_wrong)}, only 1989 wrong "
      f"{np.sum(lecun_wrong & ~chollet_wrong)}, only Chollet wrong {np.sum(~lecun_wrong & chollet_wrong)}")
only_lecun = np.flatnonzero(lecun_wrong & ~chollet_wrong)
F.save(F.misclassified_grid(x_test_raw[only_lecun], y_test[only_lecun],
                            runs["lecun"][0].test_predictions[only_lecun], n=24),
       FIG / "fig06_lecun_only_errors.png")
display(Image(FIG / "fig06_lecun_only_errors.png", width=600))

# %%
# The learned H1 kernels and their feature maps for one test digit, the view of
# Figure 3 in Le Cun et al. (1989, p. 399).
model = runs["lecun"][0].model
kernels = keras.ops.convert_to_numpy(model.get_layer("H1").kernel)[:, :, 0, :]
probe = keras.Model(model.inputs, model.get_layer("H1").output)
lecun_data = load_mnist("symmetric", "pm1")
digit = int(np.flatnonzero(y_test == 0)[0])
maps = keras.ops.convert_to_numpy(probe(lecun_data.test.x[digit:digit + 1]))[0]
F.save(F.h1_filters_and_maps(kernels, maps, x_test_raw[digit]), FIG / "fig07_h1_filters.png")
display(Image(FIG / "fig07_h1_filters.png", width=540))

# %% [markdown]
# ## 7 Graphical abstract and export

# %%
s = summary
delta = s.loc["chollet", "test_accuracy_pct_mean"] - s.loc["lecun", "test_accuracy_pct_mean"]
ratio = s.loc["chollet", "parameters"] / s.loc["lecun", "parameters"]
numbers = {
    "chollet_err": s.loc["chollet", "test_error_pct_mean"],
    "lecun_err": s.loc["lecun", "test_error_pct_mean"],
    "chollet_params": s.loc["chollet", "parameters"],
    "lecun_params": s.loc["lecun", "parameters"],
    "chollet_time": s.loc["chollet", "train_seconds_mean"],
    "lecun_time": s.loc["lecun", "train_seconds_mean"],
    "conclusion": (f"Yes. The rebuilt\nnetwork matches all\nthree published totals\nand reaches "
                   f"{s.loc['lecun', 'test_accuracy_pct_mean']:.2f}%.\n\nIt trails Chollet by\n"
                   f"{delta:.2f} points with\n{ratio:.1f}× fewer weights."),
}
F.save(F.graphical_abstract(x_test_raw[digit], numbers), FIG / "fig00_graphical_abstract.png")
display(Image(FIG / "fig00_graphical_abstract.png", width=640))

# %%
# A single machine-readable record of every headline number, so the README and
# any write-up can be checked against what this notebook actually produced.
record = {
    "environment": ENV,
    "seeds": SEEDS,
    "paper_check": check.to_dict(orient="records"),
    "selected_optimiser": best_opt_label,
    "selected_recipe": LECUN_FINAL.to_dict(),
    "selected_dropout": best_drop_label,
    "selection": selection.to_dict(orient="records"),
    "summary": summary.reset_index().to_dict(orient="records"),
    "statistics": stats.to_dict(orient="records"),
}
(MET / "results.json").write_text(json.dumps(record, indent=2, default=float))
print("wrote", (MET / "results.json").relative_to(OUT.parent))
print("tables:", sorted(p.name for p in TAB.glob("*.csv")))
print("figures:", sorted(p.name for p in FIG.glob("*.png")))
