"""Figures for the notebook.

Colour never identifies a series on its own: every series also has its own
marker and line style, so the figures stay readable in greyscale and for
readers with colour-vision deficiency.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from .layers import LECUN_1989_TABLE1

# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"

SERIES = {
    "chollet": {"label": "Chollet (2021)", "color": "#2a78d6", "marker": "o", "ls": "-"},
    "lecun": {"label": "Le Cun et al. (1989), reconstructed", "color": "#eb6834", "marker": "s", "ls": "--"},
    "lecun_dropout": {"label": "Le Cun topology, dropout for Table 1", "color": "#1baf7a", "marker": "^", "ls": "-."},
    "lecun_chollet_recipe": {"label": "Le Cun topology, Chollet's training", "color": "#eda100", "marker": "D", "ls": ":"},
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "lines.linewidth": 1.6,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
        }
    )


def save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def mnist_samples(images: np.ndarray, labels: np.ndarray, per_class: int = 8, seed: int = 0):
    """A grid with one row per digit class, drawn at random from the given split."""
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(10, per_class, figsize=(per_class * 0.42, 10 * 0.42))
    for digit in range(10):
        idx = rng.choice(np.flatnonzero(labels == digit), per_class, replace=False)
        for j, i in enumerate(idx):
            ax = axes[digit, j]
            ax.imshow(images[i], cmap="gray_r", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
        axes[digit, 0].set_ylabel(str(digit), rotation=0, labelpad=8, va="center",
                                  fontsize=8, color=INK_2)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    return fig


def class_distribution(train_counts, val_counts, test_counts):
    """Share of each class per split, as grouped dots rather than bars:
    the classes differ by a few percentage points, and dots on a common scale
    make that visible without a misleading zero-based bar."""
    fig, ax = plt.subplots(figsize=(4.6, 2.2))
    x = np.arange(10)
    for k, (counts, name, key) in enumerate(
        [(train_counts, "train", "chollet"), (val_counts, "validation", "lecun"), (test_counts, "test", "lecun_dropout")]
    ):
        share = 100 * np.asarray(counts) / np.sum(counts)
        s = SERIES[key]
        ax.plot(x + (k - 1) * 0.18, share, s["marker"], color=s["color"], ms=4.5,
                label=f"{name} (n = {int(np.sum(counts)):,})", ls="none")
    ax.axhline(10, color=MUTED, lw=0.8)
    ax.set_xticks(x)
    ax.set_xlabel("Digit class")
    ax.set_ylabel("Share of split (%)")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    return fig


# --------------------------------------------------------------------------- #
# Method
# --------------------------------------------------------------------------- #
def _block(ax, x, y, w, h, title, subtitle, color, alpha=0.16):
    style = "round,pad=0,rounding_size=0.012"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style,
                                linewidth=0, facecolor=color, alpha=alpha))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style,
                                linewidth=0.9, edgecolor=color, facecolor="none"))
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=7.2,
            fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center", fontsize=6.3,
            color=INK_2, linespacing=1.15)


def _pipeline(ax, y, blocks, color, header, params):
    n = len(blocks)
    left, right = 0.005, 0.995
    gap = 0.026
    w = (right - left - gap * (n - 1)) / n
    h = 0.30
    for i, (title, sub) in enumerate(blocks):
        x = left + i * (w + gap)
        _block(ax, x, y, w, h, title, sub, color)
        if i < n - 1:
            ax.annotate("", xy=(x + w + gap, y + h / 2), xytext=(x + w, y + h / 2),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8,
                                        shrinkA=0, shrinkB=0, mutation_scale=7))
    ax.text(left, y + h + 0.035, header, fontsize=8, fontweight="bold", color=INK, va="bottom")
    ax.text(right, y + h + 0.035, params, fontsize=7.5, color=INK_2, va="bottom", ha="right")


def architecture_diagram():
    """Both networks as aligned pipelines, with Table 1 drawn underneath."""
    fig = plt.figure(figsize=(6.3, 3.5))
    ax = fig.add_axes([0.0, 0.36, 1.0, 0.64])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _pipeline(
        ax, 0.60,
        [
            ("Input", "28×28×1\n[0, 1]"),
            ("Conv", "32 @ 3×3\nReLU → 26×26"),
            ("Max pool", "2×2\n→ 13×13×32"),
            ("Conv", "64 @ 3×3\nReLU → 11×11"),
            ("Max pool", "2×2\n→ 5×5×64"),
            ("Dropout", "rate 0.5\non 1,600 units"),
            ("Dense", "10, softmax\ncross-entropy"),
        ],
        SERIES["chollet"]["color"], "Chollet (2021)", "34,826 parameters",
    )
    _pipeline(
        ax, 0.08,
        [
            ("Input", "28×28×1\n[−1, 1]"),
            ("H1 conv", "4 @ 5×5\nf(·) → 24×24"),
            ("H2 average", "2×2, 1 w + 1 b\nper map → 12×12"),
            ("H3 conv", "12 @ 5×5\nTable 1 → 8×8"),
            ("H4 average", "2×2, 1 w + 1 b\nper map → 4×4"),
            ("Flatten", "192 units"),
            ("Output", "10, f(·)\n±1 targets, MSE"),
        ],
        SERIES["lecun"]["color"], "Le Cun et al. (1989), reconstructed", "2,578 parameters",
    )

    # Table 1 inset
    tab = fig.add_axes([0.10, 0.0, 0.50, 0.25])
    _table1(tab)
    fig.text(0.63, 0.12,
             "f(a) = 1.7159 tanh(2a/3)\n"
             "Table 1: each H3 map reads one or two\n"
             "H2 maps, giving two almost independent\n"
             "modules and 20 of the 48 possible\n"
             "connections (pp. 400, 402).",
             fontsize=6.6, color=INK_2, va="center", linespacing=1.3)
    return fig


def _table1(ax):
    t = LECUN_1989_TABLE1
    color = SERIES["lecun"]["color"]
    ax.imshow(np.zeros_like(t), cmap="Greys", vmin=0, vmax=1, aspect="equal")
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            if t[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.42, i - 0.42), 0.84, 0.84, color=color, lw=0))
    ax.set_xticks(range(12), labels=[str(j + 1) for j in range(12)], fontsize=6.3)
    ax.set_yticks(range(4), labels=[f"H2.{i + 1}" for i in range(4)], fontsize=6.3)
    ax.set_xlabel("Table 1 (Le Cun et al., 1989, p. 400): H3 feature map", fontsize=6.6, labelpad=4)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(-0.5, 12, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax.grid(which="minor", color="#cfcdc6", lw=0.5)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0)
    ax.axvline(5.5, color=INK_2, lw=1.0)
    for s in ax.spines.values():
        s.set_visible(False)


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
def learning_curves(curves: dict, metric: str = "val_accuracy", ylim=(0.95, 1.0)):
    """Validation curves, mean over seeds with a min-max band, labelled at the end.

    ``curves`` maps a series key to an array of shape (n_seeds, n_epochs).
    """
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    ends = []
    for key, runs in curves.items():
        runs = np.asarray(runs)
        s = SERIES[key]
        epochs = np.arange(1, runs.shape[1] + 1)
        mean = runs.mean(axis=0)
        ax.fill_between(epochs, runs.min(axis=0), runs.max(axis=0), color=s["color"], alpha=0.15, lw=0)
        ax.plot(epochs, mean, color=s["color"], ls=s["ls"], marker=s["marker"],
                ms=3.2, markevery=max(1, len(epochs) // 6), label=s["label"])
        ends.append((epochs[-1], mean[-1], s))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy" if "acc" in metric else metric)
    ax.set_ylim(*ylim)
    ax.set_xlim(0.5, max(e for e, _, _ in ends) + 0.5)
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    return fig


def metric_dots(summary, models: list[str], panels: list[tuple[str, str, bool]]):
    """Small multiples, one measure per panel on its own axis (never a dual axis).

    ``summary`` is a DataFrame indexed by series key with columns
    ``<measure>_mean``, ``<measure>_lo``, ``<measure>_hi``.
    ``panels`` lists (measure, axis label, log-scale?).
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(2.15 * len(panels), 2.0), sharey=True)
    axes = np.atleast_1d(axes)
    y = np.arange(len(models))[::-1]
    for ax, (measure, label, log) in zip(axes, panels):
        for yi, key in zip(y, models):
            s = SERIES[key]
            mean = summary.loc[key, f"{measure}_mean"]
            lo = summary.loc[key, f"{measure}_lo"]
            hi = summary.loc[key, f"{measure}_hi"]
            ax.plot([lo, hi], [yi, yi], color=s["color"], lw=2.2, solid_capstyle="round")
            ax.plot(mean, yi, s["marker"], color=s["color"], ms=5.5, mec="white", mew=0.8)
        ax.set_xlabel(label)
        if log:
            ax.set_xscale("log")
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.18)
        if not log:
            ax.set_xlim(left=0)
    axes[0].set_yticks(y, labels=[SERIES[k]["label"].replace(", ", ",\n") for k in models], fontsize=7)
    axes[0].set_ylim(-0.6, len(models) - 0.4)
    fig.subplots_adjust(wspace=0.12)
    return fig


def confusion_pair(cms: dict, titles: dict):
    """Row-normalised confusion matrices with the diagonal suppressed, so the
    colour scale is spent on the errors rather than on the ~99% correct cells."""
    keys = list(cms)
    fig, axes = plt.subplots(1, len(keys), figsize=(3.0 * len(keys), 2.8))
    axes = np.atleast_1d(axes)
    vmax = max(
        ((cm / cm.sum(axis=1, keepdims=True)) * (1 - np.eye(10))).max() for cm in cms.values()
    ) * 100
    for ax, key in zip(axes, keys):
        cm = cms[key]
        rate = 100 * cm / cm.sum(axis=1, keepdims=True)
        off = np.where(np.eye(10, dtype=bool), np.nan, rate)
        im = ax.imshow(off, cmap="Blues", vmin=0, vmax=vmax)
        for i in range(10):
            for j in range(10):
                if i != j and cm[i, j] >= 3:
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=5.6,
                            color="white" if off[i, j] > vmax * 0.55 else INK)
            ax.text(i, i, f"{rate[i, i]:.1f}", ha="center", va="center", fontsize=5.0, color=MUTED)
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xlabel("Predicted digit")
        ax.set_ylabel("True digit")
        ax.set_title(titles[key], fontsize=7.8)
        ax.grid(False)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Errors as % of true class", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)
    return fig


def misclassified_grid(images, y_true, y_pred, n: int = 24, cols: int = 12):
    idx = np.flatnonzero(y_true != y_pred)[:n]
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.52, rows * 0.66))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, i in zip(np.ravel(axes), idx):
        ax.imshow(images[i], cmap="gray_r", vmin=0, vmax=255)
        ax.set_title(f"{y_true[i]}→{y_pred[i]}", fontsize=6.5, color=INK, pad=1.5,
                     fontweight="normal", loc="center")
    fig.subplots_adjust(wspace=0.08, hspace=0.35)
    return fig


def h1_filters_and_maps(kernels: np.ndarray, feature_maps: np.ndarray, image: np.ndarray):
    """The four learned H1 kernels and the feature maps they produce for one
    digit, the same view as Figure 3 of Le Cun et al. (1989, p. 399)."""
    fig, axes = plt.subplots(2, 5, figsize=(5.4, 2.3))
    axes[0, 0].imshow(image, cmap="gray_r")
    axes[0, 0].set_title("Input", fontsize=7, loc="center")
    axes[1, 0].axis("off")
    lim = np.abs(kernels).max()
    for k in range(4):
        axes[0, k + 1].imshow(kernels[..., k], cmap="RdBu_r", vmin=-lim, vmax=lim)
        axes[0, k + 1].set_title(f"H1.{k + 1} kernel", fontsize=7, loc="center")
        axes[1, k + 1].imshow(feature_maps[..., k], cmap="gray_r")
        axes[1, k + 1].set_title(f"H1.{k + 1} map", fontsize=7, loc="center")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.subplots_adjust(wspace=0.12, hspace=0.32)
    return fig


# --------------------------------------------------------------------------- #
# Graphical abstract
# --------------------------------------------------------------------------- #
def graphical_abstract(sample_digit: np.ndarray, numbers: dict):
    """Motivation, method, result and conclusion in one strip.

    ``numbers`` holds the headline values to print, so the figure is always
    drawn from the measured results rather than typed in by hand. Keys:
    chollet_err, lecun_err, chollet_params, lecun_params, chollet_time,
    lecun_time, conclusion.
    """
    fig = plt.figure(figsize=(6.3, 2.55))

    def panel(x, w, title):
        ax = fig.add_axes([x, 0.04, w, 0.80])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        fig.text(x + 0.005, 0.90, title, fontsize=8.2, fontweight="bold", color=INK)
        return ax

    # 1. Motivation
    ax = panel(0.00, 0.22, "1  Question")
    ax.text(0.03, 0.86,
            "Can the first CNN for\ndigit recognition\n(Le Cun et al., 1989)\nbe rebuilt in modern\n"
            "Keras, and how far\nis it from a 2021\nreference model?",
            fontsize=7.0, color=INK_2, va="top", linespacing=1.3)

    # 2. Method
    ax = panel(0.235, 0.27, "2  Method")
    img = fig.add_axes([0.24, 0.30, 0.075, 0.30])
    img.imshow(sample_digit, cmap="gray_r")
    img.axis("off")
    img.set_title("MNIST\n60k / 10k", fontsize=6.3, color=INK_2, pad=2, loc="center")
    for yy, key, text in [(0.70, "chollet", "Chollet (2021)\nReLU, max pooling,\ndropout"),
                          (0.30, "lecun", "Le Cun (1989) rebuilt\nTable 1, averaging,\n±1 targets, MSE")]:
        c = SERIES[key]["color"]
        ax.add_patch(FancyBboxPatch((0.38, yy - 0.15), 0.60, 0.30,
                                    boxstyle="round,pad=0,rounding_size=0.03",
                                    facecolor=c, alpha=0.16, edgecolor="none"))
        ax.add_patch(FancyBboxPatch((0.38, yy - 0.15), 0.60, 0.30,
                                    boxstyle="round,pad=0,rounding_size=0.03",
                                    facecolor="none", edgecolor=c, lw=0.9))
        ax.text(0.68, yy, text, ha="center", va="center", fontsize=6.1, color=INK, linespacing=1.2)
        ax.annotate("", xy=(0.38, yy), xytext=(0.30, 0.47),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8, mutation_scale=7))
    ax.text(0.03, 0.02, "Same data split, 5 seeds each,\nsame CPU", fontsize=6.2, color=INK_2,
            va="bottom")

    # 3. Results. Every bar starts at zero, so bar length is proportional to the
    # value; error rather than accuracy is shown because a zero-based accuracy
    # bar would hide a difference of tenths of a percent.
    fig.text(0.525, 0.90, "3  Results (mean of 5 seeds)", fontsize=8.2, fontweight="bold", color=INK)
    rows = [
        ("Test error (%)", numbers["chollet_err"], numbers["lecun_err"], "{:.2f}"),
        ("Free parameters", numbers["chollet_params"], numbers["lecun_params"], "{:,.0f}"),
        ("Training time (s)", numbers["chollet_time"], numbers["lecun_time"], "{:.0f}"),
    ]
    for r, (label, a, b, fmt) in enumerate(rows):
        y0 = 0.60 - r * 0.235
        ax = fig.add_axes([0.53, y0, 0.18, 0.14])
        for yi, (v, key) in enumerate(zip((a, b), ("chollet", "lecun"))):
            ax.barh(1 - yi, v, color=SERIES[key]["color"], height=0.62)
            ax.text(v, 1 - yi, "  " + fmt.format(v), va="center", ha="left", fontsize=6.5, color=INK)
        ax.set_xlim(0, max(a, b) * 1.55)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.axvline(0, color=MUTED, lw=0.6)
        ax.set_title(label, fontsize=6.6, color=INK_2, loc="left", pad=1.5, fontweight="normal")
    # Legend: a coloured mark carries identity, the text stays in ink.
    for i, key in enumerate(("chollet", "lecun")):
        yy = 0.075 - i * 0.06
        fig.patches.append(FancyBboxPatch((0.532, yy - 0.012), 0.012, 0.03, boxstyle="square,pad=0",
                                          transform=fig.transFigure, facecolor=SERIES[key]["color"],
                                          edgecolor="none"))
        fig.text(0.55, yy + 0.003, "Chollet (2021)" if key == "chollet" else "Le Cun (1989) rebuilt",
                 fontsize=6.4, color=INK, va="center")

    # 4. Conclusion
    ax = panel(0.81, 0.19, "4  Answer")
    ax.text(0.03, 0.86, numbers["conclusion"], fontsize=6.8, color=INK_2, va="top", linespacing=1.3)
    return fig
