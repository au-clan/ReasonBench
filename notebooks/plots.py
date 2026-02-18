import numpy as np
import pandas as pd

from scipy.stats import t

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.collections import PolyCollection

import seaborn as sns

from utils import get_cis

METHODS = {
    "react": "ReAct",
    "foa": "FoA",
    "io": "IO",
    "cot": "CoT",
    "cot_sc": "CoT-SC",
    "tot_bfs": "ToT-BFS",
    "tot_dfs": "ToT-DFS",
    "got": "GoT",
    "mcts": "MCTS*",
    "rap": "RAP",
    "reflexion": "Reflexion"

}

BENCHMARKS = {
    "game24": "Game of 24",
    "hle": "HLE",
    "hotpotqa": "HotpotQA",
    "humaneval": "HumanEval",
    "scibench": "SciBench",
    "sonnetwriting": "Sonnet Writing",
    "matharena": "Math Arena"
}

Y_LABEL_FONT = 18
X_LABEL_FONT = 18
X_TICK_FONT = 16
Y_TICK_FONT = 14
LEGEND_FONT = 12

def mean_ci95(values: pd.Series):
    """Return (mean, lo, hi) 95% CI using t-interval."""
    x = values.dropna().to_numpy(dtype=float)
    n = x.size
    if n == 0:
        return np.nan, np.nan, np.nan
    m = x.mean()
    if n == 1:
        return m, m, m  # no uncertainty with 1 sample
    
    # Following + get_cis is the same but just for consistency I'm using get_cis everywhere
    # s = x.std(ddof=1)
    # se = s / np.sqrt(n)
    # crit = t.ppf(0.975, df=n-1)
    # lo = m - crit * se
    # hi = m + crit * se

    lo, hi = get_cis(x)
    return m, lo, hi

def plot_errorbar(
    df: pd.DataFrame,
    benchmark_col="Benchmark",
    method_col="Method",
    value_col="Accuracy",
    benchmark_order=None,
    method_order=None,
    figsize=(10, 4),
    capsize=4,
):
    df["Method"] = df["Method"].map(METHODS)
    df["Benchmark"] = df["Benchmark"].map(BENCHMARKS)
    
    # Aggregate to mean and CI
    agg = (
        df.groupby([benchmark_col, method_col])[value_col]
          .apply(mean_ci95)
          .apply(pd.Series)
          .rename(columns={0: "mean", 1: "lo", 2: "hi"})
          .reset_index()
    )

    # Orders
    if benchmark_order is None:
        benchmark_order = list(pd.unique(df[benchmark_col]))
    if method_order is None:
        method_order = list(pd.unique(df[method_col]))

    # Ensure categorical ordering
    agg[benchmark_col] = pd.Categorical(agg[benchmark_col], categories=benchmark_order, ordered=True)
    agg[method_col] = pd.Categorical(agg[method_col], categories=method_order, ordered=True)
    agg = agg.sort_values([benchmark_col, method_col])

    # X positions
    benchmarks = benchmark_order
    methods = method_order
    x_base = np.arange(len(benchmarks))
    b_to_x = {b: i for i, b in enumerate(benchmarks)}

    # Dodging offsets (side-by-side within each benchmark)
    k = len(methods)
    span = 0.6                     # total width occupied by methods in a benchmark
    offsets = np.linspace(-span/2, span/2, k)

    # Simple styling (auto colors/markers)
    markers = ["o", "s", "v", "D", "^", "P", "X", "<", ">"]
    method_to_marker = {m: markers[i % len(markers)] for i, m in enumerate(methods)}

    fig, ax = plt.subplots(figsize=figsize)

    palette = sns.color_palette("colorblind") #palette = plt.get_cmap("Accent")  # or "Set2", "Dark2", "Paired"
    method_colors = {
        m: palette[i] for i, m in enumerate(methods)
    }

    for j, m in enumerate(methods):
        d = agg[agg[method_col] == m]
        x = d[benchmark_col].map(b_to_x).astype(float).to_numpy() + offsets[j]
        y = d["mean"].to_numpy()
        yerr = np.vstack([y - d["lo"].to_numpy(), d["hi"].to_numpy() - y])

        ax.errorbar(
            x, y, yerr=yerr,
            fmt=method_to_marker[m],
            color=method_colors[m],
            ecolor=method_colors[m],
            capsize=capsize,
            elinewidth=2.5,
            markersize=7,
            linewidth=0,
            label=str(m),
            alpha=0.95,
        )

    # Reference line at 0 if your accuracy is centered around 0.
    # If accuracy is in [0,1], you probably *don't* want this; comment out if not needed.
    # ax.axhline(0, color="black", linestyle="--", linewidth=2, alpha=0.6)

    ax.set_xticks(x_base)
    ax.set_xticklabels([str(b) for b in benchmarks])
    #ax.set_xlabel(benchmark_col)
    ax.set_ylabel(f"Quality", fontsize=Y_LABEL_FONT)
    ax.legend(frameon=False, ncol=min(4, len(methods)), fontsize=LEGEND_FONT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.05)

    ax.tick_params(axis='x', labelsize=X_TICK_FONT)
    ax.tick_params(axis='y', labelsize=Y_TICK_FONT)

    

    plt.tight_layout()
    return fig, ax, agg

def plot_panel_errorbar(
    df: pd.DataFrame,
    benchmark_col="Benchmark",
    method_col="Method",
    value_col="Accuracy",
    benchmark_order=None,
    method_order=None,
    ncols=3,
    figsize_per_col=4,
    figsize_per_row=3,
    capsize=4,
):
    df["Method"] = df["Method"].map(METHODS)
    df["Benchmark"] = df["Benchmark"].map(BENCHMARKS)
    
    # Aggregate mean + 95% CI
    agg = (
        df.groupby([benchmark_col, method_col])[value_col]
          .apply(mean_ci95)
          .apply(pd.Series)
          .rename(columns={0: "mean", 1: "lo", 2: "hi"})
          .reset_index()
    )

    if benchmark_order is None:
        benchmark_order = list(pd.unique(df[benchmark_col]))
    if method_order is None:
        method_order = list(pd.unique(df[method_col]))

    agg[benchmark_col] = pd.Categorical(
        agg[benchmark_col], categories=benchmark_order, ordered=True
    )
    agg[method_col] = pd.Categorical(
        agg[method_col], categories=method_order, ordered=True
    )

    # Layout
    n_bench = len(benchmark_order)
    ncols = min(ncols, n_bench)
    nrows = int(np.ceil(n_bench / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_col * ncols, figsize_per_row * nrows),
        sharex='col',
        #sharey='row'
    )
    axes = np.atleast_1d(axes).ravel()

    x = np.arange(len(method_order))

    markers = ["o", "s", "v", "D", "^", "P", "X", "<", ">"]
    method_to_marker = {
        m: markers[i % len(markers)] for i, m in enumerate(method_order)
    }

    palette = sns.color_palette("colorblind") 
    
    methods = list(pd.unique(df[method_col]))
    method_colors = {
        m: palette[i] for i, m in enumerate(methods)
    }

    for idx, bench in enumerate(benchmark_order):
        ax = axes[idx]
        d = agg[agg[benchmark_col] == bench].sort_values(method_col)

        for i, m in enumerate(method_order):
            dm = d[d[method_col] == m]
            if dm.empty:
                continue

            mean = dm["mean"].iloc[0]
            lo = dm["lo"].iloc[0]
            hi = dm["hi"].iloc[0]

            ax.errorbar(
                i,
                mean,
                yerr=[[mean - lo], [hi - mean]],
                fmt=method_to_marker[m],
                color=method_colors[m],
                ecolor=method_colors[m],
                capsize=capsize,
                elinewidth=2.5,
                markersize=7,
                linewidth=0,
                label=str(m),
                alpha=0.95
            )

        ax.set_title(str(bench), fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(method_order, rotation=25, ha="right")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        #ax.legend(frameon=False, ncol=min(4, len(methods)), fontsize=LEGEND_FONT)
        ax.tick_params(axis="x", labelbottom=False)
    
    # Disable empty panels
    for ax in axes[len(benchmark_order):]:
        ax.axis("off")
        ax.tick_params(axis='x', labelsize=X_TICK_FONT)
        ax.tick_params(axis='y', labelsize=Y_TICK_FONT)

     # ---- Shared legend ----
    handles = [
        Patch(facecolor=method_colors[m], edgecolor="black", label=str(m))
        for m in method_order
    ]

    
    legend_ncol = min(len(method_order), 11)

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=legend_ncol,
        frameon=False,
        fontsize=LEGEND_FONT,
    )

    #fig.supxlabel(method_col)
    fig.supylabel(f"Quality", fontsize=Y_LABEL_FONT)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.08)
    return fig, axes, agg

def plot_panel_boxplot(
    df: pd.DataFrame,
    benchmark_col="Benchmark",
    method_col="Method",
    value_col="Accuracy",
    benchmark_order=None,
    method_order=None,
    ncols=3,
    figsize_per_col=4,
    figsize_per_row=3,
    showfliers=True,
    legend_loc="upper center",
    legend_ncol=None,
):
    
    df["Method"] = df["Method"].map(METHODS)
    df["Benchmark"] = df["Benchmark"].map(BENCHMARKS)

    if benchmark_order is None:
        benchmark_order = list(pd.unique(df[benchmark_col]))
    if method_order is None:
        method_order = list(pd.unique(df[method_col]))


    # Layout
    n_bench = len(benchmark_order)
    ncols = min(ncols, n_bench)
    nrows = int(np.ceil(n_bench / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_col * ncols, figsize_per_row * nrows),
        sharex='col',
        sharey='row'
    )
    axes = np.atleast_1d(axes).ravel()

    x = np.arange(len(method_order))

    palette = sns.color_palette("colorblind") 
    methods = list(pd.unique(df[method_col]))
    method_colors = {
        m: palette[i] for i, m in enumerate(methods)
    }

    for idx, bench in enumerate(benchmark_order):
        ax = axes[idx]
        d = df[df[benchmark_col] == bench]

        data = [
            d.loc[d[method_col] == m, value_col].dropna().to_numpy()
            for m in method_order
        ]

        bp = ax.boxplot(
            data,
            positions=x,
            widths=0.6,
            patch_artist=True,
            showfliers=showfliers,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
        )

        # Color boxes by method
        for box, m in zip(bp["boxes"], method_order):
            box.set_facecolor(method_colors[m])
            box.set_alpha(0.8)
            box.set_edgecolor("black")

        ax.set_title(str(bench), fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(method_order, rotation=25, ha="right")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelbottom=False)

    # Turn off unused panels
    for ax in axes[len(benchmark_order):]:
        ax.axis("off")
        ax.tick_params(axis='x', labelsize=X_TICK_FONT)
        ax.tick_params(axis='y', labelsize=Y_TICK_FONT)

    # ---- Shared legend ----
    handles = [
        Patch(facecolor=method_colors[m], edgecolor="black", label=str(m))
        for m in method_order
    ]
    
    legend_ncol = min(len(method_order), 11)

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=legend_ncol,
        frameon=False,
        fontsize=LEGEND_FONT,
    )

    #fig.supxlabel(method_col)
    fig.supylabel(f"Quality", fontsize=Y_LABEL_FONT)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.08)
    return fig, axes

def plot_panel_violin(
    df: pd.DataFrame,
    benchmark_col="Benchmark",
    method_col="Method",
    value_col="Accuracy",
    benchmark_order=None,
    method_order=None,
    ncols=3,
    figsize_per_col=4,
    figsize_per_row=3,
    inner="quartile",   # 'box', 'quartile', 'point', or None
    legend_loc="upper center",
    legend_ncol=None,
):
    
    df["Method"] = df["Method"].map(METHODS)
    df["Benchmark"] = df["Benchmark"].map(BENCHMARKS)

    if benchmark_order is None:
        benchmark_order = list(pd.unique(df[benchmark_col]))
    if method_order is None:
        method_order = list(pd.unique(df[method_col]))

    # Ensure categorical ordering
    df = df.copy()
    df[benchmark_col] = pd.Categorical(
        df[benchmark_col], categories=benchmark_order, ordered=True
    )
    df[method_col] = pd.Categorical(
        df[method_col], categories=method_order, ordered=True
    )

    # Layout
    n_bench = len(benchmark_order)
    ncols = min(ncols, n_bench)
    nrows = int(np.ceil(n_bench / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_col * ncols, figsize_per_row * nrows),
        sharex="col",
        sharey="row",
    )
    axes = np.atleast_1d(axes).ravel()

    palette = sns.color_palette("colorblind") 
    methods = list(pd.unique(df[method_col]))
    method_colors = {
        m: palette[i] for i, m in enumerate(methods)
    }

    for ax, bench in zip(axes, benchmark_order):
        d = df[df[benchmark_col] == bench]

        sns.violinplot(
            data=d,
            x=method_col,
            y=value_col,
            order=method_order,
            palette=method_colors,
            inner="stick",
            hue="Method",
            #cut=0,
            linewidth=1.2,
            ax=ax,
        )

        ax.set_title(str(bench), fontsize=14)
        ax.set_xlabel("")
        ax.set_ylabel("")

        ax.tick_params(axis="x", rotation=25)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelbottom=False)

        # Remove seaborn's automatic legend (we'll add one global legend)
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # Turn off unused axes
    for ax in axes[len(benchmark_order):]:
        ax.axis("off")

    # ---- Shared legend ----
    handles = [
        plt.Line2D(
            [0], [0],
            color=method_colors[m],
            lw=8,
            label=str(m),
        )
        for m in method_order
    ]

    legend_ncol = min(len(method_order), 11)

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=legend_ncol,
        frameon=False,
        fontsize=LEGEND_FONT,
    )
    
    fig.supylabel(f"Quality", fontsize=Y_LABEL_FONT)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.08)
    return fig, axes

def plot_panel_dual_errorbar(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a="A",
    label_b="B",
    benchmark_col="Benchmark",
    method_col="Method",
    value_col="Accuracy",
    benchmark_order=None,
    method_order=None,
    ncols=3,
    figsize_per_col=4,
    figsize_per_row=3,
    capsize=4,
    dx=0.18,  # horizontal offset between the two errorbars per method
):
    # --- Copy to avoid mutating inputs ---
    df_a = df_a.copy()
    df_b = df_b.copy()

    # Optional: keep your mappings if they exist in scope
    if "METHODS" in globals():
        df_a[method_col] = df_a[method_col].map(METHODS)
        df_b[method_col] = df_b[method_col].map(METHODS)
    if "BENCHMARKS" in globals():
        df_a[benchmark_col] = df_a[benchmark_col].map(BENCHMARKS)
        df_b[benchmark_col] = df_b[benchmark_col].map(BENCHMARKS)

    # Tag datasets and combine
    df_a["_dataset"] = label_a
    df_b["_dataset"] = label_b
    df = pd.concat([df_a, df_b], ignore_index=True)

    # Aggregate mean + 95% CI (expects your mean_ci95 -> (mean, lo, hi))
    agg = (
        df.groupby([benchmark_col, method_col, "_dataset"])[value_col]
          .apply(mean_ci95)
          .apply(pd.Series)
          .rename(columns={0: "mean", 1: "lo", 2: "hi"})
          .reset_index()
    )

    # Orders
    if benchmark_order is None:
        benchmark_order = list(pd.unique(df[benchmark_col]))
    if method_order is None:
        method_order = list(pd.unique(df[method_col]))

    agg[benchmark_col] = pd.Categorical(
        agg[benchmark_col], categories=benchmark_order, ordered=True
    )
    agg[method_col] = pd.Categorical(
        agg[method_col], categories=method_order, ordered=True
    )

    # Layout
    n_bench = len(benchmark_order)
    ncols = min(ncols, n_bench)
    nrows = int(np.ceil(n_bench / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_col * ncols, figsize_per_row * nrows),
        sharex="col",
    )
    axes = np.atleast_1d(axes).ravel()

    x = np.arange(len(method_order))

    # Colors by method
    palette = sns.color_palette("colorblind")
    method_colors = {m: palette[i % len(palette)] for i, m in enumerate(method_order)}

    # Dataset styling (distinct markers, keep method->color)
    dataset_order = [label_a, label_b]
    dataset_to_marker = {label_a: "o", label_b: "s"}
    dataset_to_offset = {label_a: -dx, label_b: +dx}

    for idx, bench in enumerate(benchmark_order):
        ax = axes[idx]
        d = agg[agg[benchmark_col] == bench]

        for i, m in enumerate(method_order):
            for ds in dataset_order:
                dm = d[(d[method_col] == m) & (d["_dataset"] == ds)]
                if dm.empty:
                    continue

                mean = float(dm["mean"].iloc[0])
                lo = float(dm["lo"].iloc[0])
                hi = float(dm["hi"].iloc[0])

                ax.errorbar(
                    i + dataset_to_offset[ds],
                    mean,
                    yerr=[[mean - lo], [hi - mean]],
                    fmt=dataset_to_marker[ds],
                    color=method_colors[m],
                    ecolor=method_colors[m],
                    capsize=capsize,
                    elinewidth=2.5,
                    markersize=7,
                    linewidth=0,
                    alpha=0.95,
                )

        ax.set_title(str(bench), fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(method_order, rotation=25, ha="right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelbottom=False)

    # Disable empty panels
    for ax in axes[len(benchmark_order):]:
        ax.axis("off")

    # ---- Legends ----
    # Method legend (colors)
    method_handles = [
        Patch(facecolor=method_colors[m], edgecolor="black", label=str(m))
        for m in method_order
    ]

    # Dataset legend (markers)
    dataset_handles = [
        Line2D([0], [0], marker=dataset_to_marker[ds], color="black",
               linestyle="None", markersize=7, label=str(ds))
        for ds in dataset_order
    ]

    # Put both legends at bottom (stacked)
    legend_ncol = min(len(method_order), 11)
    fig.legend(
        handles=method_handles,
        loc="lower center",
        ncol=legend_ncol,
        frameon=False,
        fontsize=LEGEND_FONT if "LEGEND_FONT" in globals() else 10,
        bbox_to_anchor=(0.5, 0.03),
    )
    fig.legend(
        handles=dataset_handles,
        loc="lower center",
        ncol=len(dataset_order),
        frameon=False,
        fontsize=LEGEND_FONT if "LEGEND_FONT" in globals() else 10,
        bbox_to_anchor=(0.5, -0.005),
    )

    fig.supylabel("Quality", fontsize=Y_LABEL_FONT if "Y_LABEL_FONT" in globals() else 12)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.14)  # more bottom space for 2 legends
    return fig, axes, agg

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

def plot_panel_dual_boxplot(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a="A",
    label_b="B",
    benchmark_col="Benchmark",
    method_col="Method",
    value_col="Accuracy",
    benchmark_order=None,
    method_order=None,
    ncols=3,
    figsize_per_col=4,
    figsize_per_row=3,
    showfliers=True,
    legend_loc="lower center",
    legend_ncol=None,
    dx=0.18,          # horizontal offset for the two boxes per method
    box_width=0.28,   # width of each individual box
):
    # --- Copy to avoid mutating inputs ---
    df_a = df_a.copy()
    df_b = df_b.copy()

    # Optional: keep your mappings if they exist in scope
    if "METHODS" in globals():
        df_a[method_col] = df_a[method_col].map(METHODS)
        df_b[method_col] = df_b[method_col].map(METHODS)
    if "BENCHMARKS" in globals():
        df_a[benchmark_col] = df_a[benchmark_col].map(BENCHMARKS)
        df_b[benchmark_col] = df_b[benchmark_col].map(BENCHMARKS)

    if benchmark_order is None:
        benchmark_order = list(pd.unique(pd.concat([df_a[benchmark_col], df_b[benchmark_col]])))
    if method_order is None:
        method_order = list(pd.unique(pd.concat([df_a[method_col], df_b[method_col]])))

    # Layout
    n_bench = len(benchmark_order)
    ncols = min(ncols, n_bench)
    nrows = int(np.ceil(n_bench / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_col * ncols, figsize_per_row * nrows),
        sharex="col",
        sharey="row",
    )
    axes = np.atleast_1d(axes).ravel()

    x = np.arange(len(method_order))

    # Colors by method
    palette = sns.color_palette("colorblind")
    method_colors = {m: palette[i % len(palette)] for i, m in enumerate(method_order)}

    dataset_order = [label_a, label_b]
    dataset_to_offset = {label_a: -dx, label_b: +dx}
    dataset_to_hatch = {label_a: "", label_b: "///"}  # change hatch patterns if you want

    # helper to draw a styled boxplot set and return bp dict
    def _boxplot(ax, data, positions):
        return ax.boxplot(
            data,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=showfliers,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
        )

    for idx, bench in enumerate(benchmark_order):
        ax = axes[idx]

        da = df_a[df_a[benchmark_col] == bench]
        db = df_b[df_b[benchmark_col] == bench]

        # Build data arrays per method for each dataset
        data_a = [
            da.loc[da[method_col] == m, value_col].dropna().to_numpy()
            for m in method_order
        ]
        data_b = [
            db.loc[db[method_col] == m, value_col].dropna().to_numpy()
            for m in method_order
        ]

        pos_a = x + dataset_to_offset[label_a]
        pos_b = x + dataset_to_offset[label_b]

        bp_a = _boxplot(ax, data_a, pos_a)
        bp_b = _boxplot(ax, data_b, pos_b)

        # Style boxes: color by method; hatch by dataset
        for box, m in zip(bp_a["boxes"], method_order):
            box.set_facecolor(method_colors[m])
            box.set_alpha(0.8)
            box.set_edgecolor("black")
            box.set_hatch(dataset_to_hatch[label_a])

        for box, m in zip(bp_b["boxes"], method_order):
            box.set_facecolor(method_colors[m])
            box.set_alpha(0.8)
            box.set_edgecolor("black")
            box.set_hatch(dataset_to_hatch[label_b])

        ax.set_title(str(bench), fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(method_order, rotation=25, ha="right")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelbottom=False)

    # Turn off unused panels
    for ax in axes[len(benchmark_order):]:
        ax.axis("off")

    # ---- Legends ----
    # Method legend (colors)
    method_handles = [
        Patch(facecolor=method_colors[m], edgecolor="black", label=str(m))
        for m in method_order
    ]
    legend_ncol = legend_ncol or min(len(method_order), 11)

    fig.legend(
        handles=method_handles,
        loc="lower center",
        ncol=legend_ncol,
        frameon=False,
        fontsize=LEGEND_FONT if "LEGEND_FONT" in globals() else 10,
        bbox_to_anchor=(0.5, 0.03),
    )

    # Dataset legend (hatches)
    dataset_handles = [
        Patch(facecolor="white", edgecolor="black", hatch=dataset_to_hatch[ds], label=str(ds))
        for ds in dataset_order
    ]
    fig.legend(
        handles=dataset_handles,
        loc="lower center",
        ncol=len(dataset_order),
        frameon=False,
        fontsize=LEGEND_FONT if "LEGEND_FONT" in globals() else 10,
        bbox_to_anchor=(0.5, -0.005),
    )

    fig.supylabel("Quality", fontsize=Y_LABEL_FONT if "Y_LABEL_FONT" in globals() else 12)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.14)  # extra space for two legends
    return fig, axes

def plot_panel_dual_violin(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a="A",
    label_b="B",
    benchmark_col="Benchmark",
    method_col="Method",
    value_col="Accuracy",
    benchmark_order=None,
    method_order=None,
    ncols=3,
    figsize_per_col=4,
    figsize_per_row=3,
    inner="quartile",   # 'box', 'quartile', 'point', or None
    legend_loc="lower center",
    legend_ncol=None,
):
    # --- Copy to avoid mutating inputs ---
    df_a = df_a.copy()
    df_b = df_b.copy()

    # Optional: keep your mappings if they exist in scope
    if "METHODS" in globals():
        df_a[method_col] = df_a[method_col].map(METHODS)
        df_b[method_col] = df_b[method_col].map(METHODS)
    if "BENCHMARKS" in globals():
        df_a[benchmark_col] = df_a[benchmark_col].map(BENCHMARKS)
        df_b[benchmark_col] = df_b[benchmark_col].map(BENCHMARKS)

    # Tag + combine
    df_a["_dataset"] = label_a
    df_b["_dataset"] = label_b
    df = pd.concat([df_a, df_b], ignore_index=True)

    if benchmark_order is None:
        benchmark_order = list(pd.unique(df[benchmark_col]))
    if method_order is None:
        method_order = list(pd.unique(df[method_col]))

    # Ensure categorical ordering
    df[benchmark_col] = pd.Categorical(df[benchmark_col], categories=benchmark_order, ordered=True)
    df[method_col] = pd.Categorical(df[method_col], categories=method_order, ordered=True)
    df["_dataset"] = pd.Categorical(df["_dataset"], categories=[label_a, label_b], ordered=True)

    # Layout
    n_bench = len(benchmark_order)
    ncols = min(ncols, n_bench)
    nrows = int(np.ceil(n_bench / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_col * ncols, figsize_per_row * nrows),
        sharex="col",
        sharey="row",
    )
    axes = np.atleast_1d(axes).ravel()

    # Method colors (same for both dataset violins)
    palette = sns.color_palette("colorblind")
    method_colors = {m: palette[i % len(palette)] for i, m in enumerate(method_order)}

    # We'll let seaborn position/dodge by dataset, but recolor violins per method manually.
    neutral_palette = {label_a: (0.7, 0.7, 0.7), label_b: (0.7, 0.7, 0.7)}
    hatch_for = {label_a: "", label_b: "///"}  # hatch one dataset

    for ax, bench in zip(axes, benchmark_order):
        d = df[df[benchmark_col] == bench]

        sns.violinplot(
            data=d,
            x=method_col,
            y=value_col,
            order=method_order,
            hue="_dataset",
            hue_order=[label_a, label_b],
            dodge=True,          # <-- makes two separate violins per method
            split=False,
            palette=neutral_palette,  # recolor after
            inner=inner,
            linewidth=1.2,
            ax=ax,
            cut=0,
        )

        # Recolor + hatch
        polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
        # With dodge=True and 2 datasets, seaborn typically creates 2 violins per method.
        # Ordering is usually method-major, hue-minor: (m1,a), (m1,b), (m2,a), (m2,b), ...
        k = 0
        for m in method_order:
            for ds in [label_a, label_b]:
                if k >= len(polys):
                    break
                poly = polys[k]
                poly.set_facecolor(method_colors[m])
                poly.set_edgecolor("black")
                poly.set_alpha(0.8)
                poly.set_hatch(hatch_for[ds])
                k += 1

        ax.set_title(str(bench), fontsize=14)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=25)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelbottom=False)

        # Remove per-axes legend (we'll do global)
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # Turn off unused axes
    for ax in axes[len(benchmark_order):]:
        ax.axis("off")

    # ---- Legends ----
    # Method legend (colors)
    method_handles = [
        Patch(facecolor=method_colors[m], edgecolor="black", label=str(m))
        for m in method_order
    ]
    legend_ncol = legend_ncol or min(len(method_order), 11)

    fig.legend(
        handles=method_handles,
        loc="lower center",
        ncol=legend_ncol,
        frameon=False,
        fontsize=LEGEND_FONT if "LEGEND_FONT" in globals() else 10,
        bbox_to_anchor=(0.5, 0.03),
    )

    # Dataset legend (hatches)
    dataset_handles = [
        Patch(facecolor="white", edgecolor="black", hatch=hatch_for[ds], label=str(ds))
        for ds in [label_a, label_b]
    ]
    fig.legend(
        handles=dataset_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=LEGEND_FONT if "LEGEND_FONT" in globals() else 10,
        bbox_to_anchor=(0.5, -0.005),
    )

    fig.supylabel("Quality", fontsize=Y_LABEL_FONT if "Y_LABEL_FONT" in globals() else 12)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.14)
    return fig, axes

def plot_panel_dual_violin_split(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a="A",
    label_b="B",
    benchmark_col="Benchmark",
    method_col="Method",
    value_col="Accuracy",
    benchmark_order=None,
    method_order=None,
    ncols=3,
    figsize_per_col=4,
    figsize_per_row=3,
    inner="quartile",   # 'box', 'quartile', 'point', or None
    legend_loc="lower center",
    legend_ncol=None,
):
    # --- Copy to avoid mutating inputs ---
    df_a = df_a.copy()
    df_b = df_b.copy()

    # Optional: keep your mappings if they exist in scope
    if "METHODS" in globals():
        df_a[method_col] = df_a[method_col].map(METHODS)
        df_b[method_col] = df_b[method_col].map(METHODS)
    if "BENCHMARKS" in globals():
        df_a[benchmark_col] = df_a[benchmark_col].map(BENCHMARKS)
        df_b[benchmark_col] = df_b[benchmark_col].map(BENCHMARKS)

    # Tag + combine
    df_a["_dataset"] = label_a
    df_b["_dataset"] = label_b
    df = pd.concat([df_a, df_b], ignore_index=True)

    if benchmark_order is None:
        benchmark_order = list(pd.unique(df[benchmark_col]))
    if method_order is None:
        method_order = list(pd.unique(df[method_col]))

    # Ensure categorical ordering
    df[benchmark_col] = pd.Categorical(df[benchmark_col], categories=benchmark_order, ordered=True)
    df[method_col] = pd.Categorical(df[method_col], categories=method_order, ordered=True)
    df["_dataset"] = pd.Categorical(df["_dataset"], categories=[label_a, label_b], ordered=True)

    # Layout
    n_bench = len(benchmark_order)
    ncols = min(ncols, n_bench)
    nrows = int(np.ceil(n_bench / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_col * ncols, figsize_per_row * nrows),
        sharex="col",
        sharey="row",
    )
    axes = np.atleast_1d(axes).ravel()

    # Method colors (same for both halves)
    palette = sns.color_palette("colorblind")
    method_colors = {m: palette[i % len(palette)] for i, m in enumerate(method_order)}

    # We’ll draw split violins by dataset (hue), but we want *method* colors.
    # So we pass a neutral palette to seaborn (same for both halves), then recolor per method manually.
    neutral_palette = {label_a: (0.7, 0.7, 0.7), label_b: (0.7, 0.7, 0.7)}
    hatch_for = {label_a: "", label_b: "///"}  # hatch one dataset half

    for ax, bench in zip(axes, benchmark_order):
        d = df[df[benchmark_col] == bench]

        sns.violinplot(
            data=d,
            x=method_col,
            y=value_col,
            order=method_order,
            hue="_dataset",
            hue_order=[label_a, label_b],
            split=True,
            palette=neutral_palette,  # recolor after
            inner=inner,
            linewidth=1.2,
            ax=ax,
            cut=0,
        )

        # Recolor and hatch the halves:
        # Seaborn violins are PolyCollections; with split=True there are 2 per method (one per dataset).
        polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
        # Expect 2 * len(method_order) polygons (if both datasets present for all methods)
        # They come in method-major order; for each method: first hue, second hue.
        k = 0
        for m in method_order:
            for ds in [label_a, label_b]:
                if k >= len(polys):
                    break
                poly = polys[k]
                poly.set_facecolor(method_colors[m])
                poly.set_edgecolor("black")
                poly.set_alpha(0.8)
                poly.set_hatch(hatch_for[ds])
                k += 1

        ax.set_title(str(bench), fontsize=14)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=25)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelbottom=False)

        # Remove per-axes legend (we'll do global)
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # Turn off unused axes
    for ax in axes[len(benchmark_order):]:
        ax.axis("off")

    # ---- Legends ----
    # Method legend (colors)
    method_handles = [
        Patch(facecolor=method_colors[m], edgecolor="black", label=str(m))
        for m in method_order
    ]
    legend_ncol = legend_ncol or min(len(method_order), 11)

    fig.legend(
        handles=method_handles,
        loc="lower center",
        ncol=legend_ncol,
        frameon=False,
        fontsize=LEGEND_FONT if "LEGEND_FONT" in globals() else 10,
        bbox_to_anchor=(0.5, 0.03),
    )

    # Dataset legend (hatches)
    dataset_handles = [
        Patch(facecolor="white", edgecolor="black", hatch=hatch_for[ds], label=str(ds))
        for ds in [label_a, label_b]
    ]
    fig.legend(
        handles=dataset_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=LEGEND_FONT if "LEGEND_FONT" in globals() else 10,
        bbox_to_anchor=(0.5, -0.005),
    )

    fig.supylabel("Quality", fontsize=Y_LABEL_FONT if "Y_LABEL_FONT" in globals() else 12)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.14)
    return fig, axes