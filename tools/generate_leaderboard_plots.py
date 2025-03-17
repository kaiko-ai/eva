# Run this script with `python tools/generate_leaderboard_plot.py`
# to create the image of the leaderboard heatmap and starplot displayed
# in docs/leaderboards.md
#
# Note: the code below assumes that the eva results are stored in
# `eva/logs/<task>/<fm_identifier>/results`.

import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import seaborn as sns

from typing import Optional


_tasks_to_metric = {
    "bach": "MulticlassAccuracy",
    "crc": "MulticlassAccuracy",
    "mhist": "BinaryBalancedAccuracy",
    "patch_camelyon": "BinaryBalancedAccuracy",
    "camelyon16_small": "BinaryBalancedAccuracy",
    "panda_small": "MulticlassAccuracy",
    "consep": "GeneralizedDiceScore",
    "monusac": "GeneralizedDiceScore",
}
_fm_name_map = {
    "paige_virchow2": "Virchow2 - DINOv2 ViT-H14 | 3.1M slides",
    "lunit_vits16": "Lunit - DINO ViT-S16 | TCGA",
    "owkin_phikon": "Phikon - iBOT ViT-B16 | TCGA",
    "owkin_phikon_v2": "Phikon-v2 - DINOv2 ViT-L16 | PANCAN-XL",
    "mahmood_uni": "UNI - DINOv2 ViT-L16 | Mass-100k",
    "mahmood_uni2_h": "UNI2-h - DINOv2 ViT-H14 | 250k slides",
    "bioptimus_h_optimus_0": "H-optimus-0 - ViT-G14 | 500k slides",
    "prov_gigapath": "Prov-GigaPath - DINOv2 ViT-G14 | 181k slides",
    "histai_hibou_l": "hibou-L - DINOv2 ViT-B14 | 1M slides",
    "kaiko_vits16": "kaiko.ai - DINO ViT-S16 | TCGA",
    "kaiko_vits8": "kaiko.ai - DINO ViT-S8 | TCGA",
    "kaiko_vitb16": "kaiko.ai - DINO ViT-B16 | TCGA",
    "kaiko_vitb8": "kaiko.ai - DINO ViT-B8 | TCGA",
    "kaiko_vitl14": "kaiko.ai - DINOv2 ViT-L14 | TCGA",
}
_tasks_names_map = {
    "bach": "BACH",
    "bracs": "BRACS",
    "bracs/test": "BRACS/test",
    "breakhis": "BreakHis",
    "gleason_arvaniti": "Gleason",
    "crc": "CRC",
    "mhist": "MHIST",
    "patch_camelyon": "PCam",
    "patch_camelyon/test": "PCam/test",
    "camelyon16_small": "Cam16Small",
    "camelyon16_small/test": "Cam16Small/test",
    "panda_small": "PANDASmall",
    "panda_small/test": "PANDASmall/test",
    "consep": "CoNSeP",
    "monusac": "MoNuSAC",
}
_colors_for_startplot = [
    "#7F7F7F",
    "#FFC000",
    "#C1E814",
    "#FF0000",
    "#D400FF",
    "#4FFF87",
    "#00A735",
    "#6666FF",
    "#0000FF",
    "#00007F",
    "#0000FF",
]
_label_offsets_startplot = {
    "BACH": (0, -0.1),
    "CRC": (0, -0.1),
    "MHIST": (0.07, -0.1),
    "PCam": (0.05, 0),
    "Camelyon16": (-0.05, -0.08),
    "PANDA": (0, -0.1),
    "CoNSeP": (0.1, -0.07),
    "MoNuSAC": (0.15, 0.08),
}


def get_leaderboard(logs_dir: Optional[str] = None) -> pd.DataFrame:
    """Get the leaderboard data frame."""

    # load existing leaderboard if available:
    if os.path.isfile("tools/data/leaderboard.csv"):
        df_existing = pd.read_csv("tools/data/leaderboard.csv")
    else:
        df_existing = pd.DataFrame()

    # load results into data frame:
    if logs_dir:
        all_scores = []
        for model in _fm_name_map.keys():
            scores = []
            for task in _tasks_to_metric.keys():
                run_folder = [
                    d
                    for d in sorted(os.listdir(f"{logs_dir}/{task}/{model}/results"))
                    if d.startswith("20")
                ][-1]
                with open(
                    os.path.join(f"{logs_dir}/{task}/{model}/results/{run_folder}/results.json")
                ) as f:
                    d = json.load(f)
                split = "test" if d["metrics"]["test"] else "val"
                metric = _tasks_to_metric.get(task)
                if metric is None:
                    raise Exception(f"no metric defined for task {task}")
                scores.append(round(d["metrics"][split][0][f"{split}/{metric}"]["mean"], 3))
            all_scores.append(scores)
        df = pd.DataFrame(all_scores, columns=_tasks_to_metric.keys())
        df["model"] = _fm_name_map.keys()

        # combine existing and new data frame
        df = pd.concat([df, df_existing]).drop_duplicates()
        df.to_csv("tools/data/leaderboard.csv", index=False)
    else:
        df = df_existing

    df = df.set_index("model", drop=True)
    df = df[[fm in _fm_name_map.keys() for fm in df.index]]
    df.index = df.reset_index()["model"].apply(lambda x: _fm_name_map.get(x) or x)
    return df


def plot_leaderboard(df: pd.DataFrame, output_file: str = "docs/images/leaderboard.svg"):
    """Plot the leaderboard heatmap."""

    # create colormap:
    colors = [[0, "white"], [1, "#0000ff"]]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

    # prepare data frame:
    df.index.names = [""]

    # exclude BACH from the average
    df_for_avg = df[[c for c in df.columns if c != "bach"]]

    df.loc[:, "overall_performance"] = df_for_avg.mean(axis=1)
    df = df.sort_values(by="overall_performance", ascending=False)
    df = df.drop(columns=["overall_performance"])
    df.columns = [_tasks_names_map.get(c) or c for c in df.columns]
    scaled_df = (df - df.min(axis=0)) / (df.max(axis=0) - df.min(axis=0))

    # create plot:
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(scaled_df, annot=df, cmap=cmap, ax=ax, cbar=False, fmt=".3f")
    plt.tick_params(
        axis="x",
        which="major",
        labelsize=10,
        labelbottom=False,
        bottom=False,
        top=False,
        labeltop=True,
        rotation=20,
    )
    plt.savefig(output_file, format="svg", dpi=1200, bbox_inches="tight")


def plot_startplot(df: pd.DataFrame, output_file: str = "docs/images/starplot.png"):
    """Plot the star plot."""

    plt.style.use("seaborn-v0_8-ticks")

    df = df[_tasks_to_metric.keys()]
    datasets = _label_offsets_startplot.keys()
    models = df.index.tolist()
    accuracy_values_new = df.to_numpy()
    angles = np.linspace(0, 2 * np.pi, len(datasets), endpoint=False).tolist()
    angles += angles[:1]
    accuracy_values = np.concatenate((accuracy_values_new, accuracy_values_new[:, [0]]), axis=1)
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for angle, label in zip(angles[:-1], datasets):
        ha = "left" if angle < np.pi else "right"
        va = "bottom" if angle < np.pi else "top"
        offset_x, offset_y = _label_offsets_startplot[label]
        ax.text(
            angle + offset_x,
            1.1 + offset_y,
            label,
            horizontalalignment=ha,
            size=20,
            color="black",
            verticalalignment=va,
        )
        ax.plot([angle, angle], [0, 1], color="grey", linestyle="-", linewidth=0.5)

    ax.set_yticklabels([])
    ax.xaxis.set_visible(False)

    ax.set_rlabel_position(0)
    y_ticks = [0.5, 0.6, 0.7, 0.8, 0.9]
    plt.ylim(0.3, 0.98)

    for idx, (model, values) in enumerate(zip(models, accuracy_values)):
        # if np.any(np.isnan(values)):
        if np.any(pd.isna(values)):
            values = np.nan_to_num(values)
        color = _colors_for_startplot[idx % len(_colors_for_startplot)]
        ax.plot(
            angles, values, label=model, color=color, linewidth=5, linestyle="solid", alpha=0.45
        )

    # Annotate y tick values slightly further from the axes
    for tick in y_ticks:
        for angle in angles[:-1]:
            alignment = "left" if angle < np.pi else "right"
            ax.text(
                angle,
                tick + 0.02,
                f"{tick}",
                horizontalalignment=alignment,
                size=8,
                color="grey",
                verticalalignment="center",
            )

    legend = ax.legend(loc="upper right", bbox_to_anchor=(1.95, 1), title="", fontsize=18)
    plt.savefig(output_file, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_dir", type=str, default=None)
    parser.add_argument("--output_leaderboard", type=str, default="docs/images/leaderboard.svg")
    parser.add_argument("--output_starplot", type=str, default="docs/images/starplot.png")
    args = parser.parse_args()

    leaderboard_df = get_leaderboard(args.logs_dir)
    plot_leaderboard(leaderboard_df, args.output_leaderboard)
    plot_startplot(leaderboard_df, args.output_starplot)


if __name__ == "__main__":
    main()
