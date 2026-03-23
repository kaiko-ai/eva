"""
Run this script with `python tools/generate_leaderboard_plot.py`
to create the image of the leaderboard heatmap displayed
in docs/leaderboards.md

Note: the code below assumes that the eva results are stored in
`eva/logs/<task>/<fm_identifier>/results`.
"""

from typing import Literal

import matplotlib.patheffects as path_effects
import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Literal
from jsonargparse import CLI

import matplotlib.patheffects as path_effects
import os
import pandas as pd
import matplotlib.pyplot as plt
from jsonargparse import CLI


class LeaderboardConfig:
    """Configurations for the leaderboard."""

    def __init__(self, modality: str):
        self.modality = modality
        self.fm_name_map = self._fm_name_map[modality]
        self.task_name_map = self._task_name_map[modality]
        self.exclude_for_average = self._exclude_for_average[modality]

    _fm_name_map = {
        "pathology": {
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
            "kaiko_midnight_12k": "kaiko.ai - DINOv2 Midnight-12k | TCGA",
        },
        "radiology": {
            "voco_b": "VoCo-B",
            "voco_h": "VoCo-H",
            "vit_base_patch16_224_dino_1chan": "ViT-B16 (DINO)",
        },
    }
    _task_name_map = {
        "pathology": {
            "bach": "BACH",
            "breakhis": "BreakHis",
            "crc": "CRC",
            "gleason_arvaniti": "Gleason",
            "mhist": "MHIST",
            "patch_camelyon": "PCam",
            "camelyon16_small": "Cam16\nSmall",
            "panda_small": "PANDA\nSmall",
            "consep": "CoNSeP",
            "monusac": "MoNuSAC",
        },
        "radiology": {
            "btcv": "BTCV",
            "lits17": "LiTS17",
            "msd_task7_pancreas": "MSD Task 7 Pancreas",
        },
    }

    _exclude_for_average = {
        "pathology": ["bach"],
        "radiology": [],
    }
    """The tasks to exclude from the average column in the leaderboard table."""


def get_leaderboard(csv_path: str, config: LeaderboardConfig) -> pd.DataFrame:
    """Get the leaderboard data frame."""

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    df = df.set_index("model", drop=True)
    df = df[[fm in config.fm_name_map.keys() for fm in df.index]]
    df.index = df.reset_index()["model"].apply(lambda x: config.fm_name_map.get(x) or x)
    return df


def plot_leaderboard(
    df: pd.DataFrame,
    config: LeaderboardConfig,
    output_file: str = "docs/images/leaderboards/pathology.svg",
):
    """Minimalist table with unified header styling for rows and columns."""

    # 1. Prepare Data
    tasks_for_avg = [
        f"{t}/test" if f"{t}/test" in df.columns else t
        for t in config.task_name_map.keys()
        if t not in config.exclude_for_average
    ]
    df["Average"] = df[tasks_for_avg].mean(axis=1)
    df = df.sort_values(by="Average", ascending=False)

    def _task_name(name: str) -> str:
        name_clean = name.removesuffix("/test")
        return config.task_name_map.get(name_clean) or name_clean

    df.columns = [_task_name(c) for c in df.columns]
    vals = df.values

    # 2. Styling Config
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Inter", "Arial", "sans-serif"]

    fig, ax = plt.subplots(figsize=(14, len(df) * 0.5 + 1))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    rows, cols = df.shape
    for i in range(rows):
        # Row zebra striping
        ax.axhspan(i - 0.4, i + 0.4, color="gray", alpha=0.05, lw=0)

        # ROW HEADERS: Styled exactly like column headers
        ax.text(
            -0.6,
            i,
            df.index[i],
            ha="right",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#4b5563",
        )

        for j in range(cols):
            val = vals[i, j]
            is_avg = df.columns[j] == "Average"

            if val > df.iloc[:, j].max() * 0.98:
                ax.plot(j, i, marker="", markersize=22, color="#4f46e5", alpha=0.12)

            weight = "black" if is_avg else "normal"
            color = "#4f46e5" if is_avg else "#1f2937"

            ax.text(
                j,
                i,
                f"{val:.3f}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight=weight,
                color=color,
            )

    # 3. COLUMN HEADERS
    ax.set_xticks(range(cols))
    ax.set_xticklabels(
        df.columns,
        rotation=0,
        fontsize=9,
        fontweight="bold",
        color="#4b5563",
        ha="center",
        va="center",
    )
    ax.xaxis.set_tick_params(pad=10)
    ax.xaxis.tick_top()

    # 4. Final Clean-up
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_xlim(-0.5, cols - 0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(axis="both", which="both", length=0)
    ax.yaxis.set_visible(False)

    plt.savefig(output_file, format="svg", transparent=True, bbox_inches="tight")
    plt.close()


def generate_leaderboard_plot(
    csv_path: str | None = None,
    modality: Literal["pathology", "radiology"] = "pathology",
    save_path: str | None = None,
):
    """Generating leaderboard plots from a csv file.

    Args:
        csv_path: Path to the CSV file. Defaults to None.
        modality: "pathology" or "radiology". Defaults to "pathology".
        save_path: Path to save the SVG plot. Defaults to None.
    """
    config = LeaderboardConfig(modality)
    leaderboard_df = get_leaderboard(
        csv_path=csv_path or f"tools/data/leaderboards/{modality}.csv", config=config
    )
    plot_leaderboard(
        df=leaderboard_df.copy(),
        config=config,
        output_file=save_path or f"docs/images/leaderboards/{modality}.svg",
    )


if __name__ == "__main__":
    CLI(generate_leaderboard_plot)
