"""
Run this script with `python tools/generate_leaderboard_plot.py`
to create the image of the leaderboard heatmap displayed
in docs/leaderboards.md

Note: the code below assumes that the eva results are stored in
`eva/logs/<task>/<fm_identifier>/results`.
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors


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
            "camelyon16_small": "Cam16Small",
            "panda_small": "PANDASmall",
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


import pandas as pd
import matplotlib.pyplot as plt


def plot_leaderboard(
    df: pd.DataFrame,
    config: LeaderboardConfig,
    output_file: str = "docs/images/leaderboards/pathology.svg",
):
    """Minimalist table-style heatmap with modular internal logic."""

    def _prepare_data(data: pd.DataFrame) -> pd.DataFrame:
        # 1. Calc Average
        tasks = [
            f"{t}/test" if f"{t}/test" in data.columns else t
            for t in config.task_name_map
            if t not in config.exclude_for_average
        ]
        data["Average"] = data[tasks].mean(axis=1)
        data = data.sort_values(by="Average", ascending=False)

        # 2. Map Column Names
        data.columns = [config.task_name_map.get(c.removesuffix("/test"), c) for c in data.columns]
        return data

    def _setup_canvas(rows: int):
        plt.rcParams["font.sans-serif"] = ["Inter", "Arial", "sans-serif"]
        fig, ax = plt.subplots(figsize=(14, rows * 0.5 + 1))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        return fig, ax

    def _draw_rows(ax, df_display):
        rows, cols = df_display.shape
        for i in range(rows):
            # Row Background & Header
            ax.axhspan(i - 0.4, i + 0.4, color="gray", alpha=0.05, lw=0)
            ax.text(
                -0.6,
                i,
                df_display.index[i],
                ha="right",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="#4b5563",
            )

            # Cell Values
            for j in range(cols):
                val = df_display.iloc[i, j]
                is_avg = df_display.columns[j] == "Average"

                # Highlight top performers
                if val > df_display.iloc[:, j].max() * 0.98:
                    ax.plot(j, i, marker="s", markersize=22, color="#4f46e5", alpha=0.12)

                ax.text(
                    j,
                    i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="black" if is_avg else "normal",
                    color="#4f46e5" if is_avg else "#1f2937",
                )

    def _format_axes(ax, columns, rows):
        # Column Headers
        ax.set_xticks(range(len(columns)))
        ax.set_xticklabels(
            columns,
            rotation=0,
            fontsize=9,
            fontweight="bold",
            color="#4b5563",
            ha="center",
            va="center",
        )
        ax.xaxis.tick_top()
        ax.xaxis.set_tick_params(pad=10, length=0)

        # Clean up spines and visibility
        ax.set_ylim(rows - 0.5, -0.5)
        ax.set_xlim(-0.5, len(columns) - 0.5)
        ax.yaxis.set_visible(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    processed_df = _prepare_data(df.copy())
    fig, ax = _setup_canvas(len(processed_df))

    _draw_rows(ax, processed_df)
    _format_axes(ax, processed_df.columns, len(processed_df))

    plt.savefig(output_file, format="svg", transparent=True, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument(
        "--modality", type=str, default="pathology", choices=["pathology", "radiology"]
    )
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()
    config = LeaderboardConfig(args.modality)

    leaderboard_df = get_leaderboard(
        csv_path=args.csv_path or f"tools/data/leaderboards/{args.modality}.csv", config=config
    )

    plot_leaderboard(
        df=leaderboard_df.copy(),
        config=config,
        output_file=args.save_path or f"docs/images/leaderboards/{args.modality}.svg",
    )


if __name__ == "__main__":
    main()
