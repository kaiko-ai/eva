"""
Run this script with `python tools/generate_leaderboard_plot.py`
to create the image of the leaderboard heatmap displayed
in docs/leaderboards.md

Note: the code below assumes that the eva results are stored in
`eva/logs/<task>/<fm_identifier>/results`.
"""


import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns

from typing import Optional


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
            "voco_h": "VoCo-H"
        }
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
        }
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


def plot_leaderboard(df: pd.DataFrame, config: LeaderboardConfig, output_file: str = "docs/images/leaderboards/pathology.svg"):
    """Plot the leaderboard heatmap."""

    def _task_name(name: str) -> str:
        if "/test" in name:
            return f"{config.task_name_map[name.removesuffix('/test')]}/test"
        return config.task_name_map.get(name) or name

    # create colormap:
    colors = [[0, "white"], [1, "#0000ff"]]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

    # prepare data frame:
    df.index.names = [""]

    # calculate average column
    tasks_for_average = []
    for task in config.task_name_map.keys():
        if task in config.exclude_for_average:
            continue
        if f"{task}/test" in df.columns:
            tasks_for_average.append(f"{task}/test")
        else:
            tasks_for_average.append(task)
    df["Average"] = df[tasks_for_average].mean(axis=1)
    df = df.sort_values(by="Average", ascending=False)

    # create plot:
    df.columns = [_task_name(c) or c for c in df.columns]
    height = max(2, len(df) * 0.5) if len(df) < 4 else 6
    fig, ax = plt.subplots(figsize=(12, height))
    scaled_df = (df - df.min(axis=0)) / (df.max(axis=0) - df.min(axis=0))
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
    plt.tick_params(
        axis="y",
        which="major",
        labelsize=10,
        rotation=0,
    )
    plt.savefig(output_file, format="svg", dpi=1200, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--modality", type=str, default="pathology", choices=["pathology", "radiology"])
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()
    config = LeaderboardConfig(args.modality)

    leaderboard_df = get_leaderboard(
        csv_path=args.csv_path or f"tools/data/leaderboards/{args.modality}.csv",
        config=config)

    plot_leaderboard(
        df=leaderboard_df.copy(),
        config=config,
        output_file=args.save_path or f"docs/images/leaderboards/{args.modality}.svg")

if __name__ == "__main__":
    main()
