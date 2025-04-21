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


_tasks_to_metric = {
    "bach": "MulticlassAccuracy",
    "breakhis": "MulticlassAccuracy",
    "crc": "MulticlassAccuracy",
    "gleason_arvaniti": "MulticlassAccuracy",
    "mhist": "BinaryBalancedAccuracy",
    "patch_camelyon": "BinaryBalancedAccuracy",
    "camelyon16_small": "BinaryBalancedAccuracy",
    "panda_small": "MulticlassAccuracy",
    "consep": "MonaiDiceScore",
    "monusac": "MonaiDiceScore",
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
    "kaiko_midnight_12k": "kaiko.ai - DINOv2 Midnight-12k | TCGA",
}
_tasks_names_map = {
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
}
_exclude_for_average = ["bach"]

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

    def _task_name(name: str) -> str:
        if "/test" in name:
            return f"{_tasks_names_map[name.removesuffix('/test')]}/test"
        return _tasks_names_map.get(name) or name

    # create colormap:
    colors = [[0, "white"], [1, "#0000ff"]]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

    # prepare data frame:
    df.index.names = [""]

    # calculate average column
    tasks_for_average = []
    for task in _tasks_names_map.keys():
        if task in _exclude_for_average:
            continue
        if f"{task}/test" in df.columns:
            tasks_for_average.append(f"{task}/test")
        else:
            tasks_for_average.append(task)
    df["Average"] = df[tasks_for_average].mean(axis=1)
    df = df.sort_values(by="Average", ascending=False)

    # create plot:
    df.columns = [_task_name(c) or c for c in df.columns]
    fig, ax = plt.subplots(figsize=(12, 6))
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
    plt.savefig(output_file, format="svg", dpi=1200, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_dir", type=str, default=None)
    parser.add_argument("--output_leaderboard", type=str, default="docs/images/leaderboard.svg")
    args = parser.parse_args()

    leaderboard_df = get_leaderboard(args.logs_dir)
    plot_leaderboard(leaderboard_df.copy(), args.output_leaderboard)

if __name__ == "__main__":
    main()
