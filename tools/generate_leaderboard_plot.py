# Run this script with `python tools/generate_leaderboard_plot.py`
# to create the image of the leaderboard heatmap displayed in
# docs/leaderboards.md
#
# Note: the code below assumes that the eva results are stored in
# `eva/logs/<task>/<fm_identifier>`.

import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns


def main():
    _tasks_to_metric = {
        "bach": "MulticlassAccuracy",
        "crc": "MulticlassAccuracy",
        "mhist": "BinaryBalancedAccuracy",
        "patch_camelyon": "BinaryBalancedAccuracy",
        "camelyon16": "BinaryBalancedAccuracy",
        "panda": "MulticlassAccuracy",
        "consep": "GeneralizedDiceScore",
        "monusac": "GeneralizedDiceScore",
    }
    _fm_name_map = {
        "dino_vits16_lunit": "Lunit - ViT-S16 | TCGA",
        "owkin_phikon": "Owkin (Phikon) - iBOT ViT-B16 | TCGA",
        "dino_vitl16_uni": "UNI - DINOv2 ViT-L16 | Mass-100k",
        "dino_vits16_kaiko": "kaiko.ai - DINO ViT-S16 | TCGA",
        "dino_vits8_kaiko": "kaiko.ai - DINO ViT-S8 | TCGA",
        "dino_vitb16_kaiko": "kaiko.ai - DINO ViT-B16 | TCGA",
        "dino_vitb8_kaiko": "kaiko.ai - DINO ViT-B8 | TCGA",
        "dino_vitl14_kaiko": "kaiko.ai - DINOv2 ViT-L14 | TCGA",
    }
    _tasks_names_map = {
        "bach": "BACH",
        "crc": "CRC",
        "mhist": "MHIST",
        "patch_camelyon": "PCam",
        "camelyon16": "Camelyon16",
        "panda": "PANDA",
        "consep": "CoNSeP",
        "monusac": "MoNuSAC",
    }

    # get log_dir from arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_dir", type=str, default="logs")
    parser.add_argument("--output_file", type=str, default="docs/images/leaderboard.svg")
    args = parser.parse_args()

    # load existing leaderboard if available:
    if os.path.isfile("tools/data/leaderboard.csv"):
        df_existing = pd.read_csv("tools/data/leaderboard.csv")
    else:
        df_existing = pd.DataFrame()

    # load results into data frame:
    all_scores = []
    for model in _fm_name_map.keys():
        scores = []
        for task in _tasks_to_metric.keys():
            results_folder = [
                d for d in os.listdir(f"{args.logs_dir}/{task}/{model}") if d.startswith("20")
            ][0]
            with open(
                os.path.join(f"{args.logs_dir}/{task}/{model}/{results_folder}/results.json")
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
    df = df.set_index("model", drop=True)

    # create colormap:
    colors = [[0, "white"], [1, "#0000ff"]]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

    # prepare data frame:
    df = df[[fm in _fm_name_map.keys() for fm in df.index]]
    df.index = df.reset_index()["model"].apply(lambda x: _fm_name_map.get(x) or x)
    df.index.names = [""]
    df.loc[:, "overall_performance"] = df.mean(axis=1)
    df = df.sort_values(by="overall_performance", ascending=False)
    df = df.drop(columns=["overall_performance"])
    df.columns = [_tasks_names_map.get(c) or c for c in df.columns]
    scaled_df = (df - df.min(axis=0)) / (df.max(axis=0) - df.min(axis=0))

    # create plot:
    fig, ax = plt.subplots(figsize=(8, 5))
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
    plt.savefig(args.output_file, format="svg", dpi=1200, bbox_inches="tight")


if __name__ == "__main__":
    main()
