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

import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import textwrap


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



def plot_leaderboard(
    df: pd.DataFrame,
    config: LeaderboardConfig,
    output_file: str = "docs/images/leaderboards/pathology.svg",
):
    """Heatmap with merged test columns and formatted value (test_value) display."""
    
    def _prepare_data(data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        # 1. Define targets for merging
        merge_targets = ["patch_camelyon", "camelyon16_small", "panda_small"]
        display_df = data.copy()
        numeric_df = data.copy()

        for target in merge_targets:
            test_col = f"{target}/test"
            if target in data.columns and test_col in data.columns:
                # Average for heatmap color
                avg_series = data[[target, test_col]].mean(axis=1)
                numeric_df[target] = avg_series
                
                # Format as "val (test_val)" on a single line with space
                display_df[target] = data.apply(
                    lambda row: f"{row[target]:.3f} ({row[test_col]:.3f})", axis=1
                )
                
                display_df = display_df.drop(columns=[test_col])
                numeric_df = numeric_df.drop(columns=[test_col])


        # 2. Handle Average Column
        tasks_for_average = []
        for task in config.task_name_map.keys():
            if task in config.exclude_for_average:
                continue
            if f"{task}/test" in data.columns:
                tasks_for_average.append(f"{task}/test")
            else:
                tasks_for_average.append(task)

        final_avg = data[tasks_for_average].mean(axis=1)
        display_df["Average"] = final_avg.map(lambda x: f"{x:.3f}")
        numeric_df["Average"] = final_avg

        # 3. Sort
        display_df = display_df.loc[numeric_df.sort_values(by="Average", ascending=False).index]
        numeric_df = numeric_df.sort_values(by="Average", ascending=False)

        # 4. Map Column Names and add "(test)" to the right
        new_cols = []
        for c in display_df.columns:
            mapped = config.task_name_map.get(c, c)
            if c in merge_targets:
                # Place (test) to the right of the name
                full_label = f"{mapped} (test)"
            else:
                full_label = mapped
            
            # Simple wrap for long headers
            if len(full_label) > 15:
                new_cols.append(textwrap.fill(full_label, width=12))
            else:
                new_cols.append(full_label)
            
        display_df.columns = new_cols
        numeric_df.columns = new_cols
        return display_df, numeric_df

    def _draw_smooth_heatmap(ax, df_display, df_numeric):
        rows, cols = df_display.shape
        cmap = plt.get_cmap("Blues")

        for j in range(cols):
            col_data = df_numeric.iloc[:, j]
            v_min, v_max = col_data.min(), col_data.max()
            
            for i in range(rows):
                val_num = col_data.iloc[i]
                val_text = df_display.iloc[i, j]
                is_avg = df_display.columns[j].startswith("Average")
                
                norm_val = (val_num - v_min) / (v_max - v_min + 1e-9)
                alpha = 0.05 + (norm_val * 0.15)
                color = "#4f46e5" if is_avg else cmap(norm_val)

                rect = patches.FancyBboxPatch(
                    (j - 0.45, i - 0.4), 0.9, 0.8,
                    boxstyle="round,pad=0.02", 
                    facecolor=color, alpha=alpha, zorder=1
                )
                ax.add_patch(rect)

                ax.text(j, i, val_text, ha="center", va="center",
                        fontsize=10, fontweight="bold" if is_avg else 500,
                        color="#1e293b" if not is_avg else "#4338ca", 
                        zorder=2, linespacing=0.9)

        for i in range(rows):
            ax.text(-0.7, i, df_display.index[i], ha="right", va="center", 
                    fontsize=11, fontweight=600, color="#0f172a")

        def _format_axes(ax, columns, rows):
            ax.set_xticks(range(len(columns)))
            
            # Initial call to set the labels
            ax.set_xticklabels(
                columns, 
                fontweight=700, 
                color="#64748b", 
                ha="center", 
                va="top"
            )

            # Iterate through the labels we just set to vary the fontsize
            for label in ax.get_xticklabels():
                text = label.get_text()
                # If the name is long or has multiple lines, shrink the font
                if len(text) > 12 or "\n" in text:
                    label.set_fontsize(7.5)
                else:
                    label.set_fontsize(9)

            ax.xaxis.tick_top()
            ax.tick_params(axis='both', which='both', length=0, pad=10)
            
            # Standardize plot limits
            ax.set_ylim(rows - 0.4, -0.8)
            ax.set_xlim(-0.5, len(columns) - 0.5)
            for s in ax.spines.values(): 
                s.set_visible(False)
            ax.yaxis.set_visible(False)

    # Final execution flow
    display_df, numeric_df = _prepare_data(df.copy())

    plt.rcParams["font.family"] = "sans-serif"
    # Adjusted figsize width to 1.8 to fit the horizontal "val (test_val)" layout better
    fig, ax = plt.subplots(figsize=(len(display_df.columns) * 1.8, len(display_df) * 0.6 + 1.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    
    _draw_smooth_heatmap(ax, display_df, numeric_df)
    
    # Set the ticks
    ax.set_xticks(range(len(display_df.columns)))
    
    # 1. Apply the labels with the top alignment
    ax.set_xticklabels(
        display_df.columns, 
        fontweight=700, 
        color="#64748b", 
        ha="center", 
        va="center"
    )

    ax.xaxis.tick_top()
    ax.tick_params(axis='x', which='both', length=0, pad=7)
    
    ax.set_ylim(len(display_df) - 0.4, -0.8)
    ax.set_xlim(-0.5, len(display_df.columns) - 0.5)
    
    for s in ax.spines.values(): 
        s.set_visible(False)
    ax.yaxis.set_visible(False)

    plt.savefig(output_file, format="svg", bbox_inches="tight")
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
