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
    """Smooth heatmap with wrapped headers and a borderless UI."""
    
    def _prepare_data(data: pd.DataFrame) -> pd.DataFrame:
        tasks = [t for t in config.task_name_map if t in data.columns or f"{t}/test" in data.columns]
        data["Average"] = data[[f"{t}/test" if f"{t}/test" in data.columns else t for t in tasks]].mean(axis=1)
        data = data.sort_values(by="Average", ascending=False)

        def manual_wrap(text: str):
            if len(text) <= 12: return text
            # Find a natural break point (/, _, or space) near the middle
            mid = len(text) // 2
            for i in range(mid, len(text)):
                if text[i] in "/_ ":
                    return f"{text[:i+1]}\n{text[i+1:]}"
            # Fallback: Hard split at midpoint
            return f"{text[:mid]}\n{text[mid:]}"

        new_cols = []
        for c in data.columns:
            if c == "Average":
                new_cols.append(c)
                continue
            
            # Map the name but keep the /test suffix if it exists
            base = c.removesuffix("/test")
            mapped_base = config.task_name_map.get(base, base)
            full_name = f"{mapped_base}/test" if c.endswith("/test") else mapped_base
            
            new_cols.append(manual_wrap(full_name))
            
        data.columns = new_cols
        return data

    def _setup_canvas(rows: int, cols: int):
        plt.rcParams["font.family"] = "sans-serif"
        # Adjusted height for wrapped headers
        fig, ax = plt.subplots(figsize=(cols * 1.4, rows * 0.5 + 1.5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        return fig, ax

    def _draw_smooth_heatmap(ax, df_display):
        rows, cols = df_display.shape
        cmap = plt.get_cmap("Blues") # Smooth blue gradient

        for j in range(cols):
            col_data = df_display.iloc[:, j]
            v_min, v_max = col_data.min(), col_data.max()
            
            for i in range(rows):
                val = col_data.iloc[i]
                is_avg = df_display.columns[j].startswith("Average")
                
                # Normalize value for the "smooth" background intensity
                norm_val = (val - v_min) / (v_max - v_min + 1e-9)
                alpha = 0.05 + (norm_val * 0.15) # Subtle variation
                
                color = "#4f46e5" if is_avg else cmap(norm_val)

                # Draw a rounded "cell" for a smooth look
                rect = patches.FancyBboxPatch(
                    (j - 0.45, i - 0.4), 0.9, 0.8,
                    boxstyle="round,pad=0.02", 
                    facecolor=color, alpha=alpha, zorder=1
                )
                ax.add_patch(rect)

                # Text styling
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=10, fontweight="bold" if is_avg else 500,
                        color="#1e293b" if not is_avg else "#4338ca", zorder=2)

        # Draw Row Labels (Models)
        for i in range(rows):
            ax.text(-0.7, i, df_display.index[i], ha="right", va="center", 
                    fontsize=10, fontweight=600, color="#0f172a")

    def _format_axes(ax, columns, rows):
        ax.set_xticks(range(len(columns)))
        # ax.set_xticklabels(columns, fontsize=9, fontweight=700, color="#64748b", linespacing=0.9)
        ax.set_xticklabels(columns, fontsize=9, fontweight=700, color="#64748b", linespacing=0.9, ha="center", multialignment="center")

        ax.xaxis.tick_top()
        
        # Remove tick lines and spines
        ax.tick_params(axis='both', which='both', length=0, pad=0)
        ax.set_ylim(rows - 0.4, -0.8)
        ax.set_xlim(-0.5, len(columns) - 0.5)
        for s in ax.spines.values(): s.set_visible(False)
        ax.yaxis.set_visible(False)

    processed_df = _prepare_data(df.copy())
    fig, ax = _setup_canvas(len(processed_df), len(processed_df.columns))
    
    _draw_smooth_heatmap(ax, processed_df)
    _format_axes(ax, processed_df.columns, len(processed_df))

    plt.savefig(output_file, format="svg", bbox_inches="tight", transparent=False)
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
