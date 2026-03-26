import os
from jsonargparse import CLI

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


def get_leaderboard(csv_path: str, config: LeaderboardConfig) -> pd.DataFrame:
    """Load and prepare leaderboard DataFrame."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.set_index("model")
    df = df[df.index.isin(config.fm_name_map.keys())]
    df.index = df.index.map(lambda x: config.fm_name_map.get(x, x))
    return df


def _prepare_data(df: pd.DataFrame, config: LeaderboardConfig):
    """Prepare display and numeric DataFrames for plotting."""
    merge_targets = {"patch_camelyon", "camelyon16_small", "panda_small"}

    display_df = df.copy()
    numeric_df = df.copy()

    for target in merge_targets:
        test_col = f"{target}/test"
        if target in df.columns and test_col in df.columns:
            numeric_df[target] = df[[target, test_col]].mean(axis=1)
            display_df[target] = df.apply(
                lambda row: f"{row[target]:.3f} ({row[test_col]:.3f})", axis=1
            )
            display_df.drop(columns=[test_col], inplace=True)
            numeric_df.drop(columns=[test_col], inplace=True)

    # Average column
    tasks_for_avg = []
    for task in config.task_name_map:
        if task in config.exclude_for_average:
            continue
        col = f"{task}/test" if f"{task}/test" in df.columns else task
        if col in df.columns:
            tasks_for_avg.append(col)

    avg_series = df[tasks_for_avg].mean(axis=1)
    display_df["Average"] = avg_series.map("{:.3f}".format)
    numeric_df["Average"] = avg_series

    numeric_df = numeric_df.sort_values(by="Average", ascending=False)
    display_df = display_df.loc[numeric_df.index]

    # Rename columns
    new_columns = []
    for col in display_df.columns:
        name = config.task_name_map.get(col, col)
        if col in merge_targets:
            name = f"{name} (test)"
        if len(name) > 15:
            name = textwrap.fill(name, width=12)
        new_columns.append(name)

    display_df.columns = new_columns
    numeric_df.columns = new_columns

    return display_df, numeric_df


def _draw_heatmap(ax, display_df: pd.DataFrame, numeric_df: pd.DataFrame):
    """Draw the heatmap with Kaiko.ai's branding: Navy, Indigo, and Slate."""
    rows, cols = display_df.shape

    # Kaiko.ai Color Palette
    KAIKO_PRIMARY = "#000b1c"  # Deep Navy background
    KAIKO_ACCENT = "#4f46e5"  # Vibrant Indigo for 'Average'
    KAIKO_SLATE = "#94a3b8"  # Muted Slate for labels
    KAIKO_TEXT_DARK = "#1e293b"  # Dark slate for cell text

    # Custom Gradient for heat (Light Slate to a clean Blue/Teal)
    from matplotlib.colors import LinearSegmentedColormap

    kaiko_cmap = LinearSegmentedColormap.from_list("kaiko", ["#f1f5f9", "#cbd5e1", "#3b82f6"])

    for j in range(cols):
        col_numeric = numeric_df.iloc[:, j]
        vmin, vmax = col_numeric.min(), col_numeric.max()
        column_name = display_df.columns[j]
        is_avg = "Average" in column_name

        for i in range(rows):
            val_num = col_numeric.iloc[i]
            val_text = display_df.iloc[i, j]

            norm = (val_num - vmin) / (vmax - vmin + 1e-9)

            # Formatting cells
            if is_avg:
                color = KAIKO_ACCENT
                alpha = 0.1 + (norm * 0.2)  # Subtle indigo glow
                text_color = KAIKO_ACCENT
                weight = "bold"
            else:
                color = kaiko_cmap(norm)
                alpha = 0.6
                text_color = KAIKO_TEXT_DARK
                weight = 500

            rect = patches.FancyBboxPatch(
                (j - 0.42, i - 0.38),
                0.84,
                0.76,
                boxstyle="round,pad=0.03",
                facecolor=color,
                alpha=alpha,
                edgecolor="none",
                zorder=1,
            )
            ax.add_patch(rect)

            ax.text(
                j,
                i,
                val_text,
                ha="center",
                va="center",
                fontsize=10,
                fontweight=weight,
                color=text_color,
                zorder=2,
            )

    # Model names: Kaiko uses clean, semi-bold slate typography
    for i, model_name in enumerate(display_df.index):
        ax.text(
            -0.7,
            i,
            model_name,
            ha="right",
            va="center",
            fontsize=11,
            fontweight=600,
            color="#334155",  # Darker slate for readability
        )


def generate_leaderboard(
    csv_path: str = None,
    modality: str = "pathology",
    save_path: str = None,
) -> None:
    """
    Generate transparent leaderboard heatmap that works on both white and black backgrounds.
    """
    config = LeaderboardConfig(modality)

    csv_path = csv_path or f"tools/data/leaderboards/{modality}.csv"
    save_path = save_path or f"docs/images/leaderboards/{modality}.svg"

    df = get_leaderboard(csv_path, config)
    display_df, numeric_df = _prepare_data(df, config)

    plt.rcParams["font.family"] = "sans-serif"

    fig, ax = plt.subplots(figsize=(len(display_df.columns) * 1.8, len(display_df) * 0.6 + 1.5))

    # Transparent backgrounds
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    _draw_heatmap(ax, display_df, numeric_df)

    ax.set_xticks(range(len(display_df.columns)))
    ax.set_xticklabels(
        display_df.columns,
        fontsize=10,
        fontweight=700,
        color="#94a3b8",  # muted slate — good contrast on both bg
        ha="center",
        va="center",
    )

    ax.xaxis.tick_top()
    ax.tick_params(axis="both", which="both", length=0, pad=10)

    ax.set_ylim(len(display_df) - 0.4, -0.8)
    ax.set_xlim(-0.5, len(display_df.columns) - 0.5)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.yaxis.set_visible(False)

    # Save with transparent background
    plt.savefig(save_path, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    CLI(generate_leaderboard)
