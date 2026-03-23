"""Color mapping constants."""

from typing import List, Tuple

COLORS = [
    (0, 0, 0),
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (128, 128, 0),  # Olive
    (128, 0, 128),  # Purple
    (0, 128, 128),  # Teal
    (192, 192, 192),  # Silver
    (128, 128, 128),  # Gray
    (255, 165, 0),  # Orange
    (210, 105, 30),  # Chocolate
    (0, 128, 0),  # Lime
    (255, 192, 203),  # Pink
    (255, 69, 0),  # Red-Orange
    (255, 140, 0),  # Dark Orange
    (0, 255, 255),  # Sky Blue
    (0, 255, 127),  # Spring Green
    (0, 0, 139),  # Dark Blue
    (255, 20, 147),  # Deep Pink
    (139, 69, 19),  # Saddle Brown
    (0, 100, 0),  # Dark Green
    (106, 90, 205),  # Slate Blue
    (138, 43, 226),  # Blue-Violet
    (218, 165, 32),  # Goldenrod
    (199, 21, 133),  # Medium Violet Red
    (70, 130, 180),  # Steel Blue
    (165, 42, 42),  # Brown
    (128, 0, 0),  # Maroon
    (255, 0, 255),  # Fuchsia
    (210, 180, 140),  # Tan
    (0, 0, 128),  # Navy
    (139, 0, 139),  # Dark Magenta
    (144, 238, 144),  # Light Green
    (46, 139, 87),  # Sea Green
    (255, 255, 0),  # Gold
    (154, 205, 50),  # Yellow Green
    (0, 191, 255),  # Deep Sky Blue
    (0, 250, 154),  # Medium Spring Green
    (250, 128, 114),  # Salmon
    (255, 105, 180),  # Hot Pink
    (204, 255, 204),  # Pastel Light Green
    (51, 0, 51),  # Very Dark Magenta
    (255, 102, 0),  # Dark Orange
    (0, 255, 0),  # Bright Green
    (51, 153, 255),  # Blue-Purple
    (51, 51, 255),  # Bright Blue
    (204, 0, 0),  # Dark Red
    (90, 90, 90),  # Very Dark Gray
    (255, 255, 51),  # Pastel Yellow
    (255, 153, 255),  # Pink-Magenta
    (153, 0, 76),  # Dark Pink
    (51, 25, 0),  # Very Dark Brown
    (102, 51, 0),  # Dark Brown
    (0, 0, 51),  # Very Dark Blue
    (180, 180, 180),  # Dark Gray
    (102, 255, 204),  # Pastel Green
    (0, 102, 0),  # Dark Green
    (220, 245, 20),  # Lime Yellow
    (255, 204, 204),  # Pastel Pink
    (0, 204, 255),  # Pastel Blue
    (240, 240, 240),  # Light Gray
    (153, 153, 0),  # Dark Yellow
    (102, 0, 51),  # Dark Red-Pink
    (0, 51, 0),  # Very Dark Green
    (255, 102, 204),  # Magenta Pink
    (204, 0, 102),  # Red-Pink
]
"""RGB colors."""

COLORMAP = dict(enumerate(COLORS)) | {255: (255, 255, 255)}
"""Class id to RGB color mapping."""


def get_colors(num_colors: int) -> List[Tuple[int, int, int]]:
    """Get a list of RGB colors.

    If the number of colors is greater than the predefined colors, it will
    repeat the colors until it reaches the requested number

    Args:
        num_colors: The number of colors to return.

    Returns:
        A list of RGB colors.
    """
    colors = COLORS
    while len(colors) < num_colors:
        colors = colors + COLORS[1:]
    return colors
