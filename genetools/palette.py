import collections
from dataclasses import dataclass


def convert_palette_list_to_dict(palette, hue_names, sort_hues=True):
    """
    If palette is a list, convert it to a dict, assigning a color to each value in hue_names (with sort enabled by default).
    If palette is already a dict, pass it through with no changes.
    """
    if isinstance(palette, collections.abc.Mapping):
        # already a dict. do nothing.
        return palette
    if len(palette) < len(hue_names):
        raise ValueError(
            f"Not enough colors in palette. palette has {len(palette)} colors, but hue_names has {len(hue_names)} values."
        )
    if sort_hues:
        hue_names = sorted(hue_names, key=lambda hue: str(hue))
    return {k: v for k, v in zip(hue_names, palette)}


@dataclass
class HueValueStyle:
    """Describes how to style scatterplot markers for a particular value (i.e. category) of a categorical hue column."""

    color: str
    marker: str = None
    marker_size_scale_factor: float = 1.0
    legend_size_scale_factor: float = 1.0
    # for face and edge colors, None is the default value. to disable them, set to string "none"
    facecolors: str = None
    edgecolors: str = None
    linewidths: float = None
    zorder: int = 1
    alpha: float = None
    hatch: str = None

    @staticmethod
    def huestyles_to_colors_dict(d: dict) -> dict:
        """cast any HueValueStyle values in dict to be color strings"""
        return {k: v.color if isinstance(v, HueValueStyle) else v for k, v in d.items()}

    @classmethod
    def from_color(cls, s):
        """construct from color string only; keep all other marker parameters set to defaults. if already a HueValueStyle, pass through"""
        if isinstance(s, cls):
            return s
        return cls(color=s)
