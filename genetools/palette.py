import collections
import dataclasses
import matplotlib as mpl


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


@dataclasses.dataclass
class HueValueStyle:
    """
    Describes how to style a particular value (category) of a categorical hue column.

    Use palettes mapping hue values to HueValueStyles to make plots with different marker shapes, transparencies, z-orders, etc. for different groups.

    The plotting functions accept a hue_key, which identifies a dataframe column that contains hue values.
    They also accept a palette mapping each hue value to a HueValueStyle that defines not just the color to use for that hue value, but also other styles:

    * Scatterplot marker shape, size, primary color, face color, edge color, line width, transparency, and line width.
    * Rectangle/barplot color and hatch pattern.
    * Size scale factor for legend entries.

    For face and edge colors, ``None`` is the default value; to disable them, set to string ``'none'``.
    """

    color: str
    marker: str = None
    marker_size: int = None
    legend_size_scale_factor: float = 1.0
    facecolors: str = None
    edgecolors: str = None
    linewidths: float = None
    zorder: int = 1
    alpha: float = None
    hatch: str = None

    @staticmethod
    def huestyles_to_colors_dict(d: dict) -> dict:
        """Cast any HueValueStyle values in dict to be color strings."""
        return {k: v.color if isinstance(v, HueValueStyle) else v for k, v in d.items()}

    @classmethod
    def from_color(cls, s):
        """Construct from color string only; keep all other marker parameters set to defaults.
        If already a HueValueStyle, pass through without modification."""
        if isinstance(s, cls):
            return s
        return cls(color=s)

    def apply_defaults(self, defaults: "HueValueStyle"):
        """
        Returns new HueValueStyle that applies defaults:
        Modifies this style to fill any missing values with the values from another HueValueStyle.

        Use case: supply global style defaults for an entire scatterplot, then override with customizations in any individual hue value style.
        """
        # The type hint is quoted because inside of the HueValuStyle class, the class itself is not defined yet. https://stackoverflow.com/a/44798831/130164

        # we are detecting unset values by checking whether they are None
        # to make this more robust, we should compare to each field's default value or default factory
        # see https://stackoverflow.com/a/56452356/130164
        changes = {
            field.name: getattr(defaults, field.name)
            for field in dataclasses.fields(defaults)
            if getattr(self, field.name) is None
        }

        # returns copy
        return dataclasses.replace(self, **changes)

    def render_scatter_props(self):
        """Returns kwargs to pass to ax.scatter() to apply this style."""
        return dict(
            color=self.color,
            marker=self.marker,
            s=self.marker_size,
            facecolors=self.facecolors,
            edgecolors=self.edgecolors,
            linewidths=self.linewidths,
            zorder=self.zorder,
            alpha=self.alpha,
        )

    def render_scatter_continuous_props(self):
        """Returns kwargs to pass to ax.scatter() to apply this style, in the context of continuous cmap scatterplots."""
        return dict(
            marker=self.marker,
            s=self.marker_size,
            edgecolors=self.edgecolors,
            linewidths=self.linewidths,
            zorder=self.zorder,
            alpha=self.alpha,
        )

    def render_scatter_legend_props(self):
        """Returns kwargs to pass to ax.legend() to apply this style."""
        # apply scaling to the default marker size we'd get by calling ax.scatter without any size arguments.
        legend_marker_size = (
            mpl.rcParams["lines.markersize"] ** 2
        ) * self.legend_size_scale_factor

        return dict(
            color=self.color,
            marker=self.marker,
            s=legend_marker_size,
            facecolors=self.facecolors,
            edgecolors=self.edgecolors,
            linewidths=self.linewidths,
        )

    def render_rectangle_props(self):
        """Returns kwargs to pass to ax.bar() to apply this style."""
        return dict(color=self.color, hatch=self.hatch)
