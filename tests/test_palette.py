import pytest
import seaborn as sns
from genetools.palette import HueValueStyle, convert_palette_list_to_dict


def test_HueValueStyle_apply_defaults():
    style_for_overall_function = HueValueStyle(color="red", marker="o")
    assert style_for_overall_function.zorder != 100

    style_for_this_hue_value = HueValueStyle(color="blue", zorder=100)
    assert style_for_this_hue_value.marker != "o"

    computed_style = style_for_this_hue_value.apply_defaults(style_for_overall_function)
    assert computed_style.color == "blue"
    assert computed_style.zorder == 100
    assert computed_style.marker == "o"


def test_cast():
    hvs = HueValueStyle(color="red")
    assert HueValueStyle.from_color(hvs) == hvs
    assert HueValueStyle.from_color("red") == hvs

    palette = {"groupA": HueValueStyle(color="red"), "groupB": "blue"}
    assert HueValueStyle.huestyles_to_colors_dict(palette) == {
        "groupA": "red",
        "groupB": "blue",
    }

    assert convert_palette_list_to_dict(
        ["red", "blue"], ["groupB", "groupA"], sort_hues=True
    ) == {"groupA": "red", "groupB": "blue"}
    assert convert_palette_list_to_dict(
        ["red", "blue"], ["groupB", "groupA"], sort_hues=False
    ) == {"groupB": "red", "groupA": "blue"}
    assert convert_palette_list_to_dict(
        {"groupB": "red", "groupA": "blue"}, ["groupB", "groupA"], sort_hues=True
    ) == {"groupB": "red", "groupA": "blue"}


@pytest.mark.xfail(raises=ValueError)
def test_make_palette_with_not_enough_colors():
    convert_palette_list_to_dict(
        sns.color_palette("muted", n_colors=2), ["groupA", "groupB", "groupC"]
    )
