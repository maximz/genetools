from genetools.palette import HueValueStyle


from genetools.palette import HueValueStyle


def test_HueValueStyle_apply_defaults():
    style_for_overall_function = HueValueStyle(color="red", marker="o")
    assert style_for_overall_function.zorder != 100

    style_for_this_hue_value = HueValueStyle(color="blue", zorder=100)
    assert style_for_this_hue_value.marker != "o"

    computed_style = style_for_this_hue_value.apply_defaults(style_for_overall_function)
    assert computed_style.color == "blue"
    assert computed_style.zorder == 100
    assert computed_style.marker == "o"
