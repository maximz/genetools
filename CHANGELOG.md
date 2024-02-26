# Changelog

## 0.7.0

* Many new stats, plotting, and helper functions. See PRs [#24](https://github.com/maximz/genetools/pull/24) and [#73](https://github.com/maximz/genetools/pull/73).

## 0.6.0

* Two new functions to customize tick labels on any existing plot:
    * `wrap_tick_labels`: add text wrapping
    * `add_sample_size_to_labels`: add group sample sizes with a `(n=N)` suffix
* Make writing PDF figures a deterministic process and make the PDF text editable.
* Scatterplot improvements:
    - Adjust default marker shape and size to work better for most plots.
    - Adjust HueValueStyle so that an explicit marker size is not specified there, only a marker size scaling factor. The scatterplot itself is responsible for defining the base marker size, while the palette of HueValueStyles should be drawable at any marker size.
    - Change legends so that only one marker is drawn to indicate a group's style.

## 0.5.0 (2022-01-10)

* Improve scatter plots and stacked bar plots.
* Introduce `HueValueStyle` for granular styling of each hue.

## 0.4.0 (2020-07-22)

* Centered log ratio (CLR) normalization for Cite-seq protein data.

## 0.3.0 (2020-06-03)

* Pandas helpers for easier normalization

## 0.2.0 (2020-06-03)

* Far faster implementation of `stats.coclustering`
* Introducing `helpers.make_slurm_command`
* Global submodule import (no longer need to import submodules individually)

## 0.1.0 (2020-03-06)

* First release on PyPI.
