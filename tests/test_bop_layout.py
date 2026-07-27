"""BOP dataset layout table: the seven Classic-Core sets resolve to the
directory shapes bop_toolkit's dataset_params declares, unknown names refuse
to guess, and bop_frame_manifest builds paths from the layout values.

The failure mode this table exists to prevent is not a crash: under the old
hardcoded rgb/png layout, itodd (gray/*.tif) completed as a clean-looking
all-zero CSV.
"""
import json

import pytest

from popoe.datasets.bop import BOP_LAYOUTS, bop_frame_manifest, bop_layout


def test_seven_core_sets_are_covered():
    assert set(BOP_LAYOUTS) == {"lmo", "tless", "tudl", "icbin", "itodd",
                                "hb", "ycbv"}


def test_tless_layout_matches_bop_toolkit_defaults():
    lay = bop_layout("tless")
    assert lay["split"] == "test_primesense"
    assert lay["models_dir"] == "models_cad"   # tless has no models/


def test_hb_split_is_primesense():
    assert bop_layout("hb")["split"] == "test_primesense"


def test_itodd_is_gray_tif():
    lay = bop_layout("itodd")
    assert (lay["img_dir"], lay["img_ext"]) == ("gray", ".tif")
    assert lay["depth_ext"] == ".tif"


@pytest.mark.parametrize("name", ["lmo", "tudl", "icbin", "ycbv"])
def test_rgb_png_sets_keep_the_plain_layout(name):
    lay = bop_layout(name)
    assert lay == {"split": "test", "img_dir": "rgb", "img_ext": ".png",
                   "depth_ext": ".png", "models_dir": "models"}


def test_unknown_dataset_refuses_to_guess():
    with pytest.raises(ValueError, match="unknown BOP dataset"):
        bop_layout("hope")


def test_overrides_refine_a_known_layout():
    lay = bop_layout("hb", split="test_kinect", models_dir="models_x")
    assert lay["split"] == "test_kinect"
    assert lay["models_dir"] == "models_x"
    # untouched fields keep the table's values
    assert lay["img_dir"] == "rgb"


def test_overrides_do_not_mutate_the_table():
    bop_layout("ycbv", split="something_else")
    assert BOP_LAYOUTS["ycbv"]["split"] == "test"


def test_frame_manifest_builds_paths_from_layout(tmp_path):
    lay = bop_layout("itodd")
    sdir = tmp_path / lay["split"] / "000001"
    sdir.mkdir(parents=True)
    (sdir / "scene_camera.json").write_text(json.dumps(
        {"3": {"cam_K": [500, 0, 32, 0, 500, 24, 0, 0, 1],
               "depth_scale": 0.1}}))
    m = bop_frame_manifest(tmp_path, lay["split"], 1, 3,
                           img_dir=lay["img_dir"], img_ext=lay["img_ext"],
                           depth_ext=lay["depth_ext"])
    assert m.rgb_path.endswith("gray/000003.tif")
    assert m.depth_path.endswith("depth/000003.tif")
    # depth_scale mm-per-unit -> metres-per-unit
    assert m.depth_scale == pytest.approx(0.1 / 1000.0)
