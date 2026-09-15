"""BOP dataset layout table: the seven Classic-Core sets resolve to the
directory shapes bop_toolkit's dataset_params declares, unknown names refuse
to guess, and bop_frame_manifest builds paths from the layout values.

The failure mode this table exists to prevent is not a crash: under the old
hardcoded rgb/png layout, itodd (gray/*.tif) completed as a clean-looking
all-zero CSV.
"""
import json

import pytest

from popoe.datasets.bop import (
    BOP_LAYOUTS, bop_frame_manifest, bop_layout, depth_image_path,
    find_instances, resolve_dataset_layout, rgb_image_path, scene_split_dir,
)


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


def test_resolve_dataset_layout_uses_basename():
    name, lay = resolve_dataset_layout("/data/tless")
    assert name == "tless"
    assert lay["split"] == "test_primesense"
    assert lay["models_dir"] == "models_cad"


def test_resolve_dataset_layout_explicit_name_beats_basename():
    name, lay = resolve_dataset_layout("/data/itodd_copy", dataset="itodd")
    assert name == "itodd"
    assert (lay["img_dir"], lay["img_ext"]) == ("gray", ".tif")


def test_path_helpers_follow_layout(tmp_path):
    _, tless = resolve_dataset_layout(tmp_path / "tless")
    assert scene_split_dir(tmp_path / "tless", tless, 3).as_posix().endswith(
        "test_primesense/000003")
    assert rgb_image_path(tmp_path / "tless", tless, 3, 7).as_posix().endswith(
        "test_primesense/000003/rgb/000007.png")
    _, itodd = resolve_dataset_layout(tmp_path / "itodd")
    assert depth_image_path(tmp_path / "itodd", itodd, 1, 2).as_posix().endswith(
        "test/000001/depth/000002.tif")
    assert rgb_image_path(tmp_path / "itodd", itodd, 1, 2).as_posix().endswith(
        "test/000001/gray/000002.tif")


def test_find_instances_uses_primesense_split(tmp_path):
    root = tmp_path / "tless"
    sdir = root / "test_primesense" / "000001"
    (sdir / "mask_visib").mkdir(parents=True)
    (sdir / "scene_gt.json").write_text(json.dumps(
        {"5": [{"obj_id": 8}, {"obj_id": 1}]}))
    (sdir / "mask_visib" / "000005_000000.png").write_bytes(b"x")
    # A dummy test/ tree must not be consulted for T-LESS.
    (root / "test" / "000001" / "mask_visib").mkdir(parents=True)
    (root / "test" / "000001" / "scene_gt.json").write_text(json.dumps(
        {"9": [{"obj_id": 8}]}))
    (root / "test" / "000001" / "mask_visib" / "000009_000000.png").write_bytes(b"x")
    assert find_instances(root, 8, n=5) == [(1, 5, 0)]


def test_grasp_load_gt_uses_layout_split(tmp_path):
    from popoe.metrics import grasp

    root = tmp_path / "tless"
    sdir = root / "test_primesense" / "000002"
    sdir.mkdir(parents=True)
    (sdir / "scene_gt.json").write_text(json.dumps({
        "4": [{"obj_id": 3, "cam_R_m2c": [1, 0, 0, 0, 1, 0, 0, 0, 1],
               "cam_t_m2c": [1.0, 2.0, 3.0]}],
    }))
    gt = grasp.load_gt(root, [2])
    assert (2, 4, 3) in gt
    assert gt[(2, 4, 3)][0]["t"].shape == (3, 1)


def test_ar_probe_scene_width_reads_tif_depth(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from popoe.metrics.ar import _probe_scene_width

    root = tmp_path / "itodd"
    sdir = root / "test" / "000001" / "depth"
    sdir.mkdir(parents=True)
    img = __import__("numpy").zeros((48, 1280), dtype="uint16")
    assert cv2.imwrite(str(sdir / "000003.tif"), img)
    _, layout = resolve_dataset_layout(root)
    assert _probe_scene_width(root, layout, 1, ["3"]) == 1280.0
