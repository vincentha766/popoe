import json
from pathlib import Path

import numpy as np
import pytest

from popoe.bop_muse import (
    BOPTargetImage,
    _main,
    bop_frame_manifest,
    build_muse_classes,
    generate_bop_muse_detections,
    load_bop_target_images,
)
from popoe.interfaces import Detection, ObjectModel, Scene
from popoe.segmentor_muse import MUSE_SOURCE, MuseClass


def _write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def _obj(obj_id: int) -> ObjectModel:
    return ObjectModel(obj_id, f"obj_{obj_id:06d}.ply", diameter=0.1)


def _scene(scene_id: int, im_id: int) -> Scene:
    return Scene(
        rgb=np.zeros((5, 6, 3), np.uint8),
        depth=np.ones((5, 6), np.float32),
        K=np.eye(3),
        scene_id=scene_id,
        im_id=im_id,
    )


class _FakeMuseSegmentor:
    def __init__(self, obj_ids=(9, 14)):
        self.classes = [MuseClass(_obj(oid), f"/tpl/{oid}") for oid in obj_ids]
        self.calls = []

    def config(self):
        return {"class_order": [c.obj.obj_id for c in self.classes]}

    def detections_for(self, scene, obj_id, n_masks=None):
        self.calls.append((scene.scene_id, scene.im_id, int(obj_id), n_masks))
        mask = np.zeros((5, 6), bool)
        x0 = 1 if int(obj_id) == 9 else 3
        mask[1:4, x0:x0 + 2] = True
        return [Detection(
            mask=mask,
            score=float(obj_id) / 100.0,
            bbox=(float(x0), 1.0, float(x0 + 2), 4.0),
            descriptor=np.array([0.9, 0.8, 0.7, 0.1], dtype=np.float32),
            source=MUSE_SOURCE,
        )]


def test_load_bop_target_images_groups_repeated_targets_and_filters(tmp_path):
    targets = _write_json(tmp_path / "test_targets_bop19.json", [
        {"scene_id": 2, "im_id": 3, "obj_id": 14, "inst_count": 1},
        {"scene_id": 2, "im_id": 3, "obj_id": 9, "inst_count": 2},
        {"scene_id": 2, "im_id": 3, "obj_id": 9, "inst_count": 2},
        {"scene_id": 1, "im_id": 8, "obj_id": 9, "inst_count": 1},
    ])

    grouped = load_bop_target_images(targets)
    filtered = load_bop_target_images(targets, obj_ids={14})

    assert grouped == [
        BOPTargetImage(scene_id=1, im_id=8, obj_ids=(9,), inst_counts=((9, 1),)),
        BOPTargetImage(
            scene_id=2,
            im_id=3,
            obj_ids=(9, 14),
            inst_counts=((9, 2), (14, 1)),
        ),
    ]
    assert filtered == [
        BOPTargetImage(scene_id=2, im_id=3, obj_ids=(14,), inst_counts=((14, 1),))
    ]


def test_load_bop_target_images_treats_empty_filter_as_empty_and_rejects_negative_limit(tmp_path):
    targets = _write_json(tmp_path / "test_targets_bop19.json", [
        {"scene_id": 2, "im_id": 3, "obj_id": 14, "inst_count": 1},
    ])

    assert load_bop_target_images(targets, obj_ids=set()) == []
    with pytest.raises(ValueError, match="limit_images"):
        load_bop_target_images(targets, limit_images=-1)


def test_load_bop_target_images_uses_row_count_when_inst_count_is_missing(tmp_path):
    targets = _write_json(tmp_path / "test_targets_bop19.json", [
        {"scene_id": 2, "im_id": 3, "obj_id": 14},
        {"scene_id": 2, "im_id": 3, "obj_id": 14},
    ])

    assert load_bop_target_images(targets) == [
        BOPTargetImage(scene_id=2, im_id=3, obj_ids=(14,), inst_counts=((14, 2),))
    ]


def test_bop_frame_manifest_converts_depth_scale_to_metres(tmp_path):
    _write_json(tmp_path / "test" / "000002" / "scene_camera.json", {
        "3": {
            "cam_K": [100.0, 0.0, 3.0, 0.0, 101.0, 2.0, 0.0, 0.0, 1.0],
            "depth_scale": 1.0,
        }
    })

    manifest = bop_frame_manifest(tmp_path, "test", scene_id=2, im_id=3)

    assert manifest.scene_id == 2
    assert manifest.im_id == 3
    assert manifest.depth_scale == pytest.approx(0.001)
    assert manifest.rgb_path.endswith("test/000002/rgb/000003.png")
    assert manifest.depth_path.endswith("test/000002/depth/000003.png")
    assert manifest.K.shape == (3, 3)


def test_build_muse_classes_from_bop_metadata_and_template_root(tmp_path):
    models_info = _write_json(tmp_path / "models" / "models_info.json", {
        "9": {"diameter": 130.0},
        "14": {"diameter": 125.0},
    })
    (tmp_path / "templates" / "obj_000009").mkdir(parents=True)
    (tmp_path / "templates" / "obj_000014").mkdir(parents=True)

    classes = build_muse_classes(
        [14, 9],
        models_info=models_info,
        models_dir=tmp_path / "models",
        template_root=tmp_path / "templates",
    )

    assert [c.obj.obj_id for c in classes] == [9, 14]
    assert classes[0].obj.diameter == pytest.approx(0.130)
    assert classes[0].obj.mesh_path.endswith("models/obj_000009.ply")
    assert classes[1].template_dir.endswith("templates/obj_000014")


def test_build_muse_classes_accepts_explicit_template_overrides(tmp_path):
    info = {"9": {"diameter": 130.0}}
    custom = tmp_path / "custom_mug"
    custom.mkdir()

    classes = build_muse_classes(
        [9],
        models_info=info,
        template_overrides={9: str(custom)},
    )

    assert classes[0].template_dir == str(custom)
    with pytest.raises(FileNotFoundError, match="template directory"):
        build_muse_classes([9], models_info=info, template_overrides={9: str(tmp_path / "missing")})


def test_generate_bop_muse_detections_filters_target_objects_after_full_run():
    pytest.importorskip("pycocotools")
    seg = _FakeMuseSegmentor()
    targets = [
        BOPTargetImage(scene_id=1, im_id=1, obj_ids=(9,)),
        BOPTargetImage(scene_id=1, im_id=2, obj_ids=(9, 14)),
    ]

    records = generate_bop_muse_detections(
        bop_root="/unused",
        split="test",
        targets=targets,
        segmentor=seg,
        n_masks=1,
        target_object_only=True,
        scene_loader=lambda _root, _split, sid, iid: _scene(sid, iid),
        record_time=True,
    )

    assert [(r["scene_id"], r["image_id"], r["category_id"]) for r in records] == [
        (1, 1, 9),
        (1, 2, 9),
        (1, 2, 14),
    ]
    assert all(r["source"] == MUSE_SOURCE for r in records)
    assert all("time" in r and r["time"] >= 0.0 for r in records)
    assert seg.calls == [
        (1, 1, 9, 1),
        (1, 1, 14, 1),
        (1, 2, 9, 1),
        (1, 2, 14, 1),
    ]


def test_generate_bop_muse_detections_resumes_from_full_output_shards(tmp_path):
    pytest.importorskip("pycocotools")
    targets = [BOPTargetImage(scene_id=1, im_id=1, obj_ids=(9,))]
    shard_dir = tmp_path / "shards"

    full_records = generate_bop_muse_detections(
        bop_root="/unused",
        split="test",
        targets=targets,
        segmentor=_FakeMuseSegmentor(),
        n_masks=1,
        target_object_only=False,
        shard_dir=shard_dir,
        scene_loader=lambda _root, _split, sid, iid: _scene(sid, iid),
        record_time=False,
    )

    def fail_loader(*_args):
        raise AssertionError("resume should not reload the BOP image")

    resumed = generate_bop_muse_detections(
        bop_root="/unused",
        split="test",
        targets=targets,
        segmentor=_FakeMuseSegmentor(),
        n_masks=1,
        target_object_only=True,
        shard_dir=shard_dir,
        resume=True,
        scene_loader=fail_loader,
        record_time=False,
    )

    assert [(r["category_id"], r["source"]) for r in full_records] == [
        (9, MUSE_SOURCE),
        (14, MUSE_SOURCE),
    ]
    assert [(r["category_id"], r["source"]) for r in resumed] == [(9, MUSE_SOURCE)]


def test_generate_bop_muse_detections_shard_key_depends_on_segmentor_config(tmp_path):
    pytest.importorskip("pycocotools")
    targets = [BOPTargetImage(scene_id=1, im_id=1, obj_ids=(9,))]
    shard_dir = tmp_path / "shards"

    generate_bop_muse_detections(
        bop_root="/unused",
        split="test",
        targets=targets,
        segmentor=_FakeMuseSegmentor((9, 14)),
        n_masks=1,
        shard_dir=shard_dir,
        scene_loader=lambda _root, _split, sid, iid: _scene(sid, iid),
    )
    seg = _FakeMuseSegmentor((9, 14, 21))
    records = generate_bop_muse_detections(
        bop_root="/unused",
        split="test",
        targets=targets,
        segmentor=seg,
        n_masks=1,
        shard_dir=shard_dir,
        resume=True,
        scene_loader=lambda _root, _split, sid, iid: _scene(sid, iid),
    )

    assert len(list(shard_dir.glob("*.json"))) == 2
    assert sorted({r["category_id"] for r in records}) == [9, 14, 21]
    assert seg.calls == [(1, 1, 9, 1), (1, 1, 14, 1), (1, 1, 21, 1)]


def test_generate_bop_muse_detections_shard_key_depends_on_bop_split(tmp_path):
    pytest.importorskip("pycocotools")
    targets = [BOPTargetImage(scene_id=1, im_id=1, obj_ids=(9,))]
    shard_dir = tmp_path / "shards"

    generate_bop_muse_detections(
        bop_root=tmp_path / "bop",
        split="test",
        targets=targets,
        segmentor=_FakeMuseSegmentor(),
        n_masks=1,
        shard_dir=shard_dir,
        scene_loader=lambda _root, _split, sid, iid: _scene(sid, iid),
    )
    seg = _FakeMuseSegmentor()
    generate_bop_muse_detections(
        bop_root=tmp_path / "bop",
        split="train",
        targets=targets,
        segmentor=seg,
        n_masks=1,
        shard_dir=shard_dir,
        resume=True,
        scene_loader=lambda _root, _split, sid, iid: _scene(sid, iid),
    )

    assert len(list(shard_dir.glob("*.json"))) == 2
    assert seg.calls == [(1, 1, 9, 1), (1, 1, 14, 1)]


def test_generate_bop_muse_detections_floors_n_masks_to_inst_count():
    pytest.importorskip("pycocotools")
    seg = _FakeMuseSegmentor()
    targets = [
        BOPTargetImage(
            scene_id=1,
            im_id=1,
            obj_ids=(9,),
            inst_counts=((9, 3),),
        )
    ]

    generate_bop_muse_detections(
        bop_root="/unused",
        split="test",
        targets=targets,
        segmentor=seg,
        n_masks=1,
        scene_loader=lambda _root, _split, sid, iid: _scene(sid, iid),
    )

    assert seg.calls == [(1, 1, 9, 3), (1, 1, 14, 3)]


def test_resume_requires_shard_dir_and_record_time_uses_separate_shards(tmp_path):
    pytest.importorskip("pycocotools")
    targets = [BOPTargetImage(scene_id=1, im_id=1, obj_ids=(9,))]
    shard_dir = tmp_path / "shards"

    with pytest.raises(ValueError, match="requires shard_dir"):
        generate_bop_muse_detections(
            bop_root="/unused",
            split="test",
            targets=targets,
            segmentor=_FakeMuseSegmentor(),
            resume=True,
        )

    generate_bop_muse_detections(
        bop_root="/unused",
        split="test",
        targets=targets,
        segmentor=_FakeMuseSegmentor(),
        n_masks=1,
        shard_dir=shard_dir,
        scene_loader=lambda _root, _split, sid, iid: _scene(sid, iid),
        record_time=True,
    )
    seg = _FakeMuseSegmentor()
    resumed = generate_bop_muse_detections(
        bop_root="/unused",
        split="test",
        targets=targets,
        segmentor=seg,
        n_masks=1,
        shard_dir=shard_dir,
        resume=True,
        scene_loader=lambda _root, _split, sid, iid: _scene(sid, iid),
        record_time=False,
    )

    assert resumed
    assert seg.calls == [(1, 1, 9, 1), (1, 1, 14, 1)]
    assert len(list(shard_dir.glob("*.json"))) == 2
    assert all("time" not in rec for rec in resumed)


def test_resume_reports_corrupt_shard_path(tmp_path):
    pytest.importorskip("pycocotools")
    targets = [BOPTargetImage(scene_id=1, im_id=1, obj_ids=(9,))]
    shard_dir = tmp_path / "shards"

    generate_bop_muse_detections(
        bop_root="/unused",
        split="test",
        targets=targets,
        segmentor=_FakeMuseSegmentor(),
        n_masks=1,
        shard_dir=shard_dir,
        scene_loader=lambda _root, _split, sid, iid: _scene(sid, iid),
    )
    shard = next(shard_dir.glob("*.json"))
    shard.write_text("{")

    with pytest.raises(ValueError, match=f"corrupt shard {shard}"):
        generate_bop_muse_detections(
            bop_root="/unused",
            split="test",
            targets=targets,
            segmentor=_FakeMuseSegmentor(),
            n_masks=1,
            shard_dir=shard_dir,
            resume=True,
            scene_loader=lambda *_args: pytest.fail("resume should read shard"),
        )


def test_generate_bop_muse_detections_rejects_zero_masks():
    with pytest.raises(ValueError, match="n_masks"):
        generate_bop_muse_detections(
            bop_root="/unused",
            split="test",
            targets=[BOPTargetImage(scene_id=1, im_id=1, obj_ids=(9,))],
            segmentor=_FakeMuseSegmentor(),
            n_masks=0,
            scene_loader=lambda _root, _split, sid, iid: _scene(sid, iid),
        )


def test_cli_rejects_invalid_resume_limit_topk_and_specs():
    base = ["--bop-root", "/unused", "--out", "/tmp/out.json"]

    with pytest.raises(SystemExit, match="--resume requires --shard-dir"):
        _main(base + ["--resume"])
    with pytest.raises(SystemExit, match="--limit-images"):
        _main(base + ["--limit-images", "-1"])
    with pytest.raises(SystemExit, match="--topk"):
        _main(base + ["--topk", "0"])
    with pytest.raises(SystemExit, match="--objs entries"):
        _main(base + ["--objs", "not-an-int"])
    with pytest.raises(SystemExit, match="obj_id must be an integer"):
        _main(base + ["--classes", "x=/tpl"])
