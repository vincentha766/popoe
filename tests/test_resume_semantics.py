"""adapters.resolve_resume — count-only, inst_count-aware resume classification.

The writer's completion invariant (examples/bop_eval.py): a finished target
emits EXACTLY inst_count rows, zero-padded when fewer champions were found.
So resume needs only counts — row contents are never consulted, because
"crashed after two rows" and "completed with two champions" are
indistinguishable from contents (and real scores can format as 0.000000).

The invariant that guards LMO/YCB-V: with inst_count==1 everywhere, ANY
existing row marks its target done and partial is impossible — byte-identical
to the old any-row rule.
"""

from popoe.adapters import resolve_resume

K = (48, 1, 5)   # (scene, im, obj)


def test_single_instance_any_row_is_done():
    done, partial = resolve_resume({K: 1}, {K: 1})
    assert done == {K} and partial == set()


def test_multi_instance_complete_is_done():
    done, partial = resolve_resume({K: 3}, {K: 3})
    assert done == {K} and partial == set()


def test_padded_target_is_done():
    # A completed inst_count=3 target with only 2 usable champions is
    # zero-padded to 3 rows by the writer — count says done. (This is the
    # codex round-3 major: without padding, a legitimately under-detected
    # target was indistinguishable from a crash and re-ran forever.)
    done, partial = resolve_resume({K: 3}, {K: 3})
    assert done == {K} and partial == set()


def test_multi_instance_partial_is_flagged_for_rerun():
    done, partial = resolve_resume({K: 1}, {K: 3})
    assert done == set() and partial == {K}
    done, partial = resolve_resume({K: 2}, {K: 3})
    assert done == set() and partial == {K}


def test_excess_rows_still_done():
    # More rows than inst_count (e.g. targets file changed between runs):
    # treat as done — never silently delete scored rows for a "complete" key.
    done, partial = resolve_resume({K: 5}, {K: 3})
    assert done == {K} and partial == set()


def test_stale_row_for_untargeted_key_defaults_to_done():
    # A row from an earlier run with different --objs: not in target_counts,
    # treated as inst_count=1. It is not in this run's targets anyway.
    other = (99, 99, 99)
    done, partial = resolve_resume({other: 1}, {K: 3})
    assert other in done and partial == set()


def test_empty_inputs():
    assert resolve_resume({}, {K: 3}) == (set(), set())
    assert resolve_resume({K: 0}, {K: 3}) == (set(), set())
