"""Validate the REAL source RewardForwardFilter class without importing Atari/EnvPool."""

import ast
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "cleanrl" / "ppo_rnd_envpool.py"


def get_source_reward_filter():
    source = ast.parse(SOURCE.read_text())
    (cls,) = (node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "RewardForwardFilter")
    ns = {"np": np}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(SOURCE), "exec"), ns)
    return ns["RewardForwardFilter"]


def test_original_no_mask_semantics_are_unchanged():
    cls = get_source_reward_filter()
    obj = cls(0.9)
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 3.0, 4.0])
    np.testing.assert_allclose(obj.update(x), x)
    np.testing.assert_allclose(obj.update(y), 0.9 * x + y)


def test_reset_dummy_never_changes_discount_filter_state():
    cls = get_source_reward_filter()
    obj = cls(0.5)
    baseline = np.array([2.0, 3.0, 4.0])
    mask = np.array([True, False, True])
    np.testing.assert_allclose(obj.update(baseline, mask), [2.0, 0, 4.0])
    # A dummy carrying an arbitrarily huge fake intrinsic reward has no effect
    # and in particular causes no spurious extra gamma discount.
    np.testing.assert_allclose(
        obj.update(np.array([10000.0, 10000.0, 10000.0]), [False, False, False]),
        [2.0, 0, 4.0],
    )
    np.testing.assert_allclose(
        obj.update(np.array([1.0, 500.0, 2.0]), [True, False, True]),
        [2.0, 0, 4.0],
    )
    # Only valid positions belong to the RMS observation sample.
    observed_returns = obj.rewems[mask]
    assert observed_returns.size == 2


@pytest.mark.parametrize("bad_mask", [np.array([True]), np.ones((1, 3)), np.ones((3, 1))])
def test_reject_reward_mask_broadcasting(bad_mask):
    cls = get_source_reward_filter()
    with pytest.raises(ValueError):
        cls(0.9).update(np.arange(3.0), valid_mask=bad_mask)
