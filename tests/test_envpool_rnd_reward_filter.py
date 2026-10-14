"""Validate the REAL source RewardForwardFilter class without importing Atari/EnvPool."""

import ast
from pathlib import Path

import numpy as np
import pytest

from cleanrl_utils.envpool_rnd_gae import discounted_intrinsic_returns

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


# This catches a real axis inversion in the original RND EnvPool integration:
# the reward filter received [steps] one environment at a time, but its
# recurrence MUST receive [envs] one timestep at a time.
def independent_time_major_oracle(rewards, valid_mask, gamma, previous=None):
    state = np.zeros(rewards.shape[1], dtype=np.float64) if previous is None else previous.copy()
    trace = []
    for t in range(rewards.shape[0]):
        for env in range(rewards.shape[1]):
            if valid_mask[t, env]:
                state[env] = gamma * state[env] + rewards[t, env]
        trace.append(state.copy())
    return np.array(trace)


def test_discounted_rewards_recur_through_time_per_environment():
    reward = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    valid = np.ones_like(reward, dtype=bool)
    got = discounted_intrinsic_returns(get_source_reward_filter()(0.5), reward, valid)
    np.testing.assert_array_equal(got, [[1.0, 10.0], [2.5, 25.0], [4.25, 42.5]])
    # A transposed loop across envs produces [[1, 10.5], [2, 21], [3, 31.5]].
    np.testing.assert_array_equal(got, independent_time_major_oracle(reward, valid, 0.5))


def test_discounted_rewards_skip_dummy_without_discount_or_cross_env_leakage():
    reward = np.array([[1.0, 10.0], [1e6, 20.0], [3.0, 1e6], [4.0, 40.0]])
    valid = np.array([[True, True], [False, True], [True, False], [True, True]])
    obj = get_source_reward_filter()(0.5)
    got = discounted_intrinsic_returns(obj, reward, valid)
    expected = independent_time_major_oracle(reward, valid, 0.5)
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(got[-1], [5.75, 52.5])


def test_discounted_rewards_rollout_chunks_equal_one_continuous_rollout():
    rng = np.random.default_rng(20261014)
    reward = rng.uniform(0, 4, size=(51, 7))
    valid = rng.integers(0, 2, size=(51, 7)).astype(bool)
    filter_full = get_source_reward_filter()(0.97)
    full = discounted_intrinsic_returns(filter_full, reward, valid)
    filter_split = get_source_reward_filter()(0.97)
    split = np.concatenate(
        [
            discounted_intrinsic_returns(filter_split, reward[:17], valid[:17]),
            discounted_intrinsic_returns(filter_split, reward[17:], valid[17:]),
        ]
    )
    np.testing.assert_allclose(full, split, atol=0, rtol=0)
    np.testing.assert_allclose(full, independent_time_major_oracle(reward, valid, 0.97), atol=1e-14)


def test_discounted_rewards_equivariant_to_environment_permutation():
    rng = np.random.default_rng(451)
    reward = rng.uniform(0, 2, size=(13, 5))
    valid = rng.integers(0, 2, size=(13, 5)).astype(bool)
    permutation = np.array([4, 1, 3, 0, 2])
    plain = discounted_intrinsic_returns(get_source_reward_filter()(0.8), reward, valid)
    permuted = discounted_intrinsic_returns(
        get_source_reward_filter()(0.8), reward[:, permutation], valid[:, permutation]
    )
    np.testing.assert_allclose(permuted, plain[:, permutation], atol=0, rtol=0)


def test_discounted_rewards_reject_invalid_time_major_arrays():
    obj = get_source_reward_filter()(0.5)
    with pytest.raises(ValueError):
        discounted_intrinsic_returns(obj, np.ones((3, 2)), np.ones((2, 3)))
    with pytest.raises(ValueError):
        discounted_intrinsic_returns(obj, np.ones((0, 2)), np.ones((0, 2), dtype=bool))
    with pytest.raises(ValueError):
        discounted_intrinsic_returns(obj, np.ones((3, 2)), np.ones((3, 2)) * 2)
    assert obj.rewems is None  # Invalid input cannot partially mutate filter state.
