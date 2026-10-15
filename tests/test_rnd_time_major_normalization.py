# Copyright 2026 The CleanRL contributors.
# SPDX-License-Identifier: MIT
"""Time-major RND intrinsic-reward normalization regression contracts.

Load only the two actual source definitions so the numerical tests do not
depend on a GPU, EnvPool or Atari ROMs.
"""

import ast
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "cleanrl/ppo_rnd_envpool.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
wanted = {
    "RewardForwardFilter",
    "update_intrinsic_reward_running_stats",
}
definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted]
assert {node.name for node in definitions} == wanted
scope = {"np": np}
exec(
    compile(ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[])), str(SOURCE), "exec"),
    scope,
)
RewardForwardFilter = scope["RewardForwardFilter"]
update_stats = scope["update_intrinsic_reward_running_stats"]


class RecordingRunningMeanStd:
    def __init__(self):
        self.samples = []

    def update(self, observations):
        self.samples.extend(np.asarray(observations).tolist())

    @property
    def var(self):
        return np.var(np.asarray(self.samples))


@pytest.mark.parametrize("gamma", [0.0, 0.5, 0.99])
@pytest.mark.parametrize("num_envs", [1, 3])
def test_vector_discounting_runs_along_time_not_across_environments(gamma, num_envs):
    raw = np.arange(1, 5 * num_envs + 1, dtype=np.float64).reshape(5, num_envs)
    expected = []
    running = np.zeros(num_envs)
    for step in raw:
        running = step + gamma * running
        expected.append(running.copy())
    expected = np.asarray(expected)

    stats = RecordingRunningMeanStd()
    observed = update_stats(raw, RewardForwardFilter(gamma), stats)
    np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(stats.samples, expected.ravel(), rtol=0, atol=1e-12)
    assert len(stats.samples) == raw.size
    np.testing.assert_array_equal(raw, np.arange(1, 5 * num_envs + 1).reshape(5, num_envs))


def test_discounted_state_carries_across_rollouts_in_step_order():
    gamma = 0.75
    first = np.array([[1.0, 10.0], [2.0, 20.0]])
    second = np.array([[3.0, 30.0], [4.0, 40.0]])
    stats, state = RecordingRunningMeanStd(), RewardForwardFilter(gamma)
    out_a = update_stats(first, state, stats)
    out_b = update_stats(second, state, stats)
    concat = np.concatenate([out_a, out_b], axis=0)
    expected = []
    accum = np.zeros(2)
    for step in np.concatenate([first, second]):
        accum = gamma * accum + step
        expected.append(accum.copy())
    np.testing.assert_allclose(concat, expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(stats.samples, concat.ravel(), atol=1e-12, rtol=0)


def test_first_return_does_not_alias_mutable_cpu_reward_buffer():
    source = np.array([[1.0, 2.0], [3.0, 4.0]])
    rewards = source.copy()
    filt = RewardForwardFilter(0.5)
    stats = RecordingRunningMeanStd()
    actual = update_stats(rewards, filt, stats)
    expected_state = np.array([3.5, 5.0])
    np.testing.assert_allclose(actual[-1], expected_state)
    # The main algorithm divides the CPU tensor in-place by the running
    # standard deviation. The filter must not point into that storage.
    rewards[:] /= 10
    np.testing.assert_allclose(filt.rewems, expected_state)
    np.testing.assert_allclose(filt.update(np.array([1.0, 1.0])), expected_state * 0.5 + 1)


def test_running_variance_uses_every_time_env_pair_not_only_env_count():
    rewards = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]], dtype=np.float64)
    stats = RecordingRunningMeanStd()
    actual = update_stats(rewards, RewardForwardFilter(0.5), stats)
    expected = np.array([[1.0, 10.0], [3.5, 25.0], [6.75, 42.5]])
    np.testing.assert_allclose(actual, expected)
    assert len(stats.samples) == 6
    np.testing.assert_allclose(stats.var, np.var(expected.reshape(-1)))


def test_numpy_row_view_is_not_retained_after_first_update():
    rewards = np.array([[1.0, 2.0], [3.0, 4.0]])
    filtered = RewardForwardFilter(0.9)
    first = filtered.update(rewards[0])
    rewards[0] *= 1000
    np.testing.assert_array_equal(first, [1.0, 2.0])
    np.testing.assert_array_equal(filtered.rewems, [1.0, 2.0])
