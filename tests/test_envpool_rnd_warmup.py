"""Regression oracle for RND warmup: temporal semantics and simulator state."""

from types import SimpleNamespace

import numpy as np
import pytest

from cleanrl_utils.envpool_rnd_warmup import initialize_rnd_observation_stats


class TrackedRMS:
    def __init__(self):
        self.samples = []

    def update(self, samples):
        self.samples.append(np.asarray(samples).copy())


class StubDelayedResetEnv:
    """Deterministic two-env simulator with an observable delayed-reset dummy."""

    def __init__(self, frames=None, dones=None):
        self.single_action_space = SimpleNamespace(n=4)
        self.frames = frames or [[10, 20], [9999, 30], [40, 50], [60, 70]]
        self.dones = dones or [[True, False], [False, False], [False, True], [False, False]]
        self.simulator_clock = 0
        self.reset_calls = 0

    def reset(self):
        self.simulator_clock = 0
        self.reset_calls += 1
        return self._frame([-1, -2])

    def _frame(self, markers):
        result = np.zeros((2, 4, 2, 2), dtype=np.float64)
        result[:, 3, :, :] = np.asarray(markers)[:, None, None]
        return result

    def step(self, actions):
        assert np.asarray(actions).shape == (2,)
        idx = self.simulator_clock
        if idx >= len(self.frames):
            raise AssertionError("unexpected simulator action")
        self.simulator_clock += 1
        return self._frame(self.frames[idx]), np.zeros(2), np.asarray(self.dones[idx]), {}


def observed_markers(stats):
    return [chunk[:, 0, 0, 0].tolist() for chunk in stats.samples]


def test_drops_only_reset_dummy_but_keeps_real_terminal_observation():
    env = StubDelayedResetEnv()
    prior_observation = env.reset().copy()
    stats = TrackedRMS()
    current_observation = initialize_rnd_observation_stats(env, stats, 2, 3, 1)
    # First step terminates env0 (its image 10 is real), second step is an
    # automatic reset (9999 is invalid), third step is a real transition.
    assert observed_markers(stats) == [[10, 20, 30, 40, 50]]
    assert env.reset_calls == 2
    assert env.simulator_clock == 0
    np.testing.assert_array_equal(current_observation, prior_observation)


def test_warmup_must_reset_before_the_first_policy_action():
    env = StubDelayedResetEnv()
    env.reset()
    stats = TrackedRMS()
    initial = initialize_rnd_observation_stats(env, stats, 2, 2, 1)
    assert env.simulator_clock == 0  # first training step really acts on reset state
    np.testing.assert_equal(initial[:, 3, 0, 0], [-1, -2])
    env.step(np.zeros(2, dtype=np.int64))
    assert env.simulator_clock == 1


def test_chunks_never_mix_dummy_frames_and_ignore_current_terminal_only_next_step():
    env = StubDelayedResetEnv()
    env.reset()
    stats = TrackedRMS()
    initialize_rnd_observation_stats(env, stats, 2, 2, 2)
    assert observed_markers(stats) == [[10, 20, 30], [40, 50, 60]]


def test_zero_warmup_leaves_existing_reset_untouched():
    env = StubDelayedResetEnv()
    env.reset()
    stats = TrackedRMS()
    out = initialize_rnd_observation_stats(env, stats, 2, 3, 0)
    assert out is None
    assert env.reset_calls == 1 and env.simulator_clock == 0
    assert stats.samples == []


@pytest.mark.parametrize("args", [(0, 1, 1), (2, 0, 1), (2, 1, -1)])
def test_invalid_warmup_configuration_is_rejected_before_steps(args):
    env = StubDelayedResetEnv()
    stats = TrackedRMS()
    with pytest.raises(ValueError):
        initialize_rnd_observation_stats(env, stats, *args)
    assert env.simulator_clock == 0


def test_reject_malformed_done_flags_before_stats_update():
    env = StubDelayedResetEnv(dones=[[2, False]])
    env.reset()
    stats = TrackedRMS()
    with pytest.raises(ValueError):
        initialize_rnd_observation_stats(env, stats, 2, 1, 1)
    assert stats.samples == []


@pytest.mark.parametrize("seed", range(48))
def test_randomized_per_environment_terminal_masks_match_independent_oracle(seed):
    """Different terminal clocks and stats-chunk sizes cannot mix dummy rows."""
    rng = np.random.default_rng(seed)
    steps_per_chunk = int(rng.integers(1, 9))
    chunks = int(rng.integers(1, 5))
    total = steps_per_chunk * chunks
    frames = rng.integers(0, 10000, size=(total, 2)).tolist()
    dones = rng.integers(0, 2, size=(total, 2)).astype(bool).tolist()
    env = StubDelayedResetEnv(frames=frames, dones=dones)
    env.reset()
    stats = TrackedRMS()
    initialize_rnd_observation_stats(env, stats, 2, steps_per_chunk, chunks)

    previous_done = [False, False]
    expected = []
    for chunk_index in range(chunks):
        chunk = []
        for step in range(chunk_index * steps_per_chunk, (chunk_index + 1) * steps_per_chunk):
            chunk.extend(frames[step][env_id] for env_id in range(2) if not previous_done[env_id])
            previous_done = dones[step]
        if chunk:
            expected.append(chunk)
    assert observed_markers(stats) == expected
    assert env.reset_calls == 2
    assert env.simulator_clock == 0
