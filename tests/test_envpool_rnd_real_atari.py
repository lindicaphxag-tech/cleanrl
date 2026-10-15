"""Opt-in REAL Atari/EnvPool contract for RND warmup validity masks.

To run: CLEANRL_TEST_REAL_ATARI=1 python -m pytest -q -s tests/test_envpool_rnd_real_atari.py
Requires envpool 0.8.4, Atari ROMs and a compatible Linux CPU environment.
This checks the source-level statistical sample contract, NOT PPO learning returns.
"""

import os

import numpy as np
import pytest

from cleanrl_utils.envpool_rnd_warmup import initialize_rnd_observation_stats


@pytest.mark.skipif(
    os.environ.get("CLEANRL_TEST_REAL_ATARI") != "1",
    reason="Set CLEANRL_TEST_REAL_ATARI=1 to run the genuine Atari/EnvPool integration test",
)
def test_real_breakout_terminal_actions_are_counted_but_dummy_resets_are_not():
    # An explicit integration-test opt-in must fail when EnvPool is missing.
    import envpool
    raw = envpool.make(
        "Breakout-v5",
        env_type="gym",
        num_envs=2,
        episodic_life=True,
        reward_clip=True,
        seed=20261015,
    )

    class InspectEnv:
        def __init__(self):
            self.single_action_space = raw.action_space
            self.previous_done = np.zeros(2, dtype=bool)
            self.valid_count = 0
            self.dummy_count = 0
            self.terminal_count = 0
            self.reset_count = 0

        def reset(self):
            self.reset_count += 1
            self.previous_done[:] = False
            return raw.reset()

        def step(self, actions):
            observation, reward, done, info = raw.step(actions)
            # The oracle depends only on the REAL EnvPool terminal stream,
            # never on production RND masks or a replay of warmup logic.
            self.valid_count += int((~self.previous_done).sum())
            self.dummy_count += int(self.previous_done.sum())
            self.terminal_count += int(np.asarray(done).sum())
            self.previous_done = np.asarray(done, dtype=bool)
            return observation, reward, done, info

    class Moments:
        def __init__(self):
            self.count = 0

        def update(self, batch):
            assert batch.shape[1:] == (1, 84, 84)
            self.count += len(batch)

    env = InspectEnv()
    original = env.reset()
    stats = Moments()
    aligned = initialize_rnd_observation_stats(env, stats, 2, 3200, 1)
    assert aligned.shape == original.shape == (2, 4, 84, 84)
    assert env.reset_count == 2
    assert env.terminal_count > 0
    assert env.dummy_count > 0
    assert env.dummy_count + env.valid_count == 6400
    assert stats.count == env.valid_count

    next_observation, reward, done, info = env.step(np.zeros(2, dtype=np.int64))
    assert next_observation.shape == original.shape
