"""RND observation-stat warmup with EnvPool delayed-reset semantics.

Warmup calls env.step() and therefore MUTATES the simulator. Returning a fresh
reset observation is part of the contract: the rollout's initial observation
must correspond to the simulator state that will execute the first action.
"""

import numpy as np


def initialize_rnd_observation_stats(envs, obs_rms, num_envs, num_steps, num_iterations):
    """Warm up observation moments with action-bearing transitions only.

    Args:
        envs: EnvPool Gym-compatible vector environment, already reset.
        obs_rms: RunningMeanStd for last-frame images, shape (1, 1, H, W).
        num_envs: Number of vectorized environments.
        num_steps: Number of simulated actions per stats-update chunk.
        num_iterations: Number of stats-update chunks (zero skips warmup).

    Returns:
        The fresh initial observation after a second envs.reset(), or None
        when zero iterations are requested, leaving the original reset intact.

    EnvPool's action immediately after done=True is an automatic-reset dummy.
    Its returned observation is not a next-state for an action-bearing sample.
    A *current* done=True is a genuine terminal action, so retain its result.
    """
    if num_envs < 1 or num_steps < 1 or num_iterations < 0:
        raise ValueError("Require num_envs >= 1, num_steps >= 1, num_iterations >= 0")
    if num_iterations == 0:
        return None

    previous_done = np.zeros(num_envs, dtype=bool)
    pending = []
    for step in range(num_steps * num_iterations):
        actions = np.random.randint(0, envs.single_action_space.n, size=num_envs)
        observations, _, done, _ = envs.step(actions)
        observations = np.asarray(observations)
        done = np.asarray(done)
        if observations.ndim != 4 or observations.shape[0] != num_envs or observations.shape[1] != 4:
            raise ValueError("Expected [envs, 4, height, width] Atari observations")
        if done.shape != (num_envs,) or not np.isin(done, (0, 1)).all():
            raise ValueError("Expected Boolean terminal signals for every environment")

        # Each row represents an action-bearing transition from the preceding
        # observation, not the automatic reset command after a true terminal.
        real_observations = observations[~previous_done, 3:4, :, :]
        if len(real_observations):
            pending.append(real_observations)
        previous_done = done.astype(bool)

        if (step + 1) % num_steps == 0:
            if pending:
                obs_rms.update(np.concatenate(pending, axis=0))
                pending.clear()

    # The simulator moved during the warmup. Never reuse the previous
    # initial observation/termination flag for the first actual rollout.
    return envs.reset()
