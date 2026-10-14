"""CleanRL/EnvPool rollout validity under delayed automatic reset semantics.

EnvPool's Atari API emits a terminal observation and then a dummy transition
to the next reset observation. The transition immediately *after* the
preceding done=True is not an action-bearing MDP sample.

This only selects valid training samples: it does not rewrite rewards,
terminations, or compute counterfactual replay probabilities.
"""

import numpy as np


def valid_transition_indices(previous_dones: np.ndarray) -> np.ndarray:
    """Flatten time-major [steps, envs] rollouts, excluding reset dummy steps.

    A preceding done=True at buffer position (t,e) means this recorded
    observation follows a terminal transition. EnvPool's t-th step is a
    dummy automatic reset transition; it must NOT contribute to PPO losses.

    The genuine terminal *transition* t-1 stays in training. The existing
    GAE recursion already ends its trace at nextnonterminal=1-dones[t].
    """
    previous_dones = np.asarray(previous_dones)
    if previous_dones.ndim != 2:
        raise ValueError("Expected time-major [num_steps, num_envs] done buffer")
    if not np.isin(previous_dones, (0, 1)).all():
        raise ValueError("Done buffer must consist only of Boolean flags")
    valid = np.flatnonzero(previous_dones.reshape(-1) == 0)
    if not len(valid):
        raise ValueError("No real EnvPool transitions available for PPO update")
    return valid
