"""Intrinsic GAE for EnvPool's explicit post-terminal reset dummy step.

Unlike extrinsic reward, RND's intrinsic return remains *continuing*
across real episode terminals. The correct transition chain therefore
skips dummy reset actions but can span a genuine terminal boundary.
"""

import numpy as np
import torch


def intrinsic_gae_skip_reset_dummy(reward, value, previous_done, bootstrap, gamma, gae_lambda):
    """Compute RND continuing-task GAE on the compressed valid transition chain.

    Args:
      reward/value/previous_done: T x num_envs time-major tensors.
      bootstrap: latest value of the next real observation, [num_envs] or
        [1,num_envs]; this supplies the successor of a trailing dummy step.

    The dummy value has no semantic right to affect Bellman targets.
    The next value for a valid t is the next valid t' > t, or bootstrap.
    This also ensures 1 dummy step does not add an extra discount factor.
    """
    if reward.ndim != 2 or value.shape != reward.shape or previous_done.shape != reward.shape:
        raise ValueError("Expected aligned [steps, envs] tensors")
    if reward.shape[0] == 0 or not 0 < gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("Empty or invalid time and discount contract")
    if not torch.isfinite(reward).all() or not torch.isfinite(value).all():
        raise ValueError("Non-finite RND intrinsic samples")
    if not torch.all((previous_done == 0) | (previous_done == 1)):
        raise ValueError("Invalid EnvPool previous-terminal masks")
    successor = bootstrap.reshape(-1).to(value.device)
    if successor.shape != value.shape[1:]:
        raise ValueError("Bootstrap shape inconsistent with number of envs")
    gae = torch.zeros_like(successor)
    advantages = torch.zeros_like(reward)
    for t in reversed(range(reward.shape[0])):
        valid = ~previous_done[t].bool()
        delta = reward[t] + gamma * successor - value[t]
        gae = torch.where(valid, delta + gamma * gae_lambda * gae, gae)
        advantages[t] = torch.where(valid, gae, torch.zeros_like(gae))
        successor = torch.where(valid, value[t], successor)
    return advantages


def discounted_intrinsic_returns(reward_filter, rewards, valid_mask):
    """Advance independent per-environment discounted rewards along TIME.

    Inputs are time-major [steps, envs]. EnvPool reset-dummy steps
    neither add a reward nor consume a discount step. The filter retains
    per-environment state across consecutive rollout chunks.
    """
    rewards = np.asarray(rewards)
    valid_mask = np.asarray(valid_mask)
    if rewards.ndim != 2 or rewards.shape != valid_mask.shape:
        raise ValueError("Expected aligned time-major [steps, envs] arrays")
    if not np.issubdtype(rewards.dtype, np.number) or not np.isfinite(rewards).all():
        raise ValueError("Expected finite numeric intrinsic rewards")
    if not np.isin(valid_mask, (False, True)).all():
        raise ValueError("EnvPool validity mask must be Boolean")
    if rewards.shape[0] == 0:
        raise ValueError("Expected at least one timestep")
    return np.stack(
        [
            reward_filter.update(reward_at_t, valid_mask=valid_at_t).copy()
            for reward_at_t, valid_at_t in zip(rewards, valid_mask)
        ]
    )
