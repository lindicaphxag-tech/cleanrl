"""RND continuing intrinsic GAE must ignore EnvPool reset-only steps."""
import numpy as np
import pytest
import torch

from cleanrl_utils.envpool_rnd_gae import intrinsic_gae_skip_reset_dummy


def independent_numpy_oracle(reward, value, previous_done, bootstrap, gamma, decay):
    """Remove each environment's dummy steps before running scalar reference GAE."""
    reward, value, mask = map(np.asarray, (reward, value, previous_done))
    expected = np.zeros_like(reward)
    for env in range(reward.shape[1]):
        idxs = np.flatnonzero(mask[:, env] == 0)
        continuation = float(bootstrap[env])
        gae = 0.0
        for index in idxs[::-1]:
            delta = reward[index, env] + gamma * continuation - value[index, env]
            gae = delta + gamma * decay * gae
            expected[index, env] = gae
            continuation = float(value[index, env])
    return expected


def test_dummy_rewards_and_values_do_not_change_compressed_gae():
    rew = torch.tensor([[1.], [2.], [3333.], [3.]], dtype=torch.float64)
    value = torch.tensor([[.5], [1.], [-10000.], [1.5]], dtype=torch.float64)
    done = torch.tensor([[0.], [0.], [1.], [0.]])
    bootstrap = torch.tensor([5.], dtype=torch.float64)
    result = intrinsic_gae_skip_reset_dummy(rew, value, done, bootstrap, .99, .95)
    old_dummy_reward, old_dummy_value = rew[2].clone(), value[2].clone()
    rew[2] = -1e10
    value[2] = 1e10
    again = intrinsic_gae_skip_reset_dummy(rew, value, done, bootstrap, .99, .95)
    torch.testing.assert_close(result, again, atol=0, rtol=0)
    assert result[2].item() == 0.0
    assert old_dummy_reward.item() != rew[2].item() and old_dummy_value.item() != value[2].item()


@pytest.mark.parametrize("gamma,lam", [(1., 1.), (.99, .95), (.5, 0.), (.001, .6)])
def test_100_random_multienv_instances_agree_with_compressed_scalar_oracle(gamma, lam):
    rng = np.random.default_rng(20261014)
    for _ in range(100):
        t = int(rng.integers(1, 17))
        n = int(rng.integers(1, 9))
        rew = rng.standard_normal((t, n))
        values = rng.standard_normal((t, n))
        done = rng.integers(0, 2, (t, n))
        boot = rng.standard_normal(n)
        expected = independent_numpy_oracle(rew, values, done, boot, gamma, lam)
        got = intrinsic_gae_skip_reset_dummy(
            torch.tensor(rew, dtype=torch.float64),
            torch.tensor(values, dtype=torch.float64),
            torch.tensor(done, dtype=torch.bool),
            torch.tensor(boot, dtype=torch.float64),
            gamma, lam)
        np.testing.assert_allclose(got.numpy(), expected, atol=1e-13, rtol=1e-13)


def test_no_dummy_behavior_matches_original_recursion():
    rew = torch.tensor([[1., 2.], [2., 3.], [1., 4.]])
    values = torch.tensor([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]])
    bootstrap = torch.tensor([0.4, 0.5])
    actual = intrinsic_gae_skip_reset_dummy(
        rew, values, torch.zeros_like(rew), bootstrap, .99, .95)
    last = torch.zeros_like(bootstrap)
    expected = torch.empty_like(rew)
    for t in reversed(range(rew.shape[0])):
        nextvalue = bootstrap if t == rew.shape[0] - 1 else values[t + 1]
        last = rew[t] + .99 * nextvalue - values[t] + .99 * .95 * last
        expected[t] = last
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_all_dummy_advantages_are_exact_zero():
    rew = torch.randn(4, 3)
    values = torch.randn(4, 3)
    got = intrinsic_gae_skip_reset_dummy(
        rew, values, torch.ones_like(rew), torch.randn(3), .99, .95)
    assert torch.equal(got, torch.zeros_like(rew))


@pytest.mark.parametrize("bad", [
    (torch.ones(3), torch.ones(3), torch.zeros(3)),
    (torch.ones(3, 2), torch.ones(3, 1), torch.zeros(3, 2)),
    (torch.ones(3, 2), torch.ones(3, 2), torch.ones(3, 2) * 2),
])
def test_reject_invalid_shapes_and_nonboolean_masks(bad):
    with pytest.raises(ValueError):
        intrinsic_gae_skip_reset_dummy(*bad, torch.ones(2), .99, .95)
