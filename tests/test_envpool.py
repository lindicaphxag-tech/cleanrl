import subprocess

import torch

from cleanrl.ppo_atari_envpool import _valid_transition_indices, compute_gae


def test_ppo_atari_envpool():
    subprocess.run(
        "python cleanrl/ppo_atari_envpool.py --num-envs 8 --num-steps 32 --total-timesteps 256",
        shell=True,
        check=True,
    )


def test_ppo_rnd_envpool():
    subprocess.run(
        "python cleanrl/ppo_rnd_envpool.py --num-envs 8 --num-steps 32 --num-iterations-obs-norm-init 1 --total-timesteps 256",
        shell=True,
        check=True,
    )


def test_ppo_atari_envpool_xla_jax():
    subprocess.run(
        "python cleanrl/ppo_atari_envpool_xla_jax.py --num-envs 8 --num-steps 6 --update-epochs 1 --num-minibatches 1 --total-timesteps 256",
        shell=True,
        check=True,
    )


def test_ppo_atari_envpool_xla_jax_scan():
    subprocess.run(
        "python cleanrl/ppo_atari_envpool_xla_jax_scan.py --num-envs 8 --num-steps 6 --update-epochs 1 --num-minibatches 1 --total-timesteps 256",
        shell=True,
        check=True,
    )


def test_ppo_atari_envpool_xla_jax_scan_eval():
    subprocess.run(
        "python cleanrl/ppo_atari_envpool_xla_jax_scan.py --save-model --num-envs 8 --num-steps 6 --update-epochs 1 --num-minibatches 1 --total-timesteps 256",
        shell=True,
        check=True,
    )


def test_envpool_gae_excludes_dummy_transition():
    rewards_a = torch.tensor([[1.0], [2.0], [1234.0], [3.0]])
    rewards_b = rewards_a.clone()
    rewards_b[2] = -9876.0
    dones = torch.tensor([[0.0], [0.0], [1.0], [0.0]])
    values_a = torch.tensor([[0.5], [0.75], [1000.0], [1.25]])
    values_b = values_a.clone()
    values_b[2] = -2000.0
    next_done = torch.tensor([0.0])
    next_value = torch.tensor([1.5])

    advantages_a = compute_gae(rewards_a, dones, values_a, next_done, next_value, gamma=0.99, gae_lambda=0.95)
    advantages_b = compute_gae(rewards_b, dones, values_b, next_done, next_value, gamma=0.99, gae_lambda=0.95)

    assert advantages_a[2].item() == 0.0
    assert advantages_b[2].item() == 0.0
    torch.testing.assert_close(advantages_a[[0, 1, 3]], advantages_b[[0, 1, 3]])
    assert _valid_transition_indices(dones).tolist() == [0, 1, 3]
