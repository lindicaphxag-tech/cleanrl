import numpy as np
import pytest
from cleanrl_utils.envpool_dummy_steps import valid_transition_indices


def test_real_terminal_transition_kept_but_reset_dummy_removed():
    # EnvPool's auto-reset protocol:
    # 0,1 are real steps; 2 observes terminal S_T from done[1] and its
    # action causes an automatic-reset dummy jump to S_0.
    done_before_step = np.array([[0], [0], [1], [0], [0], [1]])
    np.testing.assert_array_equal(
        valid_transition_indices(done_before_step), [0, 1, 3, 4]
    )


def test_vectorized_rollout_keeps_independent_env_boundaries():
    # Time-major flattening matches PPO's obs.reshape((-1,) + ...).
    previous = np.array([[0, 0, 0],
                         [0, 1, 0],
                         [1, 0, 0],
                         [0, 0, 1]], dtype=np.int64)
    np.testing.assert_array_equal(
        valid_transition_indices(previous), [0, 1, 2, 3, 5, 7, 8, 9, 10]
    )


def test_real_terminal_advantage_has_no_next_state_bootstrap():
    # The source PPO GAE recursion uses nextnonterminal=1-dones[t+1].
    # A terminal transition at t=1 is *kept*, while next dummy t=2 is
    # removed. Its action's artificial value must not enter learning.
    done_before_step = np.array([[0], [0], [1], [0]])
    reward = np.array([0., 3., 0., 0.])
    value = np.array([1., 2., 500., 1.])
    discount = 0.99
    nextnonterminal = 1. - done_before_step[2, 0]
    actual_terminal_delta = reward[1] + discount * value[2] * nextnonterminal - value[1]
    assert actual_terminal_delta == 1.0
    selected = valid_transition_indices(done_before_step)
    assert 1 in selected and 2 not in selected


def test_nondone_rollout_is_identical_to_existing_training_distribution():
    np.testing.assert_array_equal(
        valid_transition_indices(np.zeros((5, 4), dtype=bool)),
        np.arange(20)
    )


def test_tensor_like_flags_are_accepted_without_backprop():
    data = [[False, True], [True, False]]
    np.testing.assert_array_equal(valid_transition_indices(data), [0, 3])


@pytest.mark.parametrize("bad", [
    np.array([0, 1]), np.zeros((1, 2, 3)),
    np.array([[0, 2]]), np.array([[0., np.nan]]),
    np.array([[-1., 0.]]),
])
def test_bad_or_ambiguous_done_buffers_fail_closed(bad):
    with pytest.raises(ValueError):
        valid_transition_indices(bad)


def test_no_real_transitions_is_not_a_silent_empty_minibatch():
    with pytest.raises(ValueError, match="No real"):
        valid_transition_indices(np.ones((4, 2), dtype=bool))
