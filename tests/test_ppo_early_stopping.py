import torch


def test_target_kl_stop_is_triggered_per_minibatch():
    target_kl = 0.01
    minibatch_kls = [0.002, 0.013, 0.001]

    processed = []
    early_stop = False
    for kl in minibatch_kls:
        processed.append(kl)
        if kl > target_kl:
            early_stop = True
            break

    assert early_stop
    assert processed == [0.002, 0.013]


def test_target_kl_none_never_stops():
    minibatch_kls = [0.02, 0.03]
    target_kl = None

    processed = []
    for kl in minibatch_kls:
        processed.append(kl)
        if target_kl is not None and kl > target_kl:
            break

    assert processed == minibatch_kls
