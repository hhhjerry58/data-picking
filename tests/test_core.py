import copy
import random

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from data_picking.checkpoint import latest_checkpoint, load_training_checkpoint, save_checkpoint, validate_checkpoint
from data_picking.common import load_config, paths, seed_all
from data_picking.data import duplicate_groups, quotas
from data_picking.loss import per_example_nll
from data_picking.tokenization import tokenize_example


def test_causal_mask_and_per_example_normalization():
    logits = torch.tensor([[[1., 0., -1.], [0., 2., -2.], [0., 0., 0.], [3., 1., 2.]],
                           [[2., 0., 1.], [1., 0., -1.], [0., 1., 2.], [0., 0., 0.]]], requires_grad=True)
    ids = torch.tensor([[2, 1, 0, 2], [0, 2, 1, 0]])
    mask = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 0]], dtype=torch.bool)
    sums, means, counts = per_example_nll(logits, ids, mask, chunk_size=1)
    expected0 = -logits[0, 1].log_softmax(0)[0] - logits[0, 2].log_softmax(0)[2]
    expected1 = -logits[1, 0].log_softmax(0)[2]
    torch.testing.assert_close(sums, torch.stack([expected0, expected1]))
    torch.testing.assert_close(means, torch.stack([expected0/2, expected1]))
    assert counts.tolist() == [2, 1]
    means.mean().backward()
    assert logits.grad[0, 0].abs().sum() == 0
    assert logits.grad[0, -1].abs().sum() == 0
    assert logits.grad[0, 1].abs().sum() > 0
    with pytest.raises(ValueError):
        per_example_nll(logits, ids, torch.zeros_like(mask))


def test_real_template_mask_excludes_prompt_and_control_tokens():
    cfg = load_config("configs/pilot.yaml")
    tokenizer = AutoTokenizer.from_pretrained(paths(cfg)["model"], local_files_only=True)
    response = "A repeated answer: $x=3$."
    row = tokenize_example(tokenizer, "Quote: " + response + " Now explain why.", response)
    selected = [t for t,m in zip(row["input_ids"], row["response_mask"]) if m]
    assert tokenizer.decode(selected) == response
    assert not set(selected) & set(tokenizer.all_special_ids)


def test_grouping_and_quota_capacity():
    rows = [{"id": "1", "prompt": "What is 2 + 2?"}, {"id": "2", "prompt": " WHAT is 2 + 2? "},
            {"id": "3", "prompt": "Unrelated long question about a polynomial."}]
    groups, edges = duplicate_groups(rows, 0.8)
    assert sorted(len(g) for g in groups) == [1, 2]
    assert len(edges) == 1
    q = quotas(10, {"a": 1, "b": 20}, {"a": 9, "b": 1})
    assert q == {"a": 1, "b": 9}


def test_resume_restores_optimizer_scheduler_rng_and_data_position(tmp_path):
    def setup():
        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(0.3), torch.nn.Linear(8, 2))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
        return model, optimizer, scheduler

    def step(model, optimizer, scheduler):
        optimizer.zero_grad()
        x = torch.randn(4, 4) + random.random() + float(np.random.random())
        loss = model(x).square().mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        return loss.detach()

    seed_all(123)
    model, optimizer, scheduler = setup()
    step(model, optimizer, scheduler)
    state = {"global_step": 1, "orders": [[3, 0, 1, 2]], "position": 4}
    directory = save_checkpoint(tmp_path, model, optimizer, scheduler, state, "test-fingerprint")
    loss_expected = step(model, optimizer, scheduler)
    expected = copy.deepcopy(model.state_dict())
    loaded, opt2, sched2 = setup()
    found = latest_checkpoint(tmp_path, "test-fingerprint")
    assert found == directory
    restored = load_training_checkpoint(found, loaded, opt2, sched2, "test-fingerprint")
    assert restored == state
    assert sched2.last_epoch == 1
    assert all(int(s["step"]) == 1 for s in opt2.state.values())
    loss_actual = step(loaded, opt2, sched2)
    torch.testing.assert_close(loss_actual, loss_expected, rtol=0, atol=0)
    for name, value in loaded.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="mismatch"):
        latest_checkpoint(tmp_path, "wrong")
    # Incomplete writes are ignored if a previous valid checkpoint exists.
    (tmp_path / "step-000002-incomplete").mkdir()
    assert latest_checkpoint(tmp_path, "test-fingerprint") == directory
    with (directory / "state.pt").open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(RuntimeError, match="Corrupt"):
        validate_checkpoint(directory)
