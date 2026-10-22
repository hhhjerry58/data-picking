import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from data_picking.analysis import auc, evaluate, match_lengths, precision_at


def test_auc_ties_and_reversed_direction():
    y = np.array([0, 0, 1, 1])
    assert auc(y, np.ones(4)) == 0.5
    assert auc(y, np.array([3, 2, 1, 0])) == 0
    s = np.array([0.3, 0.5, 0.5, 0.8])
    assert auc(y, s) == roc_auc_score(y, s)
    assert precision_at(y, s, 0.25, np.arange(4)) == 1


def test_length_matching_and_paired_identical_baselines():
    rows = []
    for domain, n in [("A", 12), ("B", 8)]:
        for i in range(n):
            value = float(i)
            rows.append({"id": f"{domain}{i}", "domain": domain, "group_id": f"{domain}{i//2}",
                         "prompt_tokens": 10+i%3, "response_tokens": 20+i%4,
                         "delta_loss": value, "relative_gain": value/2,
                         "neg_ref_loss": value, "neg_sft_loss": value,
                         "pos_ref_loss": -value, "pos_sft_loss": -value})
    matched, info = match_lengths(rows, 42)
    assert len({r["id"] for r in matched}) == len(matched)
    assert sum(r["domain"]=="A" for r in matched) == sum(r["domain"]=="B" for r in matched)
    result = evaluate(rows, {"analysis": {"bootstrap_seed": 9, "bootstrap_samples": 20}})
    difference = result["paired_delta_auc_minus_baseline"]["neg_ref_loss"]
    assert difference == {"difference": 0., "ci95": [0.,0.]}
    assert 0 < info["retained_fraction"] <= 1
