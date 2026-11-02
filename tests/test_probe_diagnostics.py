import pandas as pd
import pytest

from data_picking.probe_diagnostics import QA_METADATA, pair_qa
from data_picking.probe_diagnostic_analysis import compare_auc, fixed_subset


def test_qa_pairing_rejects_changed_masks_and_missing_examples():
    data = {k: [f"{k}:0", f"{k}:1"] for k in QA_METADATA}
    data.update(response_tokens=[2, 4], scored_tokens=[2, 4], mean_nll=[1., 2.], sum_nll=[2., 8.])
    ref = pd.DataFrame(data)
    sft = ref.copy(); sft["mean_nll"] = [.5, 3.]; sft["sum_nll"] = [1., 12.]
    paired = pair_qa(ref, sft.iloc[::-1])
    assert paired.delta_loss.tolist() == [.5, -1.]
    assert paired.relative_gain.tolist() == [.5, -.5]
    changed = sft.copy(); changed.loc[0, "tokenization_hash"] = "different mask"
    with pytest.raises(AssertionError): pair_qa(ref, changed)
    with pytest.raises(AssertionError): pair_qa(ref, sft.iloc[:1])
    with pytest.raises(ValueError): pair_qa(ref, pd.concat([sft, sft.iloc[:1]]))


def test_frozen_matching_rejects_missing_or_duplicated_ids():
    frame = pd.DataFrame({"id": ["a", "b", "c"], "score": [9., -2., 3.]})
    assert fixed_subset(frame, ["c", "a"]).id.tolist() == ["c", "a"]
    for bad in [["a", "a"], ["missing"]]:
        with pytest.raises(ValueError): fixed_subset(frame, bad)


def test_auc_change_bootstrap_is_paired_and_does_not_reverse_direction():
    frame = pd.DataFrame({"id": ["a", "b", "c", "d"], "group_id": ["a", "b", "c", "d"],
        "injected_math_source": [1, 1, 0, 0], "delta_loss": [4., 3., 2., 1.], "relative_gain": [4., 3., 2., 1.]})
    after = frame.copy()
    after[["delta_loss", "relative_gain"]] *= -1
    result = compare_auc(frame, after.iloc[::-1], ["delta_loss", "relative_gain"], 30, 42)
    assert all(v == {"auc_difference_after_minus_before": -1., "ci95": [-1., -1.]} for v in result.values())
    changed = after.copy(); changed.loc[0, "group_id"] = "changed"
    with pytest.raises(AssertionError): compare_auc(frame, changed, ["delta_loss"], 30, 42)
