import pandas as pd
import pytest

from data_picking.gradient_review import align_gradient_rows


def frame():
    return pd.DataFrame([{"id":str(i), "group_id":str(i), "domain":"A", "split":"A2", "text_hash":f"text{i}",
                          "tokenization_hash":f"tokens{i}", "prompt_tokens":10, "response_tokens":20,
                          "ref_mean_nll":1.0+i, "sft_mean_nll":0.8+i} for i in range(2)])


def test_gradient_join_uses_ids_and_rejects_mismatched_tokens_or_losses():
    target = frame()
    measured = target.iloc[::-1].copy()
    aligned, errors = align_gradient_rows(target, measured, 1e-4)
    assert aligned.id.tolist() == target.id.tolist() and errors == {"ref":0.0, "sft":0.0}
    measured.loc[0,"tokenization_hash"] = "different-mask"
    with pytest.raises(AssertionError): align_gradient_rows(target, measured, 1e-4)
    measured = frame()
    measured.loc[0,"ref_mean_nll"] += 0.01
    with pytest.raises(ValueError, match="NLL mismatch"): align_gradient_rows(target, measured, 1e-4)
