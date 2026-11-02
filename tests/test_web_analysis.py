import numpy as np
import pandas as pd
import pytest

from data_picking.web_analysis import aggregate_documents, match_document_lengths, pair_windows, ranking_points, retrieval_metrics


def test_document_nll_is_weighted_by_actual_scored_tokens():
    windows = pd.DataFrame([{"id": "w0", "document_id": "d", "ref_sum_nll": 10., "sft_sum_nll": 8., "scored_tokens": 2},
                            {"id": "w1", "document_id": "d", "ref_sum_nll": 30., "sft_sum_nll": 16., "scored_tokens": 3}])
    manifest = pd.DataFrame([{"id": "d", "scored_tokens": 5, "token_count": 6, "url": "https://example.org/a"}])
    labels = pd.DataFrame([{"id": "d", "injected_math_source": True}])
    result = aggregate_documents(windows, manifest, labels).iloc[0]
    assert result.ref_mean_nll == 8
    assert result.sft_mean_nll == 4.8
    assert result.delta_loss == 3.2
    assert result.relative_gain == .4


def test_pairing_rejects_changed_tokenization_even_when_losses_match():
    reference = pd.DataFrame([{"id": "w", "document_id": "d", "window_index": 0, "token_start": 0,
        "token_end": 3, "tokenization_hash": "original", "scored_tokens": 2, "sum_nll": 2.}])
    sft = reference.copy();sft["tokenization_hash"] = "changed"
    with pytest.raises(AssertionError): pair_windows(reference, sft)


def test_length_control_preserves_one_to_nine_and_does_not_use_scores():
    frame = pd.DataFrame([{"id": f"{label}:{i}", "injected_math_source": label, "token_count": 10+i//factor,
                           "delta_loss": i} for label, factor in [(True, 1), (False, 9)] for i in range(12*factor)])
    cfg = {"length_quantiles": [.25, .5, .75], "matched_background_per_math": 9, "length_matching_seed": 42}
    first, info = match_document_lengths(frame, cfg)
    frame["delta_loss"] = -1000*frame["delta_loss"]
    second, _ = match_document_lengths(frame, cfg)
    assert first.id.tolist() == second.id.tolist()
    assert len(first) == 10*first.injected_math_source.sum()
    assert info["retained_fraction"] == 1
    assert all(r["retained_background"] == 9*r["retained_math"] for r in info["bins"])


def test_source_retrieval_budget_and_bootstrap_known_separation():
    y = np.array([1]*3+[0]*27)
    scores = np.arange(len(y), 0, -1)
    top = ranking_points(y, scores, np.arange(len(y)), [.1])['0.1']
    assert top == {"k": 3, "retrieved_math_source": 3, "source_precision": 1., "source_recall": 1., "source_enrichment": 10.}
    keys = ["delta_loss", "relative_gain", "neg_ref_loss", "neg_sft_loss", "pos_ref_loss", "pos_sft_loss"]
    frame = pd.DataFrame({"injected_math_source": y.astype(bool), "group_id": [str(i) for i in range(len(y))],
                          "token_count": np.arange(len(y))+10, **{k: scores.astype(float) for k in keys}})
    cfg = {"reported_loss_scores": keys[:2], "raw_loss_baselines": keys[2:], "top_fractions": [.01, .05, .1],
           "bootstrap_seed": 42, "tie_seed": 43, "bootstrap_samples": 20}
    result = retrieval_metrics(frame, cfg)
    for key in keys:
        assert result["scores"][key]["roc_auc_ci95"] == [1., 1.]
    assert result["paired_auc_differences"]["relative_gain"]["neg_ref_loss"] == {"difference": 0., "ci95": [0., 0.]}
