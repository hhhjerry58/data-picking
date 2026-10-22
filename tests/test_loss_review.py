import pandas as pd

from data_picking.loss_review import match_a2_dolly_lengths


def test_pair_matching_ignores_other_domains_even_when_lengths_change():
    rows = [{"id": f"{d}{i}", "domain": d, "prompt_tokens": 30+i, "response_tokens": 60+i}
            for d in ["A", "B", "C"] for i in range(12)]
    frame = pd.DataFrame(rows)
    matched, summary = match_a2_dolly_lengths(frame, 42)
    frame.loc[frame.domain == "B", ["prompt_tokens", "response_tokens"]] = 100000
    changed, second_summary = match_a2_dolly_lengths(frame, 42)
    pair, pair_summary = match_a2_dolly_lengths(frame.loc[frame.domain != "B"], 42)
    pd.testing.assert_frame_equal(matched, changed)
    pd.testing.assert_frame_equal(matched, pair)
    assert summary == second_summary == pair_summary
    assert matched.groupby("domain").size().to_dict() == {"A": 12, "C": 12}
    assert matched["id"].is_unique and summary["retained_fraction"] == 1
