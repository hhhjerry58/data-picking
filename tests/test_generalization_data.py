import pytest
from transformers import AutoTokenizer

from data_picking.common import load_config, paths
from data_picking.generalization_data import build_candidates, duplicate_summary, overlap_report, sample_candidates
from data_picking.tokenization import tokenize_example


SPEC = {"name": "dolly", "revision": "fixed", "category": "general_qa", "max_length": 2048}


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(paths(load_config("configs/pilot.yaml"))["model"], local_files_only=True)


def item(prompt, response):
    return {"category": "general_qa", "context": "", "instruction": prompt, "response": response}


def test_content_and_duplicate_questions_are_retained(tokenizer):
    raw = [item("What is 4 * 3?", "12"), item("Write Python code", "```python\nprint(1)\n```"),
           item("你好", "你好！"), item("Is this true?", "Yes"),
           item("  WHAT is 4 * 3?  ", "Four times three equals twelve."),
           item("Explain the figure below", "See the diagram.")]
    metadata, candidates = build_candidates(raw, SPEC, tokenizer)
    assert len(candidates) == len(raw)
    assert all(r["exclusion_reason"] is None for r in metadata)
    assert candidates[0]["group_id"] == candidates[4]["group_id"]
    assert duplicate_summary(candidates)["same_question_different_answer_groups"] == 1
    selected = sample_candidates(candidates, len(candidates), 42)
    assert {r["id"] for r in selected} == {r["id"] for r in candidates}
    assert sample_candidates(candidates, 4, 42) == sample_candidates(list(reversed(candidates)), 4, 42)
    metadata[0]["selected"] = True
    matches = overlap_report(metadata, {"A1": [{"id": "train:0", "prompt": "What is 4 * 3?"}]})
    assert len(matches) == 2 and matches[0]["selected"]
    assert len(candidates) == len(raw)  # Reporting must not remove matching rows.


def test_complete_sequence_limit_and_basic_validity(tokenizer):
    overhead = tokenize_example(tokenizer, "p", "x")["sequence_tokens"] - 1
    boundary = " ".join(["x"] * (2048 - overhead))
    assert tokenize_example(tokenizer, "p", boundary)["sequence_tokens"] == 2048
    raw = [item("p", boundary), item("p", boundary + " x"), item(" ", "answer"),
           item("p", " \n"), item("p", "<|im_end|>")]
    metadata, candidates = build_candidates(raw, SPEC, tokenizer)
    assert [r["exclusion_reason"] for r in metadata] == [None, "overlong", "empty_prompt", "empty_response", "invalid_response_mask"]
    assert len(candidates) == 1
    assert candidates[0]["response"] == boundary
    assert candidates[0]["sequence_tokens"] == 2048
