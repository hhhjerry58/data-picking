import pytest
from transformers import AutoTokenizer

from data_picking.common import load_config, paths
from data_picking.generalization_data import sample_candidates
from data_picking.tokenization import tokenize_example
from data_picking.ultrachat_data import build_ultrachat_candidate

SPEC = {"name": "ultrachat", "revision": "fixed", "split": "test_sft", "max_length": 2048}


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(paths(load_config("configs/pilot.yaml"))["model"], local_files_only=True)


def conversation(prompt, response):
    return {"prompt_id": "source-id", "messages": [{"role": "user", "content": prompt},
                                                    {"role": "assistant", "content": response}]}


def test_first_pair_keeps_content_and_duplicate_questions(tokenizer):
    candidates = []
    raw = [conversation("What is 4 * 3?", "12"), conversation("WHAT is 4 * 3?", "Twelve."),
           conversation("Write code", "```python\nprint(1)\n```"), conversation("你好", "好")]
    raw[0]["messages"].extend([{"role": "user", "content": "later"}, {"role": "assistant", "content": "x "*4096}])
    for i, item in enumerate(raw):
        meta, row = build_ultrachat_candidate(item, i, SPEC, tokenizer)
        assert meta["exclusion_reason"] is None
        candidates.append(row)
    assert candidates[0]["response"] == "12"
    assert candidates[0]["source_message_count"] == 4
    assert candidates[0]["source_prompt_id"] == "source-id"
    assert candidates[0]["group_id"] == candidates[1]["group_id"]
    assert len(sample_candidates(candidates, 4, 42)) == 4


def test_invalid_or_overlong_first_pair_is_not_replaced_by_later_turns(tokenizer):
    overhead = tokenize_example(tokenizer, "p", "x")["sequence_tokens"]-1
    boundary = " ".join(["x"]*(2048-overhead))
    meta, row = build_ultrachat_candidate(conversation("p", boundary), 0, SPEC, tokenizer)
    assert row["sequence_tokens"] == 2048 and row["response"] == boundary
    long = conversation("p", boundary+" x")
    long["messages"].extend(conversation("short", "ok")["messages"])
    invalid = [{"messages": []}, conversation(" ", "yes"), conversation("p", " "),
               conversation("p", "<|im_end|>"), long]
    reasons = ["invalid_first_pair", "empty_prompt", "empty_response", "invalid_response_mask", "overlong"]
    for i, (item, expected) in enumerate(zip(invalid, reasons)):
        meta, row = build_ultrachat_candidate(item, i, SPEC, tokenizer)
        assert row is None and meta["exclusion_reason"] == expected
