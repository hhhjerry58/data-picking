import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_picking.document_utils import TargetOverlap, document_windows, grouped_duplicates
from data_picking.web_retrieval import selected_rows
from data_picking.web_retrieval import prepare_targets
from data_picking.web_retrieval import split_target_pools


@pytest.mark.parametrize("count", [2, 4, 5, 7, 11])
def test_windows_score_each_noninitial_token_once_including_tail(count):
    ids = list(range(count))
    windows = list(document_windows("document", ids, 4))
    scored = [token for row in windows for token, mask in zip(row["input_ids"], row["response_mask"]) if mask]
    assert scored == ids[1:]
    assert sum(r["scored_tokens"] for r in windows) == count-1
    assert all(len(r["input_ids"]) <= 4 and r["response_mask"][0] == 0 for r in windows)
    for left, right in zip(windows, windows[1:]): assert left["input_ids"][-1] == right["input_ids"][0]


def test_duplicates_group_web_rows_without_removing_them():
    content = " ".join(f"word{i}" for i in range(100))
    rows = [{"id": "a", "url": "https://example.com/a", "text": content},
            {"id": "b", "url": "http://example.com/a#section", "text": "different extraction"},
            {"id": "c", "url": "https://other.example/x", "text": content.replace("word50", "changed")}]
    groups, edges = grouped_duplicates(rows, "text", .8, 42, "url")
    assert len(rows) == 3 and len(groups) == 1
    assert len({r["group_id"] for r in rows}) == 1
    assert {e["kind"] for e in edges} >= {"same_url", "near_text"}


def test_target_containment_matches_passages_but_ignores_short_generic_answers():
    prompt = " ".join(f"item{i}" for i in range(100))
    guard = TargetOverlap([{"id": "target", "prompt": prompt, "response": "1"}],
                          {"problem_min_tokens": 12, "solution_min_tokens": 40,
                           "containment_threshold": .8, "anchors_per_text": 8})
    assert guard.matches("Introduction " + prompt + " other text")[0]["exact_normalized_substring"]
    near = guard.matches(prompt.replace("item50", "replacement"))
    assert near and near[0]["shingle_containment"] >= .8
    assert not guard.matches("The result is 1, an ordinary number in an unrelated page.")
    assert guard.skipped_short["response"] == 1


def test_parquet_row_selection_keeps_absolute_source_row_identity(tmp_path):
    path = tmp_path/"source.parquet"
    pq.write_table(pa.table({"id": list(range(10)), "text": [f"row {i}" for i in range(10)]}), path, row_group_size=3)
    result = selected_rows(path, [8, 0, 4])
    assert {i: row["id"] for i, row in result.items()} == {0: 0, 4: 4, 8: 8}


def test_target_split_reserves_legacy_evaluation_and_its_duplicate_group(tmp_path, monkeypatch):
    import json
    rows = [{"problem": " ".join(f"topic{i}word{j}" for j in range(20)), "solution": "answer",
             "level": "Level 1", "type": "Algebra"} for i in range(8)]
    rows.append({**rows[0], "solution": "another answer"})
    raw = tmp_path/"raw.parquet"
    pq.write_table(pa.Table.from_pylist(rows), raw)
    reserved = tmp_path/"reserved.jsonl"
    reserved.write_text(json.dumps({"source_subset": "algebra", "source_row": 0})+'\n')
    monkeypatch.setattr("data_picking.web_retrieval.raw_path", lambda *args: raw)
    monkeypatch.setattr("data_picking.web_retrieval.tokenize_example", lambda *args: {
        "input_ids": [1, 2], "response_mask": [0, 1], "sequence_tokens": 2, "response_tokens": 1})
    cfg = {"artifact_root": str(tmp_path), "target": {"name": "math", "revision": "fixed",
        "files": [{"path": "algebra/train.parquet"}], "reserved_evaluation_files": [str(reserved)],
        "train_size": 3, "validation_size": 2, "seed": 42, "max_length": 2048},
        "web": {"near_duplicate_threshold": .8}, "statistics": {"minhash_seed": 42}}
    train, validation, summary = prepare_targets(cfg, tmp_path, None)
    assert len(train) == 3 and len(validation) == 2
    assert not {r["source_row"] for r in train+validation} & {0, 8}
    assert not {r["group_id"] for r in train} & {r["group_id"] for r in validation}
    assert summary["status_counts"]["reserved_legacy_evaluation_group"] == 2


def test_target_expansion_preserves_validation_and_includes_old_training(tmp_path):
    from data_picking.common import write_jsonl
    pools = {subject: [{"id": f"{subject}{i}", "group_id": f"{subject}{i}", "text_hash": f"hash{subject}{i}",
                        "input_ids": [i, 9], "response_mask": [0, 1]} for i in range(6)] for subject in ["algebra", "geometry"]}
    old_train = [pools[s][0] for s in pools]
    old_valid = [pools[s][1] for s in pools]
    write_jsonl(tmp_path/"train.jsonl", old_train)
    write_jsonl(tmp_path/"validation.jsonl", old_valid)
    spec = {"seed": 42, "train_size": 8, "validation_size": 2, "preserve_splits_from": str(tmp_path)}
    train, valid = split_target_pools(pools, spec)
    assert len(train) == 8 and valid == old_valid
    assert {r["id"] for r in old_train} <= {r["id"] for r in train}
    assert not {r["group_id"] for r in train} & {r["group_id"] for r in valid}
    pools["algebra"][0]["input_ids"] = [99, 99]
    with pytest.raises(ValueError, match="tokenization changed"):
        split_target_pools(pools, spec)
