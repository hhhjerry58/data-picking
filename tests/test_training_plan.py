from copy import deepcopy

import pytest

from data_picking.common import file_hash, write_json, write_jsonl
from data_picking.training_plan import training_plan


def prepared_config(tmp_path):
    data = tmp_path/"prepared"
    data.mkdir()
    write_jsonl(data/"A1.jsonl", [{"id": f"train:{i}", "split": "A1"} for i in range(4096)])
    write_jsonl(data/"A2.jsonl", [{"id": f"val:{i}", "split": "A2"} for i in range(512)])
    write_jsonl(data/"manifest.jsonl", [{"frozen": True}])
    model = {"name": "test-model", "revision": "fixed"}
    write_json(data/"summary.json", {"model": model})
    write_json(data/"COMPLETED.json", {"files": {name: file_hash(data/name) for name in
        ["A1.jsonl", "A2.jsonl", "manifest.jsonl", "summary.json"]}})
    return {"artifact_root": str(tmp_path), "run_id": "test-run", "model": model,
            "data": {"prepared_dir": str(data), "a1_size": 4096},
            "training": {"save_optimizer": True, "auto_resume": True, "parameter_dtype": "float32",
                "autocast_dtype": "bfloat16", "attention_backend": "math", "epochs": 1,
                "effective_batch_size": 64, "micro_batch_size": 2, "save_steps": 16,
                "warmup_ratio": .1, "learning_rate": 2e-5, "expected_total_steps": 64}}


def test_effective_4096_schedule_and_resume_settings(tmp_path):
    cfg = prepared_config(tmp_path)
    plan = training_plan(cfg)
    assert plan["train_examples"] == 4096 and plan["total_steps"] == 64
    assert plan["gradient_accumulation_steps"] == 32
    assert plan["checkpoint_steps"] == [16, 32, 48, 64]
    assert plan["warmup_steps"] == 7
    assert plan["save_optimizer"] and plan["auto_resume"]
    assert plan["validation_only"]
    for key in ["save_optimizer", "auto_resume"]:
        invalid = deepcopy(cfg)
        invalid["training"][key] = False
        with pytest.raises(ValueError, match="optimizer state"):
            training_plan(invalid)


def test_preflight_rejects_step_mismatch_and_changed_data(tmp_path):
    cfg = prepared_config(tmp_path)
    cfg["training"]["expected_total_steps"] = 32
    with pytest.raises(ValueError, match="give 64"):
        training_plan(cfg)
    cfg["training"]["expected_total_steps"] = 64
    with (tmp_path/"prepared/A1.jsonl").open("a") as f:
        f.write('\n')
    with pytest.raises(ValueError, match="checksum mismatch"):
        training_plan(cfg)
