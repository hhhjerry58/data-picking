"""Check real command routing without starting training or changing run artifacts."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("experiment_cli", Path(__file__).resolve().parents[1] / "scripts/experiment.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


@pytest.mark.parametrize("kind,module,function", [
    ("dolly", "data_picking.generalization_data", "prepare_extension"),
    ("ultrachat", "data_picking.ultrachat_data", "prepare_ultrachat"),
])
def test_prepare_uses_the_configured_dataset(tmp_path, monkeypatch, kind, module, function):
    config = tmp_path / "experiment.yaml"
    config.write_text(json.dumps({"c_data": {"kind": kind}}))
    called = []
    monkeypatch.setattr(f"{module}.{function}", lambda path: called.append(path))
    assert cli.main(["prepare", "--config", str(config)]) == 0
    assert called == [str(config)]


def test_final_score_resolves_saved_checkpoint(tmp_path, monkeypatch):
    import data_picking.common as common
    import data_picking.scoring as scoring
    config = tmp_path / "pilot.yaml"
    config.write_text("{}")
    (tmp_path / "training_complete.json").write_text(json.dumps({"checkpoint": "saved checkpoint"}))
    cfg = {"test": "config"}
    monkeypatch.setattr(common, "load_config", lambda path: cfg)
    monkeypatch.setattr(common, "paths", lambda cfg: {"run": tmp_path})
    called = []
    monkeypatch.setattr(scoring, "score", lambda *args: called.append(args))
    assert cli.main(["score", "--config", str(config), "--checkpoint", "final", "--sanity-only"]) == 0
    assert called == [(cfg, "saved checkpoint", True)]


@pytest.mark.parametrize("extension", [False, True])
def test_export_selects_the_right_template_and_preserves_arguments(tmp_path, monkeypatch, extension):
    run = tmp_path / "run with spaces"
    (run / "metrics").mkdir(parents=True)
    (run / "metrics/metrics.json").write_text(json.dumps({"comparisons": {}} if extension else {"main": {}}))
    called = []
    monkeypatch.setattr(cli, "delegate", lambda script, args: called.append((script, args)) or 0)
    assert cli.main(["export", "--run-dir", str(run), "--overwrite"]) == 0
    expected = "export_generalization.py" if extension else "export_pilot.py"
    assert called == [(f"notebooks/{expected}", ["--run-dir", str(run), "--overwrite"])]


def test_training_rejects_an_evaluation_config(tmp_path):
    config = tmp_path / "extension.yaml"
    config.write_text('{"c_data": {}}')
    with pytest.raises(SystemExit) as exc:
        cli.main(["train", "--config", str(config)])
    assert exc.value.code == 2


def test_web_retrieval_config_is_preparation_only(tmp_path, monkeypatch):
    config = tmp_path/"web.yaml"
    config.write_text('{"dataset_kind": "web_retrieval"}')
    called = []
    monkeypatch.setattr("data_picking.web_retrieval.prepare_bundle", lambda p: called.append(p))
    assert cli.main(["prepare", "--config", str(config)]) == 0
    assert called == [str(config)]
    with pytest.raises(SystemExit) as exc:
        cli.main(["train", "--config", str(config)])
    assert exc.value.code == 2


def test_training_check_only_does_not_start_training(tmp_path, monkeypatch):
    config = tmp_path/"train.yaml"
    config.write_text('{}')
    monkeypatch.setattr("data_picking.common.load_config", lambda path: {"fixture": True})
    called = []
    monkeypatch.setattr("data_picking.training_plan.write_training_plan", lambda *args: called.append(args) or {"total_steps": 64})
    def forbidden(*args, **kwargs):
        raise AssertionError("check-only must not launch training")
    monkeypatch.setattr("data_picking.training.train", forbidden)
    assert cli.main(["train", "--config", str(config), "--check-only"]) == 0
    assert called == [({"fixture": True}, None, "")]


def test_code_provenance_includes_nested_tools_and_dependencies():
    from data_picking.common import code_state
    hashes = code_state()["file_hashes"]
    assert {"scripts/experiment.py", "scripts/notebooks/_shared.py", "scripts/workflows/run_pilot.py",
            "scripts/maintenance/stage_weights.py", "requirements/training.txt", "requirements/notebook.txt",
            "pyproject.toml"} <= set(hashes)
