import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import verify_data
from annotations import write_image_manifests
from verification_log import load_review_log


@pytest.fixture
def review_data(tmp_path, monkeypatch):
    monkeypatch.setattr(verify_data, "ROOT", tmp_path)
    monkeypatch.setattr(verify_data, "IMAGES_DIR", tmp_path / "images")
    monkeypatch.setattr(verify_data, "codex_is_ready", lambda: (True, "ready"))
    samples = {}
    for sample_id, tool_id in [("7", "11"), ("9", "12")]:
        rows = []
        for angle in (0, 45):
            filepath = f"images/{tool_id}/{angle}.jpg"
            path = tmp_path / filepath
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), "white").save(path)
            rows.append({
                "Image ID": f"image-{tool_id}-{angle}",
                "Sample ID": sample_id,
                "Filepath": filepath,
                "Tool IDs": tool_id,
                "Tool ID": tool_id,
                "Tool Class": "pliers",
                "Tool Name": "lineman",
                "Angle": str(angle),
            })
        samples[sample_id] = rows
    calls = []

    def run(command, **kwargs):
        prompt = command[2]
        metadata = json.loads(prompt.split("Sample metadata: ")[1].split("\n\n")[0])
        calls.append(metadata)
        reviews = [
            {
                "sample_id": sample["sample_id"],
                "status": "review",
                "summary": "Check the angle.",
                "confidence": 0.8,
                "issues": [{
                    "type": "possible_wrong_angle",
                    "filepath": sample["images"][0]["filepath"],
                    "reason": "Angle may be ambiguous.",
                    "confidence": 0.7,
                }],
            }
            for sample in metadata
        ]
        result = {"sample_reviews": reviews, "taxonomy_suggestions": []}
        Path(command[command.index("-o") + 1]).write_text(json.dumps(result))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(verify_data.subprocess, "run", run)
    return samples, calls, tmp_path / "visual_review_log.json"


def run_review(samples, path, **kwargs):
    objects = [row for rows in samples.values() for row in rows]
    return verify_data.run_agent_review(
        samples, objects, 1, kwargs.pop("model", None),
        log_path=path, yes=True, **kwargs,
    )


def test_reuses_findings_without_starting_codex(review_data, monkeypatch):
    samples, calls, path = review_data
    reviews, _, errors = run_review(samples, path)
    assert not errors
    assert len(calls) == 2
    assert all(not review["cached"] for review in reviews)
    monkeypatch.setattr(
        verify_data, "codex_is_ready", lambda: pytest.fail("Cached run must not need Codex")
    )
    reused, _, errors = run_review(samples, path)
    assert not errors
    assert len(calls) == 2
    assert all(review["cached"] and review["status"] == "review" for review in reused)
    assert reused[0]["issues"] == reviews[0]["issues"]
    assert reused[0]["reviewed_at"] == reviews[0]["reviewed_at"]


def test_renumber_and_row_order_preserve_reviews_and_update_log(review_data):
    samples, calls, path = review_data
    run_review(samples, path)
    renamed = {}
    for index, rows in enumerate(samples.values()):
        for row in rows:
            row["Sample ID"] = str(index)
        renamed[str(index)] = list(reversed(rows))
    reviews, _, errors = run_review(renamed, path)
    assert not errors
    assert len(calls) == 2
    assert [review["sample_id"] for review in reviews] == ["0", "1"]
    assert all(review["cached"] for review in reviews)
    assert {
        entry["result"]["sample_id"]
        for entry in load_review_log(path)["samples"].values()
    } == {"0", "1"}


@pytest.mark.parametrize(
    "change", ["content", "angle", "label", "membership", "model", "policy"]
)
def test_changed_inputs_trigger_review(review_data, monkeypatch, change):
    samples, calls, path = review_data
    run_review(samples, path)
    kwargs = {}
    if change == "content":
        Image.new("RGB", (8, 8), "black").save(verify_data.ROOT / samples["7"][0]["Filepath"])
    elif change == "angle":
        samples["7"][0]["Angle"] = "90"
    elif change == "label":
        samples["7"][0]["Tool Name"] = "surgical_lineman"
    elif change == "membership":
        samples["7"].pop()
    elif change == "model":
        kwargs["model"] = "another-model"
    else:
        original = verify_data.agent_prompt
        monkeypatch.setattr(
            verify_data, "agent_prompt", lambda *args: original(*args) + "\nNew policy."
        )
    reviews, _, errors = run_review(samples, path, **kwargs)
    assert not errors
    changed = {review["sample_id"] for review in reviews if not review["cached"]}
    assert changed == ({"7", "9"} if change in {"model", "policy"} else {"7"})
    assert len(calls) > 2


def test_annotation_and_manufacturer_edits_do_not_invalidate_review(review_data):
    samples, calls, path = review_data
    run_review(samples, path)
    samples["7"][0].update({"Manufacturer": "Brand", "Reviewed": "TRUE", "Mask Path": "mask.npz"})
    reviews, _, errors = run_review(samples, path)
    assert not errors
    assert len(calls) == 2
    assert all(review["cached"] for review in reviews)


def test_taxonomy_change_does_not_repeat_visual_reviews(review_data):
    samples, calls, path = review_data
    run_review(samples, path)
    objects = [row for rows in samples.values() for row in rows]
    objects.append({"Tool Class": "scissors", "Tool Name": "iris"})
    reviews, _, errors = verify_data.run_agent_review(
        samples, objects, 1, None, log_path=path, yes=True
    )
    assert not errors
    assert all(review["cached"] for review in reviews)
    assert calls[-1] == []
    assert len(calls) == 3


def test_force_review_ignores_saved_results(review_data):
    samples, calls, path = review_data
    run_review(samples, path)
    reviews, _, errors = run_review(samples, path, force=True)
    assert not errors
    assert len(calls) == 4
    assert not any(review["cached"] for review in reviews)


@pytest.mark.parametrize(
    "failure",
    ["exit", "missing", "duplicate", "bad_status", "foreign_path", "changed_image"],
)
def test_failed_or_invalid_results_are_not_cached(review_data, monkeypatch, failure):
    samples, _, path = review_data
    original = verify_data.subprocess.run

    def run(command, **kwargs):
        result = original(command, **kwargs)
        if failure == "exit":
            return SimpleNamespace(returncode=1)
        result_path = Path(command[command.index("-o") + 1])
        data = json.loads(result_path.read_text())
        if failure == "missing":
            data["sample_reviews"] = []
        elif failure == "duplicate":
            data["sample_reviews"] *= 2
        elif failure == "bad_status":
            data["sample_reviews"][0]["status"] = "invalid"
        elif failure == "foreign_path":
            data["sample_reviews"][0]["issues"][0]["filepath"] = "images/foreign.jpg"
        elif failure == "changed_image":
            sample_id = data["sample_reviews"][0]["sample_id"]
            image = verify_data.ROOT / samples[sample_id][0]["Filepath"]
            Image.new("RGB", (8, 8), "black").save(image)
        result_path.write_text(json.dumps(data))
        return result

    monkeypatch.setattr(verify_data.subprocess, "run", run)
    reviews, _, errors = run_review(samples, path)
    assert errors
    assert not reviews
    assert not path.exists()


def test_completed_batches_survive_a_later_failure(review_data, monkeypatch):
    samples, calls, path = review_data
    original = verify_data.subprocess.run

    def run(command, **kwargs):
        if calls:
            return SimpleNamespace(returncode=1)
        return original(command, **kwargs)

    monkeypatch.setattr(verify_data.subprocess, "run", run)
    reviews, _, errors = run_review(samples, path)
    assert errors
    assert len(reviews) == 1
    monkeypatch.setattr(verify_data.subprocess, "run", original)
    reviews, _, errors = run_review(samples, path)
    assert not errors
    assert len(reviews) == 2
    assert len(calls) == 2
    assert reviews[0]["cached"]


def test_corrupt_log_stops_before_spending_credits(review_data):
    samples, calls, path = review_data
    path.write_text("not json")
    _, _, errors = run_review(samples, path)
    assert errors
    assert not calls
    assert path.read_text() == "not json"


def test_log_write_failure_stops_further_paid_calls(review_data, monkeypatch):
    samples, calls, path = review_data

    def fail_save(*args):
        raise OSError("Disk is full")

    monkeypatch.setattr(verify_data, "save_review_log", fail_save)
    reviews, _, errors = run_review(samples, path)
    assert len(calls) == 1
    assert len(reviews) == 1
    assert "Could not save" in errors[0]


def test_cli_reuses_log_and_static_only_leaves_it_untouched(review_data, monkeypatch):
    samples, calls, path = review_data
    rows = [dict(row) for sample_rows in samples.values() for row in sample_rows]
    objects = [
        {
            **row,
            "Instance ID": "instance-0",
            "Confidence": "0.8",
        }
        for row in rows
    ]
    for row in rows:
        row["Time Taken"] = "2026-10-23T10:00:00-04:00"
    manifest = path.with_name("image_manifest.csv")
    write_image_manifests(
        manifest, rows, path.with_name("image_object_manifest.csv"), objects
    )
    report_path = path.with_name("report.json")
    monkeypatch.setattr(verify_data, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(verify_data, "REPORT_JSON_PATH", report_path)
    monkeypatch.setattr(verify_data, "REPORT_HTML_PATH", path.with_name("report.html"))
    argv = ["verify_data.py", "--yes", "--review-log", str(path)]
    monkeypatch.setattr(sys, "argv", argv)
    verify_data.main()
    assert len(calls) == 1
    verify_data.main()
    assert len(calls) == 1
    report = json.loads(report_path.read_text())
    assert all(review["cached"] for review in report["sample_reviews"])
    before = path.read_bytes()
    monkeypatch.setattr(sys, "argv", [*argv, "--static-only"])
    verify_data.main()
    assert path.read_bytes() == before
    assert len(calls) == 1


def test_surgical_context_and_per_image_labels_are_in_prompt(review_data):
    samples, _, _ = review_data
    prompt = verify_data.agent_prompt([("7", samples["7"], Path("sheet.jpg"))], None)
    assert "surgical instruments" in prompt
    assert "surgical\nlineman (linemen) pliers" in prompt
    metadata = json.loads(prompt.split("Sample metadata: ")[1].split("\n\n")[0])
    assert metadata[0]["images"][0]["tool_name"] == "lineman"
