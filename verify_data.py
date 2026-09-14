import argparse
import csv
import html
import json
import math
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from annotations import numeric_sort_key, tool_id_summary

from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "image_manifest.csv"
IMAGES_DIR = ROOT / "images"
REPORT_JSON_PATH = ROOT / "verification_report.json"
REPORT_HTML_PATH = ROOT / "verification_report.html"
IMAGE_EXTENSIONS = {".avif", ".jpeg", ".jpg", ".png", ".webp"}
EXPECTED_ANGLES = set(range(0, 360, 45))
REQUIRED_COLUMNS = {
    "Tool IDs",
    "Image ID",
    "Sample ID",
    "Filepath",
    "Angle",
    "Time Taken",
}
AGENT_SCHEMA = {
    "type": "object",
    "properties": {
        "sample_reviews": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sample_id": {"type": "string"},
                    "status": {"type": "string", "enum": ["pass", "review"]},
                    "summary": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "issues": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": [
                                        "possible_wrong_sample",
                                        "possible_wrong_angle",
                                        "tool_identity_mismatch",
                                        "image_quality",
                                        "other",
                                    ],
                                },
                                "filepath": {"type": "string"},
                                "reason": {"type": "string"},
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                            },
                            "required": ["type", "filepath", "reason", "confidence"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["sample_id", "status", "summary", "confidence", "issues"],
                "additionalProperties": False,
            },
        },
        "taxonomy_suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "current_class": {"type": "string"},
                    "current_name": {"type": "string"},
                    "suggested_class": {"type": "string"},
                    "suggested_name": {"type": "string"},
                    "action": {
                        "type": "string",
                        "enum": ["keep", "rename", "merge", "split", "review"],
                    },
                    "reason": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": [
                    "current_class",
                    "current_name",
                    "suggested_class",
                    "suggested_name",
                    "action",
                    "reason",
                    "confidence",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["sample_reviews", "taxonomy_suggestions"],
    "additionalProperties": False,
}


def issue(severity: str, code: str, location: str, message: str) -> dict[str, str]:
    return {
        "severity": severity,
        "code": code,
        "location": location,
        "message": message,
    }


def normalized_label(value: str) -> str:
    return " ".join(value.casefold().split())


def load_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError("The manifest does not have a header row.")
        rows = [
            {column: row.get(column) or "" for column in reader.fieldnames}
            for row in reader
        ]
        return reader.fieldnames, rows


def resolved_manifest_image(filepath: str) -> Path | None:
    relative = PurePosixPath(filepath)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    path = (ROOT / Path(*relative.parts)).resolve()
    try:
        path.relative_to(IMAGES_DIR.resolve())
    except ValueError:
        return None
    return path


def validate_manifest(
    rows: list[dict[str, str]],
    objects: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, list[dict[str, str]]]]:
    issues: list[dict[str, str]] = []
    samples: dict[str, list[dict[str, str]]] = defaultdict(list)
    filepaths: Counter[str] = Counter()

    by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    image_ids = [row["Image ID"] for row in rows]
    if len(set(image_ids)) != len(image_ids):
        issues.append(issue("error", "duplicate_image_id", "Images", "Image IDs must be unique."))
    known_ids = set(image_ids)
    image_by_id = {row["Image ID"]: row for row in rows}
    instance_keys = set()
    for obj in objects:
        key = (obj["Image ID"], obj["Instance ID"])
        location = f"Instance {key}"
        if key in instance_keys:
            issues.append(issue("error", "duplicate_instance", location, "Instance key is repeated."))
        instance_keys.add(key)
        if obj["Image ID"] not in known_ids:
            issues.append(issue("error", "unknown_image", location, "Image ID is not in the image manifest."))
        for field in ("Image ID", "Instance ID", "Tool ID", "Tool Class", "Tool Name"):
            if not obj[field].strip():
                issues.append(issue("error", "blank_field", location, f"{field} is blank."))
        try:
            if not 0 <= float(obj["Confidence"]) <= 1:
                raise ValueError
        except ValueError:
            issues.append(issue("error", "invalid_confidence", location, "Confidence must be from 0 to 1."))
        image = image_by_id.get(obj["Image ID"])
        if image is not None:
            for field in ("Sample ID", "Filepath"):
                if obj.get(field, "") != image.get(field, ""):
                    issues.append(issue(
                        "error", "stale_instance_reference", location,
                        f"{field} differs from the image manifest. Run basic.py to refresh it.",
                    ))
        by_image[obj["Image ID"]].append(obj)
    for row in rows:
        instances = sorted(by_image[row["Image ID"]], key=lambda obj: obj["Instance ID"])
        if not instances:
            issues.append(issue("error", "missing_instances", row["Filepath"], "Image has no instance metadata."))
        if row.get("Tool IDs", "") != tool_id_summary(instances):
            issues.append(issue(
                "error", "stale_tool_summary", row["Filepath"],
                "Tool IDs differs from instance metadata. Run basic.py to refresh it.",
            ))
        # Joined strings are only for the report; CSV storage stays one row per instance.
        for field in ("Tool ID", "Tool Class", "Tool Name"):
            row[field] = "; ".join(obj[field] for obj in instances)

    for line_number, row in enumerate(rows, start=2):
        location = f"CSV row {line_number}"
        for column in REQUIRED_COLUMNS - {"Sample ID", "Angle"}:
            if not row[column].strip():
                issues.append(issue("error", "blank_field", location, f"{column} is blank."))

        filepath = row["Filepath"].strip()
        filepaths[filepath] += 1
        image_path = resolved_manifest_image(filepath)
        if image_path is None:
            issues.append(
                issue(
                    "error",
                    "invalid_filepath",
                    location,
                    f"{filepath!r} is not a safe path inside images/.",
                )
            )
        elif not image_path.is_file():
            issues.append(issue("error", "missing_image", filepath, "The image file does not exist."))

        path_parts = PurePosixPath(filepath).parts
        folder_tool_id = path_parts[1] if len(path_parts) >= 3 and path_parts[0] == "images" else None
        if (folder_tool_id is not None and folder_tool_id.isdigit()
                and len(by_image[row["Image ID"]]) == 1
                and row["Tool ID"].strip() != folder_tool_id):
            issues.append(
                issue(
                    "error",
                    "tool_folder_mismatch",
                    filepath,
                    f"Tool ID is {row['Tool ID']!r}, but the folder is {folder_tool_id!r}.",
                )
            )

        sample_id = row["Sample ID"].strip()
        angle = row["Angle"].strip()
        if bool(sample_id) != bool(angle):
            issues.append(
                issue(
                    "error",
                    "partial_sample_assignment",
                    location,
                    "Sample ID and Angle must either both be filled or both be blank.",
                )
            )
        if sample_id:
            samples[sample_id].append(row)
        else:
            issues.append(
                issue(
                    "warning",
                    "ungrouped_image",
                    filepath or location,
                    "The image has no sample ID or angle and cannot be visually checked by sample.",
                )
            )

    for filepath, count in filepaths.items():
        if filepath and count > 1:
            issues.append(
                issue("error", "duplicate_filepath", filepath, f"The filepath appears in {count} CSV rows.")
            )

    documented = {filepath for filepath in filepaths if filepath}
    if IMAGES_DIR.is_dir():
        for image_path in sorted(IMAGES_DIR.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            filepath = image_path.relative_to(ROOT).as_posix()
            if filepath not in documented:
                issues.append(
                    issue("warning", "undocumented_image", filepath, "The image is not in the manifest.")
                )

    for sample_id, sample_rows in samples.items():
        location = f"Sample {sample_id}"
        if len(sample_rows) != 8:
            issues.append(
                issue(
                    "error",
                    "sample_size",
                    location,
                    f"The sample has {len(sample_rows)} images instead of 8.",
                )
            )
        for column in ("Tool ID", "Tool Class", "Tool Name"):
            values = {row[column].strip() for row in sample_rows}
            if len(values) != 1:
                issues.append(
                    issue(
                        "error",
                        "mixed_sample_values",
                        location,
                        f"{column} has multiple values: {', '.join(sorted(values))}.",
                    )
                )
        angles: list[int] = []
        invalid_angles: list[str] = []
        for row in sample_rows:
            try:
                angles.append(int(row["Angle"].strip()))
            except ValueError:
                invalid_angles.append(row["Angle"])
        if invalid_angles or set(angles) != EXPECTED_ANGLES or len(angles) != len(set(angles)):
            shown = ", ".join(row["Angle"] or "blank" for row in sample_rows)
            issues.append(
                issue(
                    "error",
                    "invalid_angles",
                    location,
                    f"Expected each of 0, 45, ..., 315 once; found: {shown}.",
                )
            )

    tool_values: dict[str, set[tuple[str, str]]] = defaultdict(set)
    label_spellings: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for row in objects:
        tool_id = row["Tool ID"].strip()
        labels = row["Tool Class"].strip(), row["Tool Name"].strip()
        tool_values[tool_id].add(labels)
        label_spellings[tuple(map(normalized_label, labels))].add(labels)
    for tool_id, values in tool_values.items():
        if tool_id and len(values) > 1:
            shown = "; ".join(f"{tool_class} / {tool_name}" for tool_class, tool_name in sorted(values))
            issues.append(
                issue(
                    "warning",
                    "inconsistent_tool_label",
                    f"Tool {tool_id}",
                    f"The same tool ID uses multiple labels: {shown}.",
                )
            )
    for spellings in label_spellings.values():
        if len(spellings) > 1:
            shown = "; ".join(f"{tool_class} / {tool_name}" for tool_class, tool_name in sorted(spellings))
            issues.append(
                issue(
                    "warning",
                    "label_spelling",
                    "Taxonomy",
                    f"Labels differ only by capitalization or spacing: {shown}.",
                )
            )

    return issues, dict(samples)


def make_contact_sheet(sample_id: str, rows: list[dict[str, str]], destination: Path) -> None:
    columns = 4
    tile_width, image_height, label_height = 420, 420, 38
    header_height = 52
    row_count = math.ceil(len(rows) / columns)
    sheet = Image.new(
        "RGB",
        (columns * tile_width, header_height + row_count * (image_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)
    first = rows[0]
    header = (
        f"Sample {sample_id} | Tool {first['Tool ID']} | "
        f"{first['Tool Class']} / {first['Tool Name']}"
    )
    draw.text((12, 14), header, fill="black", font=font)

    def angle_key(row: dict[str, str]) -> tuple[int, str]:
        try:
            return int(row["Angle"]), row["Filepath"]
        except ValueError:
            return 999, row["Filepath"]

    for index, row in enumerate(sorted(rows, key=angle_key)):
        image_path = resolved_manifest_image(row["Filepath"])
        if image_path is None or not image_path.is_file():
            continue
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((tile_width - 16, image_height - 16))
            x = (index % columns) * tile_width + (tile_width - image.width) // 2
            y_base = header_height + (index // columns) * (image_height + label_height)
            y = y_base + (image_height - image.height) // 2
            sheet.paste(image, (x, y))
        label = f"{Path(row['Filepath']).name} | {row['Angle'] or '?'} degrees"
        draw.text(
            ((index % columns) * tile_width + 8, y_base + image_height + 7),
            label,
            fill="black",
            font=font,
        )

    sheet.save(destination, "JPEG", quality=82, optimize=True)


def taxonomy_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    counts = Counter((row["Tool Class"].strip(), row["Tool Name"].strip()) for row in rows)
    return [
        {"tool_class": tool_class, "tool_name": tool_name, "image_count": count}
        for (tool_class, tool_name), count in sorted(counts.items())
    ]


def agent_prompt(
    batch: list[tuple[str, list[dict[str, str]], Path]],
    taxonomy: list[dict[str, object]] | None,
) -> str:
    sample_details = [
        {
            "sample_id": sample_id,
            "tool_id": sample_rows[0]["Tool ID"],
            "tool_class": sample_rows[0]["Tool Class"],
            "tool_name": sample_rows[0]["Tool Name"],
            "images": [
                {"filepath": row["Filepath"], "recorded_angle": row["Angle"]}
                for row in sample_rows
            ],
        }
        for sample_id, sample_rows, _ in batch
    ]
    taxonomy_instruction = (
        "Review the complete taxonomy below for inconsistent naming, synonyms that should be "
        "merged, or labels that combine meanings and should be split. Return only actionable "
        "suggestions; do not return keep entries.\n"
        f"Taxonomy: {json.dumps(taxonomy, ensure_ascii=False)}"
        if taxonomy is not None
        else "Return an empty taxonomy_suggestions array for this batch."
    )
    return f"""You are reviewing an image dataset of tools. Each attached contact sheet is one
sample and its tiles are labeled with filepath and recorded angle. For each sample, check that:
1. Every tile shows the same physical tool; flag a tile if another meaningful property differs.
2. The tool appears to rotate in approximately 45-degree steps in the recorded order.
3. The visible object plausibly agrees with the recorded tool class and name.
4. Image quality is sufficient to identify and compare the tool.

Interpret angles as clockwise rotations from image north (the top edge): north is 0 degrees,
east is 90 degrees, south is 180 degrees, and west is 270 degrees. Use the tool's directed
centerline from its handle or base toward its working end or tip. For an opened symmetric tool,
such as a pair of scissors, use its axis of symmetry: the bisector between the working ends,
pointing from the handles through the hinge toward the blades or jaws. If a double-ended or
nearly symmetric tool has no visually distinguishable forward end, describe the angle as
ambiguous rather than guessing; a possible 180-degree ambiguity alone is not evidence that its
sample ID is wrong.

Do not treat lighting, shadows, framing, focus, or small camera movement as a different sample.
Angle judgments from appearance are approximate: report uncertainty and use
possible_wrong_angle only when the sequence looks genuinely suspicious. Never suggest edits as
facts. Review every listed sample exactly once and use only filepaths from its metadata when
identifying an image. If no problem is visible, use status pass and an empty issues array.

Sample metadata: {json.dumps(sample_details, ensure_ascii=False)}

{taxonomy_instruction}
"""


def codex_is_ready() -> tuple[bool, str]:
    if shutil.which("codex") is None:
        return False, "Codex CLI was not found. Install it, then run 'codex login'."
    result = subprocess.run(
        ["codex", "login", "status"],
        capture_output=True,
        text=True,
        check=False,
    )
    message = (result.stdout or result.stderr).strip()
    if result.returncode != 0:
        return False, message or "Codex is not signed in. Run 'codex login'."
    return True, message


def run_agent_review(
    samples: dict[str, list[dict[str, str]]],
    rows: list[dict[str, str]],
    batch_size: int,
    model: str | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str]]:
    sample_reviews: list[dict[str, object]] = []
    taxonomy_suggestions: list[dict[str, object]] = []
    errors: list[str] = []

    with tempfile.TemporaryDirectory(prefix="flowsis-review-") as temp_name:
        temp_dir = Path(temp_name)
        schema_path = temp_dir / "agent_schema.json"
        schema_path.write_text(json.dumps(AGENT_SCHEMA), encoding="utf-8")
        prepared: list[tuple[str, list[dict[str, str]], Path]] = []
        for index, (sample_id, sample_rows) in enumerate(sorted(samples.items(), key=lambda item: numeric_sort_key(item[0]))):
            sheet_path = temp_dir / f"sample-{index:04d}.jpg"
            try:
                make_contact_sheet(sample_id, sample_rows, sheet_path)
            except (OSError, UnidentifiedImageError) as error:
                errors.append(f"Could not create contact sheet for sample {sample_id}: {error}")
                continue
            prepared.append((sample_id, sample_rows, sheet_path))

        for start in range(0, len(prepared), batch_size):
            batch = prepared[start : start + batch_size]
            result_path = temp_dir / f"result-{start // batch_size:04d}.json"
            taxonomy = taxonomy_summary(rows) if start == 0 else None
            command = [
                "codex",
                "exec",
                agent_prompt(batch, taxonomy),
                "--ephemeral",
                "--sandbox",
                "read-only",
                "--output-schema",
                str(schema_path),
                "-o",
                str(result_path),
            ]
            if model:
                command.extend(["--model", model])
            for _, _, sheet_path in batch:
                command.extend(["--image", str(sheet_path)])
            sample_names = ", ".join(sample_id for sample_id, _, _ in batch)
            print(f"Reviewing sample(s) {sample_names}...")
            result = subprocess.run(command, cwd=ROOT, check=False)
            if result.returncode != 0 or not result_path.is_file():
                errors.append(f"Codex review failed for sample(s) {sample_names}.")
                continue
            try:
                batch_result = json.loads(result_path.read_text(encoding="utf-8"))
                batch_reviews = batch_result["sample_reviews"]
                expected_ids = Counter(sample_id for sample_id, _, _ in batch)
                returned_ids = Counter(review["sample_id"] for review in batch_reviews)
                if returned_ids != expected_ids:
                    errors.append(
                        f"Codex did not return exactly one review for each of "
                        f"sample(s) {sample_names}."
                    )
                sample_reviews.extend(batch_reviews)
                taxonomy_suggestions.extend(batch_result["taxonomy_suggestions"])
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                errors.append(f"Codex returned an invalid result for sample(s) {sample_names}: {error}")

    return sample_reviews, taxonomy_suggestions, errors


def render_report(report: dict[str, object]) -> str:
    static_issues = report["static_issues"]
    sample_reviews = report["sample_reviews"]
    suggestions = report["taxonomy_suggestions"]
    agent_errors = report["agent_errors"]
    assert isinstance(static_issues, list)
    assert isinstance(sample_reviews, list)
    assert isinstance(suggestions, list)
    assert isinstance(agent_errors, list)

    def table(headers: list[str], body: list[list[object]]) -> str:
        if not body:
            return "<p>None.</p>"
        head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
        rows = "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>"
            for row in body
        )
        return f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"

    issue_rows = [
        [item["severity"], item["location"], item["message"]]
        for item in static_issues
        if isinstance(item, dict)
    ]
    review_rows: list[list[object]] = []
    for review in sample_reviews:
        if not isinstance(review, dict):
            continue
        details = "; ".join(
            f"{item.get('filepath') or 'sample'}: {item.get('reason', '')}"
            for item in review.get("issues", [])
            if isinstance(item, dict)
        )
        review_rows.append(
            [review.get("sample_id", ""), review.get("status", ""), review.get("summary", ""), details]
        )
    suggestion_rows = [
        [
            f"{item.get('current_class', '')} / {item.get('current_name', '')}",
            item.get("action", ""),
            f"{item.get('suggested_class', '')} / {item.get('suggested_name', '')}",
            item.get("reason", ""),
        ]
        for item in suggestions
        if isinstance(item, dict)
    ]
    error_html = "".join(f"<li>{html.escape(str(error))}</li>" for error in agent_errors)
    if not error_html:
        error_html = "<li>None.</li>"
    counts = Counter(item["severity"] for item in static_issues if isinstance(item, dict))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>FlowSIS verification report</title>
<style>
body {{ font: 16px/1.45 system-ui, sans-serif; margin: 2rem auto; max-width: 1200px; padding: 0 1rem; color: #222; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; }}
th, td {{ border: 1px solid #ccc; padding: .55rem; text-align: left; vertical-align: top; }}
th {{ background: #eee; }} h1, h2 {{ line-height: 1.2; }} .summary {{ padding: 1rem; background: #f4f6f8; }}
</style></head><body>
<h1>FlowSIS verification report</h1>
<p>Generated {html.escape(str(report['generated_at']))}. This report contains suggestions only; it did not edit the manifest.</p>
<p class="summary">Checked {report['row_count']} manifest rows and {report['sample_count']} documented samples.
Static results: {counts['error']} error(s), {counts['warning']} warning(s). Agent-reviewed samples: {len(sample_reviews)}.</p>
<h2>CSV and filesystem checks</h2>
{table(['Severity', 'Location', 'Finding'], issue_rows)}
<h2>Visual sample review</h2>
{table(['Sample', 'Status', 'Summary', 'Details'], review_rows)}
<h2>Class and name suggestions</h2>
{table(['Current', 'Action', 'Suggested', 'Reason'], suggestion_rows)}
<h2>Agent errors or skipped steps</h2><ul>{error_html}</ul>
</body></html>"""


def write_reports(report: dict[str, object]) -> None:
    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    REPORT_HTML_PATH.write_text(render_report(report), encoding="utf-8")
    print(f"Wrote {REPORT_HTML_PATH.name} and {REPORT_JSON_PATH.name}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check image_manifest.csv and optionally ask Codex to review sample images."
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="run CSV and filesystem checks without using Codex",
    )
    parser.add_argument("--yes", action="store_true", help="start the Codex review without prompting")
    parser.add_argument("--batch-size", type=int, default=4, help="contact sheets per Codex run (default: 4)")
    parser.add_argument("--model", help="optional Codex model override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1.")
    if not MANIFEST_PATH.is_file():
        raise SystemExit(f"Could not find {MANIFEST_PATH.name}.")

    try:
        fieldnames, rows = load_manifest(MANIFEST_PATH)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Could not read {MANIFEST_PATH.name}: {error}") from error
    missing_columns = REQUIRED_COLUMNS.difference(fieldnames)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise SystemExit(f"{MANIFEST_PATH.name} is missing columns: {missing}")

    object_path = MANIFEST_PATH.with_name("image_object_manifest.csv")
    try:
        object_fields, objects = load_manifest(object_path)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Could not read {object_path.name}: {error}") from error
    missing = {"Image ID", "Instance ID", "Tool ID", "Tool Class", "Tool Name", "Confidence", "Sample ID", "Filepath"} - set(object_fields)
    if missing:
        raise SystemExit(f"{object_path.name} is missing columns: {', '.join(sorted(missing))}")
    static_issues, samples = validate_manifest(rows, objects)
    sample_reviews: list[dict[str, object]] = []
    taxonomy_suggestions: list[dict[str, object]] = []
    agent_errors: list[str] = []
    error_count = sum(item["severity"] == "error" for item in static_issues)
    warning_count = sum(item["severity"] == "warning" for item in static_issues)
    print(
        f"Static checks found {error_count} error(s) and {warning_count} warning(s) "
        f"in {len(rows)} rows."
    )

    if not args.static_only and samples:
        ready, status = codex_is_ready()
        if not ready:
            print(status)
            agent_errors.append(status)
        else:
            print(f"Codex is ready. {len(samples)} sample(s) can be visually reviewed.")
            proceed = args.yes or input("Continue with the visual review? [y/N]: ").strip().lower() in {
                "y",
                "yes",
            }
            if proceed:
                sample_reviews, taxonomy_suggestions, agent_errors = run_agent_review(
                    samples, rows, args.batch_size, args.model
                )
            else:
                agent_errors.append("Visual review was skipped by the user.")
    elif args.static_only:
        agent_errors.append("Visual review was skipped because --static-only was used.")
    else:
        agent_errors.append("No assigned samples were available for visual review.")

    report: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "manifest": MANIFEST_PATH.name,
        "row_count": len(rows),
        "sample_count": len(samples),
        "static_issues": static_issues,
        "sample_reviews": sample_reviews,
        "taxonomy_suggestions": taxonomy_suggestions,
        "agent_errors": agent_errors,
    }
    write_reports(report)


if __name__ == "__main__":
    main()
