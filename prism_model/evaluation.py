"""Development-only evaluation and reproducible source manifests."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

from .orchestrator import Orchestrator

ROOT = Path(__file__).resolve().parents[1]


def load_cases(path, config):
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate case IDs")
    for case in cases:
        if case["contract_version"] != config["contract_version"]:
            raise ValueError("Case/config contract mismatch")
        if case["split"] != "development_smoke":
            raise ValueError("This runner must not open a locked test set")
        if not any(tag.startswith("slice:") for tag in case["tags"]):
            raise ValueError("Every development case needs at least one slice tag")
        expected = case["expected"]
        if expected["group"] is not None and expected["group"] not in config["catalog"]:
            raise ValueError("Unknown expected group")
        if expected["identification"] not in (None, *config["identification_options"]):
            raise ValueError("Unknown identification label")
        for key in ("clarification_ids", "notice_ids"):
            if not isinstance(expected[key], list) or len(set(expected[key])) != len(expected[key]):
                raise ValueError("Expected corrections/notices must be unique ID lists")
    return cases


def evaluate(cases, config, judge, thresholds=None):
    engine = Orchestrator(config, judge, thresholds)
    rows = []
    for case in cases:
        predicted = engine.evaluate(case["request"], lookup_failure=case.get("lookup_failure", False), policy_id=case.get("policy_id"))
        differences = {}
        for key, value in case["expected"].items():
            equal = set(predicted[key]) == set(value) if key in ("clarification_ids", "notice_ids") else predicted[key] == value
            if not equal:
                differences[key] = {"expected": value, "predicted": predicted[key]}
        rows.append({"case_id": case["case_id"], "tags": case["tags"], "prediction": predicted, "differences": differences})
    def score(subset):
        return {"cases": len(subset), "exact_matches": sum(not row["differences"] for row in subset)}
    tags = sorted({tag for case in cases for tag in case["tags"] if tag.startswith("slice:")})
    return {"kind": "offline_development_comparison", "judge": type(judge).__name__,
            "contract_version": config["contract_version"], "renderer_version": config["renderer_version"],
            "probabilistic_judge": judge.probabilistic, "gold_labels_reviewed": False,
            "qualifies_cpu_budget": False, "overall": score(rows),
            "slices": {tag.removeprefix("slice:"): score([row for row in rows if tag in row["tags"]]) for tag in tags},
            "expected_status_counts": dict(Counter(case["expected"]["status"] or "http_422" for case in cases)),
            "results": rows}


def source_manifest(case_path, config_path):
    paths = [case_path, config_path, ROOT / "data/multi_question_probe.json", ROOT / "pyproject.toml", ROOT / "uv.lock",
             *sorted((ROOT / "scripts").glob("*.py")), *sorted((ROOT / "prism_model").glob("*.py"))]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    try:
        commit = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", *hashes], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
        matches = bool(commit) and all(hashlib.sha256(subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT, stderr=subprocess.DEVNULL)).hexdigest() == digest for name, digest in hashes.items())
    except subprocess.CalledProcessError:
        commit, matches = None, False
    return {"files_sha256": hashes, "source_commit": commit if matches else None,
            "sources_match_commit": matches}


def case_rows(case, config, judge):
    """Observed routing supplies rows; expected labels never select the policy."""
    trace = []
    result = Orchestrator(config, judge).evaluate(case["request"], lookup_failure=case.get("lookup_failure", False), policy_id=case.get("policy_id"), trace=trace)
    return {"case_id": case["case_id"], "rows": trace, "route_status": result["status"]}
