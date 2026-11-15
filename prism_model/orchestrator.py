"""Fixed validation, evidence resolution, gating, and routing for every judge."""
from dataclasses import dataclass
from datetime import date
import math
import re
from typing import Protocol

from .renderer import group_row, requirement_row

READY = "ready_for_buyer_review"
CLARIFY = "clarification_needed"
REVIEW = "buyer_review_required"
UNAVAILABLE = "evaluation_unavailable"
FIELDS = {"item_id": str, "description": str, "quantity": int, "unit": str,
          "plant": str, "required_date": str, "reference_id": str}
NOTICE_MESSAGES = {
    "reference_not_found": "The supplied reference ID was not found. Check the ID; readiness can use complete independent specifications.",
    "reference_master_data_conflict": "The reference service reports inconsistent master data. Buyer review is required.",
}


@dataclass(frozen=True)
class Judgment:
    choice: str
    probabilities: dict[str, float] | None = None


class Judge(Protocol):
    probabilistic: bool

    def judge_group(self, row: dict, context: dict) -> Judgment: ...

    def judge_requirement(self, row: dict, context: dict, policy: dict) -> Judgment: ...


def validate_request(request):
    if not isinstance(request, dict) or set(request) - FIELDS.keys():
        raise ValueError("Request must be an object with offered public fields only")
    for key, kind in FIELDS.items():
        value = request.get(key)
        if key in ("item_id", "description") and (type(value) is not str or not value.strip()):
            raise ValueError(f"{key} must be a nonempty string")
        if value is not None and type(value) is not kind:
            raise ValueError(f"{key} has the wrong type")


def structured_issues(request, config):
    issues = []
    if request.get("quantity") is None or request["quantity"] <= 0:
        issues.append("quantity")
    if request.get("unit") != "EA":
        issues.append("unit")
    if request.get("plant") not in config["plants"]:
        issues.append("destination")
    value = request.get("required_date") or ""
    try:
        valid = bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)) and date.fromisoformat(value) >= date.fromisoformat(config["evaluation_date"])
    except ValueError:
        valid = False
    if not valid:
        issues.append("timing")
    return issues


def resolve_reference(request, config, lookup_failure=False):
    ref_id = request.get("reference_id")
    components = config["references"].get(ref_id, []) if ref_id else []
    status = "not_provided" if not ref_id else "unavailable" if lookup_failure else "found" if components else "not_found"
    return {"reference_id": ref_id, "status": status,
            "source": "mock_reference_service", "reference_version": config["contract_version"],
            "components": components if status == "found" else [],
            "integrity_issues": config.get("reference_integrity", {}).get(ref_id, []) if status == "found" else []}


def accepted(judgment, row, probabilistic, threshold):
    if not isinstance(judgment, Judgment) or judgment.choice not in row["option_ids"]:
        raise ValueError("Judge returned an invalid offered choice")
    probabilities = judgment.probabilities
    if probabilities is None:
        if probabilistic:
            raise ValueError("Probabilistic judge omitted its distribution")
        return True
    if set(probabilities) != set(row["option_ids"]):
        raise ValueError("Distribution does not cover exactly the offered choices")
    if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities.values()):
        raise ValueError("Invalid probability")
    if abs(sum(probabilities.values()) - 1) > 1e-6:
        raise ValueError("Probabilities are not normalized")
    if probabilities[judgment.choice] < max(probabilities.values()) - 1e-12:
        raise ValueError("Choice is not a highest-probability outcome")
    if threshold is None:
        return False
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid trusted threshold")
    return probabilities[judgment.choice] >= threshold


class Orchestrator:
    def __init__(self, config, judge: Judge, thresholds=None):
        self.config, self.judge, self.thresholds = config, judge, thresholds or {}

    def evaluate(self, request, *, lookup_failure=False, policy_id=None, trace=None):
        result = {"http_status": 200, "group": None, "identification": None, "status": None,
                  "clarification_ids": [], "notice_ids": [], "notices": [], "review_ids": [], "judgments": {}}
        try:
            validate_request(request)
        except ValueError as error:
            return {**result, "http_status": 422, "error": str(error)}
        result["clarification_ids"] = structured_issues(request, self.config)
        reference = resolve_reference(request, self.config, lookup_failure)
        result["reference_status"] = reference["status"]
        if reference["status"] == "unavailable":
            return {**result, "status": UNAVAILABLE}
        if reference["status"] == "not_found":
            result["notice_ids"].append("reference_not_found")
        if reference["integrity_issues"]:
            result["notice_ids"].append("reference_master_data_conflict")
        result["notices"] = [{"id": key, "message": NOTICE_MESSAGES[key]} for key in result["notice_ids"]]
        if reference["integrity_issues"]:
            return {**result, "status": REVIEW, "review_ids": ["reference_master_data"]}
        context = {"request": request, "reference": reference}
        try:
            row = group_row(context, self.config)
            if trace is not None:
                trace.append(row)
            group = self.judge.judge_group(row, context)
            is_accepted = accepted(group, row, self.judge.probabilistic,
                                   self.thresholds.get("group", {}).get(group.choice))
            result.update(group=group.choice, group_accepted=is_accepted)
            result["judgments"]["material_group"] = {"choice": group.choice, "probabilities": group.probabilities,
                "accepted": is_accepted, "input_sha256": row["text_sha256"], "renderer_version": row["renderer_version"],
                "option_ids": row["option_ids"], "source": type(self.judge).__name__}
            if not is_accepted:
                return {**result, "status": CLARIFY, "clarification_ids": result["clarification_ids"] + ["confirm_item_scope"]}
            if group.choice in ("insufficient_evidence", "mixed_scope"):
                return {**result, "status": CLARIFY, "clarification_ids": result["clarification_ids"] + ["item_scope"]}
            if group.choice == "no_suitable_group":
                return {**result, "status": REVIEW, "review_ids": ["unsupported_scope"]}
            policy_id = policy_id or self.config["default_policies"][group.choice]
            policy = self.config["policies"].get(policy_id)
            if not policy or policy["group"] != group.choice:
                return {**result, "status": UNAVAILABLE}
            context["policy_id"] = policy_id
            code_type_conflict = len(reference["components"]) == 1 and reference["components"][0]["kind"] != policy["kind"]
            if code_type_conflict:
                # Typed source facts establish this mismatch without a model judgment.
                check, is_accepted = Judgment("conflicting"), True
            else:
                row = requirement_row(context, self.config, policy)
                if trace is not None:
                    trace.append(row)
                check = self.judge.judge_requirement(row, context, policy)
                is_accepted = accepted(check, row, self.judge.probabilistic,
                                       self.thresholds.get("replacement_identification", {}).get(check.choice))
            result.update(identification=check.choice, policy_id=policy_id, identification_accepted=is_accepted)
            result["judgments"]["replacement_identification"] = {"choice": check.choice,
                "probabilities": check.probabilities, "accepted": is_accepted,
                "source": "code_type_check" if code_type_conflict else type(self.judge).__name__,
                "input_sha256": None if code_type_conflict else row["text_sha256"],
                "renderer_version": None if code_type_conflict else row["renderer_version"]}
            if not is_accepted:
                result["clarification_ids"].append("confirm_identification")
            elif check.choice == "conflicting":
                result["clarification_ids"].append("confirm_reference" if request.get("reference_id") else "confirm_identification")
            elif check.choice in ("missing", "unclear"):
                result["clarification_ids"].append("replacement_identification")
            result["status"] = CLARIFY if result["clarification_ids"] else READY
            return result
        except Exception:
            # A judge/renderer failure must preserve known corrections and never yield Ready.
            return {**result, "status": UNAVAILABLE, "error": "decision_evaluation_failed"}
