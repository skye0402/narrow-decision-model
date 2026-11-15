"""Versioned shared routing for neural evidence composition.

Validation, reference resolution and routing retain the v0.3 contract. Composite
judgments have no fabricated overall probability: each constituent is validated
and gated, and a complete set is required for accepted satisfaction.
"""
from dataclasses import dataclass

from .orchestrator import (Orchestrator as LegacyOrchestrator,Judgment,validate_request,
    structured_issues,resolve_reference,accepted,NOTICE_MESSAGES,READY,CLARIFY,REVIEW,UNAVAILABLE)
from .renderer import group_row as legacy_group_row,requirement_row as legacy_requirement_row
from .evidence_format import canonical_row
from .evidence_data import compose


class InputLimitError(ValueError):
    def __init__(self,question_id,tokens,limit):
        self.question_id,self.tokens,self.limit=question_id,tokens,limit
        super().__init__('Rendered question exceeds the model limit')


def group_row(context,config):
    return canonical_row(legacy_group_row(context,config))


def requirement_row(context,config,policy):
    return {**canonical_row(legacy_requirement_row(context,config,policy)),
            'required_attributes':list(policy['required'])}


@dataclass(frozen=True)
class EvidencePart:
    row: dict
    judgment: Judgment
    threshold: float | None


@dataclass(frozen=True)
class CompositeJudgment(Judgment):
    parts: tuple = ()


def summarize_parts(parts):
    if not parts:raise ValueError('Empty evidence composition')
    valid=[accepted(p.judgment,p.row,True,p.threshold) for p in parts]
    negatives=[p.judgment.choice for p,ok in zip(parts,valid) if ok and p.judgment.choice!='satisfied']
    if negatives:return compose(negatives),True
    if all(valid) and all(p.judgment.choice=='satisfied' for p in parts):return 'satisfied',True
    return compose([p.judgment.choice for p in parts]),False


def accepted_evidence(judgment,row,probabilistic,threshold):
    if not isinstance(judgment,CompositeJudgment):return accepted(judgment,row,probabilistic,threshold)
    if row['question_id']!='replacement_identification' or judgment.probabilities is not None:
        raise ValueError('Invalid composite judgment')
    expected={'replacement_identification',*('attribute:'+k for k in row['required_attributes'])}
    ids=[p.row['question_id'] for p in judgment.parts]
    if len(ids)!=len(expected) or set(ids)!=expected:raise ValueError('Missing or repeated required evidence question')
    if any(set(p.row['option_ids'])!=set(row['option_ids']) for p in judgment.parts):raise ValueError('Changed evidence outcomes')
    choice,is_accepted=summarize_parts(judgment.parts)
    if judgment.choice!=choice or choice not in row['option_ids']:raise ValueError('Inconsistent evidence composition')
    return is_accepted


class EvidenceOrchestrator(LegacyOrchestrator):
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
            is_accepted = accepted_evidence(group, row, self.judge.probabilistic,
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
                is_accepted = accepted_evidence(check, row, self.judge.probabilistic,
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
        except InputLimitError as error:
            return {**result, "status": UNAVAILABLE, "error": "input_too_long",
                    "input_limit": {"question_id":error.question_id,"tokens":error.tokens,"max_tokens":error.limit}}
        except Exception:
            # A judge/renderer failure must preserve known corrections and never yield Ready.
            return {**result, "status": UNAVAILABLE, "error": "decision_evaluation_failed"}
