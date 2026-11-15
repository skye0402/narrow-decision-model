"""Versioned model text. Offered IDs stay in metadata, not answer descriptions."""
import hashlib
import json

RENDERER_VERSION = "decision-row-1"


def compact(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def render_row(context, config, question_id, instructions, options, policy=None):
    if config["renderer_version"] != RENDERER_VERSION:
        raise ValueError("Unsupported renderer version")
    evidence = {key: value for key, value in context["request"].items() if key != "item_id"}
    evidence["resolved_reference"] = context["reference"]
    policy_text = None if policy is None else {
        "kind": policy["kind"], "required_attributes": policy["required"],
        "evidence_rules": config["identification_rules"],
    }
    text = (f"FORMAT {RENDERER_VERSION}\nEVIDENCE {compact(evidence)}\n"
            f"POLICY {compact(policy_text)}\nQUESTION {compact(instructions)}\nOPTIONS\n")
    spans = []
    for index, (option_id, description) in enumerate(options.items()):
        prefix, rendered = f"{index} ", compact(description)
        start = len(text) + len(prefix)
        spans.append({"option_id": option_id, "start": start, "end": start + len(rendered)})
        text += prefix + rendered + "\n"
    return {"renderer_version": RENDERER_VERSION, "contract_version": config["contract_version"],
            "catalog_version": config["catalog_version"], "question_id": question_id,
            "option_ids": list(options), "option_spans": spans, "text": text,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest()}


def group_row(context, config):
    return render_row(context, config, "material_group",
                      "Classify the requested purchase using the offered catalog. Ignore negated items and unrelated background. Distinguish unresolved alternatives from jointly requested scope.",
                      config["catalog"])


def requirement_row(context, config, policy, question=None):
    question = question or {"id": "replacement_identification",
                           "instructions": "Assess replacement identification under the supplied policy.",
                           "options": list(config["identification_options"])}
    options = {key: config["identification_options"].get(key, key) for key in question["options"]}
    row = render_row(context, config, question["id"], question["instructions"], options, policy)
    row["policy_id"] = context["policy_id"]
    return row


def consistency_issues(answers):
    """Cross-question diagnostics, independent of any particular judge."""
    if answers.get("replacement_identification") != "satisfied":
        return []
    attributes = ("power_evidence", "voltage_evidence", "phase_evidence")
    return [f"satisfied_identification_requires_{key}" for key in attributes
            if answers.get(key) != "satisfied"]
