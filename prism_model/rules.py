"""Small literal rules judge; fixed application behavior lives in Orchestrator."""
import re

from .orchestrator import Judgment

RULES_VERSION = "rules-2"

SMALL = dict(zip("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split(), range(20)))
TENS = dict(zip("twenty thirty forty fifty sixty seventy eighty ninety".split(), range(20, 100, 10)))
WORDS = "|".join([*SMALL, *TENS, "hundred", "point", "and"])


def normalize_numbers(text):
    def convert(match):
        words = match[1].replace("-", " ").split()
        integer, dot, decimal = " ".join(words).partition(" point ")
        value = 0
        for word in integer.split():
            if word == "hundred":
                value = max(1, value) * 100
            else:
                value += SMALL.get(word, TENS.get(word, 0))
        if dot:
            if any(word not in SMALL or SMALL[word] > 9 for word in decimal.split()):
                return match[0]
            return str(value) + "." + "".join(str(SMALL[word]) for word in decimal.split()) + " " + match[2]
        return str(value) + " " + match[2]
    return re.sub(rf"\b((?:(?:{WORDS})[ -]+)*(?:{WORDS}))\s+(kilowatts?|volts?|mm)\b", convert, text)


def scope_text(text):
    text=text.lower()
    marker=re.search(r"\b(?:the requested purchase is only|the only purchase requested is)\b",text)
    if marker:text=text[marker.end():]
    text=re.sub(r"\b(?:do not|don't)\s+(?:replace|order|supply|purchase)\s+(?:(?:the|a|any)\s+)?(?:motor|roller|gearbox|belt)s?\b", "", text)
    # A component being inspected is the subject of a service, not a goods line.
    return re.sub(r"\b(?:motor|roller)\s+(?=(?:inspection|repair|maintenance)\s+service)","",text)


def classify(text, components):
    text = scope_text(text)
    patterns={"motor":r"\bmotors?\b","roller":r"\brollers?\b","gearbox":r"\bgearbox(?:es)?\b",
        "belt":r"\bbelts?\b","sensor":r"\bsensors?\b","bearing":r"\bbearings?\b",
        "software":r"\b(?:software|licen[cs]es?)\b","consumable":r"\b(?:printer paper|office paper|paper)\b",
        "service":r"\b(?:services?|inspection|labou?r)\b"}
    kinds={kind for kind,pattern in patterns.items() if re.search(pattern,text)}
    if len(kinds) > 1:
        return "insufficient_evidence" if re.search(r"\bor\b", text) else "mixed_scope"
    if not kinds and len(components) == 1:
        kinds = {components[0]["kind"]}
    if kinds == {"motor"}:
        return "MG_MOTORS"
    if kinds == {"roller"}:
        return "MG_CONVEYOR"
    if kinds or re.search(r"\b(software|licen[cs]e|inspection|services?)\b", text):
        return "no_suitable_group"
    return "insufficient_evidence"


def read_facts(text):
    number = r"(\d+(?:\.\d+)?)"
    patterns = {
        "power_kw": [number + r"\s*(?:kw|kilowatts?)\b"],
        "voltage_v": [number + r"\s*(?:v|volts?)\b"],
        "diameter_mm": [r"diameter\s*" + number + r"\s*mm\b", number + r"\s*mm\s+(?:in\s+)?diameter\b"],
        "length_mm": [r"length\s*" + number + r"\s*mm\b", number + r"\s*mm\s+(?:in\s+length|long)\b"],
    }
    facts = {}
    for name, expressions in patterns.items():
        for expression in expressions:
            if match := re.search(expression, text):
                facts[name] = float(match[1])
                break
    if re.search(r"\b(?:three[ -]phase|3\s*ph(?:ase)?|3\s*~)", text):
        facts["phases"] = 3
    elif re.search(r"\b(?:single[ -]phase|one[ -]phase|1[ -]*(?:ph(?:ase)?))", text):
        facts["phases"] = 1
    if match := re.search(r"ø\s*" + number + r"\s*[x×]\s*" + number + r"\s*mm\b", text):
        facts.update(diameter_mm=float(match[1]), length_mm=float(match[2]))
    return facts


def identify(text, components, policy):
    text = normalize_numbers(text.lower())
    text=" ".join(sentence for sentence in re.split(r"(?<=[.!?])\s+",text)
        if not re.search(r"(?:belongs?\s+to|(?:is|are)\s+for)\s+(?:a\s+)?(?:different|another|unrelated)\s+(?:asset|component)",sentence))
    text=re.sub(r"\b(?:do not|don't)\s+(?:order|supply|purchase)\b[^.;!?]*","",text)
    negated = r"\bnot\s+(?:the\s+)?(\d+(?:\.\d+)?)\s*(?:v|volts?)\b(?:\s+one)?"
    excluded_voltages = {float(value) for value in re.findall(negated, text)}
    affirmative = re.sub(negated, "", text)
    # Explicit unresolved alternatives are not two affirmative requirements.
    if re.search(r"\b(?:either|or)\b",affirmative) and re.search(r"\d\s*(?:v|volts?|mm|kw)\b",affirmative):
        return "unclear"
    number=r"(\d+(?:\.\d+)?)"
    values=[re.findall(number+r"\s*(?:v|volts?)\b",affirmative),
            re.findall(number+r"\s*(?:kw|kilowatts?)\b",affirmative),
            re.findall(r"diameter\s*"+number+r"\s*mm\b",affirmative),
            re.findall(r"length\s*"+number+r"\s*mm\b",affirmative)]
    if any(len({float(v) for v in found})>1 for found in values):return "conflicting"
    facts = read_facts(affirmative)
    if len(components) > 1:
        components = [c for c in components if c.get("voltage_v") not in excluded_voltages
                      and all(key not in c or c[key] == value for key, value in facts.items())]
        if not components:
            return "conflicting"
        if len(components) > 1:
            return "unclear"
    if len(components) == 1:
        component = components[0]
        if component["kind"] != policy["kind"] or component.get("voltage_v") in excluded_voltages:
            return "conflicting"
        if any(key in component and component[key] != value for key, value in facts.items()):
            return "conflicting"
    if policy["kind"] == "roller" and re.search(r"\d+\s*[×x]\s*\d+\s*mm", text) and not {"diameter_mm", "length_mm"} <= facts.keys():
        return "unclear"
    explicit = all(name in facts for name in policy["required"])
    same = bool(re.search(r"\b(same|identical|like[ -]for[ -]like)\b", text))
    referenced = len(components) == 1 and same and all(name in components[0] for name in policy["required"])
    return "satisfied" if explicit or referenced else "missing"


class RulesJudge:
    probabilistic = False

    def judge_group(self, row, context):
        return Judgment(classify(context["request"]["description"], context["reference"]["components"]))

    def judge_requirement(self, row, context, policy):
        return Judgment(identify(context["request"]["description"], context["reference"]["components"], policy))
