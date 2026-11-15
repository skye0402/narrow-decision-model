"""Review-improved comparator; never a fallback for neural evidence inference.

The only new behavior removes explicitly historical or unrelated clauses before
the existing literal parser. Changes use exposed review/development failures.
"""
import re

from .orchestrator import Judgment
from .rules_v3 import purchase_text
from .rules_v2 import classify,identify

VERSION='rules-5'


def relevant_text(text):
    clauses=re.split(r'(?<=[.!?])\s+|;\s*',text)
    keep=[]
    for clause in clauses:
        historical=bool(re.search(r'^\s*(?:the\s+)?old\s+(?:unit|motor|part|drawing)\s+(?:had|was|lists|listed)\b',clause,re.I))
        unrelated=bool(re.search(r'\b(?:different|another|unrelated)\s+(?:machine|sensor|part|asset|component)\b',clause,re.I))
        obsolete=bool(re.search(r'\b(?:those|its|these)\s+(?:values|specifications|facts)\s+(?:are\s+)?obsolete\b',clause,re.I))
        if not (historical or unrelated or obsolete):keep.append(clause)
    return purchase_text('. '.join(keep))


class RulesV4:
    probabilistic=False
    def judge_group(self,row,context):return Judgment(classify(relevant_text(context['request']['description'])))
    def judge_requirement(self,row,context,policy):
        return Judgment(identify(relevant_text(context['request']['description']),context['reference']['components'],policy))
