"""Development-improved literal comparator for contextual requests."""
import re
from .rules_v2 import classify,identify
from .orchestrator import Judgment

VERSION='rules-4'


def purchase_text(text):
 text=re.sub(r'\b3[- ]phase\b','three-phase',text,flags=re.I)
 # Keep the requested clause when a line explicitly separates it from context.
 markers=r'(?:requested purchase|(?:actual )?order note|this requisition says|requisition wording|the requested order is described as follows|the buying request itself reads|parts request|the actual order note|order text|part requested)\s*[:—-]\s*'
 matches=list(re.finditer(markers,text,flags=re.I))
 if matches:text=text[matches[-1].end():]
 text=re.sub(r'^replacement needed on[^:]*:\s*','',text,flags=re.I)
 text=re.sub(r'^.*?\bthe component to buy is\s+','',text,flags=re.I|re.S)
 text=re.sub(r'\bIt is for repairing\b[^.!?]*[.!?]\s*','',text,flags=re.I)
 text=re.sub(r'\b(?:It is for repairing|This is for work on|Installation context only|Equipment mentioned only for context|For context, this supports|Background equipment, outside the requested scope)\b.*','',text,flags=re.I|re.S)
 text=re.sub(r'\bThe\s+.*?\bexplains where it will be used\b.*','',text,flags=re.I|re.S)
 text=re.sub(r'\bFor avoidance of ambiguity,.*','',text,flags=re.I|re.S)
 text=' '.join(sentence for sentence in re.split(r'(?<=[.!?])\s+',text) if not re.search(r'(?:background information|Job context, not an extra purchase|Equipment background|Site background concerns|Maintenance is working on)',sentence,re.I))
 return text


class RulesV3:
 probabilistic=False
 def judge_group(self,row,context):return Judgment(classify(purchase_text(context['request']['description'])))
 def judge_requirement(self,row,context,policy):return Judgment(identify(purchase_text(context['request']['description']),context['reference']['components'],policy))
