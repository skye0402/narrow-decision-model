"""Configuration-aware literal comparator for the six-category experiment."""
import re
from .orchestrator import Judgment
from .rules import normalize_numbers,read_facts,scope_text,SMALL,TENS

VERSION='rules-3'
PATTERNS={
 'MG_MOTORS':r'\bmotors?\b', 'MG_CONVEYOR':r'\b(?:rollers?|idlers?)\b',
 'MG_GEARBOXES':r'\b(?:gearbox(?:es)?|speed reducers?|gear reducers?)\b',
 'MG_BELTS':r'\bbelts?\b','MG_BEARINGS':r'\bbearings?\b',
 'MG_SENSORS':r'\b(?:sensors?|proximity switch(?:es)?)\b'}


def classify(text):
 text=text.lower()
 text=re.sub(r'\b(?:do not|don\x27t)\s+(?:order|replace|supply|buy|purchase)\b[^.!?]*[.!?]?','',text)
 text=scope_text(text)
 if re.search(r'(?:service|labor|labour)\s+only|inspection\s+service',text) and not re.search(r'\b(?:plus|together|both)\b',text):return 'no_suitable_group'
 # A roller bearing is a bearing, not a conveyor roller.
 roller_text=re.sub(r'\broller bearings?\b','bearing',text)
 groups={key for key,pattern in PATTERNS.items() if re.search(pattern,roller_text if key=='MG_CONVEYOR' else text)}
 service=bool(re.search(r'\b(?:services?|labor|labour)\b',text))
 if len(groups)>1 or (groups and service):return 'insufficient_evidence' if re.search(r'\b(?:either|or)\b',text) else 'mixed_scope'
 if groups:return next(iter(groups))
 if re.search(r'\b(?:paper|chair|pump|valve|laptop|software|licen[cs]e|services?|labor|labour|inspection)\b',text):return 'no_suitable_group'
 return 'insufficient_evidence'


def normalize(text):
 def convert(match):
  integer,_,decimal=match[1].replace('-',' ').partition(' point ');total=0;current=0
  for word in integer.split():
   if word=='hundred':current=max(1,current)*100
   elif word=='thousand':total+=max(1,current)*1000;current=0
   else:current+=SMALL.get(word,TENS.get(word,0))
  value=str(total+current)
  if decimal:value+='.'+''.join(str(SMALL[word]) for word in decimal.split())
  return value+' '+match[2]
 words='|'.join([*SMALL,*TENS,'hundred','thousand','point','and'])
 return re.sub(rf'\b((?:(?:{words})[ -]+)*(?:{words}))\s+(kw|kilowatts?|v|volts?|mm)\b',convert,text.lower())


def facts(text):
 text=normalize(text)
 text=re.sub(r'\bratio ([a-z -]+):1',lambda m:'ratio '+normalize_numbers(m[1]+' mm').removesuffix(' mm')+':1',text)
 n=r'(\d+(?:\.\d+)?)'
 patterns={
  'power_kw':n+r'\s*(?:kw|kilowatts?)\b','voltage_v':n+r'\s*(?:v|volts?)\b',
  'diameter_mm':r'(?<!outer )(?<!shaft )diameter\s*'+n+r'\s*mm\b',
  'length_mm':r'length\s*'+n+r'\s*mm\b','gear_ratio':r'ratio\s*'+n+r'\s*:1',
  'shaft_diameter_mm':r'shaft diameter\s*'+n+r'\s*mm\b','width_mm':r'width\s*'+n+r'\s*mm\b',
  'bore_mm':r'bore\s*'+n+r'\s*mm\b','outer_diameter_mm':r'outer diameter\s*'+n+r'\s*mm\b',
  'sensing_distance_mm':r'sensing distance\s*'+n+r'\s*mm\b','supply_voltage_v':n+r'\s*(?:v|volts?)\b'}
 result={key:{float(value) for value in re.findall(pattern,text)} for key,pattern in patterns.items()}
 for key,value in read_facts(text).items():result.setdefault(key,set()).add(value)
 # Include every phase mention, so contradictory affirmative phases conflict.
 if re.search(r'\b(?:three.phase|3\s*ph|3\s*~)',text):result.setdefault('phases',set()).add(3)
 if re.search(r'\b(?:single.phase|one.phase|1\s*ph)',text):result.setdefault('phases',set()).add(1)
 result['output_type']={value.upper() for value in re.findall(r'\b(pnp|npn)\b',text)}
 result['belt_profile']={value.upper() for value in re.findall(r'\bprofile\s+([a-z0-9-]+)\b',text)}
 return {key:values for key,values in result.items() if values}


def identify(text,components,policy):
 text=normalize(text)
 text=' '.join(sentence for sentence in re.split(r'(?<=[.!?])\s+',text) if not re.search(r'(?:different|another|unrelated)\s+(?:part|asset|component)',sentence))
 text=re.sub(r'\b(?:do not|don\x27t)\s+(?:order|replace|supply|buy|purchase)\b[^.!?]*[.!?]?','',text)
 excluded={float(x) for x in re.findall(r'\bnot (?:the )?(\d+(?:\.\d+)?)\s*v\b',text)}
 text=re.sub(r'\bnot (?:the )?\d+(?:\.\d+)?\s*v\b','',text)
 found=facts(text);required=policy['required']
 if any(len(found.get(key,()))>1 for key in required):return 'conflicting'
 supplied={key:next(iter(value)) for key,value in found.items() if len(value)==1}
 if len(components)>1:
  components=[c for c in components if c.get('voltage_v') not in excluded and all(key not in c or c[key]==value for key,value in supplied.items())]
  if not components:return 'conflicting'
  if len(components)>1:return 'unclear'
 if len(components)==1:
  c=components[0]
  if c['kind']!=policy['kind'] or c.get('voltage_v') in excluded or any(key in c and c[key]!=value for key,value in supplied.items()):return 'conflicting'
 if re.search(r'\b(?:either|or)\b',text) and re.search(r'\d\s*(?:v|mm|kw)\b',text):return 'unclear'
 if policy['kind']=='roller' and re.search(r'\d+\s*[x×]\s*\d+\s*mm',text) and not {'diameter_mm','length_mm'}<=supplied.keys():return 'unclear'
 same=bool(re.search(r'\b(?:same|identical|like.for.like)\b',text))
 return 'satisfied' if all(key in supplied for key in required) or (same and len(components)==1 and all(key in components[0] for key in required)) else 'missing'


class RulesV2:
 probabilistic=False
 def judge_group(self,row,context):return Judgment(classify(context['request']['description']))
 def judge_requirement(self,row,context,policy):return Judgment(identify(context['request']['description'],context['reference']['components'],policy))
