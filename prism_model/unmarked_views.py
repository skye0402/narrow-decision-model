"""Training-only plain-fact views with exact writer-recorded source intervals.

Current facts follow the purchase/identity context directly, before any
noncurrent clauses. Reference names vary without changing reference facts.
This optional generator is not part of the running model or its inference.
"""
import copy
import hashlib
import random

from .attribution_views import CLAUSES, UNRESOLVED_REFERENCE, marked_fact
from .data_v2 import PRODUCTS
from .evidence_span_data import VERSION as ANNOTATION_VERSION, Writer

VERSION='unmarked-attribution-view-1'
LIKE=('Keep the specifications of the unit in service. ',
      'A like-for-like part is required. ',
      'The replacement must have identical specifications. ',
      'Order the same specification as the existing unit. ',
      "Match the original unit's specification. ",
      "Retain the installed component's ratings. ",
      'An identical replacement is wanted. ',
      'Copy the recorded specifications for the replacement. ')


def view(case, *, epoch, seed=83):
    if case['split']!='train':raise ValueError('Plain-fact views are training-only')
    if case.get('author') not in ('assistant_relational_curriculum','assistant_facts_first_broadening',
                                  'assistant_writer_recorded_spans'):
        return None
    state=case.get('intended_facts',{});group=case['semantic_labels']['group']
    if group not in PRODUCTS or 'current_specifications' not in state or 'evidence_annotations' in case:
        return None
    rng=random.Random(f'unmarked-view:{seed}:{epoch}:{case["family_id"]}')
    result=copy.deepcopy(case);state=result['intended_facts'];refs=result['reference_fixture']
    style=rng.randrange(4);serial=rng.randrange(1000,9999)
    aliases={}
    for index,component in enumerate(refs):
        old=component['component_id']
        name=(old if style==0 else f'CV{serial%90+10}-C{index+1}' if style==1 else
              f'AS-{serial}-P{index+1}' if style==2 else f'R{serial:04d}.{index+1}')
        aliases[old]=name;component['component_id']=name
    selected=state['selected_reference_component']
    state['selected_reference_component']=aliases[selected] if selected is not None else None
    state['reference_components']=copy.deepcopy(refs)
    selected=state['selected_reference_component'];permitted=state['reference_substitution_allowed']
    writer=Writer();item=rng.choice(PRODUCTS[group][2])
    writer.append(rng.choice(('Please supply {item}. ','Replacement {item} required. ',
                              'Purchase {item} for maintenance. ','Order a replacement {item}. ')).format(item=item))
    if rng.random()<.6:
        code=rng.choice(('MX-','TR-','Z',''))+str(rng.randrange(100,9999))
        writer.append(rng.choice(('Type ','Product code ','Model identifier ')))
        writer.marked(code,attribute=None,role='type',value=code);writer.append('. ')
    neutral=False
    if refs:
        if permitted:
            writer.append(rng.choice(LIKE))
            if selected is not None and len(refs)>1:
                writer.append(rng.choice(('Select component {id}. ','The requested component is {id}. ',
                                          'Component {id} is the one to replace. ')).format(id=selected))
            elif selected is None and len(refs)>1 and rng.random()<.5:
                writer.append(rng.choice(UNRESOLVED_REFERENCE).format(
                    candidates=' or '.join(c['component_id'] for c in refs)))
        elif selected is not None and len(refs)>1:
            writer.append('The record identifies '+selected+'; do not copy omitted ratings from it. ')
        elif rng.random()<.5:
            # A supplied record alone is not positive permission to fill ratings.
            neutral=True
        else:
            writer.append(rng.choice(('The attached record is background only, for a different item. ',
                                      'Do not fill missing ratings from the linked record. ')))
    current=state['current_specifications'];conflicts=state.get('contradictory_current_values',{})
    ambiguous=state.get('ambiguous_current_values',{})
    facts=[(key,value) for key,value in current.items() if key not in conflicts and key not in ambiguous]
    facts.extend((key,value) for key,values in conflicts.items() for value in values)
    rng.shuffle(facts);orders=[]
    for i,(key,value) in enumerate(facts):
        if i:writer.append(rng.choice(('; ', ', ')))
        start=len(writer.text);marked_fact(writer,key,value,'current',rng,version=2)
        orders.append({'attribute':key,'value_first':writer.spans[-1]['start']==start})
    if facts:writer.append('. ')
    for key,value in state.get('unsupported_units',{}).items():
        if key!='power_hp':raise ValueError('Unknown unsupported training unit')
        writer.append('Power ');writer.marked(f'{value:g} hp',attribute='power_kw',role='current',value=value)
        writer.append('. ')
    for key,values in ambiguous.items():
        writer.append(rng.choice(CLAUSES['ambiguous']))
        for i,value in enumerate(values):
            if i:writer.append(' or ')
            marked_fact(writer,key,value,'ambiguous',rng,version=2)
        writer.append('. ')
    ignored=state.get('ignored_specifications',{});chunks=[]
    if ignored:
        role='obsolete' if state['blueprint'].startswith('obsolete') else 'background'
        chunks.append((role,list(ignored.items())))
    available=state.get('base_specifications',current)
    if available and rng.random()<.75:
        key=rng.choice(list(available));chunks.append((rng.choice(('background','obsolete','negated')),[(key,available[key])]))
    rng.shuffle(chunks)
    for role,values in chunks:
        writer.append(rng.choice(CLAUSES[role]));rng.shuffle(values)
        for i,(key,value) in enumerate(values):
            if i:writer.append('; ')
            marked_fact(writer,key,value,role,rng,version=2)
        writer.append('. ')
    writer.append('Job location: line '+str(rng.randrange(1,9999))+'.')
    result['case_id']=case['case_id']+'-UV'+str(epoch);result['request']['item_id']=result['case_id']
    result['request']['description']=writer.text;result['primary']=False
    result['evidence_annotations']={'version':ANNOTATION_VERSION,
        'description_sha256':hashlib.sha256(writer.text.encode()).hexdigest(),'spans':writer.spans}
    result['training_view']={'version':VERSION,'epoch':epoch,'parent_case_id':case['case_id'],
        'counts_as_independent_family':False,'current_facts_have_role_prefix':False,
        'current_fact_orders':orders,'neutral_reference_without_permission':neutral,
        'reference_id_aliases':aliases,'reference_id_style':style}
    return result
