"""Fresh relational families for a future training corpus, not runtime rules.

Labels come from constructed fictional facts. Four variants share a request
within each fixed-request pair; only source evidence changes. This module does
not read requests, labels or predictions from previous evaluation packages.
"""
import copy
import random

from . import broad_data
from .data_v2 import PRODUCTS
from .evidence_data import alternative_value,compose
from .task_data import expected_route,opaque


def labels(case,config):
    group=case['semantic_labels']['group'];state=case['intended_facts']
    policy=config['policies'][case.get('policy_id') or config['default_policies'][group]]
    current=state['current_specifications'];refs=state['reference_components']
    selected=next((r for r in refs if r['component_id']==state['selected_reference_component']),None)
    outcomes={}
    for key in policy['required']:
        if key in state['contradictory_current_values']:outcomes[key]='conflicting'
        elif key in state['ambiguous_current_values']:outcomes[key]='unclear'
        elif key in current:
            # Unknown source ratings do not contradict an explicit current fact.
            outcomes[key]='conflicting' if selected is not None and key in selected and current[key]!=selected[key] else 'satisfied'
        elif state['reference_substitution_allowed']:
            sources=[selected] if selected is not None else refs
            possible={r[key] for r in sources if key in r}
            outcomes[key]='missing' if not possible else 'satisfied' if len(possible)==1 else 'unclear'
        else:outcomes[key]='missing'
    overall=compose(list(outcomes.values()))
    if state['reference_substitution_allowed'] and len(refs)>1 and selected is None and overall!='conflicting':overall='unclear'
    return outcomes,overall


def family(config,group,attribute,mode,number,seed,role,*,frames=None,current_formatter=None):
    if group not in PRODUCTS or attribute not in PRODUCTS[group][1]:raise ValueError('Unknown group/attribute')
    if mode not in ('rating_crossing','selection_crossing','availability_crossing'):raise ValueError('Unknown relational family')
    if role not in broad_data.FRAMES:raise ValueError('Unknown data role')
    frames=broad_data.FRAMES if frames is None else frames
    rng=random.Random(f'reference-curriculum:{seed}:{number}:{group}:{attribute}:{mode}')
    token=opaque(f'reference-curriculum:{seed}:{number}:{group}:{attribute}:{mode}')
    original=broad_data.state_family(config,group,'reference_conflict',number,seed,role)[0]
    facts=copy.deepcopy(original['intended_facts']['base_specifications'])
    choices=(facts[attribute],alternative_value(attribute,facts[attribute],rng))
    primary=rng.randrange(4);cases=[]
    for request_variant in range(2):
        for reference_variant in range(2):
            variant=2*request_variant+reference_variant;c=copy.deepcopy(original)
            current=dict(facts);refs=[{'component_id':'A1','kind':PRODUCTS[group][0],**facts}];selected='A1'
            if mode=='rating_crossing':
                current[attribute]=choices[request_variant];refs[0][attribute]=choices[reference_variant]
            elif mode=='selection_crossing':
                current={};refs[0]['mounting_variant']='X'
                if reference_variant:refs.append({**refs[0],'component_id':'B2','mounting_variant':'Y'})
                selected='A1' if request_variant==0 or reference_variant==0 else None
            else:
                if request_variant:current.pop(attribute)
                if reference_variant:refs[0].pop(attribute)
            state=c['intended_facts'];state.update(blueprint=mode,target_attribute=attribute,
                current_specifications=current,reference_components=refs,selected_reference_component=selected,
                reference_substitution_allowed=True,contradictory_current_values={},ambiguous_current_values={},
                ignored_specifications={},unsupported_units={})
            c.update(case_id='REL-'+token+'-'+str(variant),family_id='REL-F-'+token,primary=variant==primary,
                     variant=variant,split=role,reference_fixture=copy.deepcopy(refs),
                     author='assistant_relational_curriculum',review_status='constructed_not_independently_reviewed',
                     tags=['synthetic','relational_counterfactual','slice:'+mode])
            c.pop('policy_id',None)
            c['request'].update(item_id=c['case_id'],reference_id='REF-REL-'+token,
                                quantity=2,unit='EA',plant=config['plants'][0],required_date='2026-12-15')
            writer=random.Random(f'relational-writing:{seed}:{token}')
            item=writer.choice(PRODUCTS[group][2]);frame=writer.randrange(len(frames[role]))
            if mode=='selection_crossing':
                headings={'train':'An exact replacement of the recorded component is needed.',
                          'development':'Please match the existing recorded component exactly.',
                          'calibration':'This purchase should match the installed component.',
                          'threshold_selection':'The recorded existing part must be replaced identically.',
                          'prototype_evaluation':'Supply an identical replacement to the component in the attached records.'}
                # Neutral wording without an outcome marker. One source candidate
                # resolves the implicit request; two require a stated selection.
                body=headings[role]+(' The requested component is A1.' if request_variant==0 else '')
            elif current_formatter is not None:
                heading={'train':'Replace the installed component like for like.',
                         'development':'Match the installed component without changing its ratings.',
                         'calibration':'We need the same specifications as the recorded part.',
                         'threshold_selection':'This purchase is an identical replacement of the recorded unit.',
                         'prototype_evaluation':'Get an exact equivalent of the existing recorded component.'}[role]
                body=heading+(' Requested: '+current_formatter(current,writer)+'.' if current else '')
            else:body=broad_data.product_body(c,writer,role)
            c['request']['description']=frames[role][frame].format(item=item,body=body,context=f'the repair cell {number}')
            c['writing_template_id']=f'relational:{mode}:{role}:{frame}'
            attrs,overall=labels(c,config);c['semantic_labels'].update(group=group,attributes=attrs,identification=overall)
            c['expected']=expected_route(c,config);cases.append(c)
    # These are data integrity requirements, not learned-outcome assertions.
    for left,right in ((0,1),(2,3)):
        assert {k:v for k,v in cases[left]['request'].items() if k!='item_id'}=={k:v for k,v in cases[right]['request'].items() if k!='item_id'}
    expected={'rating_crossing':('satisfied','conflicting','conflicting','satisfied'),
              'selection_crossing':('satisfied','satisfied','satisfied','unclear'),
              'availability_crossing':('satisfied','satisfied','satisfied','missing')}[mode]
    assert tuple(c['semantic_labels']['identification'] for c in cases)==expected
    return cases
