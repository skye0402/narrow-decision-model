"""Development probes crossing request and reference values, without text cues.

Each four-case family holds the request text fixed while changing one trusted
reference fact, then repeats the crossing for an alternative requested value.
The only source of a satisfied/conflicting label is the equality relation.
This is constructed diagnostic evidence, never a new blind evaluation cohort.
"""
import copy
import random

from . import broad_data
from .data_v2 import PRODUCTS
from .evidence_data import alternative_value,intended_attribute_labels,compose
from .task_data import expected_route,opaque


def family(config,group,attribute,number,seed=90511):
    if group not in PRODUCTS or attribute not in PRODUCTS[group][1]:raise ValueError('Unknown group/attribute')
    rng=random.Random(f'reference-grid:{seed}:{group}:{attribute}:{number}')
    token=opaque(f'reference-grid:{seed}:{group}:{attribute}:{number}')
    original=broad_data.state_family(config,group,'reference_conflict',number,seed,'development')[0]
    facts=copy.deepcopy(original['intended_facts']['base_specifications'])
    choices=(facts[attribute],alternative_value(attribute,facts[attribute],rng))
    primary=rng.randrange(4);output=[]
    for request_index in range(2):
        for reference_index in range(2):
            variant=request_index*2+reference_index;c=copy.deepcopy(original)
            ref={'component_id':'A1','kind':PRODUCTS[group][0],**facts,attribute:choices[reference_index]}
            state=c['intended_facts'];state.update(blueprint='reference_value_crossing',
                current_specifications={**facts,attribute:choices[request_index]},reference_components=[ref],
                selected_reference_component='A1',reference_substitution_allowed=True,
                contradictory_current_values={},ambiguous_current_values={},ignored_specifications={},unsupported_units={})
            c.update(case_id='GRID-'+token+'-'+str(variant),family_id='GRID-F-'+token,
                     variant=variant,primary=variant==primary,split='development',reference_fixture=[ref],
                     author='assistant_reference_grid_diagnostic',review_status='constructed_not_independently_reviewed',
                     writing_template_id='reference-grid-development',tags=['development_only','reference_sensitivity'])
            c['request'].update(item_id=c['case_id'],reference_id='REF-GRID-'+token,
                                quantity=2,unit='EA',plant=config['plants'][0],required_date='2026-12-15')
            # Identical RNG for every quadrant: no wording seed signals equality.
            body=broad_data.product_body(c,random.Random(f'grid-writing:{seed}:{token}'),'development')
            c['request']['description']=f"Please procure {PRODUCTS[group][2][0]} for the reference comparison rig {number}. {body}"
            attrs=intended_attribute_labels(c,config)
            assert attrs[attribute]==('satisfied' if request_index==reference_index else 'conflicting')
            c['semantic_labels'].update(group=group,attributes=attrs,identification=compose(list(attrs.values())))
            c['expected']=expected_route(c,config);output.append(c)
    assert output[0]['request']['description']==output[1]['request']['description']
    assert output[2]['request']['description']==output[3]['request']['description']
    return output
