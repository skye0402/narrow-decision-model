"""Provisional synthetic calibration; no human reliability claim is implied."""
import math

from .metrics import calibration_metrics,wilson

VERSION='task-calibration-1'


def probabilities(logits,temperature=1.):
    if not math.isfinite(temperature) or temperature<=0 or len(logits)<2 or not all(math.isfinite(x) for x in logits):
        raise ValueError('Invalid calibrated logits or temperature')
    maximum=max(logits)
    values=[math.exp((x-maximum)/temperature) for x in logits]
    total=math.fsum(values)
    return [x/total for x in values]


def apply_temperatures(records,temperatures):
    result=[]
    for record in records:
        p=probabilities(record['logits'],temperatures[record['question_id']])
        choice=max(range(len(p)),key=p.__getitem__)
        result.append({**record,'probabilities':p,'choice':record['option_ids'][choice],'confidence':p[choice]})
    return result


def fit_temperature(records):
    if not records:raise ValueError('Temperature fitting needs calibration observations')
    def nll(log_temperature):
        t=math.exp(log_temperature)
        total=0.
        for record in records:
            target=record['target'];values=record['logits']
            if type(target) is not int or not 0<=target<len(values):raise ValueError('Invalid calibration target')
            maximum=max(values)
            total+=(maximum-values[target])/t+math.log(math.fsum(math.exp((x-maximum)/t) for x in values))
        return total/len(records)
    lower,upper=math.log(.25),math.log(10.)
    grid=[lower+(upper-lower)*i/80 for i in range(81)]
    index=min(range(len(grid)),key=lambda i:nll(grid[i]))
    a,b=grid[max(0,index-1)],grid[min(len(grid)-1,index+1)]
    ratio=(math.sqrt(5)-1)/2
    for _ in range(36):
        c=b-ratio*(b-a);d=a+ratio*(b-a)
        if nll(c)<nll(d):b=d
        else:a=c
    candidates=(grid[index],(a+b)/2)
    chosen=min(candidates,key=nll)
    return min(10.,max(.25,math.exp(chosen)))


def fit_temperatures(records):
    temperatures={};diagnostics={}
    for question in sorted({r['question_id'] for r in records}):
        rows=[r for r in records if r['question_id']==question]
        temperature=fit_temperature(rows);temperatures[question]=temperature
        before=[{**r,'probabilities':probabilities(r['logits'])} for r in rows]
        after=[{**r,'probabilities':probabilities(r['logits'],temperature)} for r in rows]
        diagnostics[question]={'temperature':temperature,'before':calibration_metrics(before),'after':calibration_metrics(after)}
    return temperatures,diagnostics


def select_outcome_threshold(records,outcome,*,precision=.98,minimum_support=35,minimum_lower_bound=.90):
    rows=[r for r in records if r['primary'] and r['choice']==outcome]
    # One preselected variant per family supplies each question's interval.
    if len({r['family_id'] for r in rows})!=len(rows):raise ValueError('Threshold support repeats a scenario family')
    candidates=sorted({r['confidence'] for r in rows})
    for index,cutoff in enumerate(candidates):
        selected=[r for r in rows if r['confidence']>=cutoff]
        interval=wilson(sum(r['choice']==r['label'] for r in selected),len(selected))
        if len(selected)>=minimum_support and interval['estimate']>=precision and interval['lower']>=minimum_lower_bound:
            threshold=cutoff if index==0 else (candidates[index-1]+cutoff)/2
            return {'threshold':threshold,'support':len(selected),'precision':interval,
                'target_precision':precision,'minimum_support':minimum_support,'minimum_lower_bound':minimum_lower_bound}
    return {'threshold':None,'available_predictions':len(rows),'reason':'insufficient_support_or_precision',
        'target_precision':precision,'minimum_support':minimum_support,'minimum_lower_bound':minimum_lower_bound}


def fit_outcome_gates(records):
    thresholds={'group':{},'replacement_identification':{}};diagnostics={}
    for question in sorted({r['question_id'] for r in records}):
        rows=[r for r in records if r['question_id']==question]
        key='group' if question=='material_group' else 'replacement_identification'
        diagnostics[key]={}
        for outcome in sorted({option for r in rows for option in r['option_ids']}):
            settings={} if key=='group' else {'precision':.99,'minimum_support':50} if outcome=='satisfied' else {'precision':.95,'minimum_support':35,'minimum_lower_bound':.85}
            chosen=select_outcome_threshold(rows,outcome,**settings)
            thresholds[key][outcome]=chosen['threshold'];diagnostics[key][outcome]=chosen
    return thresholds,diagnostics
