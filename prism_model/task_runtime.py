"""CPU inference and the shared application path for a frozen task bundle."""
import json
import math
from pathlib import Path
from time import perf_counter

import torch

from .calibration import VERSION,probabilities
from .neural import NeuralJudge
from .orchestrator import Judgment,Orchestrator,resolve_reference
from .provenance import file_sha256
from .task_model import load_task_model

ROOT=Path(__file__).resolve().parents[1]
INFERENCE_FILES=('prism_model/neural.py','prism_model/renderer.py','prism_model/orchestrator.py',
    'prism_model/task_model.py','prism_model/calibration.py','prism_model/task_runtime.py')


def inference_hashes():
    return {name:file_sha256(ROOT/name) for name in INFERENCE_FILES}


class CalibratedJudge:
    probabilistic=True
    def __init__(self,model,temperatures,records=None):
        self.model,self.temperatures=model,temperatures
        self.records=records
        self.trace=[]

    def judge(self,row):
        values=None
        if self.records is not None:
            entry=self.records.get(row['text_sha256'])
            if entry is not None:
                if entry['option_ids']!=row['option_ids']:raise ValueError('Cached option identity mismatch')
                values=entry['logits']
        if values is None:
            if self.model is None:raise ValueError('Prediction cache lacks the actually routed question')
            with torch.inference_mode():values=self.model(row).cpu().tolist()
            if self.records is not None:
                self.records[row['text_sha256']]={'logits':values,'option_ids':row['option_ids'],'question_id':row['question_id']}
        temperature=self.temperatures[row['question_id']]
        p=probabilities(values,temperature);index=max(range(len(p)),key=p.__getitem__)
        choice=row['option_ids'][index]
        self.trace.append({'question_id':row['question_id'],'input_sha256':row['text_sha256'],'logits':values,
            'option_ids':row['option_ids'],'probabilities':dict(zip(row['option_ids'],p)),'temperature':temperature})
        return Judgment(choice,dict(zip(row['option_ids'],p)))

    def judge_group(self,row,context):return self.judge(row)
    def judge_requirement(self,row,context,policy):return self.judge(row)


def validate_calibration(calibration,bundle,model_dir,config):
    if calibration.get('version')!=VERSION or calibration.get('scope')!='synthetic_prototype':
        raise ValueError('Unsupported calibration artifact')
    if calibration['model_identity']!=bundle['model_identity']:
        raise ValueError('Calibrator belongs to a different model')
    if calibration['config_sha256']!=file_sha256(Path(model_dir)/'pilot_config.json'):
        raise ValueError('Calibration configuration changed')
    if calibration['inference_source_sha256']!=inference_hashes():raise ValueError('Inference implementation changed since calibration')
    questions={'material_group','replacement_identification'}
    if set(calibration['temperatures'])!=questions:raise ValueError('Missing question calibration')
    if any(type(t) not in (int,float) or not math.isfinite(t) or not .25<=t<=10 for t in calibration['temperatures'].values()):
        raise ValueError('Invalid fitted temperature')
    for key,options in (('group',config['catalog']),('replacement_identification',config['identification_options'])):
        gates=calibration['thresholds'][key]
        if set(gates)!=set(options):raise ValueError('Calibrated option set changed')
        if any(t is not None and (type(t) not in (int,float) or not math.isfinite(t) or not 0<=t<=1) for t in gates.values()):
            raise ValueError('Invalid outcome gate')
    if calibration.get('independently_reviewed') is not False or calibration.get('deployment_qualified') is not False:
        raise ValueError('Prototype contains unsupported qualification claims')


class TaskRuntime:
    def __init__(self,model_dir,*,threads=2,require_calibration=True):
        self.directory=Path(model_dir)
        self.model,self.config,self.manifest=load_task_model(self.directory,threads=threads)
        self.calibration=None
        if 'calibration.json' in self.manifest['files_sha256']:
            self.calibration=json.loads((self.directory/'calibration.json').read_text())
            validate_calibration(self.calibration,self.manifest,self.directory,self.config)
        elif require_calibration:raise ValueError('Fit and freeze calibration before serving this model')
        self.temperatures=self.calibration['temperatures'] if self.calibration else {'material_group':1.,'replacement_identification':1.}
        self.thresholds=self.calibration['thresholds'] if self.calibration else {}
        self.artifact_identity=file_sha256(self.directory/'bundle-manifest.json')

    def metadata(self):
        return {'model_name':'Requisition Ready v0.1','artifact_identity':self.artifact_identity,
            'model_identity':self.manifest['model_identity'],'scope':'synthetic_prototype','device':'cpu',
            'training_cases':2000,'independently_reviewed':False,'deployment_qualified':False,
            'calibration_scope':'synthetic_prototype' if self.calibration else None,
            'catalog':self.config['catalog'],'references':self.config['references'],'plants':self.config['plants'],
            'evaluation_date':self.config['evaluation_date'],'max_tokens_per_question':self.model.max_tokens}

    def evaluate(self,request,*,config=None,lookup_failure=False,policy_id=None):
        started=perf_counter();local=config or self.config
        judge=CalibratedJudge(self.model,self.temperatures)
        result=Orchestrator(local,judge,self.thresholds).evaluate(request,lookup_failure=lookup_failure,policy_id=policy_id)
        if result['http_status']==200:
            result['resolved_reference']=resolve_reference(request,local,lookup_failure)
        for question,value in result.get('judgments',{}).items():
            key='group' if question=='material_group' else question
            value['threshold']=self.thresholds.get(key,{}).get(value['choice'])
            value['calibration_scope']='synthetic_prototype' if self.calibration else None
        result['model']={'name':'Requisition Ready v0.1','artifact_identity':self.artifact_identity,
            'scope':'synthetic_prototype','device':'cpu','independently_reviewed':False}
        result['diagnostics']={'question_count':len(judge.trace),'questions':judge.trace,
            'elapsed_ms':(perf_counter()-started)*1000}
        return result
