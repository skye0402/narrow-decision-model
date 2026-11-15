"""CPU evidence inference with validated per-question gates and code composition."""
import json
import math
from pathlib import Path
import re
from time import perf_counter

from .evidence_data import ATTRIBUTES,attribute_row
from .evidence_format import canonical_row
from .evidence_orchestrator import (CompositeJudgment,EvidenceOrchestrator,EvidencePart,
                                    InputLimitError,summarize_parts)
from .model_v4 import load_model,inference_hashes as model_hashes
from .orchestrator import accepted,resolve_reference
from .provenance import file_sha256
from .task_runtime import CalibratedJudge,TaskRuntime


def inference_hashes():
    root=Path(__file__).resolve().parents[1]
    names=('prism_model/evidence_runtime.py','prism_model/evidence_orchestrator.py')
    return {**model_hashes(),**{n:file_sha256(root/n) for n in names}}


class EvidenceJudge(CalibratedJudge):
    def __init__(self,model,temperatures,thresholds,config,*,mode='composed',records=None):
        super().__init__(model,temperatures,records)
        if mode not in ('holistic','composed'):raise ValueError('Unknown evidence decision mode')
        self.thresholds,self.config,self.mode=thresholds,config,mode
        self.parts=()

    def judge(self,row):
        row=canonical_row(row)
        try:return super().judge(row)
        except ValueError as error:
            match=re.fullmatch(r'Input has (\d+) tokens; limit is (\d+); no truncation',str(error))
            if match:raise InputLimitError(row['question_id'],int(match[1]),int(match[2])) from error
            raise

    def judge_requirement(self,row,context,policy):
        holistic=self.judge(row)
        if self.mode=='holistic':return holistic
        rows=[row]+[attribute_row(context,self.config,policy,key) for key in policy['required']]
        parts=[]
        for index,question in enumerate(rows):
            question=canonical_row(question)
            result=holistic if index==0 else self.judge(question)
            threshold=self.thresholds.get(question['question_id'],{}).get(result.choice)
            parts.append(EvidencePart(question,result,threshold))
        self.parts=tuple(parts)
        choice,_=summarize_parts(self.parts)
        return CompositeJudgment(choice,None,self.parts)


class EvidenceRuntime(TaskRuntime):
    def __init__(self,directory,*,threads=2,require_calibration=True,decision_mode=None):
        self.directory=Path(directory)
        self.model,self.config,self.manifest=load_model(directory,threads)
        self.artifact_identity=file_sha256(self.directory/'bundle-manifest.json')
        adaptation=json.loads((self.directory/'adaptation.json').read_text())
        self.task_mode=adaptation['task_mode']
        if self.task_mode not in ('holistic','attributes'):raise ValueError('Unknown trained task mode')
        self.decision_mode=decision_mode or ('composed' if self.task_mode=='attributes' else 'holistic')
        if self.decision_mode not in ('holistic','composed') or (self.decision_mode=='composed' and self.task_mode!='attributes'):
            raise ValueError('Unsupported task/decision mode')
        questions={'material_group','replacement_identification'}
        if self.task_mode=='attributes':questions|={'attribute:'+key for key in ATTRIBUTES}
        self.calibration=None
        if 'calibration.json' in self.manifest['files_sha256']:
            c=json.loads((self.directory/'calibration.json').read_text());self.calibration=c
            if c['version']!='evidence-calibration-1' or c.get('scope')!='synthetic_prototype' or c['model_identity']!=self.manifest['model_identity']:
                raise ValueError('Wrong evidence calibration')
            if c['config_sha256']!=file_sha256(self.directory/'pilot_config.json') or c['inference_source_sha256']!=inference_hashes():
                raise ValueError('Evidence inference source/config changed')
            if c['independently_reviewed'] is not False or c['deployment_qualified'] is not False:
                raise ValueError('Unsupported reliability qualification')
            if set(c['temperatures'])!=questions or any(type(v) not in (float,int) or not math.isfinite(v) or not .25<=v<=10 for v in c['temperatures'].values()):
                raise ValueError('Invalid evidence temperatures')
            expected={'group' if q=='material_group' else q for q in questions}
            if set(c['thresholds'])!=expected:raise ValueError('Missing or extra gate question')
            for key,thresholds in c['thresholds'].items():
                choices=self.config['catalog'] if key=='group' else self.config['identification_options']
                if set(thresholds)!=set(choices) or any(v is not None and (type(v) not in (float,int) or not math.isfinite(v) or not 0<=v<=1) for v in thresholds.values()):
                    raise ValueError('Invalid evidence gates')
        elif require_calibration:raise ValueError('Fit evidence calibration before serving')
        self.temperatures=self.calibration['temperatures'] if self.calibration else {q:1. for q in questions}
        self.thresholds=self.calibration['thresholds'] if self.calibration else {}

    def metadata(self):
        value=super().metadata()
        value.update(model_name='Requisition Ready v0.4',training_cases=json.loads((self.directory/'training-report.json').read_text())['training']['training_cases'],
                     decision_mode=self.decision_mode,task_mode=self.task_mode)
        return value

    def evaluate(self,request,*,config=None,lookup_failure=False,policy_id=None,records=None):
        started=perf_counter();local=config or self.config
        judge=EvidenceJudge(self.model,self.temperatures,self.thresholds,local,mode=self.decision_mode,records=records)
        result=EvidenceOrchestrator(local,judge,self.thresholds).evaluate(request,lookup_failure=lookup_failure,policy_id=policy_id)
        if result['http_status']==200:result['resolved_reference']=resolve_reference(request,local,lookup_failure)
        for question,value in result.get('judgments',{}).items():
            key='group' if question=='material_group' else question
            value['threshold']=self.thresholds.get(key,{}).get(value['choice'])
            value['calibration_scope']='synthetic_prototype' if self.calibration else None
        if judge.parts:
            result['judgments']['replacement_identification']['threshold']=None
            result['judgments']['replacement_identification']['composition']='individually_gated_holistic_and_required_attributes'
            for p in judge.parts:
                q=p.row['question_id']
                if q=='replacement_identification':q='holistic_identification'
                result['judgments'][q]={'choice':p.judgment.choice,'probabilities':p.judgment.probabilities,
                    'accepted':accepted(p.judgment,p.row,True,p.threshold),'threshold':p.threshold,
                    'input_sha256':p.row['text_sha256'],'source':'EvidenceJudge','calibration_scope':'synthetic_prototype'}
        result['model']={'name':'Requisition Ready v0.4','artifact_identity':self.artifact_identity,'device':'cpu',
                         'scope':'synthetic_prototype','independently_reviewed':False,'decision_mode':self.decision_mode}
        result['diagnostics']={'question_count':len(judge.trace),'questions':judge.trace,'elapsed_ms':(perf_counter()-started)*1000}
        return result
