"""Measure one complete ownership-comparison path over real loopback HTTP.

Training-only performance inputs; no predictions are reused across requests.
Raw diagnostics do not qualify accuracy or calibrated deployment.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import http.client
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import sys
import threading
from time import perf_counter, sleep

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def read(path):
    try:return Path(path).read_text().strip()
    except OSError:return None


def meminfo():
    result={}
    for line in (read('/proc/meminfo') or '').splitlines():
        key,value=line.split(':',1);parts=value.split()
        result[key]=int(parts[0])*1024 if len(parts)>1 and parts[1]=='kB' else int(parts[0])
    return result


def distribution(values):
    if not values:return None
    ordered=sorted(values)
    return {'n':len(ordered),'p50_ms':ordered[math.ceil(.50*len(ordered))-1],
            'p95_ms':ordered[math.ceil(.95*len(ordered))-1],
            'p99_ms':ordered[math.ceil(.99*len(ordered))-1],'max_ms':ordered[-1]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--mode',required=True,choices=('learned','hybrid'))
    parser.add_argument('--precision',choices=('fp32','encoder-int8'),default='fp32')
    parser.add_argument('--dataset',type=Path,required=True,help='Small workload prepared by prepare_cpu_workload.py')
    parser.add_argument('--rounds',type=int,default=2)
    parser.add_argument('--raw-diagnostic',action='store_true',help='Measure the ungated full path; no decision-quality claim')
    parser.add_argument('--affinity',type=int,nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--emit-report',action='store_true')
    parser.add_argument('--stress-profiles',action='store_true',help='Separate derived length and invalid-schema diagnostics')
    parser.add_argument('--policy-scaling',action='store_true',help='Separate synthetic 1/2/5-attribute policies; raw diagnostics only')
    args=parser.parse_args()
    if args.rounds<1:parser.error('Positive round count required')
    if args.precision!='fp32' and not args.raw_diagnostic:parser.error('INT8 is an unfitted probe; do not reuse FP32 calibration')
    if args.policy_scaling and not args.raw_diagnostic:parser.error('Synthetic policy scaling cannot inherit fitted calibration')
    if platform.system()!='Linux':parser.error('This resource benchmark requires Linux')
    if args.output.exists():parser.error('Preserve previous benchmark results')
    if args.affinity:
        if not set(args.affinity)<=os.sched_getaffinity(0):parser.error('Unavailable affinity CPU')
        os.sched_setaffinity(0,set(args.affinity))
    if args.output.resolve().is_relative_to(args.bundle.resolve()):parser.error('Keep reports outside the model')
    started=perf_counter();stop=threading.Event();samples=[]
    def monitor():
        while not stop.is_set():
            info=meminfo()
            samples.append({'available':info.get('MemAvailable'),
                            'cgroup_memory':read('/sys/fs/cgroup/memory.current')})
            stop.wait(.05)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    import torch
    torch.set_num_interop_threads(1)
    from scripts.prepare_comparison_cpu_workload import load_workload,profile_config
    from prism_model.comparison_runtime import ComparisonRuntime
    from prism_model.ownership_bundle import load_model
    from prism_model.provenance import file_sha256
    from prism_model.task_server import ReviewServer
    config,selected,workload=load_workload(args.dataset)
    catalog_profile=workload.get('catalog_profile','six-category')
    config=profile_config(config,catalog_profile)
    manifest=workload['source_training_manifest']
    quantization=None
    if args.raw_diagnostic:
        model,model_config,bundle_manifest=load_model(args.bundle,threads=2)
        source_model_identity=bundle_manifest['model_identity']
        identity=source_model_identity
        if args.precision=='encoder-int8':
            from prism_model.quantized_probe import prepare
            model,quantization=prepare(model,source_model_identity)
            identity=quantization['variant_model_identity']
        runtime=ComparisonRuntime(model,model_config,identity,mode=args.mode,diagnostic=True)
        runtime.artifact_identity=file_sha256(args.bundle/'bundle-manifest.json')
    else:
        runtime=ComparisonRuntime.from_directory(args.bundle,mode=args.mode,threads=2)
        source_model_identity=runtime.model_identity
    training=json.loads((args.bundle/'training-report.json').read_text())
    if training['kind'] not in ('full_ownership_comparison_training','isolated_attribution_comparison_training') or training['dataset_manifest']!=manifest:
        raise ValueError('Performance input differs from the trained corpus')
    if manifest['partitions']['train']['cases']!=training['training']['training_cases']:
        raise ValueError('Unexpected gradient-training membership')
    load_ms=(perf_counter()-started)*1000
    loading_peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    raw_model_identity=runtime.artifact_identity
    by_item={c['request']['item_id']:c for c in selected}
    observed=[];original_forward=runtime.model.forward_batch
    def forward(rows):
        result=original_forward(rows)
        observed.extend(len(m['offsets']) for m in result['mappings'])
        return result
    runtime.model.forward_batch=forward
    class FixtureRuntime:
        def metadata(self):return runtime.metadata()
        def evaluate(self,request):
            observed.clear()
            fixture=by_item[request['item_id']]
            from scripts.comparison_stress import runtime_config
            result=runtime.evaluate(request,config=runtime_config(fixture,config),
                                    policy_id=fixture.get('policy_id'),lookup_failure=fixture.get('lookup_failure',False))
            result['benchmark_token_counts']=list(observed)
            return result
    server=ReviewServer(('127.0.0.1',0),FixtureRuntime())
    serving=threading.Thread(target=server.serve_forever,daemon=True);serving.start()
    def request(case, *, expected_http=200):
        start=perf_counter();conn=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=60)
        try:
            body=json.dumps(case['request']).encode()
            conn.request('POST','/evaluate-item',body,{'Content-Type':'application/json'})
            response=conn.getresponse();result=json.loads(response.read())
            if response.status!=expected_http:raise ValueError('HTTP benchmark request failed')
        finally:conn.close()
        latency=(perf_counter()-start)*1000
        tokens=result.get('benchmark_token_counts',[])
        from scripts.comparison_stress import semantic_digest
        return {'case_id':case['case_id'],'latency_ms':latency,
                'http_status':response.status,
                'service_ms':result.get('diagnostics',{}).get('elapsed_ms'),
                'model_questions':result.get('diagnostics',{}).get('question_count',0),'tokens_per_question':tokens,
                'maximum_question_tokens':max([*tokens,result.get('input_limit',{}).get('tokens',0)]),
                'predicted_group':result.get('group'),'identification':result.get('identification'),
                'status':result['status'],'error':result.get('error'),'semantic_digest':semantic_digest(result),
                'clarification_ids':result.get('clarification_ids',[]),
                'assessment_available':response.status==200 and result['status'] not in (None,'evaluation_unavailable'),
                'status_is_raw_diagnostic':args.raw_diagnostic}
    try:
        first=request(selected[0])
        for c in selected[1:4]:request(c)
        measured=[]
        for _ in range(args.rounds):
            for c in selected:measured.append(request(c))
        queued=[]
        with ThreadPoolExecutor(max_workers=3) as pool:
            for value in pool.map(request,selected[:12]):queued.append(value)
        stress=[]
        if args.stress_profiles:
            from scripts.comparison_stress import probes
            # Reuse the first preselected training fixture, without reading model outputs.
            fixture=selected[0]
            for probe in probes(fixture,config,runtime.model.tokenizer):
                case=probe['case'];by_item[case['request']['item_id']]=case
                row=request(case,expected_http=probe['expected_http'])
                if probe.get('must_reject_overlength') and (row['status']!='evaluation_unavailable' or row['error']!='input_too_long'):
                    raise ValueError('Overlength stress input did not fail explicitly')
                if probe.get('must_skip_neural_inference') and row['model_questions']:
                    raise ValueError('Invalid schema reached neural inference')
                if probe.get('must_request_quantity') and ('quantity' not in row['clarification_ids'] or row['status']=='ready_for_buyer_review'):
                    raise ValueError('Nonpositive quantity was not retained as a clarification')
                stress.append({**{k:v for k,v in probe.items() if k!='case'},'measurement':row})
        scaling=[]
        if args.policy_scaling:
            from scripts.comparison_stress import scaling_probes
            for probe in scaling_probes(selected[0],config):
                case=probe['case'];by_item[case['request']['item_id']]=case
                observations=[request(case) for _ in range(5)]
                expected=probe['required_attribute_checks']+(2 if args.mode=='learned' else 1)
                if any(r['predicted_group']!=probe['expected_group'] or r['model_questions']!=expected
                       or not r['assessment_available'] for r in observations):
                    raise ValueError('Synthetic scaling did not execute its complete declared policy')
                scaling.append({**{k:v for k,v in probe.items() if k!='case'},
                    'policy':case['performance_policy'],'actual_question_count':expected,
                    'http':distribution([r['latency_ms'] for r in observations]),'measurements':observations})
    finally:
        server.shutdown();server.server_close();serving.join(timeout=5);stop.set();watcher.join(timeout=5)
    groups=defaultdict(list)
    for row in measured:groups[str(row['model_questions'])].append(row['latency_ms'])
    available=[s['available'] for s in samples if s['available'] is not None]
    charged=[int(s['cgroup_memory']) for s in samples if s['cgroup_memory'] is not None]
    report={'kind':'complete_comparison_cpu_http_performance','model_artifact_identity':raw_model_identity,
            'model_identity':runtime.model_identity,'mode':args.mode,'calibration_identity':runtime.calibration_identity,
            'source_fp32_model_identity':source_model_identity,'quantization':quantization,
            'catalog_profile':catalog_profile,'supported_categories':len(config['default_policies']),
            'category_option_count':len(config['catalog']),'all_policy_attributes_preserved':True,
            'benchmark_source_sha256':file_sha256(__file__),'dataset_manifest_sha256':workload['source_training_manifest_sha256'],
            'workload_manifest_sha256':file_sha256(args.dataset/'workload-manifest.json'),
            'workload_preparer_sha256':workload['preparation_source_sha256'],'training_kind':training['kind'],
            'complete_training_corpus_loaded_in_measured_process':False,
            'training_requests_only':True,'test_set_used':False,'labels_used_for_accuracy':False,
            'benchmark_only_unfitted_gates':args.raw_diagnostic,'frozen_calibration_used':not args.raw_diagnostic,'model_or_calibration_files_modified':False,
            'host':{'platform':platform.platform(),'architecture':platform.machine(),'python':platform.python_version(),
                    'kernel':read('/proc/version'),'cpuinfo':read('/proc/cpuinfo'),
                    'cpu_affinity':sorted(os.sched_getaffinity(0)),'logical_cpus':os.cpu_count(),
                    'cpu_quota':read('/sys/fs/cgroup/cpu.max'),'memory_limit':read('/sys/fs/cgroup/memory.max'),
                    'memory_swap_limit':read('/sys/fs/cgroup/memory.swap.max'),'meminfo':meminfo(),
                    'torch':torch.__version__,'transformers':importlib.metadata.version('transformers'),
                    'tokenizers':importlib.metadata.version('tokenizers'),'threads':torch.get_num_threads(),
                    'interop_threads':torch.get_num_interop_threads(),'precision':args.precision,'gpu_available':torch.cuda.is_available()},
            'startup':{'imports_and_model_load_ms':load_ms,'first_http_request':first,
                       'disk_cache_state':'not controlled; new process, not guaranteed cold disk'},
            'memory':{'loading_peak_process_rss_bytes':loading_peak,
                      'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      'cgroup_memory_peak':read('/sys/fs/cgroup/memory.peak'),
                      'sampled_cgroup_memory_peak_bytes':max(charged) if charged else None,
                      'minimum_guest_mem_available_bytes':min(available) if available else None,
                      'swap_current':read('/sys/fs/cgroup/memory.swap.current'),
                      'memory_events':read('/sys/fs/cgroup/memory.events')},
            'sampled_families':len(selected),'rounds':args.rounds,
            'warm_http':distribution([r['latency_ms'] for r in measured if r['assessment_available']]),
            'warm_all_http_including_unavailable':distribution([r['latency_ms'] for r in measured]),
            'unavailable_assessments':sum(not r['assessment_available'] for r in measured),
            'warm_by_actual_question_count':{k:distribution(v) for k,v in groups.items()},
            'warm_up_to_384_tokens_each':distribution([r['latency_ms'] for r in measured if r['assessment_available'] and r['maximum_question_tokens']<=384]),
            'short_profile_bound_violations':sum(r['maximum_question_tokens']>384 for r in measured) if catalog_profile=='two-category' else None,
            'three_concurrent_requests':distribution([r['latency_ms'] for r in queued]),
            'measurements':measured,'queued_measurements':queued,
            'stress_profiles':stress,'stress_profiles_included_in_warm_latency':False,
            'policy_scaling_profiles':scaling,'policy_scaling_included_in_warm_latency':False,
            'policy_preservation_scope':'Original short/expanded workload; separate synthetic scaling policies are explicitly changed',
            'qualifies_model_quality':False,'total_system_budget_qualification':'requires external VM resource/no-swap verification',
            'notes':'Actual complete path over loopback HTTP. Primary training fixtures only. All original policy attributes are retained; the short profile restricts offered categories, not specification requirements. Catalog-profile changes are performance probes, not independent calibration or accuracy evidence. No prediction cache.'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    if args.emit_report:
        print(json.dumps(report,indent=2),flush=True)
    else:
        print(json.dumps({'benchmark_complete':True,'artifact_identity':raw_model_identity,'warm_http':report['warm_http'],
                          'peak_process_rss_bytes':report['memory']['peak_process_rss_bytes'],
                          'cgroup_memory_peak':report['memory']['cgroup_memory_peak'],'output':str(args.output)}),flush=True)


if __name__=='__main__':main()
