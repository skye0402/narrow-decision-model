"""Run public illustrative requests through the real frozen CPU model."""
from pathlib import Path
import sys,json,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prism_model.demo_server import DemoRuntime,EXAMPLES
runtime=DemoRuntime.from_directory()
reports=[]
for e in EXAMPLES:
 request={'item_id':e['id'],'description':e['description'],'quantity':2,'unit':'EA','plant':'1100','required_date':'2026-12-15','reference_id':e['reference_id'] or None}
 result=runtime.evaluate(request)
 reports.append({'id':e['id'],'status':result.get('status'),'http_status':result.get('http_status')})
assert reports[0]['status']=='clarification_needed'
assert reports[1]['status']=='ready_for_buyer_review'
print(json.dumps(reports,indent=2))
