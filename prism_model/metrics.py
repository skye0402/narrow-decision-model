"""Small dependency-free classification and selective-routing diagnostics."""
import math


def classification(truth, predicted, labels):
    labels=list(labels)
    if len(truth)!=len(predicted) or not labels or len(set(labels))!=len(labels):
        raise ValueError('Invalid classification observations')
    matrix={a:{b:0 for b in labels} for a in labels}
    for a,b in zip(truth,predicted):
        if a not in matrix or b not in matrix: raise ValueError('Unknown classification label')
        matrix[a][b]+=1
    per={}
    for label in labels:
        tp=matrix[label][label]; support=sum(matrix[label].values())
        called=sum(matrix[a][label] for a in labels)
        precision=tp/called if called else 0.
        recall=tp/support if support else 0.
        per[label]={'support':support,'predicted':called,'precision':precision,'recall':recall,
            'f1':2*precision*recall/(precision+recall) if precision+recall else 0.}
    return {'observations':len(truth),'accuracy':sum(a==b for a,b in zip(truth,predicted))/len(truth) if truth else None,
        'macro_f1':sum(v['f1'] for v in per.values())/len(labels),'per_label':per,'confusion_matrix':matrix}


def wilson(successes,total,z=1.959963984540054):
    if not 0<=successes<=total: raise ValueError('Invalid binomial support')
    if total==0:return {'estimate':None,'lower':None,'upper':None,'support':0}
    p=successes/total; den=1+z*z/total
    center=(p+z*z/(2*total))/den
    radius=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return {'estimate':p,'lower':max(0.,center-radius),'upper':min(1.,center+radius),'support':total}


def calibration_metrics(records,bins=10):
    if not records:return {'observations':0,'nll':None,'brier':None,'ece':None,'bins':[]}
    if bins<1: raise ValueError('At least one reliability bin is required')
    buckets=[[] for _ in range(bins)]; nll=brier=0.
    for record in records:
        p=record['probabilities']; target=record['target']
        if not p or any(not math.isfinite(x) or x<0 or x>1 for x in p) or abs(sum(p)-1)>1e-5:
            raise ValueError('Invalid calibration probabilities')
        choice=max(range(len(p)),key=p.__getitem__); confidence=p[choice]
        nll-=math.log(max(p[target],1e-15))
        brier+=sum((value-(i==target))**2 for i,value in enumerate(p))
        buckets[min(bins-1,int(confidence*bins))].append((confidence,choice==target))
    rows=[];ece=0.
    for i,items in enumerate(buckets):
        confidence=sum(v for v,_ in items)/len(items) if items else None
        accuracy=sum(ok for _,ok in items)/len(items) if items else None
        if items:ece+=len(items)/len(records)*abs(confidence-accuracy)
        rows.append({'lower':i/bins,'upper':(i+1)/bins,'count':len(items),'confidence':confidence,'accuracy':accuracy})
    return {'observations':len(records),'nll':nll/len(records),'brier':brier/len(records),'ece':ece,'bin_count':bins,'bins':rows}
