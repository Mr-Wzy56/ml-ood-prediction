import argparse, json
from pathlib import Path
import numpy as np
import torch
from src.data import id_loaders, ood_loaders
from src.model import make_model
from src.scores import id_confidence
from src.metrics import detection_metrics, ece, risk_coverage

@torch.no_grad()
def collect(model, loader, device, method):
    model.eval(); scores=[]; probs=[]; labels=[]
    for x,y in loader:
        logits=model(x.to(device)); scores.extend(id_confidence(logits,method).cpu().numpy()); probs.extend(torch.softmax(logits,1).cpu().numpy()); labels.extend(np.asarray(y))
    return np.asarray(scores),np.asarray(probs),np.asarray(labels)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--id-dataset',choices=['cifar10','cifar100'],default=None); p.add_argument('--root',default='data'); p.add_argument('--output',default='runs/eval'); p.add_argument('--method',choices=['msp','energy'],default='msp'); p.add_argument('--coverage',type=float,default=.95); p.add_argument('--batch-size',type=int,default=128); p.add_argument('--workers',type=int,default=4); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); a=p.parse_args()
    ck=torch.load(a.checkpoint,map_location=a.device,weights_only=False); name=a.id_dataset or ck.get('id_dataset','cifar10'); model=make_model(num_classes=10 if name=='cifar10' else 100).to(a.device); model.load_state_dict(ck['model'])
    _,val,test=id_loaders(name,a.root,a.batch_size,workers=a.workers)
    vs,_,_=collect(model,val,a.device,a.method); threshold=float(np.quantile(vs,1-a.coverage))
    ids,probs,labels=collect(model,test,a.device,a.method); correct=(probs.argmax(1)==labels)
    report={'id_dataset':name,'method':a.method,'threshold_from_validation_id':threshold,'target_validation_coverage':a.coverage,'id_test_accuracy':float(correct.mean()),'id_test_ece':ece(probs,labels),'selective':{'accepted_coverage':float((ids>=threshold).mean()),'accepted_error':float(1-correct[ids>=threshold].mean()) if (ids>=threshold).any() else None},'ood':{}}
    ood_scores={}
    for ood_name,loader in ood_loaders(a.root,a.batch_size,a.workers).items():
        os_,_,_=collect(model,loader,a.device,a.method); ood_scores[ood_name]=os_; report['ood'][ood_name]=detection_metrics(ids,os_)
    rows=risk_coverage(ids,correct); Path(a.output).mkdir(parents=True,exist_ok=True); (Path(a.output)/'metrics.json').write_text(json.dumps(report,indent=2));
    np.savez_compressed(Path(a.output)/'scores.npz', val=vs, id_test=ids, **{f'ood_{k}':v for k,v in ood_scores.items()})
    with (Path(a.output)/'risk_coverage.csv').open('w') as f:
        f.write('coverage,risk\n'); f.writelines(f'{c},{r}\n' for c,r in rows)
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()

