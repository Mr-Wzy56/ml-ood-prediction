"""Evaluate a CIFAR-10-C corruption array after validation-only threshold calibration."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.data import id_loaders
from src.model import make_model
from src.scores import id_confidence
from src.metrics import ece

@torch.no_grad()
def scores(model, loader, device, method):
    model.eval(); ss=[]; pp=[]; yy=[]
    for x,y in loader:
        logits=model(x.to(device)); ss.extend(id_confidence(logits,method).cpu().numpy()); pp.extend(torch.softmax(logits,1).cpu().numpy()); yy.extend(np.asarray(y))
    return np.asarray(ss),np.asarray(pp),np.asarray(yy)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--npy',required=True,help='One CIFAR-10-C corruption .npy file (N,32,32,3)'); p.add_argument('--labels',default='data/CIFAR-10-C/labels.npy'); p.add_argument('--output',default='runs/cifar10c'); p.add_argument('--method',choices=['msp','energy'],default='msp'); p.add_argument('--severity',type=int,default=1,help='1-5; selects the corresponding 10000-image block'); p.add_argument('--batch-size',type=int,default=128); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); a=p.parse_args()
    ck=torch.load(a.checkpoint,map_location=a.device,weights_only=False); model=make_model().to(a.device); model.load_state_dict(ck['model'])
    _,val,_=id_loaders('cifar10',batch_size=a.batch_size)
    vs,_,_=scores(model,val,a.device,a.method); threshold=float(np.quantile(vs,.05))
    arr=np.load(a.npy); labels=np.load(a.labels); start=(a.severity-1)*10000; arr=arr[start:start+10000]; labels=labels[start:start+10000]
    if len(arr)!=len(labels) or arr.ndim!=4: raise ValueError('Expected matching CIFAR-10-C images and labels in NHWC format')
    x=torch.from_numpy(arr).permute(0,3,1,2).float()/255.; mean=torch.tensor([.4914,.4822,.4465])[None,:,None,None]; std=torch.tensor([.247,.2435,.2616])[None,:,None,None]; x=(x-mean)/std
    loader=DataLoader(TensorDataset(x,torch.from_numpy(labels).long()),batch_size=a.batch_size)
    sc,pr,y=scores(model,loader,a.device,a.method); pred=pr.argmax(1); accepted=sc>=threshold; report={'method':a.method,'threshold_validation_id_95pct':threshold,'severity':a.severity,'accuracy':float((pred==y).mean()),'ece':ece(pr,y),'accepted_coverage':float(accepted.mean()),'accepted_error':float(1-(pred[accepted]==y[accepted]).mean()) if accepted.any() else None}
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True); (out/'metrics.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()

