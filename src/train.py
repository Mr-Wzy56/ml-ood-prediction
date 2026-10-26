import argparse, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from src.data import id_loaders, tinyimagenet_outliers
from src.model import make_model

def run_epoch(model, loader, optimizer, device, outlier_loader=None, energy_weight=0.1, in_margin=-5., out_margin=-1.):
    model.train(); total=correct=count=0; out_iter=iter(outlier_loader) if outlier_loader else None
    for x,y in loader:
        x,y=x.to(device),y.to(device); logits=model(x); loss=F.cross_entropy(logits,y)
        if out_iter:
            try: batch=next(out_iter)
            except StopIteration: out_iter=iter(outlier_loader); batch=next(out_iter)
            ox=batch[0].to(device); ol=model(ox)
            # Energy separation: push ID energy below in_margin and auxiliary outlier energy above out_margin.
            ein=-torch.logsumexp(logits,dim=1); eout=-torch.logsumexp(ol,dim=1)
            eloss=F.relu(ein-in_margin).pow(2).mean()+F.relu(out_margin-eout).pow(2).mean()
            loss=loss+energy_weight*eloss
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        total += loss.item()*len(y); correct += (logits.argmax(1)==y).sum().item(); count += len(y)
    return total/count, correct/count

@torch.no_grad()
def accuracy(model, loader, device):
    model.eval(); good=n=0
    for x,y in loader:
        x,y=x.to(device),y.to(device); good+=(model(x).argmax(1)==y).sum().item(); n+=len(y)
    return good/n

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',default='data'); p.add_argument('--id-dataset',choices=['cifar10','cifar100'],default='cifar10'); p.add_argument('--output',default='runs/cifar10-baseline'); p.add_argument('--epochs',type=int,default=20); p.add_argument('--batch-size',type=int,default=128); p.add_argument('--workers',type=int,default=4); p.add_argument('--seed',type=int,default=42); p.add_argument('--lr',type=float,default=.1); p.add_argument('--energy-aware',action='store_true'); p.add_argument('--aux-root',default='data/tiny-imagenet-200/train'); p.add_argument('--exclude-classes',default='data/tinyimagenet_exclude.txt'); p.add_argument('--init-checkpoint',default=None,help='fine-tune from these weights (energy-aware fine-tuning)'); p.add_argument('--energy-weight',type=float,default=.1); p.add_argument('--in-margin',type=float,default=-5.); p.add_argument('--out-margin',type=float,default=-1.); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); a=p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    tr,va,_=id_loaders(a.id_dataset,a.root,a.batch_size,seed=a.seed,workers=a.workers)
    oe=tinyimagenet_outliers(a.aux_root,a.batch_size,a.workers,a.exclude_classes) if a.energy_aware else None
    nclass=10 if a.id_dataset=='cifar10' else 100
    model=make_model(num_classes=nclass).to(a.device)
    if a.init_checkpoint:
        ck=torch.load(a.init_checkpoint,map_location=a.device,weights_only=False)
        model.load_state_dict(ck['model']); print(f'initialized from {a.init_checkpoint} (epoch {ck.get("epoch")}, val_acc {ck.get("val_acc")})')
    opt=torch.optim.SGD(model.parameters(),lr=a.lr,momentum=.9,weight_decay=5e-4); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs)
    best=0.; history=[]
    import time
    (out/'config.json').write_text(json.dumps(vars(a),indent=2))
    total_start=time.time()
    for ep in range(1,a.epochs+1):
        ep_start=time.time()
        loss,train_acc=run_epoch(model,tr,opt,a.device,oe,a.energy_weight,a.in_margin,a.out_margin); val_acc=accuracy(model,va,a.device); sched.step()
        row={'epoch':ep,'loss':loss,'train_acc':train_acc,'val_acc':val_acc,'epoch_seconds':round(time.time()-ep_start,1)}; history.append(row); print(json.dumps(row))
        if val_acc>best:
            best=val_acc; torch.save({'model':model.state_dict(),'epoch':ep,'val_acc':best,'seed':a.seed,'id_dataset':a.id_dataset,'energy_aware':a.energy_aware},out/'best.pt')
    (out/'history.json').write_text(json.dumps({'args':vars(a),'history':history,'best_val_acc':best,'total_seconds':round(time.time()-total_start,1)},indent=2))
    print(f'Best validation accuracy: {best:.4f}; checkpoint: {out / "best.pt"}')
if __name__=='__main__': main()

