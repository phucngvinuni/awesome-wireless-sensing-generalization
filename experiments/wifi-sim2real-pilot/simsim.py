# sim->sim learnability check: train on 80% of simulated clips, test on held-out 20% (same simulator)
import sys, numpy as np, torch, torch.nn.functional as F
sys.argv=['x','--train','sim','--target','ut']
torch.set_num_threads(3)
from sim import frontend
import importlib.util
H=np.load(sys.argv[4] if False else __import__('os').environ.get('D','simdata')+'/ut_H.npy',mmap_mode='r'); y=np.load(__import__('os').environ.get('D','simdata')+'/ut_y.npy')
mode=__import__('os').environ.get('MODE','clean')
g=torch.Generator().manual_seed(0)
X=frontend(torch.from_numpy(np.ascontiguousarray(H[:])),mode,g).reshape(len(y),250,90)
X=(X-X.mean(1,keepdim=True)) if __import__('os').environ.get('NORM')=='san' else (X-X.flatten(1).mean(1).view(-1,1,1))
X=X/X.flatten(1).std(1).view(-1,1,1)
n=len(y); perm=np.random.default_rng(0).permutation(n); tr,te=perm[:int(.8*n)],perm[int(.8*n):]
exec(open('train.py').read().split('class Net')[1].join(['class Net','']) if False else '')
import torch.nn as nn
net=nn.Sequential(nn.Conv1d(90,64,7,padding=3),nn.BatchNorm1d(64),nn.ReLU(),nn.MaxPool1d(2),nn.Conv1d(64,128,5,padding=2),nn.BatchNorm1d(128),nn.ReLU(),nn.MaxPool1d(2),nn.Conv1d(128,128,3,padding=1),nn.BatchNorm1d(128),nn.ReLU(),nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(128,7))
opt=torch.optim.AdamW(net.parameters(),1e-3); Y=torch.tensor(y)
for ep in range(20):
    net.train(); p=np.random.permutation(tr)
    for i in range(0,len(p),64):
        j=p[i:i+64]; loss=F.cross_entropy(net(X[j].transpose(1,2)),Y[j]); opt.zero_grad(); loss.backward(); opt.step()
    if ep%5==4:
        net.eval()
        with torch.no_grad(): pr=net(X[te].transpose(1,2)).argmax(1); ptr=net(X[tr[:700]].transpose(1,2)).argmax(1)
        print(mode,'epoch',ep+1,'sim-train acc',(ptr==Y[tr[:700]]).float().mean().item().__round__(3),'sim-heldout acc',(pr==Y[te]).float().mean().item().__round__(3))
