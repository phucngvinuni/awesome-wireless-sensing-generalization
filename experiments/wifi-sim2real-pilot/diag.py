import numpy as np, torch, json
from sim import frontend
torch.set_num_threads(2)
H=np.load(__import__('os').environ.get('D','simdata')+'/ut_H.npy',mmap_mode='r'); y=np.load(__import__('os').environ.get('D','simdata')+'/ut_y.npy')
meta=json.load(open('simdata/ut_meta.json'))
CL=["liedown","fall","walk","pickup","run","sitdown","standup"]
g=torch.Generator().manual_seed(0)
idx=np.arange(0,len(y),2)  # 700 samples
db=frontend(torch.from_numpy(np.ascontiguousarray(H[idx])),'clean',g).reshape(len(idx),250,90).numpy()
ys=y[idx]
d=np.load('real/ut_train.npz'); Xr,yr=d['X'],d['y']
def stats(X):
    X=X-X.mean(1,keepdims=True)
    tstd=X.std(1).mean(-1)                         # temporal variation (dB)
    dx=np.abs(np.diff(X,axis=1)).mean((1,2))       # fastness
    # fraction of temporal energy that is common across subcarriers (shadowing-like)
    com=X.mean(-1); common=(com.std(1)/(X.std(1).mean(-1)+1e-9))
    return tstd,dx,common
for name,X,Y in [('SIM clean',db,ys),('REAL',Xr,yr)]:
    t,dx,c=stats(X)
    print(name)
    for k in range(7):
        m=Y==k; print(f"  {CL[k]:8s} temporal-std {np.median(t[m]):5.2f} dB | step {np.median(dx[m]):5.3f} | common-mode {np.median(c[m]):4.2f}")
