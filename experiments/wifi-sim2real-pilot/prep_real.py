import numpy as np, glob, scipy.io as sio, os
os.makedirs('real',exist_ok=True)
UT_CLASSES=['liedown','fall','walk','pickup','run','sitdown','standup']
NTU_CLASSES=['box','circle','clean','fall','run','walk']
def pool_sub(x, A, K):  # x (T, A*K) -> (T, A*30)
    T=x.shape[0]; x=x.reshape(T,A,K)
    return np.stack([g.mean(-1) for g in np.array_split(x,30,axis=-1)],-1).reshape(T,A*30)
for split in ['train','test']:
    X=np.load(f'data/UT_HAR/data/X_{split}.csv').reshape(-1,250,90).astype(np.float32)
    y=np.load(f'data/UT_HAR/label/y_{split}.csv').astype(np.int64)
    np.savez(f'real/ut_{split}.npz',X=X,y=y); print('UT',split,X.shape,np.bincount(y))
    Xs,ys=[],[]
    for c,name in enumerate(NTU_CLASSES):
        for f in sorted(glob.glob(f'data/NTU-Fi_HAR/{split}_amp/{name}/*.mat')):
            a=sio.loadmat(f)['CSIamp'][:, ::4].T            # (500, 342) as in SenseFi
            a=0.5*(a[0::2]+a[1::2])                           # (250, 342)
            Xs.append(pool_sub(a,3,114)); ys.append(c)
    X=np.stack(Xs).astype(np.float32); y=np.array(ys)
    np.savez(f'real/ntu_{split}.npz',X=X,y=y); print('NTU',split,X.shape,np.bincount(y))
