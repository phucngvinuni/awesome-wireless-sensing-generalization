# sim->sim learnability with a Doppler (DFS) representation from complex CSI (antenna conjugate product)
import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(3)
D=os.environ.get('D','simv3')
H=torch.from_numpy(np.load(f'{D}/ut_H.npy')); y=np.load(f'{D}/ut_y.npy')
# per-sample random CFO-like phase per packet common to antennas -> cancelled by conjugate product
cfo=torch.exp(1j*2*np.pi*torch.rand(H.shape[0],H.shape[1],1,1)); H=H*cfo
C=H[:,:,0,:]*H[:,:,1,:].conj()                      # (N,T,K)
C=C-C.mean(1,keepdim=True)                          # remove static
w=torch.hann_window(32)
S=[]
for k in range(0,C.shape[2],3):                     # average STFT over subcarriers
    st=torch.stft(C[:,:,k],n_fft=32,hop_length=8,window=w,return_complex=True,onesided=False)  # (N,32,frames)
    S.append(st.abs()**2)
S=torch.stack(S).mean(0); S=torch.fft.fftshift(S,dim=1)
S=torch.log(S+1e-9); S=(S-S.mean((1,2),keepdim=True))/S.std((1,2),keepdim=True)
X=S.float()                                        # (N,32 freq,frames)
print('DFS shape',X.shape)
net=nn.Sequential(nn.Conv1d(32,64,5,padding=2),nn.BatchNorm1d(64),nn.ReLU(),nn.Conv1d(64,128,3,padding=1),nn.BatchNorm1d(128),nn.ReLU(),nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(128,7))
n=len(y); perm=np.random.default_rng(0).permutation(n); tr,te=perm[:int(.8*n)],perm[int(.8*n):]; Y=torch.tensor(y)
opt=torch.optim.AdamW(net.parameters(),1e-3)
for ep in range(40):
    net.train(); p=np.random.permutation(tr)
    for i in range(0,len(p),64):
        j=p[i:i+64]; loss=F.cross_entropy(net(X[j]),Y[j]); opt.zero_grad(); loss.backward(); opt.step()
    if ep%10==9:
        net.eval()
        with torch.no_grad(): a=(net(X[te]).argmax(1)==Y[te]).float().mean().item(); b=(net(X[tr]).argmax(1)==Y[tr]).float().mean().item()
        print('epoch',ep+1,'sim-train',round(b,3),'sim-heldout',round(a,3))
