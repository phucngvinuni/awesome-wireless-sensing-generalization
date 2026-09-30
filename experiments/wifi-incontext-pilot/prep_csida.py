import zipfile, json, numpy as np, numcodecs, collections
z = zipfile.ZipFile('data/CSIDA-1.zip'); blosc = numcodecs.Blosc()
def arr1d(name):
    meta = json.loads(z.read(f'{name}/.zarray'))
    return np.frombuffer(blosc.decode(z.read(f'{name}/0')), dtype=meta['dtype'])[:meta['shape'][0]]
labels = {k: arr1d(f'csi_label_{k}') for k in ['act', 'env', 'loc', 'user']}
meta = json.loads(z.read('csi_data_amp/.zarray')); N = meta['shape'][0]
groups = np.array_split(np.arange(114), 30)
X = np.zeros((N, 180, 90), np.float32)
for c in range(0, N, 2):
    a = np.frombuffer(blosc.decode(z.read(f'csi_data_amp/{c//2}.0.0.0')), dtype='<f4').reshape(-1, 1800, 3, 114)
    a = 20 * np.log10(np.abs(a) + 1e-6)                        # dB
    a = a.reshape(len(a), 180, 10, 3, 114).mean(2)              # 1 kHz -> 100 Hz
    a = np.stack([a[..., g].mean(-1) for g in groups], -1)      # 114 -> 30 subcarrier groups
    X[c:c + len(a)] = a.reshape(len(a), 180, 90)[: N - c]
np.savez('real/csida.npz', X=X, **labels)
print(X.shape, 'dB range', float(X.min()), float(X.max()))
for k, v in labels.items(): print(k, dict(collections.Counter(v.tolist())))
d = collections.Counter(zip(labels['env'], labels['loc'], labels['user'])); print(len(d), 'domains (env,loc,user); sizes', sorted(d.values())[:5], '...', sorted(d.values())[-3:])
