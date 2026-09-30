"""Train on simulated (with a front-end condition) or real CSI; evaluate on real CSI.
Examples:
  python train.py --train sim --target ut --mode phys --seed 0
  python train.py --train ut_real --target ut --seed 0            (in-domain upper bound)
  python train.py --train ntu_real --target ut --classes fall,run,walk --seed 0   (cross-dataset)
"""
import argparse, json, time, os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sim import frontend

p = argparse.ArgumentParser()
p.add_argument("--train", required=True)          # sim | ut_real | ntu_real
p.add_argument("--target", required=True)         # ut | ntu  (evaluation dataset / sim format)
p.add_argument("--mode", default="phys")          # sim front-end condition
p.add_argument("--classes", default="")           # comma list (subset); default: all target classes
p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=0)
p.add_argument("--norm", default="global")        # global | sanitized
p.add_argument("--tag", default="")
a = p.parse_args()
torch.set_num_threads(int(os.environ.get("THREADS", "4")))
torch.manual_seed(a.seed); np.random.seed(a.seed)

CL = {"ut": ["liedown", "fall", "walk", "pickup", "run", "sitdown", "standup"],
      "ntu": ["box", "circle", "clean", "fall", "run", "walk"]}
classes = a.classes.split(",") if a.classes else CL[a.target]
nc = len(classes)


def subset(X, y, names):
    keep = [names.index(c) for c in classes]
    m = np.isin(y, keep)
    remap = {k: i for i, k in enumerate(keep)}
    return X[m], np.array([remap[v] for v in y[m]])


def norm(x):  # x (B, T, 90)
    if a.norm == "sanitized":
        x = x - x.mean(1, keepdim=True)
        return x / (x.flatten(1).std(1).view(-1, 1, 1) + 1e-6)
    x = x - x.flatten(1).mean(1).view(-1, 1, 1)
    return x / (x.flatten(1).std(1).view(-1, 1, 1) + 1e-6)


def to_feat(db, fmt):  # db (B, T, 3, K) -> (B, 250, 90)
    if fmt == "ntu":
        db = 0.5 * (db[:, 0::2] + db[:, 1::2])
        groups = np.array_split(np.arange(db.shape[-1]), 30)
        db = torch.stack([db[..., g].mean(-1) for g in groups], -1)
    return db.reshape(db.shape[0], db.shape[1], -1)


class Net(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv1d(90, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, nc))

    def forward(self, x):  # x (B, T, 90)
        return self.f(x.transpose(1, 2))


def load_real(name, split):
    d = np.load(f"real/{name}_{split}.npz")
    return subset(d["X"], d["y"], CL[name])


# ---------------------------------------------------------------- training data
gen = torch.Generator().manual_seed(1000 + a.seed)
if a.train == "sim":
    H = np.load(f"simdata/{a.target}_H.npy", mmap_mode="r")
    ys = np.load(f"simdata/{a.target}_y.npy")
    keep = [CL[a.target].index(c) for c in classes]
    idx_all = np.where(np.isin(ys, keep))[0]
    remap = {k: i for i, k in enumerate(keep)}
    ytr = np.array([remap[v] for v in ys[idx_all]])
    ntr = len(idx_all)
    epochs = a.epochs or 15

    def batches():
        perm = np.random.permutation(ntr)
        for i in range(0, ntr - 63, 64):
            j = np.sort(idx_all[perm[i:i + 64]])
            order = np.argsort(np.argsort(idx_all[perm[i:i + 64]]))
            Hb = torch.from_numpy(np.ascontiguousarray(H[j]))
            db = frontend(Hb, a.mode, gen)
            x = to_feat(db, a.target)
            yb = torch.tensor(np.array([remap[v] for v in ys[j]]))
            yield norm(x), yb
else:
    src = a.train.split("_")[0]
    Xr, yr = load_real(src, "train")
    Xr = torch.tensor(Xr)
    ytr = yr
    ntr = len(yr)
    epochs = a.epochs or (30 if ntr > 2000 else 60)

    def batches():
        perm = np.random.permutation(ntr)
        for i in range(0, ntr, 64):
            j = perm[i:i + 64]
            yield norm(Xr[j]), torch.tensor(yr[j])

# ---------------------------------------------------------------- train
model = Net(nc)
opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-3)
steps = epochs * (ntr // 64 + 1)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps + 10)
t0 = time.time()
for ep in range(epochs):
    model.train(); tl, nb = 0, 0
    for x, yb in batches():
        loss = F.cross_entropy(model(x), yb, label_smoothing=0.1)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        tl += loss.item(); nb += 1
print(f"trained {a.train}/{a.mode} on {a.target} ({nc} cls, n={ntr}) {epochs} ep, last loss {tl/max(nb,1):.3f}, {time.time()-t0:.0f}s")


# ---------------------------------------------------------------- evaluate on real
def evaluate(X, y):
    model.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            preds.append(model(norm(torch.tensor(X[i:i + 256]))).argmax(1).numpy())
    pr = np.concatenate(preds)
    acc = float((pr == y).mean())
    rec = [float((pr[y == c] == c).mean()) for c in range(nc) if (y == c).any()]
    f1s = []
    for c in range(nc):
        tp = ((pr == c) & (y == c)).sum(); fp = ((pr == c) & (y != c)).sum(); fn = ((pr != c) & (y == c)).sum()
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    cm = np.zeros((nc, nc), int)
    for t_, p_ in zip(y, pr): cm[t_, p_] += 1
    return dict(acc=acc, bal_acc=float(np.mean(rec)), macro_f1=float(np.mean(f1s)), n=int(len(y)), cm=cm.tolist())


Xt, yt = load_real(a.target, "test")
res = dict(train=a.train, target=a.target, mode=a.mode if a.train == "sim" else "real", classes=classes,
           seed=a.seed, norm=a.norm, tag=a.tag, test=evaluate(Xt, yt))
if a.train == "sim":  # real train split is also unseen by sim-only models -> larger eval set
    Xa, ya = load_real(a.target, "train")
    res["all_real"] = evaluate(np.concatenate([Xa, Xt]), np.concatenate([ya, yt]))
res["majority_test"] = float(np.bincount(yt).max() / len(yt))
print(json.dumps({k: (v if k not in ("test", "all_real") else {kk: vv for kk, vv in v.items() if kk != "cm"}) for k, v in res.items()}))
with open("results.jsonl", "a") as f:
    f.write(json.dumps(res) + "\n")
