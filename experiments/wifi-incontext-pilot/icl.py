"""In-context adaptation pilot: cross-dataset (UT-HAR <-> NTU-Fi) WiFi HAR on shared classes.

Methods (same encoder everywhere):
  erm       source-only training
  erm_vd    + virtual-domain augmentation drawn i.i.d. per sample
  bnadapt   erm_vd, BN statistics re-estimated on unlabeled target data
  tent      erm_vd, entropy-minimisation TTA (BN affine params) on unlabeled target data
  arm       ARM-CML-style: mean-pooled context embedding (+ site statistics) concatenated to query embedding
  xattn     ours: query cross-attends over context tokens (+ site statistics token)
arm/xattn are meta-trained on episodes where ONE virtual domain is applied to context and query alike.
Target labels are used only for scoring.
"""
import argparse, json, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--src", required=True)            # ut | ntu
p.add_argument("--method", required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--steps", type=int, default=3000)
p.add_argument("--ctx", type=int, default=32)
p.add_argument("--tag", default="icl")
a = p.parse_args()
torch.set_num_threads(int(os.environ.get("THREADS", "2")))
torch.manual_seed(a.seed); np.random.seed(a.seed)
rng = np.random.default_rng(a.seed)
tgt = "ntu" if a.src == "ut" else "ut"
SHARED = ["fall", "run", "walk"]
CL = {"ut": ["liedown", "fall", "walk", "pickup", "run", "sitdown", "standup"],
      "ntu": ["box", "circle", "clean", "fall", "run", "walk"]}
T = 64  # 1 s at 62.5 Hz


def rate_match(X, name):
    """Return (N, 64, 90): 1 s at 62.5 Hz. UT: 250 samples over 1 s -> avg-pool x4.
    NTU: 250 samples over 4 s (62.5 Hz) -> the most active 1-s window (label-free)."""
    if name == "ut":
        Z = X[:, :248].reshape(len(X), 62, 4, 90).mean(2)
    else:
        v = np.stack([X[:, s:s + 62].std(1).mean(-1) for s in range(0, 250 - 62 + 1, 4)], 1)
        best = v.argmax(1) * 4
        Z = np.stack([X[i, b:b + 62] for i, b in enumerate(best)])
    return np.pad(Z, ((0, 0), (1, 1), (0, 0)), mode="edge").astype(np.float32)


def load(name, splits):
    Xs, ys = [], []
    for s in splits:
        d = np.load(f"real/{name}_{s}.npz"); Xs.append(d["X"]); ys.append(d["y"])
    X, y = np.concatenate(Xs), np.concatenate(ys)
    keep = [CL[name].index(c) for c in SHARED]
    m = np.isin(y, keep)
    return torch.tensor(rate_match(X[m], name)), torch.tensor([keep.index(v) for v in y[m]])


Xs, ys = load(a.src, ["train"])
Xt, yt = load(tgt, ["train", "test"])  # target: unlabeled context + evaluation (labels only for scoring)


# ------------------------------------------------------------------ virtual domains
def sample_domain(n=1):
    """Parameters of a random environment/device transform (consistent within a domain)."""
    k = torch.linspace(0, 1, 30)
    d = []
    for _ in range(n):
        ant = torch.randn(3, 1) * 3.0
        rip = sum(torch.rand(1) * 2.0 * torch.sin(2 * np.pi * (torch.rand(1) * 3 + .5) * k + torch.rand(3, 1) * 6.28) for _ in range(3))
        dyn = torch.exp(torch.randn(3, 1) * 0.3 + 0.3 * torch.sin(2 * np.pi * torch.rand(1) * 2 * k + torch.rand(1) * 6.28))
        d.append(dict(off=(ant + rip).reshape(90), dyn=dyn.reshape(90), flip=bool(rng.random() < .5),
                      perm=torch.tensor(rng.permutation(3)), stretch=float(np.exp(rng.uniform(np.log(.7), np.log(1.4)))),
                      noise=float(rng.uniform(0, 1.0)), agc=float(rng.uniform(0, 2.0))))
    return d


def apply_domain(x, dom):
    """x: (B, T, 90) dB. Apply one domain transform to all samples in x."""
    B = x.shape[0]
    x = x.view(B, T, 3, 30)[:, :, dom["perm"]]
    if dom["flip"]:
        x = x.flip(-1)
    x = x.reshape(B, T, 90)
    mu = x.mean(1, keepdim=True)
    x = mu + (x - mu) * dom["dyn"]                      # environment-dependent fluctuation strength
    x = x + dom["off"]                                   # chain gains + subcarrier response (additive in dB)
    s = dom["stretch"]                                   # residual packet-rate mismatch
    src = torch.clamp(torch.arange(T) * s, 0, T - 1)
    i0 = src.floor().long(); w = (src - i0).view(1, T, 1); i1 = torch.clamp(i0 + 1, max=T - 1)
    x = x[:, i0] * (1 - w) + x[:, i1] * w
    steps = (torch.rand(B, T) < dom["agc"] / T).float() * torch.randn(B, T) * 2.0
    x = x + steps.cumsum(1).unsqueeze(-1)                # AGC gain steps
    return x + torch.randn_like(x) * dom["noise"]


def norm(x):
    x = x - x.flatten(1).mean(1).view(-1, 1, 1)
    return x / (x.flatten(1).std(1).view(-1, 1, 1) + 1e-6)


# ------------------------------------------------------------------ models
def encoder():
    return nn.Sequential(
        nn.Conv1d(90, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
        nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
        nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
        nn.AdaptiveAvgPool1d(1), nn.Flatten())


def site_stats(xc):  # per-(antenna,subcarrier) mean and temporal std over the context set, (180,)
    return torch.cat([xc.mean((0, 1)), xc.std(1).mean(0)])


class Plain(nn.Module):
    def __init__(s):
        super().__init__(); s.f = encoder(); s.h = nn.Linear(128, 3)

    def forward(s, x, ctx=None):
        return s.h(s.f(x.transpose(1, 2)))


class ARM(nn.Module):  # mean-pooled context
    def __init__(s):
        super().__init__(); s.f = encoder(); s.g = encoder(); s.st = nn.Linear(180, 64)
        s.h = nn.Sequential(nn.Linear(128 + 128 + 64, 128), nn.ReLU(), nn.Linear(128, 3))

    def forward(s, x, ctx):
        c = s.g(ctx.transpose(1, 2)).mean(0, keepdim=True).expand(len(x), -1)
        st = s.st(site_stats(ctx)).unsqueeze(0).expand(len(x), -1)
        return s.h(torch.cat([s.f(x.transpose(1, 2)), c, st], 1))


class XAttn(nn.Module):  # ours: query cross-attends over context tokens + site-stat token
    def __init__(s):
        super().__init__(); s.f = encoder(); s.g = encoder(); s.st = nn.Linear(180, 128)
        s.att = nn.MultiheadAttention(128, 4, batch_first=True)
        s.ff = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 3))

    def forward(s, x, ctx):
        q = s.f(x.transpose(1, 2))
        tok = torch.cat([s.g(ctx.transpose(1, 2)), s.st(site_stats(ctx)).unsqueeze(0)], 0)
        tok = tok.unsqueeze(0).expand(len(x), -1, -1)
        o, _ = s.att(q.unsqueeze(1), tok, tok)
        return s.ff(torch.cat([q, o.squeeze(1)], 1))


M = {"erm": Plain, "erm_vd": Plain, "bnadapt": Plain, "tent": Plain, "arm": ARM, "xattn": XAttn}[a.method]
model = M()
opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
episodic = a.method in ("arm", "xattn")
t0 = time.time()
for it in range(a.steps):
    model.train()
    if episodic:
        loss = 0
        for _ in range(2):  # 2 episodes per step, each its own virtual domain
            idx = rng.choice(len(Xs), a.ctx + 32, replace=False)
            dom = sample_domain()[0]
            xb = norm(apply_domain(Xs[idx], dom))
            loss = loss + F.cross_entropy(model(xb[a.ctx:], xb[:a.ctx]), ys[idx[a.ctx:]], label_smoothing=0.1)
        loss = loss / 2
    else:
        idx = rng.choice(len(Xs), 64, replace=False)
        xb = Xs[idx]
        if a.method != "erm":
            doms = sample_domain(len(idx))
            xb = torch.cat([apply_domain(xb[i:i + 1], d) for i, d in enumerate(doms)])
        loss = F.cross_entropy(model(norm(xb)), ys[idx], label_smoothing=0.1)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
train_s = time.time() - t0

# ------------------------------------------------------------------ test-time on target
Xtn = norm(Xt)
if a.method == "bnadapt":
    model.train()
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d): m.reset_running_stats(); m.momentum = None
        for i in range(0, len(Xtn), 64): model(Xtn[i:i + 64])
if a.method == "tent":
    params = [p_ for m in model.modules() if isinstance(m, nn.BatchNorm1d) for p_ in m.parameters()]
    topt = torch.optim.Adam(params, 1e-3)
    model.train()
    for ep in range(1):
        for i in torch.randperm(len(Xtn)).split(64):
            pr = model(Xtn[i]).softmax(1)
            ent = -(pr * pr.clamp_min(1e-8).log()).sum(1).mean()
            topt.zero_grad(); ent.backward(); topt.step()
model.eval()
accs = []
with torch.no_grad():
    for rep in range(5):
        pred = torch.empty(len(Xtn), dtype=torch.long)
        for i in torch.arange(len(Xtn)).split(64):
            ctx = None
            if episodic:
                pool = np.setdiff1d(np.arange(len(Xtn)), i.numpy())
                ctx = Xtn[rng.choice(pool, a.ctx, replace=False)]
            pred[i] = model(Xtn[i], ctx).argmax(1)
        rec = [float((pred[yt == c] == c).float().mean()) for c in range(3)]
        accs.append(np.mean(rec))
res = dict(tag=a.tag, src=a.src, tgt=tgt, method=a.method, seed=a.seed, bal_acc=float(np.mean(accs)),
           n_src=len(ys), n_tgt=len(yt), train_s=round(train_s))
print(json.dumps(res))
with open("icl_results.jsonl", "a") as f:
    f.write(json.dumps(res) + "\n")
