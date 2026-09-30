"""In-context adaptation pilot on CSIDA raw CSI amplitude (6 gestures, 2 rooms x 3 locations x 5 users).

Domains = (user, torso location, face orientation). Protocols:
  locori : train loc 1-4 x ori 1-4 ; test loc 5 or ori 5   (loc <= 5)
  user   : train users 1-9          ; test users 10-17
Methods: erm | bnadapt | tent | arm (mean-pooled context) | xattn (cross-attention context).
Context = unlabeled samples from the same test domain (never the query itself).
Control for context models: context drawn from a *different* test domain ('other').
"""
import argparse, json, os, time, collections
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--proto", required=True)
p.add_argument("--method", required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--steps", type=int, default=2000)
p.add_argument("--tag", default="csida")
a = p.parse_args()
torch.set_num_threads(int(os.environ.get("THREADS", "2")))
torch.manual_seed(a.seed); np.random.seed(a.seed); rng = np.random.default_rng(a.seed)

d = np.load("real/csida.npz")
X, y, U, L, O = torch.tensor(d["X"]), torch.tensor(d["act"]), d["user"], d["loc"], d["env"]  # O := room
X = X - X.flatten(1).mean(1).view(-1, 1, 1); X = X / (X.flatten(1).std(1).view(-1, 1, 1) + 1e-6)
if a.proto == "roomAB":
    tr = O == 0; te = O == 1
elif a.proto == "roomBA":
    tr = O == 1; te = O == 0
elif a.proto == "random":  # in-domain sanity check: random 80/20 split over all samples
    r = np.random.default_rng(123).random(len(U)); tr = r < 0.8; te = ~tr
else:  # user
    tr = U <= 2; te = U >= 3
dom = np.array([f"{u}-{l}-{o}" for u, l, o in zip(U, L, O)])


def groups(mask):
    g = collections.defaultdict(list)
    for i in np.where(mask)[0]:
        g[dom[i]].append(i)
    return {k: np.array(v) for k, v in g.items() if len(v) >= 8}


Gtr, Gte = groups(tr), groups(te)
tr_idx = np.concatenate(list(Gtr.values()))


def encoder():
    return nn.Sequential(
        nn.Conv1d(90, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
        nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
        nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
        nn.AdaptiveAvgPool1d(1), nn.Flatten())


class Plain(nn.Module):
    def __init__(s):
        super().__init__(); s.f = encoder(); s.h = nn.Linear(128, 6)

    def forward(s, x, ctx=None):
        return s.h(s.f(x.transpose(1, 2)))


class ARM(nn.Module):
    def __init__(s):
        super().__init__(); s.f = encoder(); s.g = encoder()
        s.h = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 6))

    def forward(s, x, ctx):
        c = s.g(ctx.transpose(1, 2)).mean(0, keepdim=True).expand(len(x), -1)
        return s.h(torch.cat([s.f(x.transpose(1, 2)), c], 1))


class XAttn(nn.Module):
    def __init__(s):
        super().__init__(); s.f = encoder(); s.g = encoder()
        s.att = nn.MultiheadAttention(128, 4, batch_first=True)
        s.h = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 6))

    def forward(s, x, ctx):
        q = s.f(x.transpose(1, 2))
        tok = s.g(ctx.transpose(1, 2)).unsqueeze(0).expand(len(x), -1, -1)
        o, _ = s.att(q.unsqueeze(1), tok, tok)
        return s.h(torch.cat([q, o.squeeze(1)], 1))


model = {"erm": Plain, "bnadapt": Plain, "tent": Plain, "arm": ARM, "xattn": XAttn}[a.method]()
episodic = a.method in ("arm", "xattn")
opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
keys = list(Gtr)
t0 = time.time()
for it in range(a.steps):
    model.train()
    if episodic:
        loss = 0
        for _ in range(4):
            idx = rng.permutation(Gtr[keys[rng.integers(len(keys))]])
            q, c = idx[:8], idx[8:24]          # 8 queries, up to 16 disjoint random context samples
            loss = loss + F.cross_entropy(model(X[q], X[c]), y[q], label_smoothing=0.1)
        loss = loss / 4
    else:
        idx = rng.choice(tr_idx, 64, replace=False)
        loss = F.cross_entropy(model(X[idx]), y[idx], label_smoothing=0.1)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
train_s = time.time() - t0

te_idx = np.concatenate(list(Gte.values()))
if a.method == "bnadapt":
    model.train()
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d): m.reset_running_stats(); m.momentum = None
        for i in np.array_split(rng.permutation(te_idx), len(te_idx) // 64 + 1): model(X[i])
if a.method == "tent":
    ps = [q for m in model.modules() if isinstance(m, nn.BatchNorm1d) for q in m.parameters()]
    topt = torch.optim.Adam(ps, 1e-3); model.train()
    for i in np.array_split(rng.permutation(te_idx), len(te_idx) // 64 + 1):
        pr = model(X[i]).softmax(1); ent = -(pr * pr.clamp_min(1e-8).log()).sum(1).mean()
        topt.zero_grad(); ent.backward(); topt.step()
model.eval()


def evaluate(ctx_mode):
    correct = collections.Counter(); total = collections.Counter()
    tkeys = list(Gte)
    with torch.no_grad():
        for k, idx in Gte.items():
            for chunk in np.array_split(rng.permutation(idx), max(1, len(idx) // 8)):
                ctx = None
                if episodic:
                    if ctx_mode == "same":
                        pool = np.setdiff1d(idx, chunk)
                    else:
                        other = tkeys[rng.integers(len(tkeys))]
                        while other == k: other = tkeys[rng.integers(len(tkeys))]
                        pool = Gte[other]
                    ctx = X[rng.permutation(pool)[:16]]
                pr = model(X[chunk], ctx).argmax(1)
                for c in range(6):
                    m = y[chunk] == c
                    correct[c] += int((pr[m] == c).sum()); total[c] += int(m.sum())
    return float(np.mean([correct[c] / total[c] for c in range(6) if total[c]]))


res = dict(tag=a.tag, proto=a.proto, method=a.method, seed=a.seed, bal_acc=evaluate("same"),
           n_train_domains=len(Gtr), n_test_domains=len(Gte), n_train=int(len(tr_idx)), n_test=int(len(te_idx)),
           train_s=round(train_s))
if episodic:
    res["bal_acc_other_ctx"] = evaluate("other")
print(json.dumps(res))
with open("icl_csida_results.jsonl", "a") as f:
    f.write(json.dumps(res) + "\n")
