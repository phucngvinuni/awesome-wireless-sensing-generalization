"""Leave-one-environment-out (LOEO) on MM-Fi: raw CSI vs velocity-space input, same model.

    python loeo.py --root /data/MMFi --rep velocity   --target E04 --seeds 0 1 2
    python loeo.py --root /data/MMFi --rep raw_window --target E04 --seeds 0 1 2
    python loeo.py --root /data/MMFi --rep raw_frame  --target E04 --seeds 0 1 2   # SemPose-Fi-style input
    python summarize.py results.jsonl

Protocol per target environment T (default E04, or --target all for the 4 folds):
  train     : source envs, all subjects except the last `--val_subjects` per env
  in-domain : those held-out subjects of the source envs (same rooms, new people)
  target    : every subject of env T (new room)
  gap       = target error - in-domain error      <- the quantity the go/no-go rule is defined on
Tasks: 3D pose (MPJPE, PA-MPJPE in mm) and action recognition (balanced accuracy), trained jointly.
"""
import argparse, json, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from multiprocessing import Pool
from mmfi_seq import list_sequences, load_sequence, SeqFeatures, PROTOCOLS, N_FRAMES

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--rep", required=True, choices=["raw_frame", "raw_window", "velocity", "velocity_amp"])
p.add_argument("--target", default="E04")                 # E01..E04 or 'all'
p.add_argument("--protocol", default="P3", choices=list(PROTOCOLS))
p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
p.add_argument("--epochs", type=int, default=20)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--frame_stride", type=int, default=3)     # use every k-th frame (speed); 1 = all frames
p.add_argument("--ctx_frames", type=int, default=16)
p.add_argument("--fc", type=float, default=5.32e9)        # carrier (only scales the velocity axis)
p.add_argument("--val_subjects", type=int, default=2)
p.add_argument("--cache", default="cache")
p.add_argument("--workers", type=int, default=8)
p.add_argument("--out", default="results.jsonl")
a = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
ACTIONS = PROTOCOLS[a.protocol]


# ------------------------------------------------------------------ cache stitched sequences once
def cache_seq(s):
    f = os.path.join(a.cache, f"{s['env']}_{s['subject']}_{s['action']}.npz")
    if not os.path.exists(f):
        amp, pha, valid, pose = load_sequence(s["path"])
        np.savez(f, amp=amp, pha=pha, valid=valid, pose=pose if pose is not None else np.zeros((N_FRAMES, 17, 3)))
    return f


os.makedirs(a.cache, exist_ok=True)
seqs = list_sequences(a.root, actions=ACTIONS)
with Pool(a.workers) as pool:
    files = pool.map(cache_seq, seqs)


def build(seq_ids):
    """Materialise samples for a list of sequences -> X (N, C, F, T) float16, pose (N, 17, 3), act (N,)."""
    X, P, Y = [], [], []
    for k in seq_ids:
        d = np.load(files[k])
        sf = SeqFeatures(d["amp"], d["pha"], a.rep, ctx_frames=a.ctx_frames, fc=a.fc)
        for i in range(0, N_FRAMES, a.frame_stride):
            if not d["valid"][i]:
                continue
            X.append(sf.sample(i).astype(np.float16)); P.append(d["pose"][i]); Y.append(ACTIONS.index(seqs[k]["action"]))
    return np.stack(X), np.stack(P).astype(np.float32), np.array(Y)


class Net(nn.Module):
    """Identical for every representation except the number of input channels."""
    def __init__(s, c_in, n_act):
        super().__init__()
        def blk(i, o, st): return nn.Sequential(nn.Conv2d(i, o, 3, st, 1), nn.BatchNorm2d(o), nn.GELU())
        s.f = nn.Sequential(blk(c_in, 64, 1), blk(64, 128, 2), blk(128, 128, 2), blk(128, 256, 2),
                            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        s.pose = nn.Linear(256, 17 * 3); s.act = nn.Linear(256, n_act)

    def forward(s, x):
        h = s.f(x)
        return s.pose(h).view(-1, 17, 3), s.act(h)


def procrustes(gt, pr):
    """Similarity-aligned prediction (PA-MPJPE), per frame. gt, pr: (J, 3)."""
    mu_g, mu_p = gt.mean(0), pr.mean(0)
    g, q = gt - mu_g, pr - mu_p
    u, s, vt = np.linalg.svd(q.T @ g)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1; s[-1] *= -1; r = u @ vt
    scale = s.sum() / (q ** 2).sum()
    return scale * q @ r + mu_g


def evaluate(model, X, P, Y):
    model.eval(); preds, logits = [], []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            pp, lg = model(torch.from_numpy(X[i:i + 512]).float().to(dev))
            preds.append(pp.cpu().numpy()); logits.append(lg.argmax(1).cpu().numpy())
    pr, lg = np.concatenate(preds), np.concatenate(logits)
    mpjpe = np.linalg.norm(pr - P, axis=-1).mean() * 1000
    pa = np.mean([np.linalg.norm(procrustes(P[n], pr[n]) - P[n], axis=-1).mean() for n in range(len(P))]) * 1000
    bal = np.mean([(lg[Y == c] == c).mean() for c in np.unique(Y)])
    return dict(mpjpe=float(mpjpe), pa_mpjpe=float(pa), bal_acc=float(bal), n=int(len(Y)))


targets = ["E01", "E02", "E03", "E04"] if a.target == "all" else [a.target]
for tgt in targets:
    src_subj = {}
    for k, s in enumerate(seqs):
        if s["env"] != tgt:
            src_subj.setdefault(s["env"], set()).add(s["subject"])
    val_subj = {e: sorted(v)[-a.val_subjects:] for e, v in src_subj.items()}
    tr = [k for k, s in enumerate(seqs) if s["env"] != tgt and s["subject"] not in val_subj[s["env"]]]
    va = [k for k, s in enumerate(seqs) if s["env"] != tgt and s["subject"] in val_subj[s["env"]]]
    te = [k for k, s in enumerate(seqs) if s["env"] == tgt]
    t0 = time.time()
    (Xtr, Ptr, Ytr), (Xva, Pva, Yva), (Xte, Pte, Yte) = build(tr), build(va), build(te)
    print(f"[{a.rep}] target {tgt}: train {Xtr.shape} in-domain {len(Yva)} target {len(Yte)} "
          f"(features {time.time()-t0:.0f}s)", flush=True)
    for seed in a.seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        model = Net(Xtr.shape[1], len(ACTIONS)).to(dev)
        opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
        steps = a.epochs * (len(Ytr) // a.batch + 1)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
        for ep in range(a.epochs):
            model.train()
            for idx in np.array_split(np.random.permutation(len(Ytr)), len(Ytr) // a.batch + 1):
                x = torch.from_numpy(Xtr[idx]).float().to(dev)
                pp, lg = model(x)
                loss = torch.linalg.norm(pp - torch.from_numpy(Ptr[idx]).to(dev), dim=-1).mean() \
                    + F.cross_entropy(lg, torch.from_numpy(Ytr[idx]).to(dev))
                opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        r_in, r_tg = evaluate(model, Xva, Pva, Yva), evaluate(model, Xte, Pte, Yte)
        res = dict(rep=a.rep, target=tgt, protocol=a.protocol, seed=seed, ctx_frames=a.ctx_frames,
                   in_domain=r_in, target_env=r_tg,
                   gap_mpjpe=r_tg["mpjpe"] - r_in["mpjpe"], gap_acc=r_in["bal_acc"] - r_tg["bal_acc"])
        print(json.dumps(res), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(res) + "\n")
