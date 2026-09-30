"""Generate simulated complex CSI (pre front-end) for a target format.
usage: python gen.py {ut|ntu} N_PER_CLASS SEED OUT_PREFIX
"""
import sys, json, os, time
import numpy as np
import torch
from sim import load_clip, segment, simulate, arm_circles

torch.set_num_threads(int(os.environ.get("THREADS", "2")))
target, npc, seed, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
CLASSES = {"ut": ["liedown", "fall", "walk", "pickup", "run", "sitdown", "standup"],
           "ntu": ["box", "circle", "clean", "fall", "run", "walk"]}[target]
FMT, T, DUR = {"ut": ("iwl5300", 250, (0.9, 1.1)), "ntu": ("ath114", 500, (3.8, 4.2))}[target]
spec = json.load(open("clips.json"))
rng = np.random.default_rng(seed)

# neutral standing frame for procedural arm circles: slowest-root frame of a walk clip
bp, bnames, bsegs = load_clip("07", "01", 120)
v = np.r_[0, np.linalg.norm(np.diff(bp[:, 0, :2], axis=0), axis=1)]
base = bp[int(np.argmin(v[1:])) + 1]

K = 30 if FMT == "iwl5300" else 114
H = np.lib.format.open_memmap(out + "_H.npy", mode="w+", dtype=np.complex64, shape=(npc * len(CLASSES), T, 3, K))
y = np.zeros(npc * len(CLASSES), np.int64)
meta = []
t0, n = time.time(), 0
for i in range(npc):
    for c, cls in enumerate(CLASSES):
        dur = rng.uniform(*DUR)
        tempo = rng.uniform(0.8, 1.25)
        nfr = max(8, int(dur * 30 * tempo))
        if spec[cls] == "procedural":
            P, names, segs = arm_circles(base, bnames, nfr, rng), bnames, bsegs
            src = "procedural"
        else:
            s, t, fr, d, mode = spec[cls][rng.integers(len(spec[cls]))]
            pos, names, segs = load_clip(s, t, fr)
            P = segment(pos, names, mode, nfr, rng)
            src = f"{s}_{t}"
        P = P * rng.uniform(0.85, 1.15)
        if rng.random() < 0.5:
            P = P * np.array([1, -1, 1], np.float32)
        H[n] = simulate(P, names, segs, FMT, dur, T, rng)
        y[n] = c
        meta.append((cls, src))
        n += 1
    if i % 25 == 0:
        print(f"{target} {n}/{len(y)} {time.time()-t0:.0f}s", flush=True)
H.flush()
np.save(out + "_y.npy", y)
json.dump(meta, open(out + "_meta.json", "w"))
print("done", n, f"{time.time()-t0:.0f}s")
