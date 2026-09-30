"""Step 0: verify MM-Fi assumptions before running the experiment.

    python inspect_mmfi.py --root /path/to/MMFi

Checks: .mat keys and shapes, phase key present, empty-frame rate, ground-truth shape, and whether
CSI is continuous across frame boundaries (required for velocity windows longer than one frame).
"""
import argparse, glob, os, random
import numpy as np
import scipy.io as sio
from mmfi_seq import list_sequences, load_sequence, PKT

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--n_seq", type=int, default=20)
a = p.parse_args()

seqs = list_sequences(a.root)
print(f"{len(seqs)} sequences found; envs = {sorted({s['env'] for s in seqs})}")
f0 = sorted(glob.glob(os.path.join(seqs[0]["path"], "wifi-csi", "frame*.mat")))[0]
m = sio.loadmat(f0)
print("keys/shapes in", f0, {k: np.shape(v) for k, v in m.items() if not k.startswith("__")})

random.seed(0)
within, across, empty, gts = [], [], [], []
for s in random.sample(seqs, min(a.n_seq, len(seqs))):
    amp, pha, valid, pose = load_sequence(s["path"])
    empty.append(1 - valid.mean())
    if pose is not None:
        gts.append(pose.shape)
    x = amp.reshape(-1, PKT, *amp.shape[1:])[valid]                 # (frames, 10, 3, 114)
    d_in = np.abs(np.diff(x, axis=1)).mean()                        # packet-to-packet change inside frames
    d_x = np.abs(x[1:, 0] - x[:-1, -1]).mean()                      # last packet of frame i -> first of i+1
    within.append(d_in); across.append(d_x)
print(f"empty-frame rate: {np.mean(empty)*100:.2f}%   ground-truth shapes: {sorted(set(gts))}")
ratio = np.mean(across) / np.mean(within)
print(f"mean |change| within frames {np.mean(within):.4f} vs across frame boundaries {np.mean(across):.4f} "
      f"(ratio {ratio:.2f})")
print("-> CONTINUOUS enough for multi-frame velocity windows" if ratio < 2.0 else
      "-> WARNING: large jumps at frame boundaries; CSI may not be continuous across frames. "
      "Use --ctx_frames 1 variants or inspect timestamps before trusting velocity results.")
