"""MM-Fi WiFi-CSI sequence loading and per-frame sample construction.

MM-Fi layout (official toolbox mmfi_lib/mmfi.py):
    <root>/E0x/S0y/A0z/wifi-csi/frame001.mat ... frame297.mat   keys: 'CSIamp' (3, 114, 10), phase key (see PHASE_KEYS)
    <root>/E0x/S0y/A0z/ground_truth.npy                           (297, 17, 3) 3D keypoints (metres)
Each frame holds 10 CSI packets, 297 frames per sequence (~10 fps, 100 Hz CSI). Empty (0-byte) frames occur.

We stitch frames into one stream (T = 297*10, A = 3, K = 114). Run inspect_mmfi.py first to confirm
the stream is continuous across frame boundaries; if it is not, velocity windows spanning frames are invalid.
"""
import glob, os, re
import numpy as np
import scipy.io as sio
from velocity import raw_frame, raw_window, velocity, velocity_amp, pool_axis

PHASE_KEYS = ("CSIphase", "CSIpha", "CSI_phase", "phase")
N_FRAMES, PKT = 297, 10


def _fill_nonfinite(x):
    """Replace inf/NaN per packet with that packet's finite mean (mirrors the official loader)."""
    x = x.astype(np.float64)
    x[~np.isfinite(x)] = np.nan
    for i in range(x.shape[-1]):
        col = x[..., i]
        if np.isnan(col).any():
            good = col[~np.isnan(col)]
            col[np.isnan(col)] = good.mean() if good.size else 0.0
    return x


def load_frame(path):
    """-> amp, pha each (10, 3, 114) or None if the frame is empty/unreadable."""
    if os.path.getsize(path) == 0:
        return None
    m = sio.loadmat(path)
    amp = _fill_nonfinite(m["CSIamp"])
    pk = next((k for k in PHASE_KEYS if k in m), None)
    if pk is None:
        raise KeyError(f"No phase key in {path}; keys = {[k for k in m if not k.startswith('__')]}")
    pha = _fill_nonfinite(m[pk])
    return np.transpose(amp, (2, 0, 1)), np.transpose(pha, (2, 0, 1))


def load_sequence(seq_dir):
    """Stitch a sequence. Empty frames are filled from the nearest valid frame and flagged.
    Returns amp, pha (2970, 3, 114) float32, valid (297,) bool, pose (297, 17, 3) or None."""
    files = sorted(glob.glob(os.path.join(seq_dir, "wifi-csi", "frame*.mat")))
    frames = [load_frame(f) for f in files]
    valid = np.array([f is not None for f in frames])
    if not valid.any():
        raise ValueError(f"all frames empty in {seq_dir}")
    idx = np.where(valid)[0]
    amp, pha = [], []
    for i in range(len(frames)):
        j = i if valid[i] else idx[np.argmin(np.abs(idx - i))]
        amp.append(frames[j][0]); pha.append(frames[j][1])
    gt = os.path.join(seq_dir, "ground_truth.npy")
    pose = np.load(gt) if os.path.exists(gt) else None
    return (np.concatenate(amp).astype(np.float32), np.concatenate(pha).astype(np.float32), valid, pose)


def list_sequences(root, envs=("E01", "E02", "E03", "E04"), actions=None):
    """-> list of dicts(env, subject, action, path)."""
    out = []
    for e in envs:
        for s in sorted(glob.glob(os.path.join(root, e, "S*"))):
            for a in sorted(glob.glob(os.path.join(s, "A*"))):
                act = os.path.basename(a)
                if actions and act not in actions:
                    continue
                out.append(dict(env=e, subject=os.path.basename(s), action=act, path=a))
    return out


PROTOCOLS = {  # official MM-Fi action subsets
    "P1": ["A02", "A03", "A04", "A05", "A13", "A14", "A17", "A18", "A19", "A20", "A21", "A22", "A23", "A27"],
    "P2": ["A01", "A06", "A07", "A08", "A09", "A10", "A11", "A12", "A15", "A16", "A24", "A25", "A26"],
    "P3": [f"A{i:02d}" for i in range(1, 28)],
}


# ------------------------------------------------------------------ per-sequence features -> per-frame samples
class SeqFeatures:
    """Precomputes one representation for a whole sequence and slices per-frame samples.

    velocity/velocity_amp: STFT with hop = 10 packets so column i is centred on frame i;
    a sample is ctx_frames consecutive columns -> (C, n_v, ctx_frames).
    raw_window: raw features over exactly the same span of packets the velocity sample sees.
    raw_frame: frame i only (SemPose-Fi input), (9, 114, 10).
    """

    def __init__(self, amp, pha, rep, ctx_frames=16, fs=100.0, fc=5.32e9, n_fft=64, n_v=64, v_max=2.5):
        self.rep, self.ctx, self.n_fft = rep, ctx_frames, n_fft
        self.amp, self.pha = amp, pha
        self.span = (ctx_frames - 1) * PKT + n_fft          # packets seen by one velocity sample
        if rep in ("velocity", "velocity_amp"):
            pad = n_fft // 2 - PKT // 2                      # column j centred on packet 10*j + 5
            a = np.pad(amp, ((pad, n_fft), (0, 0), (0, 0)), mode="edge")
            p = np.pad(pha, ((pad, n_fft), (0, 0), (0, 0)), mode="edge")
            if rep == "velocity":
                v = velocity(a, p, fs=fs, fc=fc, n_fft=n_fft, hop=PKT, n_v=n_v, v_max=v_max)
            else:
                v = velocity_amp(a, fs=fs, fc=fc, n_fft=n_fft, hop=PKT, n_v=n_v // 2, v_max=v_max)
            self.cols = v[:, :, :N_FRAMES]                   # (C, n_v, 297)

    def sample(self, i):
        h = self.ctx // 2
        if self.rep in ("velocity", "velocity_amp"):
            j = np.clip(np.arange(i - h, i - h + self.ctx), 0, self.cols.shape[2] - 1)
            return self.cols[:, :, j]
        if self.rep == "raw_frame":
            s = slice(i * PKT, (i + 1) * PKT)
            return raw_frame(self.amp[s], self.pha[s])
        if self.rep == "raw_window":
            c = i * PKT + PKT // 2
            j = np.clip(np.arange(c - self.span // 2, c - self.span // 2 + self.span), 0, len(self.amp) - 1)
            return raw_window(self.amp[j], self.pha[j])
        raise ValueError(self.rep)
