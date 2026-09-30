"""CSI -> input representations for the MM-Fi velocity-space falsification test.

All functions take a *continuous* CSI stream of one sequence:
    amp, pha : (T, A, K)  amplitude and phase, T packets at fs Hz, A antennas, K subcarriers
and return float32 arrays shaped (C, F, T') so they can feed any conv encoder
(including SemPose-Fi's C x F x T encoder).

Representations
  raw_frame      SemPose-Fi-style input for ONE 10-packet frame: [z(amp), cos(phase_det), sin(phase_det)]
  raw_window     the same, over the SAME temporal window as the velocity input (control for context length)
  velocity       antenna-conjugate Doppler spectrogram mapped to path-length-rate units (m/s)
  velocity_amp   amplitude-only Doppler (no phase; sign-ambiguous, folded to |v|)

Physics note (Torun, Cuenca & Mostofi, arXiv:2609.26960): a single link observes
f_D = psi * v / lambda with geometry factor 0 <= psi <= 2, so the "velocity" axis here is the
path-length change rate psi*v (m/s). It removes carrier/sample-rate/device differences and the
static multipath, but NOT the person-to-link geometry.
"""
import numpy as np

C0 = 299_792_458.0


# ----------------------------------------------------------------------------- helpers
def detrend_phase(pha):
    """Unwrap along subcarriers and remove a per-(packet, antenna) linear fit (CFO/SFO/STO slope+offset).
    pha: (..., K) -> same shape."""
    K = pha.shape[-1]
    p = np.unwrap(pha, axis=-1)
    k = np.arange(K) - (K - 1) / 2
    slope = (p * k).sum(-1, keepdims=True) / (k ** 2).sum()
    return p - slope * k - p.mean(-1, keepdims=True)


def zscore(x, axis=None, eps=1e-6):
    return (x - x.mean(axis=axis, keepdims=True)) / (x.std(axis=axis, keepdims=True) + eps)


def pool_axis(x, n_out, axis):
    """Average-pool axis into n_out nearly-equal groups."""
    groups = np.array_split(np.arange(x.shape[axis]), n_out)
    return np.stack([np.take(x, g, axis=axis).mean(axis=axis) for g in groups], axis=axis)


def moving_average(x, win, axis=0):
    """Centered moving average along axis (edge-padded)."""
    if win <= 1:
        return x
    pad = [(0, 0)] * x.ndim
    pad[axis] = (win // 2, win - 1 - win // 2)
    xp = np.pad(x, pad, mode="edge")
    c = np.cumsum(xp, axis=axis, dtype=np.complex128 if np.iscomplexobj(x) else np.float64)
    zero = np.zeros_like(np.take(c, [0], axis=axis))
    c = np.concatenate([zero, c], axis=axis)
    hi = np.take(c, np.arange(win, win + x.shape[axis]), axis=axis)
    lo = np.take(c, np.arange(0, x.shape[axis]), axis=axis)
    return (hi - lo) / win


def first_pc(x):
    """x: (T, K) complex/real, zero-mean over T. Returns the 1st principal-component time series (T,)."""
    u, s, vh = np.linalg.svd(x, full_matrices=False)
    return u[:, 0] * s[0]


def stft(x, n_fft, hop, two_sided=True):
    """x: (T,) -> power (n_freq, n_frames) with Hann window; frequencies ordered low->high."""
    w = np.hanning(n_fft)
    n = 1 + max(0, (len(x) - n_fft) // hop)
    frames = np.stack([x[i * hop:i * hop + n_fft] * w for i in range(n)], 1)  # (n_fft, n)
    spec = np.fft.fft(frames, axis=0) if two_sided else np.fft.rfft(frames, axis=0)
    if two_sided:
        spec = np.fft.fftshift(spec, axes=0)
    return np.abs(spec) ** 2


# ----------------------------------------------------------------------------- raw inputs
def raw_frame(amp, pha):
    """One frame (10 packets): amp/pha (T, A, K) -> (3A, K, T). Mirrors SemPose-Fi preprocessing."""
    a = zscore(amp)
    p = detrend_phase(pha)
    x = np.concatenate([a, np.cos(p), np.sin(p)], axis=1)          # (T, 3A, K)
    return np.transpose(x, (1, 2, 0)).astype(np.float32)             # (3A, K, T)


def raw_window(amp, pha, k_out=38, t_out=32):
    """Same features over a longer window, pooled to (3A, k_out, t_out) to keep size manageable."""
    x = raw_frame(amp, pha)                                           # (3A, K, T)
    x = pool_axis(x, k_out, axis=1)
    return pool_axis(x, t_out, axis=2).astype(np.float32)


# ----------------------------------------------------------------------------- velocity inputs
def _ref_antenna(amp):
    """Reference antenna = highest mean/std amplitude ratio (most static-dominated), IndoTrack/Widar2-style."""
    m = amp.mean(axis=(0, 2)); s = amp.std(axis=(0, 2)) + 1e-9
    return int(np.argmax(m / s))


def _velocity_grid_power(p, fs, fc, v_max, n_v, two_sided, n_fft):
    """Map STFT power rows (frequency) onto a fixed velocity grid in m/s."""
    lam = C0 / fc
    if two_sided:
        f = np.fft.fftshift(np.fft.fftfreq(n_fft, 1 / fs))
        v_grid = np.linspace(-v_max, v_max, n_v)
    else:
        f = np.fft.rfftfreq(n_fft, 1 / fs)
        v_grid = np.linspace(0, v_max, n_v)
    v = f * lam                                                       # path-length rate (m/s)
    out = np.stack([np.interp(v_grid, v, p[:, j], left=0.0, right=0.0) for j in range(p.shape[1])], 1)
    return out, v_grid


def velocity(amp, pha, fs=100.0, fc=5.32e9, n_fft=64, hop=8, v_max=2.5, n_v=64,
             hp_win_s=1.0, alpha=1.0, beta=0.5, log=True):
    """Antenna-conjugate Doppler spectrogram in m/s.

    1. H_a = |H_a| e^{j phase_a}. Pick a static-dominated reference antenna r.
    2. Power adjustment (IndoTrack-style): |H_r| += alpha*mean|H_r|, |H_a| -= beta*mean|H_a| (a != r),
       so the product is dominated by (dynamic_a x static_r) terms.
    3. C_a = H_a * conj(H_r): common phase offsets (CFO, STO, PLL) cancel.
    4. Remove static/slow components with a moving-average high-pass (hp_win_s seconds).
    5. First principal component across subcarriers -> one complex series per antenna pair.
    6. Two-sided STFT (sign of Doppler kept), map frequency -> v = f * lambda, fixed grid [-v_max, v_max].
    Returns (A-1, n_v, n_frames) float32.
    """
    T, A, K = amp.shape
    r = _ref_antenna(amp)
    ar = amp[:, r] + alpha * amp[:, r].mean()
    Hr = ar * np.exp(1j * pha[:, r])
    feats = []
    for a in range(A):
        if a == r:
            continue
        aa = np.maximum(amp[:, a] - beta * amp[:, a].mean(), 0.0)
        Ca = aa * np.exp(1j * pha[:, a]) * np.conj(Hr)               # (T, K)
        Ca = Ca - moving_average(Ca, max(1, int(round(hp_win_s * fs))), axis=0)
        s = first_pc(Ca)
        p = stft(s, n_fft, hop, two_sided=True)
        g, _ = _velocity_grid_power(p, fs, fc, v_max, n_v, True, n_fft)
        feats.append(g)
    x = np.stack(feats, 0)
    if log:
        x = np.log(x + 1e-6 * x.max() + 1e-12)
    x = zscore(x, axis=1)                                             # column-wise normalisation per time bin
    return x.astype(np.float32)


def velocity_amp(amp, fs=100.0, fc=5.32e9, n_fft=64, hop=8, v_max=2.5, n_v=32, hp_win_s=1.0, log=True):
    """Amplitude-only Doppler (no phase needed; works on amplitude-only datasets). Returns (A, n_v, n_frames)."""
    T, A, K = amp.shape
    feats = []
    for a in range(A):
        x = amp[:, a] - moving_average(amp[:, a], max(1, int(round(hp_win_s * fs))), axis=0)
        s = first_pc(x)
        p = stft(s, n_fft, hop, two_sided=False)
        g, _ = _velocity_grid_power(p, fs, fc, v_max, n_v, False, n_fft)
        feats.append(g)
    x = np.stack(feats, 0)
    if log:
        x = np.log(x + 1e-6 * x.max() + 1e-12)
    return zscore(x, axis=1).astype(np.float32)


REPRESENTATIONS = {
    "raw_frame": lambda amp, pha, **kw: raw_frame(amp, pha),
    "raw_window": lambda amp, pha, **kw: raw_window(amp, pha),
    "velocity": lambda amp, pha, **kw: velocity(amp, pha, **kw),
    "velocity_amp": lambda amp, pha, **kw: velocity_amp(amp, **kw),
}
