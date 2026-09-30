"""Pilot simulator: CMU motion -> commodity-WiFi CSI (complex, pre front-end) + front-end models.

Physics (deliberately cheap): body = point scatterers on skeleton segments, single-bounce
Tx->body->Rx plus second-order body->wall->Rx via mirrored receivers, static LoS + first-order
wall images + random furniture scatterers, Fresnel-style LoS shadowing by the torso.
Everything that is unknown about the real deployments is randomized.
"""
import json, os
import numpy as np
import torch
from asfamc import parse_asf, parse_amc, fk

C0 = 3e8
FPS = 30.0
HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------- motion
_cache = {}


def load_clip(s, t, fr):
    key = f"{s}_{t}"
    if key in _cache:
        return _cache[key]
    f = os.path.join(HERE, "cmu_fk", key + ".npz")
    if os.path.exists(f):
        d = np.load(f, allow_pickle=True)
        out = (d["pos"], list(d["names"]), [tuple(x) for x in d["segs"]])
    else:
        b, o = parse_asf(os.path.join(HERE, "cmu", f"{s}.asf"))
        frames = parse_amc(os.path.join(HERE, "cmu", f"{s}_{t}.amc"))
        step = max(1, int(round(fr / FPS)))
        pos, names, segs = fk(b, o, frames[::step])
        os.makedirs(os.path.join(HERE, "cmu_fk"), exist_ok=True)
        np.savez(f, pos=pos.astype(np.float32), names=np.array(names), segs=np.array(segs))
        out = (pos.astype(np.float32), names, segs)
    _cache[key] = out
    return out


def _smooth(x, k=5):
    return np.convolve(np.pad(x, (k, k), mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), "valid")


def segment(pos, names, mode, nfr, rng):
    """Pick a window of nfr frames (at FPS) according to the class-specific rule."""
    i = {n: k for k, n in enumerate(names)}
    if mode == "reverse_ascent":
        pos = pos[::-1].copy()
        mode = "descent"
    T = len(pos)
    rootz = _smooth(pos[:, i["root"], 2])
    if mode == "random":
        c = rng.integers(0, max(1, T - nfr) + 1) + nfr // 2
    elif mode == "descent":
        c = int(np.argmin(np.gradient(rootz)))
    elif mode == "ascent":
        c = int(np.argmax(np.gradient(rootz)))
    elif mode == "bend":
        c = int(np.argmin(_smooth(pos[:, i["head"], 2])))
    else:
        raise ValueError(mode)
    c += int(rng.normal(0, 0.12 * nfr))  # jitter the event position inside the window
    a = c - nfr // 2
    idx = np.clip(np.arange(a, a + nfr), 0, T - 1)
    return pos[idx]


def arm_circles(base_pos, names, nfr, rng):
    """Procedural 'circling arms' built on a neutral standing frame."""
    i = {n: k for k, n in enumerate(names)}
    P = np.repeat(base_pos[None], nfr, 0).copy()
    up = np.array([0, 0, 1.0])
    lat = base_pos[i["lhipjoint"]] - base_pos[i["rhipjoint"]]
    lat[2] = 0
    lat /= np.linalg.norm(lat) + 1e-9
    fwd = np.cross(up, lat)
    period = rng.uniform(0.8, 2.0)
    plane = fwd if rng.random() < 0.6 else lat
    t = np.arange(nfr) / FPS
    for side, ph in (("l", 0.0), ("r", 0.0 if rng.random() < 0.7 else np.pi)):
        chain = [side + n for n in ("clavicle", "humerus", "radius", "wrist", "hand", "fingers")]
        L = [np.linalg.norm(base_pos[i[b]] - base_pos[i[a]]) for a, b in zip(chain[:-1], chain[1:])]
        th = 2 * np.pi * t / period + ph + rng.uniform(0, 2 * np.pi)
        d = -np.cos(th)[:, None] * up + np.sin(th)[:, None] * (plane if side == "l" else plane * (1 if plane is fwd else -1))
        cur = P[:, i[chain[0]]]
        for name, l in zip(chain[1:], L):
            cur = cur + l * d
            P[:, i[name]] = cur
        P[:, i[side + "thumb"]] = P[:, i[side + "hand"]]
    P += rng.normal(0, 0.005, P.shape) * 0  # (no jitter; kept explicit)
    return P


# ------------------------------------------------------------------------- body model
RADIUS = {"root": .13, "lowerback": .13, "upperback": .14, "thorax": .15, "lowerneck": .06, "upperneck": .06,
          "head": .10, "lhipjoint": .10, "rhipjoint": .10, "lfemur": .075, "rfemur": .075, "ltibia": .05,
          "rtibia": .05, "lfoot": .04, "rfoot": .04, "ltoes": .03, "rtoes": .03, "lclavicle": .05,
          "rclavicle": .05, "lhumerus": .045, "rhumerus": .045, "lradius": .035, "rradius": .035,
          "lwrist": .03, "rwrist": .03, "lhand": .03, "rhand": .03, "lfingers": .015, "rfingers": .015,
          "lthumb": .015, "rthumb": .015}


def scatterers(P, names, segs, rng, n_pts=80):
    """Sample body surface points (T, N, 3) and reflectivity weights (N,)."""
    Tn = P.shape[0]
    L = np.array([np.linalg.norm(P[0, c] - P[0, p]) for p, c in segs])
    r = np.array([RADIUS.get(names[c], .03) for p, c in segs])
    area = np.maximum(L, 0.02) * r
    prob = area / area.sum()
    sid = rng.choice(len(segs), n_pts, p=prob)
    u = rng.random(n_pts)
    off = rng.normal(size=(n_pts, 3))
    off /= np.linalg.norm(off, axis=1, keepdims=True)
    par = np.array([segs[k][0] for k in sid])
    chi = np.array([segs[k][1] for k in sid])
    pts = P[:, par] * (1 - u)[None, :, None] + P[:, chi] * u[None, :, None] + off[None] * r[sid][None, :, None]
    w = np.sqrt(area[sid] / prob[sid] / n_pts)  # amplitude ~ sqrt(area)
    return pts.astype(np.float32), w.astype(np.float32)


# ------------------------------------------------------------------------- radio model
def subcarrier_offsets(fmt):
    if fmt == "iwl5300":  # Intel 5300, 20 MHz, 30 grouped subcarriers
        idx = np.r_[np.arange(-28, -1, 2), -1, np.arange(1, 28, 2), 28]
    elif fmt == "ath114":  # Atheros, 40 MHz, 114 subcarriers
        idx = np.r_[np.arange(-58, -1), np.arange(2, 59)]
    else:
        raise ValueError(fmt)
    return idx * 312.5e3


def simulate(P, names, segs, fmt, dur, n_out, rng, n_ant=3, device="cpu"):
    """Return complex CSI (n_out, n_ant, K) as complex64 numpy, before any front-end effect."""
    # --- motion resampling to CSI packet times (point sampling, like decimated real CSI)
    tq = np.linspace(0, dur, n_out, endpoint=False)
    tsrc = np.arange(len(P)) / FPS
    Pi = np.stack([np.stack([np.interp(tq, tsrc, P[:, j, d]) for d in range(3)], -1) for j in range(P.shape[1])], 1)
    pts, w = scatterers(Pi, names, segs, rng)

    # --- scene
    W, D, H = rng.uniform(3, 8), rng.uniform(3, 8), rng.uniform(2.5, 3.5)
    while True:
        tx = np.array([rng.uniform(.3, W - .3), rng.uniform(.3, D - .3), rng.uniform(.5, 1.8)])
        rx = np.array([rng.uniform(.3, W - .3), rng.uniform(.3, D - .3), rng.uniform(.5, 1.8)])
        if 1.5 < np.linalg.norm(tx - rx) < 6:
            break
    # person: centre of window placed near the link, random yaw
    root = Pi[:, names.index("root")]
    mid = (tx + rx) / 2
    loc = mid[:2] + rng.normal(0, 1.0, 2)
    loc = np.clip(loc, .5, [W - .5, D - .5])
    yaw = rng.uniform(0, 2 * np.pi)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    c = root.mean(0)
    pts = (pts - np.r_[c[:2], 0]) @ Rz.T + np.r_[loc, 0]
    torso = (Pi[:, [names.index(n) for n in ("root", "lowerback", "thorax")]] - np.r_[c[:2], 0]) @ Rz.T + np.r_[loc, 0]

    fc = rng.choice([rng.uniform(2.412e9, 2.472e9), rng.uniform(5.18e9, 5.825e9)], p=[.25, .75])
    f = fc + subcarrier_offsets(fmt)
    lam = C0 / fc
    # receive array: linear, random horizontal orientation and spacing
    sp = rng.uniform(0.4, 1.0) * lam
    ang = rng.uniform(0, 2 * np.pi)
    ea = np.array([np.cos(ang), np.sin(ang), 0])
    rxa = rx[None] + (np.arange(n_ant) - (n_ant - 1) / 2)[:, None] * sp * ea[None]

    def mirror(p, axis, val):
        q = p.copy()
        q[..., axis] = 2 * val - q[..., axis]
        return q

    walls = [(0, 0.0), (0, W), (1, 0.0), (1, D), (2, 0.0), (2, H)]

    # static paths: (delay[A], amp) lists  -- v2: 1st+2nd order walls, 10-40 furniture scatterers
    s_del, s_amp = [], []
    d0 = np.linalg.norm(rxa - tx, axis=1)
    s_del.append(d0 / C0); s_amp.append(np.full(n_ant, 1.0) / d0)
    for ax, v in walls:
        ti = mirror(tx, ax, v)
        d = np.linalg.norm(rxa - ti, axis=1)
        s_del.append(d / C0); s_amp.append(rng.uniform(.3, .7) / d * np.exp(1j * rng.uniform(0, 2 * np.pi)))
        for ax2, v2 in walls:
            if ax2 == ax:
                continue
            tii = mirror(ti, ax2, v2)
            d = np.linalg.norm(rxa - tii, axis=1)
            s_del.append(d / C0); s_amp.append(rng.uniform(.1, .4) / d * np.exp(1j * rng.uniform(0, 2 * np.pi)))
    for _ in range(rng.integers(10, 40)):
        q = np.array([rng.uniform(0, W), rng.uniform(0, D), rng.uniform(0, H)])
        d1, d2 = np.linalg.norm(q - tx), np.linalg.norm(rxa - q, axis=1)
        s_del.append((d1 + d2) / C0); s_amp.append(rng.uniform(.05, .5) / (d1 * d2) * np.exp(1j * rng.uniform(0, 2 * np.pi)))
    s_del, s_amp = np.stack(s_del, 1), np.stack(s_amp, 1)  # (A, S)
    # Rician K-factor randomization: LoS power relative to all other static paths
    k_db = rng.uniform(-10, 10)
    p_rest = (np.abs(s_amp[:, 1:].sum(1)) ** 2).mean() + np.mean(np.abs(s_amp[:, 1:]) ** 2) * s_amp.shape[1]
    s_amp[:, 0] *= np.sqrt(10 ** (k_db / 10) * p_rest / np.mean(np.abs(s_amp[:, 0]) ** 2))

    # LoS shadowing by torso (Fresnel-like soft blockage) -- v2: weaker and optional
    seg = rx - tx
    tt = np.clip(((torso - tx) @ seg) / (seg @ seg), 0, 1)
    dist = np.linalg.norm(torso - (tx + tt[..., None] * seg), axis=-1).min(1)
    dlos = np.linalg.norm(seg)
    rf = np.sqrt(lam * dlos / 4) + 0.15
    alpha = rng.uniform(0, .5) if rng.random() < .5 else 0.0
    los_att = 1 - alpha * np.exp(-dist ** 2 / (2 * rf ** 2))  # (T,)

    # body paths: direct + via all 6 surfaces (mirrored receiver) -- v2
    rx_imgs = [rxa] + [mirror(rxa, ax, v) for ax, v in walls]
    gam = [1.0] + list(rng.uniform(.2, .7, len(walls)))
    phi = np.exp(1j * rng.uniform(0, 2 * np.pi, pts.shape[1]))

    dev = torch.device(device)
    fT = torch.tensor(f, dtype=torch.float64, device=dev)
    out = torch.zeros(n_out, n_ant, len(f), dtype=torch.complex128, device=dev)
    # static part
    for a in range(n_ant):
        ph = torch.exp(-2j * np.pi * torch.tensor(s_del[a], device=dev)[:, None] * fT[None])  # (S,K)
        amp = torch.tensor(s_amp[a], device=dev, dtype=torch.complex128)
        los = amp[0] * ph[0]
        rest = (amp[1:, None] * ph[1:]).sum(0)
        out[:, a] = torch.tensor(los_att, device=dev)[:, None] * los[None] + rest[None]
    stat_pow = (out.abs() ** 2).mean().item()
    # dynamic (body) part
    body = torch.zeros_like(out)
    d1 = np.linalg.norm(pts - tx, axis=-1)  # (T,N)
    for g, ri in zip(gam, rx_imgs):
        for a in range(n_ant):
            d2 = np.linalg.norm(pts - ri[a], axis=-1)
            tau = torch.tensor((d1 + d2) / C0, device=dev)
            amp = torch.tensor(g * w[None] / (d1 * d2) * phi[None], device=dev)
            body[:, a] += torch.einsum("tn,tnk->tk", amp, torch.exp(-2j * np.pi * tau[..., None] * fT))
    body_pow = (body.abs() ** 2).mean().item() + 1e-30
    ratio_db = rng.uniform(-18, -3)  # randomized dynamic-to-static power ratio
    body *= np.sqrt(stat_pow / body_pow * 10 ** (ratio_db / 10))
    H = out + body
    H = H / H.abs().pow(2).mean().sqrt()
    return H.to(torch.complex64).cpu().numpy()


# ------------------------------------------------------------------------ front-ends
def _db(x):
    return 20 * torch.log10(x.abs() + 1e-6)


def frontend(H, mode, gen):
    """H: complex tensor (B, T, A, K). Returns amplitude in dB (B, T, A, K).

    modes: clean | awgn | unstructured | phys
    """
    B, T, A, K = H.shape
    dev = H.device

    def U(lo, hi, *shape):
        return lo + (hi - lo) * torch.rand(*shape, generator=gen, device=dev)

    def N(*shape):
        return torch.randn(*shape, generator=gen, device=dev)

    if mode == "clean":
        return _db(H)
    snr = U(10, 35, B, 1, 1, 1)
    noise = (N(B, T, A, K) + 1j * N(B, T, A, K)) / np.sqrt(2) * 10 ** (-snr / 20)
    Hn = H + noise
    if mode == "awgn":
        return _db(Hn)
    phys = _phys(H, gen, snr, noise)
    if mode == "phys":
        return phys
    if mode == "unstructured":
        # same AWGN, plus i.i.d. dB perturbations whose static (per antenna/subcarrier) and
        # time-varying energies match the phys deviation, but with no physical structure
        base = _db(Hn)
        dev_ = phys - base
        stat = dev_.mean(1, keepdim=True)
        s_sig = stat.flatten(1).std(1).view(B, 1, 1, 1)
        d_sig = (dev_ - stat).flatten(1).std(1).view(B, 1, 1, 1)
        return base + s_sig * N(B, 1, A, K) + d_sig * N(B, T, A, K)
    raise ValueError(mode)


def _phys(H, gen, snr, noise):
    B, T, A, K = H.shape
    dev = H.device

    def U(lo, hi, *shape):
        return lo + (hi - lo) * torch.rand(*shape, generator=gen, device=dev)

    def N(*shape):
        return torch.randn(*shape, generator=gen, device=dev)

    k = torch.linspace(0, 1, K, device=dev)
    # 1) per-chain gain offsets + per-subcarrier (baseband filter) response, partly shared across chains
    chain_db = N(B, 1, A, 1) * U(0, 5, B, 1, 1, 1)
    rip = torch.zeros(B, 1, A, K, device=dev)
    for m in range(1, 4):
        amp_s = U(0, 1.5, B, 1, 1, 1)
        freq = U(0.5, 3, B, 1, 1, 1) * m
        ph_s = U(0, 2 * np.pi, B, 1, 1, 1)
        ph_a = U(0, 2 * np.pi, B, 1, A, 1) * U(0, 0.5, B, 1, 1, 1)
        rip = rip + amp_s * torch.sin(2 * np.pi * freq * k + ph_s + ph_a)
    edge = U(0, 3, B, 1, 1, 1) * ((2 * k - 1).abs() ** 6)
    resp_db = chain_db + rip - edge
    Hx = H * 10 ** (resp_db / 20)
    # 2) AGC: power-tracking compression (beta) + random gain steps
    p = Hx.abs().pow(2).mean(dim=(2, 3), keepdim=True)  # (B,T,1,1)
    win = int(U(3, 20, 1).item())
    ps = torch.nn.functional.avg_pool1d(p.view(B, 1, T), 2 * win + 1, 1, win, count_include_pad=False).view(B, T, 1, 1)
    beta = U(0, 0.8, B, 1, 1, 1)
    g_db = -beta * 10 * torch.log10(ps / ps.mean(1, keepdim=True))
    nj = torch.poisson(U(0, 3, B), generator=gen)
    steps = torch.zeros(B, T, device=dev)
    for b in range(B):
        for _ in range(int(nj[b].item())):
            t0 = int(torch.randint(1, T, (1,), generator=gen, device=dev).item())
            steps[b, t0:] += 2.0 * torch.randn(1, generator=gen, device=dev).item()
    g_db = g_db + steps.view(B, T, 1, 1)
    Hx = Hx * 10 ** (g_db / 20)
    # 3) thermal noise (same draw as AWGN condition) + ADC quantization of I/Q
    Hx = Hx + noise * 10 ** (g_db / 20)
    bits = U(5, 9, B, 1, 1, 1).round()
    scale = Hx.abs().flatten(1).amax(1).view(B, 1, 1, 1) / (2 ** (bits - 1))
    Hx = torch.round(Hx.real / scale) * scale + 1j * torch.round(Hx.imag / scale) * scale
    # 4) packet loss with zero-order hold
    drop = torch.rand(B, T, generator=gen, device=dev) < U(0, 0.1, B, 1)
    drop[:, 0] = False
    idx = torch.arange(T, device=dev).expand(B, T).clone()
    idx[drop] = 0
    idx = torch.cummax(idx, 1).values
    Hx = torch.gather(Hx, 1, idx.view(B, T, 1, 1).expand(B, T, A, K))
    return _db(Hx)
