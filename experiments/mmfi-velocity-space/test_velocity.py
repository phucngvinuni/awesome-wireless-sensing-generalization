"""Unit tests for the velocity representation on synthetic CSI.  Run: python test_velocity.py

A reflector changes its path length at a known rate r (m/s). The CSI also carries a random
per-packet phase common to all antennas (CFO/STO/PLL), which antenna conjugate multiplication must
cancel. Sign convention: v = f * lambda, so a path that gets LONGER gives negative v.
"""
import numpy as np
from velocity import velocity, velocity_amp, raw_frame, raw_window, C0


def synth(rate, T=1200, fs=100.0, fc=5.32e9, A=3, K=114, seed=0, snr_db=25):
    rng = np.random.default_rng(seed)
    f = fc + (np.r_[np.arange(-58, -1), np.arange(2, 59)] * 312.5e3)[:K]
    lam = C0 / fc
    t = np.arange(T) / fs
    ant = np.arange(A) * lam / 2
    H = np.zeros((T, A, K), complex)
    for _ in range(6):                                                       # static multipath
        d0, g, th = rng.uniform(3, 12), rng.uniform(.2, 1), rng.uniform(0, np.pi)
        H += g * np.exp(-2j * np.pi * f[None, None] * (d0 + ant[None, :, None] * np.cos(th)) / C0)
    d = 5.0 + rate * t                                                       # moving reflector
    H += 0.3 * np.exp(-2j * np.pi * f[None, None] * (d[:, None, None] + ant[None, :, None] * 0.5) / C0)
    H *= np.exp(1j * rng.uniform(0, 2 * np.pi, (T, 1, 1)))                   # per-packet common phase offset
    n = (rng.normal(size=H.shape) + 1j * rng.normal(size=H.shape)) * np.abs(H).mean() * 10 ** (-snr_db / 20)
    H += n / np.sqrt(2)
    return np.abs(H).astype(np.float32), np.angle(H).astype(np.float32)


def peak_velocity(x, v_max, two_sided=True):
    prof = x.mean(axis=(0, 2))                                               # average over pairs and time
    grid = np.linspace(-v_max, v_max, len(prof)) if two_sided else np.linspace(0, v_max, len(prof))
    return grid[np.argmax(prof)]


def test_velocity_recovers_rate_and_cancels_cfo():
    for rate in (0.6, -1.2, 1.8):
        amp, pha = synth(rate)
        x = velocity(amp, pha, v_max=2.5, n_v=101)
        v = peak_velocity(x, 2.5)
        assert abs(v - (-rate)) < 0.15, (rate, v)
        print(f"  rate {rate:+.1f} m/s -> velocity peak {v:+.2f} m/s (expected {-rate:+.2f})")


def test_velocity_amp_recovers_speed_magnitude():
    amp, pha = synth(1.2)
    x = velocity_amp(amp, v_max=2.5, n_v=51)
    v = peak_velocity(x, 2.5, two_sided=False)
    assert abs(v - 1.2) < 0.2, v
    print(f"  amplitude-only: |v| peak {v:.2f} m/s (expected 1.20)")


def test_shapes():
    amp, pha = synth(0.5, T=250)
    assert raw_frame(amp[:10], pha[:10]).shape == (9, 114, 10)
    assert raw_window(amp, pha).shape == (9, 38, 32)
    print("  shapes ok")


if __name__ == "__main__":
    test_velocity_recovers_rate_and_cancels_cfo()
    test_velocity_amp_recovers_speed_magnitude()
    test_shapes()
    print("all tests passed")
