# MM-Fi falsification test: does velocity-space input shrink the cross-environment gap?

One-week test for the "physics-normalized (velocity-space) representation" direction.
**Question:** on MM-Fi leave-one-environment-out, does converting CSI into an antenna-conjugate
Doppler spectrogram in m/s reduce the cross-environment gap compared with raw CSI, with the *same*
model and the *same* temporal context?

**Pre-registered rule (implemented in `summarize.py`):** GO if on E04 (and on the 4-fold average,
if run), velocity vs `raw_window`:
1. reduces the MPJPE gap (target − in-domain) by ≥ 25 %;
2. has a smaller gap than the raw_window mean in *every* seed;
3. is at most 10 % worse in-domain (so the gap doesn't shrink just because the model got worse everywhere).

Otherwise NO-GO. Decide the rule before looking at the numbers, and don't tune it afterwards.

## Files

| File | What it does |
|---|---|
| `velocity.py` | Representations: `raw_frame` (SemPose-Fi style, one frame), `raw_window` (same features over the velocity window, the **control**), `velocity` (antenna-conjugate Doppler, m/s), `velocity_amp` (amplitude-only Doppler) |
| `mmfi_seq.py` | Stitches a sequence's 297 `frameNNN.mat` files into one 100 Hz stream (3 × 114), fills empty/NaN frames like the official loader, slices per-frame samples |
| `inspect_mmfi.py` | **Run first.** Checks keys/shapes, phase key, empty-frame rate, ground truth, and CSI continuity across frame boundaries |
| `loeo.py` | Leave-one-environment-out runner (pose + action jointly, identical CNN for every representation) |
| `summarize.py` | Result table + GO / NO-GO verdict |
| `test_velocity.py` | Synthetic unit tests (passes: recovers ±0.6/1.2/1.8 m/s through random per-packet phase offsets) |

## Run

```bash
pip install numpy scipy torch
python test_velocity.py
python inspect_mmfi.py --root /data/MMFi            # confirm assumptions (see below)
for r in raw_window velocity raw_frame velocity_amp; do
  python loeo.py --root /data/MMFi --rep $r --target E04 --seeds 0 1 2
done
python summarize.py results.jsonl
# then, if E04 looks promising, all four folds:
for r in raw_window velocity; do python loeo.py --root /data/MMFi --rep $r --target all --seeds 0 1 2; done
```

`--frame_stride 3` (default) uses every third frame for speed; use `1` for the final numbers.
`--protocol P1|P2|P3` selects MM-Fi's action subsets (default P3, all 27).

## Plugging into SemPose-Fi

SemPose-Fi's encoder takes a `C × F × T` tensor, so the change is just the preprocessing step `Pre(H)`:

```python
# before (SemPose-Fi): per frame, H = [z(amp), cos(phase_det), sin(phase_det)]  -> (9, 114, 10)
# after:
from mmfi_seq import load_sequence, SeqFeatures
amp, pha, valid, pose = load_sequence(seq_dir)            # (2970, 3, 114) stitched stream
feats = SeqFeatures(amp, pha, rep="velocity", ctx_frames=16)
x_i = feats.sample(i)                                      # (2, 64, 16): antenna pairs x velocity bins x time
# feed x_i where SemPose-Fi fed H; set C=2, F=64, T=16 (N = ceil(64/4)*ceil(16/4) = 64 VQ tokens)
```

Run SemPose-Fi twice, once with `rep="raw_window"` and once with `rep="velocity"`, source-only and with
TTVQ-A, E01–E03 → E04. If velocity also shrinks the gap *before* TTVQ-A, that is the result the
new direction needs; if TTVQ-A closes the same gap either way, velocity space adds little.

## Assumptions to verify (inspect_mmfi.py)

- **Phase is stored** (key `CSIphase` or similar; SemPose-Fi uses phase, so it should be). Without phase, only `velocity_amp` works.
- **CSI is continuous across frames.** `velocity` windows span ~2.2 s (16 frames + STFT length). If the inspector reports large jumps at frame boundaries, velocity windows mix discontinuous data; results would be invalid.
- **Packet rate ≈ 100 Hz** (10 packets per ~0.1 s frame). The velocity axis assumes `fs=100`; wrong fs rescales it.
- **Carrier** `--fc 5.32e9` is a placeholder; within MM-Fi it only rescales the velocity axis, so it doesn't affect this test. It matters for cross-dataset pooling later.

## Reading notes on the two overlapping papers

**Torun, Cuenca & Mostofi, "Untangling the Geometry and Speed for RF Sensing Spectrograms", arXiv:2609.26960 (Sept 2026)** — read in full.
- Single-link Doppler: `f_D = ψ(t)·v(t)/λ` with geometry factor `ψ = |(u_tx + u_rx)ᵀ h| ∈ [0, 2]`; speed and geometry are entangled ("product ambiguity").
- Parametric spectrogram: sum of ridges `a_m K_w(f − ψ_m v_m/λ)`; a physics-informed autoencoder with a *fixed differentiable RF decoder* recovers speed, ψ, amplitude and width. Trained on synthetic walking, tested on 31 real WiFi experiments (2.4 GHz, one Tx–Rx pair); real torso-speed MAE 0.125 m/s vs 0.238 best baseline.
- No cross-environment HAR/pose, single link, walking only.
- **Implication for us:** our "velocity" axis is really ψ·v. It removes device, carrier, sample-rate and static-multipath differences, but not person-to-link geometry. On MM-Fi (same device, different rooms), any gain must come from static-multipath removal. A later step could add their ψ disentanglement.

**"Transferability Assessment of WiFi Sensing through CSI-based Doppler", ACM (2026), doi:10.1145/3743158.3783858** — full text paywalled from our side; abstract only.
- People counting; MLP and pretrained ResNet on raw CSI vs Doppler spectrum features across environments; Doppler + ResNet reaches up to 98 % in some case studies.
- **Implication:** "Doppler helps cross-environment" is already claimed for counting. Our contribution must be pretraining at scale in physical units plus held-out rooms/devices on HAR and pose. Read the full paper for its exact Doppler pipeline and protocol before writing related work.

## How to read the outcome

- **GO:** velocity space reduces the room gap with a fair context-matched control. Proceed to a second phase-bearing dataset (XRF55 or Widar3.0 raw) and cross-dataset pooling.
- **NO-GO:** the representation doesn't carry the room invariance BVP showed on Widar3.0 (whose invariance may come from multi-link geometry, not Doppler alone). Consider the fallback (SemPose-Fi + physics codebook) or stop.
- If `raw_frame` beats `raw_window`, longer context hurts raw CSI. Note it, but the decision is velocity vs raw_window.
