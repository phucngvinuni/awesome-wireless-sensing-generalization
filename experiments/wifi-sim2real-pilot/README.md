# Pilot: can WiFi activity recognition be trained purely in simulation?

A small CPU-only pilot of the idea *"Randomize the Radio, Not the Room"*: train WiFi
human-activity models only on simulated CSI (CMU motion capture rendered through cheap
propagation physics plus a randomized radio front-end), then test on real datasets.

**Status: negative for the core hypothesis (at pilot scale).** Sim-only models are near chance
on 7 activities. On 3 shared activities, the physical front-end beats clean simulation by ~10
points, but plain AWGN and unstructured noise give the same or larger gain, so the lift is generic
noise regularization, not physical front-end structure. This is a pilot, not evidence for a paper.

## What is here

| File | Purpose |
|---|---|
| `scrape_cmu.py`, `select_clips.py`, `clips.json`, `cmu_index.tsv` | CMU mocap index and the activity → clip mapping (with segmentation rule per class) |
| `asfamc.py` | Minimal CMU ASF/AMC parser + forward kinematics (metres, z-up) |
| `sim.py` | Simulator: body point scatterers, room multipath (image method, furniture), LoS shadowing, Intel 5300 / Atheros subcarrier formats, and the four front-end conditions (`clean`, `awgn`, `unstructured`, `phys`) |
| `gen.py` | Generates simulated complex CSI per target format |
| `prep_real.py` | Converts SenseFi UT-HAR / NTU-Fi HAR to a common (250 time × 3 antennas × 30 subcarriers, dB) format |
| `train.py` | Trains on sim or real, evaluates on the real test split (acc, balanced acc, macro-F1) |
| `aggregate.py`, `results.jsonl` | Result table / raw results |
| `diag.py`, `simsim.py`, `simsim_dfs.py` | Diagnostics: per-class statistics, sim→sim learnability (amplitude and Doppler inputs) |
| `run_*.sh` | Experiment queues |

## Data (not committed)

- UT-HAR, NTU-Fi HAR: SenseFi benchmark Google Drive folder (`1R0R8SlVbLI1iUFQCzh_mH90H_4CW2iwt`), unzip into `data/`.
- CMU motion capture: downloaded per clip from mocap.cs.cmu.edu into `cmu/` (see `select_clips.py`).

## Reproduce

```bash
pip install numpy scipy torch gdown
python scrape_cmu.py && python select_clips.py      # then download clips (see chat log / select_clips.py)
python prep_real.py
python gen.py ut 500 21 simdata/ut
bash run_real.sh; bash run_v3.sh; bash run_controls.sh
python aggregate.py
```

## Results (balanced accuracy on the real test split, mean ± std over 3 seeds)

**Real baselines**

| Train → test | Activities | Bal. acc. | Chance |
|---|---|---|---|
| UT-HAR → UT-HAR | 7 | 98.1 ± 0.5 | 14.3 |
| NTU-Fi → NTU-Fi | 6 | 99.6 ± 0.0 | 16.7 |
| NTU-Fi (Atheros) → UT-HAR (Intel 5300) | 3 (fall/run/walk) | 36.2 ± 1.7 | 33.3 |
| UT-HAR → NTU-Fi | 3 | 33.3 ± 0.0 | 33.3 |

Real cross-dataset / cross-chipset transfer is at chance.

**Simulation-only → real UT-HAR** (simulator v3, sanitized input)

| Front-end condition | 7 activities | 3 activities |
|---|---|---|
| clean | 19.0 ± 1.0 | 37.5 ± 1.7 |
| AWGN only | 17.9 ± 0.4 | 47.4 ± 2.4 |
| unstructured noise, matched energy | 19.5 ± 1.1 | 50.1 ± 1.1 |
| physical front-end | 17.7 ± 1.0 | 47.8 ± 2.0 |

Chance: 14.3 (7 activities), 33.3 (3 activities). The physical front-end is indistinguishable
from generic noise in both settings.

**Diagnostics**: the simulator's own classes are only ~30% separable (sim→sim, 7 classes,
amplitude CSI); ~40–47% with a Doppler (antenna conjugate-product STFT) representation.

## Caveats

- Simulator was iterated three times (v1→v3) while looking at real data: window durations
  (from dataset documentation), room multipath richness, and input normalization (chosen on
  sim→sim). Per-class real statistics were inspected once during diagnosis — a small leakage risk.
- Only one real test dataset for sim-only runs (UT-HAR); NTU-format sim data was not generated for v3.
- Stick-figure body with 80 random scatterers, no self-occlusion; amplitude-only CSI.

## Lessons from other simulators' code

- **RF-Genesis** (SenSys'23, mmWave): renders the SMPL mesh from the radar's viewpoint (Mitsuba);
  only *visible* surfaces reflect, with Lambertian intensity; ~16k points per frame.
- **Vid2Doppler** (CHI'21): skips raw-signal simulation; synthesizes the Doppler spectrum directly as a
  histogram of radial velocities of *visible* mesh vertices, discards near-zero (static) bins,
  normalizes and blurs. Suggests simulating the domain-robust *feature* (DFS/BVP for WiFi), not raw CSI.
