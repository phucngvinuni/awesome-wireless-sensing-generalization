# Pilot: in-context adaptation for cross-domain WiFi sensing

Tests whether a model that reads unlabeled CSI from the deployment site (its "context")
can adapt in one forward pass, without gradient updates, and beat test-time adaptation.

**Status: negative on the testbeds available here.** The context models do not use the domain:
feeding context from the *wrong* domain gives the same accuracy.

## Setups

| Script | Data | Protocol |
|---|---|---|
| `icl.py` | UT-HAR <-> NTU-Fi HAR (SenseFi), rate-matched to 1 s @ 62.5 Hz | Train on one dataset, test on the other, 3 shared classes; virtual-domain episodes |
| `icl_widar.py` | Widar3.0 BVP (SenseFi `Widardata`), 6 standard gestures, 440 (user, location, orientation) domains | `locori`: train loc 1-4 x ori 1-4, test loc 5 or ori 5; `user`: train users 1-9, test users 10-17 |

Data prep: `prep_widar.py` (needs `data/Widardata` from the SenseFi Drive folder) and `../wifi-sim2real-pilot/prep_real.py`.

## Results (balanced accuracy %, mean ± std over 3 seeds)

**UT-HAR <-> NTU-Fi (chance 33.3):** every method is at or below chance in both directions
(ERM 34.4 / 37.4, BN-Adapt 25.9 / 23.9, TENT 34.0 / 23.2, ARM 33.3 / 30.6, cross-attention 33.3 / 30.1).
Two domains that differ in everything give no signal to adapt from.

**Widar3.0 BVP (chance 16.7), tag `widar_v2` for the context models:**

| Protocol | ERM | BN-Adapt | TENT | ARM | Cross-attention | Cross-attention, wrong-domain context |
|---|---|---|---|---|---|---|
| Held-out location/orientation | 60.2 ± 0.2 | 61.4 ± 0.2 | 60.0 ± 0.3 | 59.3 ± 1.5 | 61.7 ± 0.2 | 61.9 ± 0.2 |
| Held-out users | 78.7 ± 0.5 | 79.6 ± 0.1 | 79.4 ± 0.1 | 70.8 ± 0.9 | 71.1 ± 1.4 | 71.7 ± 1.9 |

Rows tagged `widar` in `icl_widar_results.jsonl` for arm/xattn come from a buggy evaluation
(query chunks taken in gesture-sorted order, so context held the other gestures); ignore them.

## Caveats

- BVP is designed to remove location/orientation/environment information, so it may leave
  little domain signal to identify. Raw CSI with a large domain gap (e.g. MM-Fi E04) is the real test.
- Episodic training for context models used small per-domain batches; not tuned.
