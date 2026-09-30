# Pilot: in-context adaptation for cross-domain WiFi sensing

Tests whether a model that reads unlabeled CSI from the deployment site (its "context")
can adapt in one forward pass, without gradient updates, and beat test-time adaptation.

**Status: negative on all three testbeds available here (UT-HAR/NTU-Fi, Widar3.0 BVP, CSIDA).** The context models do not use the domain:
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

**CSIDA raw CSI amplitude (chance 16.7), 25 (room, location, user) domains:**

| Protocol | ERM | BN-Adapt | TENT | ARM | Cross-attention | Cross-attention, wrong-domain context |
|---|---|---|---|---|---|---|
| Room A -> B | 16.7 ± 0.1 | 17.8 ± 0.1 | 18.6 ± 1.6 | 15.3 ± 0.7 | 15.6 ± 1.0 | 15.6 ± 0.8 |
| Room B -> A | 16.7 ± 0.0 | 14.5 ± 0.3 | 15.5 ± 1.0 | 16.5 ± 0.2 | 16.6 ± 0.2 | 16.5 ± 0.3 |
| Users 0-2 -> 3-4 | 21.1 ± 1.3 | 19.4 ± 0.7 | 19.5 ± 0.6 | 17.5 ± 1.2 | 18.7 ± 1.5 | 19.1 ± 1.3 |

Sanity check: random in-domain 80/20 split, ERM = 76.4 (1 seed), so the pipeline learns;
cross-room and cross-user transfer is at chance for every method.
Data: `CSIDA-1.zip` from Mendeley Data (gyr6c4nbsc), decoded by `prep_csida.py`.

## Caveats

- BVP is designed to remove location/orientation/environment information, so it may leave
  little domain signal to identify. Raw CSI with a large domain gap (e.g. MM-Fi E04) is the real test.
- Episodic training for context models used small per-domain batches; not tuned.
