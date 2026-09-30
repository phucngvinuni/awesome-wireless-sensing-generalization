"""Aggregate results.jsonl and apply the pre-registered go/no-go rule.

GO if, on E04 (and on the LOEO average when --target all was run), velocity vs raw_window:
  (1) reduces the MPJPE gap (target - in-domain) by >= 25% of raw_window's gap, AND
  (2) every seed of velocity has a smaller gap than the mean raw_window gap, AND
  (3) velocity's in-domain MPJPE is not more than 10% worse than raw_window's
      (so the gap does not shrink merely because in-domain accuracy collapsed).
raw_frame is reported for reference (SemPose-Fi's input); it is NOT the control, because it
sees less temporal context than velocity.
"""
import json, sys, collections
import numpy as np

rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "results.jsonl") if l.strip()]
g = collections.defaultdict(list)
for r in rows:
    g[(r["target"], r["rep"])].append(r)


def ms(v):
    return f"{np.mean(v):7.1f} ± {np.std(v):5.1f}"


print(f"{'target':6} {'rep':13} {'n':>2} | {'in-dom MPJPE':>15} {'target MPJPE':>15} {'GAP':>15} | "
      f"{'target PA':>15} | {'in-dom acc':>13} {'target acc':>13}")
for (t, rep), rs in sorted(g.items()):
    print(f"{t:6} {rep:13} {len(rs):>2} | {ms([r['in_domain']['mpjpe'] for r in rs]):>15} "
          f"{ms([r['target_env']['mpjpe'] for r in rs]):>15} {ms([r['gap_mpjpe'] for r in rs]):>15} | "
          f"{ms([r['target_env']['pa_mpjpe'] for r in rs]):>15} | "
          f"{ms([100*r['in_domain']['bal_acc'] for r in rs]):>13} {ms([100*r['target_env']['bal_acc'] for r in rs]):>13}")


def verdict(targets):
    v = [r for r in rows if r["rep"] == "velocity" and r["target"] in targets]
    w = [r for r in rows if r["rep"] == "raw_window" and r["target"] in targets]
    if not v or not w:
        return "incomplete (need both velocity and raw_window runs)"
    gv = np.array([r["gap_mpjpe"] for r in v]); gw = np.mean([r["gap_mpjpe"] for r in w])
    iv = np.mean([r["in_domain"]["mpjpe"] for r in v]); iw = np.mean([r["in_domain"]["mpjpe"] for r in w])
    if gw <= 5.0:
        return (f"UNDECIDABLE: raw_window shows no cross-environment gap to reduce ({gw:.1f} mm). "
                "Check that training converged and the target environment differs from the sources.")
    c1 = gv.mean() <= 0.75 * gw
    c2 = bool(np.all(gv < gw))
    c3 = iv <= 1.10 * iw
    return (f"{'GO' if (c1 and c2 and c3) else 'NO-GO'}  "
            f"(gap velocity {gv.mean():.1f} vs raw_window {gw:.1f} mm -> {100*(1-gv.mean()/gw):.0f}% reduction; "
            f"all seeds better: {c2}; in-domain {iv:.1f} vs {iw:.1f} mm ok: {c3})")


print("\nE04:", verdict({"E04"}))
if len({r["target"] for r in rows}) == 4:
    print("LOEO average:", verdict({"E01", "E02", "E03", "E04"}))
