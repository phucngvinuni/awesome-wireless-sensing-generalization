import json, collections, numpy as np
rows = [json.loads(l) for l in open("results.jsonl") if l.strip()]
rows = [r for r in rows if r.get("tag") != "smoke"]
g = collections.defaultdict(list)
for r in rows:
    key = (r["target"], len(r["classes"]), r["train"], r["mode"], r["norm"], r.get("tag") or ("v1" if r["train"] == "sim" else "-"))
    g[key].append(r)


def ms(v):
    v = np.array(v) * 100
    return f"{v.mean():5.1f} ± {v.std():4.1f}" if len(v) > 1 else f"{v.mean():5.1f}"


print(f"{'target':6} {'#cls':4} {'train':9} {'mode':12} {'norm':9} {'sim':3} {'n':>2} | {'test acc':>12} {'test bal-acc':>12} {'test macroF1':>12} | {'ALL-real bal-acc':>16}  majority")
for key in sorted(g):
    rs = g[key]
    t = [r["test"] for r in rs]
    allr = [r["all_real"]["bal_acc"] for r in rs if "all_real" in r]
    print(f"{key[0]:6} {key[1]:4} {key[2]:9} {key[3]:12} {key[4]:9} {key[5]:3} {len(rs):>2} | {ms([x['acc'] for x in t]):>12} {ms([x['bal_acc'] for x in t]):>12} {ms([x['macro_f1'] for x in t]):>12} | {ms(allr) if allr else '-':>16}  {rs[0]['majority_test']*100:.1f}")
