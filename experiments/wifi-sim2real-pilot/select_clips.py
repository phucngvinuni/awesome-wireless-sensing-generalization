import re, json
rows=[l.rstrip('\n').split('\t') for l in open('cmu_index.tsv') if l.strip()]
def find(pat, exclude=None, maxn=40):
    out=[]
    for s,t,fr,d in rows:
        if re.search(pat,d,re.I) and not (exclude and re.search(exclude,d,re.I)):
            out.append((s,t,int(fr),d))
    return out[:maxn]
# segment modes: 'random' window anywhere; 'descent' = window centred on fastest root drop;
# 'ascent' = fastest root rise; 'bend' = min head height; 'reverse_ascent' = time-reversed get-up
spec = {
 'walk':   [(c,'random') for c in find(r'^(walk|walking|slow walk|normal walk|walk forward)$', maxn=40)],
 'run':    [(c,'random') for c in find(r'^(run|jog|run/jog|running|jogging|slow jog|run forward)$', maxn=40)],
 'fall':   [(c,'descent') for c in find(r'fall on face|RugPullFall|90TwistsFall|stumble|fall$', exclude=r'footfall', maxn=20)],
 'liedown':[(c,'descent') for c in find(r'^lay down$|laying down|lay down and get up|Laying down and getting up')] +
           [(c,'reverse_ascent') for c in find(r'get up from (the )?(floor|ground)|Get Up Face Down|Get Up Laying|Getting up from laying')],
 'sitdown':[(c,'descent') for c in find(r'sit on (high )?stool|sit in chair|sit down and get up|sit on stool and get up')],
 'standup':[(c,'ascent') for c in find(r'sit on (high )?stool, stand up|get up from chair|sit in chair and get up|sit down and get up|sit on stool and get up')],
 'pickup': [(c,'bend') for c in find(r'pick up|picking up|bend, pick')],
 'box':    [(c,'random') for c in find(r'^boxing$|punch')],
 'clean':  [(c,'random') for c in find(r'sweep|mop|vacuum')],
 'circle': 'procedural',
}
for k,v in spec.items():
    print(k, 'procedural' if v=='procedural' else len(v), '' if v=='procedural' else [f"{c[0]}_{c[1]}" for c,_ in v][:12])
json.dump({k:(v if v=='procedural' else [[c[0],c[1],c[2],c[3],m] for c,m in v]) for k,v in spec.items()}, open('clips.json','w'), indent=0)
