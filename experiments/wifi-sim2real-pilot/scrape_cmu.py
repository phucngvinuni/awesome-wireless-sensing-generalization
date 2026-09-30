import re, urllib.request, concurrent.futures as cf, html
def fetch(s):
    url=f"http://mocap.cs.cmu.edu/search.php?subjectnumber={s}&motion=%25%25%25&maincat=%25&subcat=%25&subjectnumber={s}"
    for _ in range(3):
        try:
            t=urllib.request.urlopen(url,timeout=60).read().decode('latin1'); break
        except Exception as e: t=None
    if t is None: return s,[]
    txt=re.sub(r'\s+',' ',html.unescape(re.sub(r'<[^>]*>',' ',t)))
    txt=txt.split('Motion Description',1)[-1]
    rows=re.findall(r'(\d+) (.+?) (?:tvd )?(?:c3d )?amc (?:mpg )?(?:Animated )?(\d+) Feedback',txt)
    return s,rows
out=[]
with cf.ThreadPoolExecutor(8) as ex:
    for s,rows in ex.map(fetch,range(1,145)):
        for tr,desc,fr in rows: out.append(f"{s:02d}\t{int(tr):02d}\t{fr}\t{desc.strip()}")
open("cmu_index.tsv","w").write("\n".join(out)+"\n"); print(len(out),"trials")
