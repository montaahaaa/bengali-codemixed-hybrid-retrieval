"""Validity test: can the regrade method find the known answer passage for the 134 drafted Romanized/mixed queries?
Reuses the setup lines of regrade_romanized.py (copied via exec, not imported)."""
import re, csv, sys, numpy as np, torch
src=open('scripts/regrade_romanized.py').read().split('q = list(csv.DictReader')[0]
exec(src)
needs=list(csv.DictReader(open('data/annotations/pilot/draft_batch1/all_needs_for_review.csv',encoding='utf-8-sig')))
pos={p:i for i,p in enumerate(pid)}
res={}
model = CrossEncoder("BAAI/bge-reranker-v2-m3", device="mps" if torch.backends.mps.is_available() else "cpu", max_length=320, activation_fn=torch.nn.Sigmoid(), model_kwargs={"torch_dtype": torch.float16})
def pseudo(text,K=1):
    out=[]
    for w in dict.fromkeys(tok(text)):
        if re.fullmatch(r"[a-z]+",w) and len(w)>=3: out+=look.get(skel_lat(w),[])[:K] or []
        elif not re.fullmatch(r"[a-z]+",w): out.append(w)
    return " ".join(out)
for form in ('q_romanized','q_mixed'):
    inidx=0; hit=0; r1=[]; r2=[]; n=0
    for nd in needs:
        p=nd['passage_id']
        if p not in pos: continue
        inidx+=1
        toks=[vocab[x] for x in expand(nd[form]) if x in vocab]
        if not toks: continue
        s=np.asarray(X[:,toks].sum(axis=1)).ravel(); top=np.argpartition(-s,30)[:30]; top=top[np.argsort(-s[top])]
        if pos[p] in top: hit+=1
        r1.append(float(model.predict([(nd[form],nd['passage_text'])])[0]))
        r2.append(float(model.predict([(pseudo(nd[form]),nd['passage_text'])])[0]))
    print(form,'| source passage in BM25 index:',inidx,'of',len(needs),'| in BM25 top-30:',hit,'| rerank raw median %.3f, pseudo-Bangla median %.3f, pseudo>=0.2: %d of %d'%(np.median(r1),np.median(r2),sum(x>=.2 for x in r2),len(r2)))
