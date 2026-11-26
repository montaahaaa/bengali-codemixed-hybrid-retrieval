"""Automatic review of the 402 drafted queries (model check, NOT a human review).
Scores (query, source passage) with bge-reranker-v2-m3 for every form; flags weak answers and forms that disagree."""
import csv, torch, collections, json
from sentence_transformers import CrossEncoder
D='data/annotations/pilot/draft_batch1/'
rd=lambda p:list(csv.DictReader(open(p,encoding='utf-8-sig')))
needs=rd(D+'all_needs_for_review.csv')
dev='mps' if torch.backends.mps.is_available() else 'cpu'
m=CrossEncoder("BAAI/bge-reranker-v2-m3",device=dev,max_length=320,activation_fn=torch.nn.Sigmoid(),model_kwargs={"torch_dtype":torch.float16} if dev=='mps' else {})
pairs=[];idx=[]
for i,n in enumerate(needs):
    for k in('q_bangla','q_romanized','q_mixed'): pairs.append((n[k],n['passage_text'])); idx.append((i,k))
sc=m.predict(pairs,batch_size=12,show_progress_bar=False)
S=collections.defaultdict(dict)
for (i,k),s in zip(idx,sc): S[i][k]=float(s)
out=[]
for i,n in enumerate(needs):
    s=S[i]; lo=min(s.values()); hi=max(s.values()); notes=[]
    if s['q_bangla']<0.30: notes.append('bangla form weakly matches passage')
    if s['q_romanized']<0.15: notes.append('romanized form weakly matches passage')
    if s['q_mixed']<0.20: notes.append('mixed form weakly matches passage')
    if hi-lo>0.5: notes.append('forms disagree (score gap %.2f)'%(hi-lo))
    out.append(dict(need_id=n['need_id'],domain=n['domain'],passage_id=n['passage_id'],score_bangla=round(s['q_bangla'],3),score_romanized=round(s['q_romanized'],3),score_mixed=round(s['q_mixed'],3),auto_flags='; '.join(notes)))
w=csv.DictWriter(open(D+'auto_review_scores.csv','w',newline='',encoding='utf-8-sig'),fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
fl=[o for o in out if o['auto_flags']]
print(len(out),'needs;',len(fl),'flagged')
import statistics as st
for k in('score_bangla','score_romanized','score_mixed'): print(k,'median',round(st.median(o[k] for o in out),3),'min',min(o[k] for o in out))
for o in fl[:40]: print(o['need_id'],o['domain'],o['score_bangla'],o['score_romanized'],o['score_mixed'],o['auto_flags'])
