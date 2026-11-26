"""Fix data/benchmark_full in place (original files backed up to _backup_2026-11-15/).
1. queries.csv keeps only answerable, owner-confirmed rows + MIRACL + the drafted queries.
2. Everything else moves to queries_excluded.csv with a reason (nothing is lost).
3. Drafted queries (402, llm_draft, unreviewed) are merged in with qrels, passages, splits.
4. Adds columns: track, tier, review_status, form_label, split (leak-free split_v2; old one kept as split_old)."""
import csv, json, random, collections, os, re, shutil
B='data/benchmark_full/'; BK=B+'_backup_2026-11-15/'
rd=lambda p:list(csv.DictReader(open(p,encoding='utf-8-sig')))
def wr(p,rows,cols):
    w=csv.DictWriter(open(p,'w',newline='',encoding='utf-8-sig'),fieldnames=cols,extrasaction='ignore'); w.writeheader(); w.writerows(rows)
if not os.path.exists(BK):
    os.makedirs(BK)
    for f in ('queries.csv','qrels.tsv','passages.csv','benchmark_flat.csv','splits_v2.csv'): shutil.copy(B+f,BK+f)
Q=rd(BK+'queries.csv'); flat={r['qid']:r for r in rd(BK+'benchmark_flat.csv')}; sp={r['qid']:r for r in rd(BK+'splits_v2.csv')}
qrels=collections.defaultdict(list)
for l in open(BK+'qrels.tsv'):
    q,z,p,g=l.split(); qrels[q].append((z,p,g))
passages={r['passage_id']:r for r in rd(BK+'passages.csv')}
label={'bn':'bangla','latin':'latin_unchecked_romanized_or_english','mixed':'mixed'}
keep=[];excl=[];fk=[]
for q in Q:
    f=flat[q['qid']]; s=sp[q['qid']]
    base=dict(qid=q['qid'],split=s['split_v2'],split_old=q['split'],group_id=s['group_id'],domain=q['domain'],script=q['script'],form_label=label[q['script']],source=q['source'],tier=f['tier'],query=q['query'])
    if f['tier']=='miracl': keep.append({**base,'track':'miracl_bn','review_status':'published_human'}); fk.append(f)
    elif f['tier'] in('core','partial') and f['grade_status']=='owner_confirmed' and f['my_check_ok_yes_no'].strip().lower()!='no':
        keep.append({**base,'track':'real_answerable','review_status':'owner_confirmed'}); fk.append(f)
    else:
        why={'unanswered':'no answer in corpus','rejected':'rejected on check','synthetic':'LLM-synthetic (not for headline)'}.get(f['tier'],'answer not confirmed by owner (auto_unchecked)')
        excl.append({**base,'reason':why})
# ---- merge drafted queries
drafts=rd('data/annotations/pilot/draft_batch1/queries_long.csv')
needs={r['need_id']:r for r in rd('data/annotations/pilot/draft_batch1/all_needs_for_review.csv')}
par={}
def fd(x):
    while par.setdefault(x,x)!=x: par[x]=par[par[x]]; x=par[x]
    return x
for n,r in needs.items(): par[fd(n)]=fd('P:'+r['passage_id'])
random.seed(42); gs=sorted({fd(n) for n in needs}); random.shuffle(gs); dev=set(gs[:int(.3*len(gs))])
text={}
for fn in ('corpus_bn','corpus_en','corpus_wikibooks','corpus_voyage'):
    for l in open(f'data/corpus/{fn}.jsonl'):
        m=re.search(r'"passage_id": "([^"]+)"',l)
        if m and m.group(1) in {r['passage_id'] for r in needs.values()}: text[m.group(1)]=json.loads(l)
dq=[];new_qr=[];new_fk=[];nq=max(int(q['qid'][1:]) for q in Q)
for i,d in enumerate(drafts,1):
    qid=f'Q{nq+i:05d}'; g=fd(d['need_id']); pid=d['passage_id']; t=text[pid]
    dq.append(dict(qid=qid,split='dev' if g in dev else 'test',split_old='',group_id=d['need_id'],domain=d['domain'].lower().replace(' ','_'),script=d['detected_script'] if d['detected_script']!='latin' else 'latin',form_label={'bangla':'bangla','romanized':'romanized','mixed':'mixed'}[d['form']],source='llm_draft',tier='drafted',query=d['query'],track='drafted_cm',review_status='unreviewed'))
    new_qr.append((qid,'0',pid,'2'))
    if pid not in passages: passages[pid]=dict(passage_id=pid,title=t['title'],text_first_700_chars=t['text'][:700],licence_note=t.get('licence','CC BY-SA 4.0'))
    new_fk.append(dict(qid=qid,split=dq[-1]['split'],tier='drafted',domain=dq[-1]['domain'],script=dq[-1]['script'],source='llm_draft',query=d['query'],grade_status='llm_draft_unreviewed',grade='2',n_relevant='1',passage_id=pid,title=t['title'],passage_text=t['text']))
keep+=dq; fk+=new_fk
cols=['qid','split','group_id','domain','script','form_label','source','track','tier','review_status','query','split_old']
wr(B+'queries.csv',keep,cols)
wr(B+'queries_excluded.csv',excl,['qid','split','group_id','domain','script','form_label','source','tier','reason','query','split_old'])
kq={r['qid'] for r in keep}
with open(B+'qrels.tsv','w') as o:
    for q in Q:
        if q['qid'] in kq:
            for z,p,g in qrels[q['qid']]: o.write(f'{q["qid"]}\t{z}\t{p}\t{g}\n')
    for q,z,p,g in new_qr: o.write(f'{q}\t{z}\t{p}\t{g}\n')
used={l.split()[2] for l in open(B+'qrels.tsv')}
wr(B+'passages.csv',[r for p,r in passages.items() if p in used],['passage_id','title','text_first_700_chars','licence_note'])
wr(B+'benchmark_flat.csv',fk,list(flat['Q00001'].keys()))
wr(B+'splits_v2.csv',[dict(qid=r['qid'],split_v2=r['split'],group_id=r['group_id']) for r in keep],['qid','split_v2','group_id'])
st=dict(total=len(keep),excluded=len(excl),by_track=collections.Counter(r['track'] for r in keep),by_form=collections.Counter(r['form_label'] for r in keep),by_split=collections.Counter(r['split'] for r in keep),excluded_reasons=collections.Counter(r['reason'] for r in excl))
json.dump(st,open('data/annotations/pilot/benchmark_full_stats.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(st,ensure_ascii=False,indent=1))
