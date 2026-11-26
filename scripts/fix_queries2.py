"""Second fix pass on data/benchmark_full/queries.csv (run after fix_queries.py). Rows are never deleted.
Adds: latin_kind (romanized_bn / english / ambiguous), query_type (question / request / statement),
multi_question, domain relabel for general/unassigned real queries (domain_old, domain_confidence),
emoji stripped (raw kept in query_raw), eval_ok + eval_reason, and rebuilds the dev/test split."""
import csv, re, json, random, collections
B='data/benchmark_full/'
rd=lambda p:list(csv.DictReader(open(p,encoding='utf-8-sig')))
Q=rd(B+'queries.csv'); old_sp={r['qid']:r for r in rd(B+'splits_v2.csv')}
EMOJI=re.compile('[\U0001F000-\U0001FAFF☀-➿⬀-⯿️‍]')
# ---- Latin kind
BN=set('ami apni tumi ki kore kori korbo korte hobe hoy hoye ache achhe nai nei na kivabe kibhabe keno kothay koto kobe kokhon jabe jay jabo dite dibo pari parbo parina lagbe lage ekta ekhon onek amar amake apnar tar tader ta er theke jonno jonne bhai vai bhaiya apu sir bolen bolun bolte jante chai chaichi hocche hoyeche hoise holo dekhai dekhacche pabo pabe pai paini nibo nite niye giye gele ashe ase asche kemon kemne ei oi ota eta kono kichu shob sob ar o abar tahole tobe kintu jodi tai kaj kajer taka din mas bochor bar'.split())
EN=set('the is are to of and in for you i it this that with on my how what can do not have be your me we they will was at from a an or if when please help need want get use does did would should could any there their which who why where about'.split())
def latin_kind(t):
    w=re.findall(r"[a-z']+",t.lower()); b=sum(x in BN for x in w); e=sum(x in EN for x in w)
    if b>=2 and b>=e: return 'romanized_bn'
    if e>=3 and e>b*2: return 'english'
    if b>=1 and b>=e: return 'romanized_bn'
    if e>=2 and b==0: return 'english'
    return 'ambiguous'
# ---- query type
QCUE=re.compile(r'[?]|কি\b|কী|কেন|কিভাবে|কীভাবে|কি ভাবে|কোথায়|কত|কখন|কোন|কিসের|কাকে|কার\b|কেমন|করব|করবো|করতে হবে|পারব|পারবো|যাবে|হবে|জানতে চাই|জানাবেন|বলবেন|বলুন|\b(ki|kivabe|kibhabe|keno|kothay|koto|kokhon|kon|kar|kemon|how|what|why|where|when|can|is|does|do|should|will)\b|\bkorbo\b|\bhobe\b|\bjabe\b|\bparbo\b|\bbolben\b|\bjante chai\b',re.I)
REQ=re.compile(r'বানাবেন|বানান|দেন|দিন|দিয়েন|দিবেন|করুন|করেন|please|plz|pls|make a video|video',re.I)
def qtype(t):
    if QCUE.search(t): return 'question'
    if REQ.search(t): return 'request'
    return 'statement'
# ---- domain
KW={'health':'জ্বর ডেঙ্গু ঔষধ ওষুধ ডাক্তার হাসপাতাল রোগ ব্যথা সর্দি কাশি ক্যান্সার ডায়াবেটিস চিকিৎসা টিকা গর্ভ প্রেসক্রিপশন শিশু ভাইরাস ঘুম ব্লাড প্রেসার jor dengue doctor medicine ushudh tika pain fever hospital',
'fees':'টাকা ফি খরচ বিল দাম ভাড়া ব্যাংক লোন সুদ বিকাশ নগদ ট্যাক্স ভ্যাট ইনকাম রিটার্ন ডিপোজিট একাউন্ট অ্যাকাউন্ট বেতন taka fee bill loan bkash nagad tax bank dam price cost salary deposit',
'procedures':'আবেদন নিবন্ধন রেজিস্ট্রেশন রেজিট্রেশন পাসপোর্ট ভিসা লাইসেন্স এনআইডি সনদ ফরম জমা চালান নামজারি পরিচয়পত্র ভোটার টিআইএন apply registration passport visa license nid challan form submit tin',
'education':'ভর্তি পরীক্ষা স্কুল কলেজ বিশ্ববিদ্যালয় এসএসসি এইচএসসি বৃত্তি স্কলারশিপ রেজাল্ট ক্লাস শিক্ষক কোর্স অনার্স hsc ssc admission exam result scholarship university college study',
'tech':'ফোন মোবাইল ইন্টারনেট ওয়াইফাই অ্যাপ অ্যান্ড্রয়েড কম্পিউটার ল্যাপটপ ফেসবুক ইউটিউব মনিটাইজেশন মনিটেশন সফটওয়্যার পাসওয়ার্ড এডিট সিম অ্যাকাউন্ট wifi phone mobile app laptop facebook youtube monetization software password internet android',
'local_services':'বিদ্যুৎ গ্যাস পানি ওয়াসা বাস ট্রেন রেল থানা পুলিশ ইউনিয়ন পৌরসভা ডাকঘর হোটেল রিকশা সিএনজি মেট্রো টিকিট bus train metro hotel police electricity gas wasa ticket'}
KWL={d:[k for k in v.split()] for d,v in KW.items()}
def dom(t):
    tl=t.lower(); s={}
    for d,ks in KWL.items():
        n=0
        for k in ks:
            if re.match(r'^[a-z]+$',k): n+=bool(re.search(r'\b'+k+r'\b',tl))
            else: n+=k in tl
        s[d]=n
    top=sorted(s.items(),key=lambda x:-x[1]); a,b=top[0],top[1]
    if a[1]>=2 and a[1]>b[1]: return a[0],'high'
    if a[1]==1 and b[1]==0: return a[0],'medium'
    return None,'low'
chg=collections.Counter(); out=[]
for r in Q:
    real=r['source'] in('youtube','facebook'); t=EMOJI.sub('',r['query']); t=re.sub(r'\s+',' ',t).strip()
    if t!=r['query']: chg['emoji_stripped']+=1; r['flags']=(r['flags']+';emoji_removed').strip(';')
    r['query']=t
    r['latin_kind']=latin_kind(t) if r['script']=='latin' and real else ''
    r['query_type']=qtype(t) if real else ('question' if r['source']!='' else '')
    r['multi_question']='yes' if real and t.count('?')>=2 else 'no'
    r['domain_old']=r['domain']; r['domain_confidence']=''
    if real:
        d,c=dom(t); r['domain_confidence']=c
        if r['domain'] in('general','unassigned') and d: r['domain']=d; chg['domain_relabelled']+=1
        elif r['domain'] in('general','unassigned'): r['domain_confidence']='low'
    why=[]
    if r['source'].startswith('llm_synthetic'): why.append('synthetic')
    if r['answerable']=='no': why.append('no_answer_in_corpus')
    if r['answerable']=='unknown': why.append('answer_status_unknown')
    if r['answerable'].startswith('unconfirmed'): why.append('answer_not_confirmed')
    if r['duplicate_of']: why.append('duplicate')
    if real and r['script']=='latin' and r['latin_kind']=='english': why.append('english_not_bengali')
    if real and r['query_type']=='statement': why.append('not_a_question')
    r['eval_ok']='yes' if not why else 'no'; r['eval_reason']=';'.join(why)
    out.append(r)
# ---- rebuild splits: groups = same answer passage or duplicate; real rows re-split 30/70 stratified by script x answerable
qr=collections.defaultdict(set)
for l in open(B+'qrels.tsv'):
    q,_,p,g=l.split(); qr[q].add(p)
par={}
def f(x):
    while par.setdefault(x,x)!=x: par[x]=par[par[x]]; x=par[x]
    return x
for r in out:
    f(r['qid'])
    if r['duplicate_of']: par[f(r['qid'])]=f(r['duplicate_of'])
    if r['answerable'] in('yes','partial','unconfirmed_yes','unconfirmed_partial') and r['source'] in('youtube','facebook'):
        for p in qr[r['qid']]: par[f(r['qid'])]=f('P:'+p)
random.seed(42); strata=collections.defaultdict(list)
real=[r for r in out if r['source'] in('youtube','facebook')]
gs={}
for r in real: gs.setdefault(f(r['qid']),r)
for g,r in gs.items(): strata[(r['script'],r['answerable'])].append(g)
asg={}
for k,v in strata.items():
    random.shuffle(v); n=int(round(len(v)*0.3))
    for i,g in enumerate(v): asg[g]='dev' if i<n else 'test'
sp_rows=[]
for r in out:
    if r['source'] in('youtube','facebook'): r['split']=asg[f(r['qid'])]
    else: r['split']=old_sp[r['qid']]['split_v2']
    sp_rows.append(dict(qid=r['qid'],split_v2=r['split'],group_id=f(r['qid']) if not f(r['qid']).startswith('P:') else f(r['qid'])[2:]))
cols=['qid','split','domain','domain_old','domain_confidence','script','latin_kind','query_type','multi_question','source','answerable','tier','eval_ok','eval_reason','n_words','duplicate_of','flags','query','query_raw']
w=csv.DictWriter(open(B+'queries.csv','w',newline='',encoding='utf-8-sig'),fieldnames=cols,extrasaction='ignore'); w.writeheader(); w.writerows(out)
w=csv.DictWriter(open(B+'splits_v2.csv','w',newline='',encoding='utf-8-sig'),fieldnames=['qid','split_v2','group_id']); w.writeheader(); w.writerows(sp_rows)
m={r['qid']:r for r in out}; fr=rd(B+'benchmark_flat.csv')
for r in fr: x=m[r['qid']]; r['query']=x['query']; r['split']=x['split']; r['domain']=x['domain']
w=csv.DictWriter(open(B+'benchmark_flat.csv','w',newline='',encoding='utf-8-sig'),fieldnames=list(fr[0].keys())); w.writeheader(); w.writerows(fr)
C=collections.Counter
st=dict(rows=len(out),eval_ok=C(r['eval_ok'] for r in out),eval_ok_by_script=C(r['script'] for r in out if r['eval_ok']=='yes'),eval_reasons=C(x for r in out for x in r['eval_reason'].split(';') if x),latin_kind=C(r['latin_kind'] for r in real if r['latin_kind']),query_type=C(r['query_type'] for r in real),multi_question=C(r['multi_question'] for r in real),domain=C(r['domain'] for r in out),domain_confidence=C(r['domain_confidence'] for r in real),split=C(r['split'] for r in out),changes=chg)
json.dump(st,open('data/annotations/pilot/benchmark_full_stats.json','w'),ensure_ascii=False,indent=1); print(json.dumps(st,ensure_ascii=False,indent=1))
