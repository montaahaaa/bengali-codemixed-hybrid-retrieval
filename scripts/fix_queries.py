"""Fix the 12,831-query file in place (data/benchmark_full/queries.csv). Every row is kept.
Original text -> query_raw. Cleans: invisible chars, whitespace, repeated punctuation, @mentions,
leading greetings. Recomputes script from the characters. Flags duplicates and very long rows.
Reads from _backup_2026-11-15 so it can be re-run safely."""
import csv, re, unicodedata, collections, json
B='data/benchmark_full/'; BK=B+'_backup_2026-11-15/'
rd=lambda p:list(csv.DictReader(open(p,encoding='utf-8-sig')))
Q=rd(BK+'queries.csv'); flat={r['qid']:r for r in rd(BK+'benchmark_flat.csv')}
GREET=re.compile(r'^\s*(?:(?:আসসালামু\s*আলাইকুম|আস্সালামু\s*আলাইকুম|assalamu?\s*alaikum|assalamualaikum|সালাম|হ্যালো|হ্যালো|hello|hi|হাই|নমস্কার)\s*[,।!.\-–:]*\s*)+(?:(?:স্যার|ভাইয়া|ভাই|আপু|আপা|ম্যাডাম|sir|vai|bhai)\s*[,।!.\-–:]*\s*)*',re.I)
def script(t):
    bn=bool(re.search(r'[ঀ-৿]',t)); la=bool(re.search(r'[A-Za-z]',t))
    return 'mixed' if bn and la else 'bn' if bn else 'latin' if la else 'other'
def clean(t):
    fl=[]; o=t
    t=unicodedata.normalize('NFC',t)
    t2=re.sub('[​‌﻿­]','',t)
    if t2!=t: fl.append('invisible_chars_removed')
    t=t2
    t2=re.sub(r'@\S+','',t)
    if t2!=t: fl.append('mention_removed')
    t=t2
    t2=re.sub(r'([?!,।])\1+',r'\1',t); t2=re.sub(r'\.{4,}','...',t2)
    if t2!=t: fl.append('repeated_punct')
    t=t2
    t2=re.sub(r'\s+',' ',t).strip()
    if t2!=t: fl.append('whitespace')
    t=t2
    g=GREET.sub('',t)
    if g.strip() and len(g.split())>=3 and g!=t: t=g.strip(); fl.append('greeting_removed')
    return t,fl
seen={}; out=[]; ch=collections.Counter()
for q in Q:
    raw=q['query']; c,fl=clean(raw)
    key=re.sub(r'\W+',' ',c.lower()).strip()
    dup=seen.get(key,''); seen.setdefault(key,q['qid'])
    if dup: fl.append('duplicate')
    nw=len(c.split())
    if nw>60: fl.append('too_long_over_60_words')
    sc=script(c)
    if sc!=q['script']: fl.append(f"script_relabelled_{q['script']}_to_{sc}")
    f=flat[q['qid']]
    ans={'core':'yes','partial':'partial','miracl':'yes','synthetic':'synthetic'}.get(f['tier'],'no')
    if f['tier'] in('core','partial') and f['grade_status']!='owner_confirmed': ans='unconfirmed_'+ans
    if ans=='no' and q['source'] in('youtube','facebook') and sc in('latin','mixed'): ans='unknown'  # grading could not reach Latin/mixed queries (validated 2026-11-15)
    for x in fl: ch[re.sub(r'_(bn|latin|mixed)_to_(bn|latin|mixed|other)$','',x)]+=1
    out.append(dict(qid=q['qid'],split=q['split'],domain=q['domain'],script=sc,source=q['source'],query=c,query_raw=raw,n_words=nw,answerable=ans,tier=f['tier'],duplicate_of=dup,flags=';'.join(fl)))
cols=list(out[0].keys())
w=csv.DictWriter(open(B+'queries.csv','w',newline='',encoding='utf-8-sig'),fieldnames=cols); w.writeheader(); w.writerows(out)
# keep benchmark_flat consistent (query + script)
fr=rd(BK+'benchmark_flat.csv'); m={o['qid']:o for o in out}
for r in fr: r['query']=m[r['qid']]['query']; r['script']=m[r['qid']]['script']
w=csv.DictWriter(open(B+'benchmark_flat.csv','w',newline='',encoding='utf-8-sig'),fieldnames=list(fr[0].keys())); w.writeheader(); w.writerows(fr)
st=dict(rows=len(out),by_script=collections.Counter(o['script'] for o in out),by_answerable=collections.Counter(o['answerable'] for o in out),by_source=collections.Counter(o['source'] for o in out),fixes=ch)
json.dump(st,open('data/annotations/pilot/benchmark_full_stats.json','w'),ensure_ascii=False,indent=1); print(json.dumps(st,ensure_ascii=False,indent=1))
