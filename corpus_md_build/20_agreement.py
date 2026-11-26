"""Agreement between reviewer 1 and reviewer 2 on R001-R050. Usage: python 20_agreement.py (after data/review/review_sheet_reviewer2.csv is filled)."""
import csv, collections as C
r1 = {r["need_id"]: r for r in csv.DictReader(open("data/review/review_verdicts_full.csv", encoding="utf-8-sig"))}
rom = {r["need_id"]: r["verdict"] for r in csv.DictReader(open("data/review/romanized_review_results.csv", encoding="utf-8-sig"))}
r2 = {r["need_id"]: r for r in csv.DictReader(open("data/review/review_sheet_reviewer2.csv", encoding="utf-8-sig")) if r["bangla_natural_yes_no"].strip()}
first = {n: {"bangla_natural": r["bangla_natural"], "bangla_answerable": r["bangla_answerable"], "mixed_natural": r["mixed_natural"], "mixed_answerable": r["mixed_answerable"],
             "romanized_natural": "no" if rom[n] == "change" else "yes", "romanized_answerable": r["romanized_answerable"], "same_meaning": r["same_meaning_all_three"]} for n, r in r1.items()}
map2 = {"bangla_natural": "bangla_natural_yes_no", "bangla_answerable": "bangla_answerable_yes_no", "mixed_natural": "mixed_natural_yes_no", "mixed_answerable": "mixed_answerable_yes_no",
        "romanized_natural": "romanized_natural_as_written_yes_no", "romanized_answerable": "romanized_answerable_yes_no", "same_meaning": "same_meaning_all_three_yes_no"}
def kappa(a, b):
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in ("yes", "no"))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")
print(f"reviewer 2 rows filled: {len(r2)}")
print(f"{'item':22s}{'n':>4s}{'agree':>8s}{'kappa':>8s}{'r1 yes':>8s}{'r2 yes':>8s}")
for k, c2 in map2.items():
    ids = [n for n in r2 if n in first]; a = [first[n][k] for n in ids]; b = [r2[n][c2].strip().lower() for n in ids]
    print(f"{k:22s}{len(ids):4d}{sum(x == y for x, y in zip(a, b)) / len(ids):8.2f}{kappa(a, b):8.2f}{a.count('yes') / len(a):8.2f}{b.count('yes') / len(b):8.2f}")
