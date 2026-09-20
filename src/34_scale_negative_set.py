#!/usr/bin/env python3
"""
34_scale_negative_set.py - harvest a large benign pool, to test a prediction this project has
                           already made and to lift the calibration ceiling it cannot clear.

Two reasons to scale the NEGATIVE side, and one reason not to scale the positive side
-------------------------------------------------------------------------------------
§10.8: the threshold is a quantile of held-out negatives, 40% of 296 is 118, so the finest
resolution available is one negative in 118 and **every specificity above 0.9915 is
extrapolation**. Calibrating a one-in-ten-thousand budget with ten negatives above the threshold
would need about 250,000 negatives. Reviewed Swiss-Prot holds **503,710** entries in the panel's
length window once Toxin, Virulence and Cytolysis keywords are excluded, so the supply is roughly
twice what the ceiling demands. This is the one binding constraint in the project that more data
actually fixes.

🔑 §10.7.1 makes a falsifiable prediction about doing this, which is the scientific reason rather
than the engineering one. The dose-response showed that the two failing classes sit inside a
**dense** benign region: removing the nearest 5, 10, 20, 40, 80 training negatives helped
monotonically, while the recovered comparison class peaked at 20 and fell back. If the failure is
density, then **adding** benign proteins should make those classes worse, and should leave the
recovered classes alone. That is a prediction with a direction, and it is cheap to test.

⚠️ What this deliberately does NOT do: scale the POSITIVE side. There are 6,720 reviewed entries
in the length window carrying the Toxin or Virulence keyword, and using them as positives would
replace 149 curated labels, each with a written reason, by whatever UniProt curators flagged.
This project's own provenance control is the argument against that: a probe that discards the
hazard label entirely and predicts lab-strain origin reaches **AUROC 0.794 +/- 0.062** on v3
(0.818 +/- 0.012 on v2), and the organism label agrees with the hazard label on **43.6%** of v3
(53.4% of v2). At keyword scale the hazard label IS the annotation, so the classifier would learn
the annotation. `§2` exists to keep that separable.

An earlier version of this docstring quoted v2's 0.818 next to v3's 44%, which are figures from two
different panels. Both are in `results/{v2,v3}/lomo_results.json` under `provenance_auroc` and
`organism_label_agreement_with_hazard`, and the audit now pins the v3 pair as well as the v2 one.

Composition is held fixed and only size varies
----------------------------------------------
`03e` separated negative-set **size** from the **decision boundary** in a 2x2 because varying both
at once confounds them. The same discipline here: the pool is sampled from reviewed Swiss-Prot
under one fixed exclusion policy, and the curve subsamples from that single pool, so composition is
constant in expectation across sizes.

⚠️ The pool is **not taxon-matched**, unlike `02d`'s and `27`'s negative blocks. That is deliberate
and it is a change of condition, not an oversight: taxon matching exists to stop provenance being
the signal when the negative set is small and narrow. A comprehensive benign reference set spans
everything, which is the realistic deployment condition, and the existing 296-negative panel stays
as the matched reference point. Results at 296 from this pool and from the panel are therefore not
the same experiment and are reported separately.

Exclusions, following 02d
-------------------------
Query side excludes the Virulence, Toxin, Cytolysis and Hemolysis keywords plus Bacteriocin and
Bacteriolytic enzyme, which `27` added because they define two v3 positive classes. Python side
re-checks the keywords, applies 02d's protein-name blocklist and its positive-class-term blocklist,
dedups on accession and sequence hash against both panels, and caps per organism so no model
organism dominates.

Usage:
    python src/34_scale_negative_set.py --target 12000
    python src/34_scale_negative_set.py --target 12000 --write-fasta
"""

import argparse
import collections
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEQ = ROOT / "data" / "sequences"
STAGE = SEQ / "_scaled_negative_pool.json"
CKPT = SEQ / "_scaled_negative_pool.partial.json"
FASTA = SEQ / "benign_pool_large.fasta"
API = "https://rest.uniprot.org/uniprotkb/search"
LEN_MIN, LEN_MAX = 100, 1400
PER_ORG = 6
PAGE = 500
RETRIES = 5

BLOCK_KEYWORDS = {"KW-0843": "Virulence", "KW-0800": "Toxin", "KW-0204": "Cytolysis",
                  "KW-0354": "Hemolysis", "KW-0078": "Bacteriocin",
                  "KW-0081": "Bacteriolytic enzyme"}
KW_NAME_BLOCK = ("virulence", "toxin", "cytolysis", "hemolysis", "haemolysis",
                 "bacteriocin", "bacteriolytic")
NAME_BLOCK = (
    "toxin", "hemolysin", "haemolysin", "cytolysin", "leukocidin", "leucocidin", "adhesin",
    "invasin", "virulence", "lethal factor", "edema factor", "oedema factor",
    "protective antigen", "aerolysin", "listeriolysin", "streptolysin", "phospholipase",
    "coagulase", "intimin", "internalin", "hemagglutinin", "haemagglutinin",
)
CLASS_BLOCK = (
    "lactamase", "beta-lactam", "penicillinase", "cephalosporinase", "carbapenemase",
    "adp-ribosyl", "ribosyltransferase", "nuclease", "dnase", "rnase", "deoxyribonuclease",
    "ribonuclease", "protease", "peptidase", "proteinase", "glycosidase",
    "ribosome-inactivating", "rrna n-glycosidase", "pore-forming", "perfringolysin",
    "pneumolysin", "superantigen", "enterotoxin", "type iii secretion", "t3ss",
    "secretion system", "effector", "neurotoxin", "contact-dependent",
    "colicin", "microcin", "endolysin", "lysozyme", "muramidase", "amidase", "holin",
    "delta-endotoxin", "crystal protein", "insecticidal",
)


def H(s):
    return hashlib.sha256(s.encode()).hexdigest()


def read_fasta_ids_and_hashes(p):
    ids, hashes, acc, seq = set(), set(), None, []
    if not p.exists():
        return ids, hashes
    for line in open(p):
        if line.startswith(">"):
            if acc:
                hashes.add(H("".join(seq)))
            acc = line[1:].split()[0]
            ids.add(acc.split("|")[1] if "|" in acc else acc)
            seq = []
        elif acc:
            seq.append(line.strip())
    if acc:
        hashes.add(H("".join(seq)))
    return ids, hashes


def protein_name(rec):
    d = rec.get("proteinDescription", {})
    v = d.get("recommendedName", {}).get("fullName", {}).get("value")
    if v:
        return v
    for alt in (d.get("submissionNames") or []) + (d.get("alternativeNames") or []):
        if alt.get("fullName", {}).get("value"):
            return alt["fullName"]["value"]
    return ""


# 🔴 A Link header must NOT be split on commas here. UniProt echoes the request's `fields`
# parameter into the next-page URL, and that parameter is itself comma-separated
# (`accession,id,organism_name,...`), so splitting turns one entry into fragments: the one
# holding `rel="next"` has no `<` and the one holding `<` has no `rel="next"`. A first version
# did exactly that, found no next page, and ended after 500 records while reporting a clean
# run. A query requesting only `accession` paginated fine, which is why the bug survived a
# spot check.
LINK_RE = re.compile(r'<([^>]+)>\s*;\s*rel="next"')


def pages(query, want):
    """Paginate the UniProt search API by following its Link header.

    Raises if a page comes back with no next link while fewer than `want` and fewer than the
    reported total have been seen. A pagination loop that ends quietly is worse than one that
    crashes: it yields a small sample that looks like a completed harvest."""
    url = API + "?" + urllib.parse.urlencode({
        "query": query, "format": "json", "size": PAGE,
        "fields": "accession,id,organism_name,protein_name,length,sequence,keyword"})
    seen, total = 0, None
    while url and seen < want:
        # 🔴 Retry transient network failures. A first run of this harvest died on a single
        # `TimeoutError: The read operation timed out` partway through the first kingdom and
        # lost everything it had collected. A harvest of thousands of records over dozens of
        # requests will hit one of these; treating it as fatal is a design error, not bad luck.
        body = link = None
        for attempt in range(RETRIES):
            try:
                req = urllib.request.Request(url, headers={"Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=180) as r:
                    body = json.load(r)
                    link = r.headers.get("Link", "")
                    if total is None:
                        total = int(r.headers.get("x-total-results", 0))
                break
            except (urllib.error.URLError, TimeoutError, OSError,
                    json.JSONDecodeError) as e:
                wait = 2 ** attempt
                print(f"    request failed ({type(e).__name__}), retry "
                      f"{attempt + 1}/{RETRIES} in {wait}s", flush=True)
                time.sleep(wait)
        if body is None:
            raise RuntimeError(f"{RETRIES} consecutive request failures at {seen} records")
        res = body.get("results", [])
        seen += len(res)
        yield res
        mt = LINK_RE.search(link)
        if not mt and seen < min(want, total):
            raise RuntimeError(
                f"pagination stopped after {seen} of {total} with {want} wanted; "
                f"Link header was {link[:200]!r}")
        url = mt.group(1) if mt else None
        time.sleep(0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=12000)
    ap.add_argument("--write-fasta", action="store_true")
    a = ap.parse_args()

    known_acc, known_hash = set(), set()
    for f in ("toxins_positive_v2.fasta", "benign_negatives_v2.fasta",
              "toxins_positive_v3.fasta", "benign_negatives_v3.fasta"):
        i, h = read_fasta_ids_and_hashes(SEQ / f)
        known_acc |= i
        known_hash |= h
    print(f"excluding {len(known_acc)} accessions already in either panel")

    # 🔴 Stratify. UniProt returns results clustered by organism, so sequential paging with a
    # per-organism cap is a pathological sampler: a first run took 500 records that were all
    # Homo sapiens, the cap rejected 493 of them, and 6 proteins survived. Slicing by
    # superkingdom and drawing each slice in proportion to its share of the eligible corpus
    # gives a documented composition instead of whatever the result order happens to be.
    excl = " ".join(f"NOT keyword:{k}" for k in BLOCK_KEYWORDS)
    base = f"reviewed:true AND length:[{LEN_MIN} TO {LEN_MAX}] {excl}"
    shares = {"Bacteria": 303421, "Eukaryota": 168417, "Archaea": 17822, "Viruses": 14253}
    tot = sum(shares.values())
    quota = {k: max(50, round(a.target * v / tot)) for k, v in shares.items()}
    cap = max(PER_ORG, a.target // 400)
    print(f"stratified quotas (proportional to the eligible corpus): {quota}")
    print(f"per-organism cap {cap}\n")

    kept, rejected, per_org = [], collections.Counter(), collections.Counter()
    fetched = 0
    for kingdom, want_k in quota.items():
        got0 = len(kept)
        query = f"{base} AND taxonomy_name:{kingdom}"
        for batch in pages(query, want=want_k * 6):
            fetched += len(batch)
            for rec in batch:
                if len(kept) - got0 >= want_k:
                    break
                acc = rec["primaryAccession"]
                s = rec.get("sequence", {}).get("value", "")
                org = rec.get("organism", {}).get("scientificName", "")
                name = protein_name(rec)
                kws = [k.get("name", "") for k in rec.get("keywords", [])]
                if not s or acc in known_acc or H(s) in known_hash:
                    rejected["already_known_or_empty"] += 1
                elif per_org[org] >= cap:
                    rejected["per_organism_cap"] += 1
                elif any(b in k.lower() for k in kws for b in KW_NAME_BLOCK):
                    rejected["blocked_keyword"] += 1
                elif any(b in name.lower() for b in NAME_BLOCK):
                    rejected["blocked_name"] += 1
                elif any(b in name.lower() for b in CLASS_BLOCK):
                    rejected["blocked_positive_class_term"] += 1
                else:
                    kept.append({"acc": f"sp|{acc}|{rec.get('uniProtkbId', acc)}",
                                 "uniprot": acc, "name": rec.get("uniProtkbId", acc),
                                 "protein_name": name, "organism": org,
                                 "kingdom": kingdom,
                                 "length": rec.get("sequence", {}).get("length"),
                                 "sequence": s, "sha256": H(s)[:16]})
                    per_org[org] += 1
                    known_acc.add(acc)
                    known_hash.add(H(s))
            if len(kept) - got0 >= want_k:
                break
        print(f"  {kingdom:<12}quota {want_k:>5}, kept {len(kept) - got0:>5}, "
              f"cumulative {len(kept):>5}", flush=True)
        # Checkpoint per kingdom, so a failure in a later slice does not discard earlier ones.
        json.dump({"partial": True, "completed_kingdoms": kingdom, "kept": len(kept),
                   "proteins": kept}, open(CKPT, "w"))

    print(f"\nfetched {fetched}, kept {len(kept)}, organisms {len(per_org)}")
    print(f"rejections: {dict(rejected)}")
    top = per_org.most_common(5)
    print(f"most represented organisms: {top}")

    json.dump({"built": time.strftime("%Y-%m-%d"), "query": query,
               "per_organism_cap": cap, "target": a.target,
               "stratified_quotas": quota,
               "kingdom_counts": dict(collections.Counter(x["kingdom"] for x in kept)),
               "fetched": fetched, "kept": len(kept),
               "n_organisms": len(per_org),
               "rejections": dict(rejected),
               "note": ("not taxon-matched, unlike 02d and 27: a comprehensive benign reference "
                        "set spans everything, which is the condition being tested. The 296 "
                        "matched negatives stay as the reference point and the two are not the "
                        "same experiment"),
               "proteins": kept}, open(STAGE, "w"), indent=2)
    print(f"wrote {STAGE}")
    if CKPT.exists():
        CKPT.unlink()

    if a.write_fasta:
        with open(FASTA, "w") as fh:
            for x in kept:
                fh.write(f">{x['acc']} {x['protein_name']} OS={x['organism']}\n")
                for i in range(0, len(x["sequence"]), 60):
                    fh.write(x["sequence"][i:i + 60] + "\n")
        print(f"wrote {FASTA} with {len(kept)} records")
    else:
        print("no FASTA written; pass --write-fasta to emit one")


if __name__ == "__main__":
    main()
