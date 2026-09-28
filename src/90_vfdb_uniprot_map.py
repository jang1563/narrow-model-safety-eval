#!/usr/bin/env python3
"""
90_vfdb_uniprot_map.py - give the 4,218 VFDB negatives a Swiss-Prot identity, and a localization.

Study H's matched contrast had to use the 133 pool-VFDB overlap as its VFDB side, because nothing else
in this repository pairs a virulence factor with UniProt localization. That bridge is small (24 in the
intracellular cell, one short of its floor) and it is selected AGAINST the hypothesis: the pool's build
query excludes the Virulence, Toxin, Cytolysis, Hemolysis, Bacteriocin and Bacteriolytic enzyme
keywords, so the bridge holds only VFDB members UniProt declines to call virulent, and the per-organism
cap of 30 truncates it further.

🔑 VFDB headers carry a RefSeq or GenBank protein accession for every record, so UniProt's ID-mapping
service can give the 4,218 admitted negatives a Swiss-Prot identity directly — including the
virulence-annotated members the pool query removed. The mapping returns `reviewed` and `keyword` in the
same call, so localization comes with it.

🔒 Frozen selection rule: among the entries an accession maps to, keep `reviewed` ones only, and among
those take the lexicographically smallest accession. Restricting to reviewed keeps both sides of the
contrast on the same annotation pipeline -- the confound that bit A2 in entry 36.

Usage:
    python src/90_vfdb_uniprot_map.py --map        # run the ID mapping (cached)
    python src/90_vfdb_uniprot_map.py
    python src/90_vfdb_uniprot_map.py --selftest
"""

import argparse
import collections
import importlib.util
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VFDB_DIR = ROOT / "data" / "external" / "vfdb"
NEGSET = ROOT / "results" / "vfdb_negative_set.json"
CACHE = ROOT / "data" / "external" / "uniprot_idmap"
OUT = ROOT / "results" / "vfdb_uniprot_map.json"
HEADER_RE = re.compile(r">(VFG\d+)\((\w+)\|([\w.]+)\)")


def _load(stem):
    spec = importlib.util.spec_from_file_location(f"_{stem}", ROOT / "src" / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M84 = _load("84_localization_control")


def vfg_accessions():
    """VFG id -> protein accession, from the FASTA headers."""
    out = {}
    for name in ("VFDB_setA_pro.fas", "VFDB_setB_pro.fas"):
        for line in (VFDB_DIR / name).read_text(errors="replace").splitlines():
            if line.startswith(">"):
                m = HEADER_RE.match(line)
                if m:
                    out.setdefault(m.group(1), m.group(3))
    return out


def _post(url, data):
    req = urllib.request.Request(url, data=urllib.parse.urlencode(data, doseq=True).encode(),
                                 method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)


def run_mapping(ids, from_db):
    """Submit one ID-mapping job and return its TSV rows, following pagination."""
    job = _post("https://rest.uniprot.org/idmapping/run",
                {"from": from_db, "to": "UniProtKB", "ids": ",".join(ids)})
    jid = job["jobId"]
    for _ in range(120):
        with urllib.request.urlopen(
                f"https://rest.uniprot.org/idmapping/status/{jid}", timeout=120) as r:
            st = json.load(r)
        if st.get("jobStatus") in (None, "FINISHED") or "results" in st or "failedIds" in st:
            break
        if st.get("jobStatus") == "ERROR":
            raise SystemExit(f"ID mapping job {jid} failed: {st}")
        time.sleep(3)
    q = urllib.parse.urlencode({"format": "tsv", "size": 500,
                                "fields": "accession,reviewed,organism_name,keyword,length"})
    url = f"https://rest.uniprot.org/idmapping/uniprotkb/results/{jid}?{q}"
    rows, header = [], None
    while url:
        with urllib.request.urlopen(url, timeout=180) as r:
            text = r.read().decode()
            link = r.headers.get("Link", "")
        lines = text.splitlines()
        if lines:
            if header is None:
                header = lines[0]
            rows.extend(lines[1:])
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
        if url:
            time.sleep(0.3)
    return rows


def selftest():
    assert HEADER_RE.match(">VFG000002(gb|WP_010930159) (bvgA) x [y] [z]").group(3) == "WP_010930159"
    assert HEADER_RE.match(">VFG037176(gb|WP_001081735) (plc1) x") .group(1) == "VFG037176"
    # 🔒 the frozen selection rule, exercised on a synthetic accession with three hits
    rows = [("A", "unreviewed", "o", "Signal", "100"), ("A", "reviewed", "o", "Secreted", "100"),
            ("A", "reviewed", "o", "Cytoplasm", "100")]
    best = pick(rows)
    assert best is not None and best[0] == "A" and best[1] == "reviewed"
    assert pick([("Z", "unreviewed", "o", "", "1")]) is None, "unreviewed-only must be dropped"
    assert M84.stratum_of({"Secreted"}) == "extracellular"
    print("selftest PASS")


def pick(hits):
    """Frozen: reviewed only, then the lexicographically smallest accession."""
    rev = sorted((h for h in hits if h[1] == "reviewed"), key=lambda h: h[0])
    return rev[0] if rev else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    acc_of = vfg_accessions()
    admitted = json.loads(NEGSET.read_text())["admitted"]
    vfgs = [r["vfg"] for r in admitted]
    missing = [v for v in vfgs if v not in acc_of]
    if missing:
        raise SystemExit(f"{len(missing)} admitted VFGs have no accession (e.g. {missing[:3]})")
    accs = [acc_of[v] for v in vfgs]
    refseq = sorted({x for x in accs if re.match(r"^(WP|NP|YP)_", x)})
    embl = sorted({x for x in accs if not re.match(r"^(WP|NP|YP)_", x)})
    print(f"{len(vfgs)} admitted negatives -> {len(set(accs))} distinct accessions "
          f"({len(refseq)} RefSeq, {len(embl)} EMBL/GenBank)")

    CACHE.mkdir(parents=True, exist_ok=True)
    shard = CACHE / "vfdb_negatives_idmap.tsv"
    if a.map or not shard.exists():
        rows = []
        for ids, db in ((refseq, "RefSeq_Protein"), (embl, "EMBL-GenBank-DDBJ_CDS")):
            if not ids:
                continue
            print(f"  mapping {len(ids)} ids from {db} ...", flush=True)
            got = run_mapping(ids, db)
            print(f"    {len(got)} rows", flush=True)
            rows.extend(got)
        shard.write_text("From\tEntry\tReviewed\tOrganism\tKeywords\tLength\n" + "\n".join(rows) + "\n")
        if a.map:
            print(f"  wrote {shard.relative_to(ROOT)}")
            return

    hits = collections.defaultdict(list)
    for line in shard.read_text().splitlines()[1:]:
        f = line.split("\t")
        if len(f) >= 5:
            hits[f[0]].append((f[1], f[2], f[3], f[4], f[5] if len(f) > 5 else ""))

    chosen, strata, no_hit, unreviewed_only = {}, collections.Counter(), 0, 0
    for r in admitted:
        h = hits.get(acc_of[r["vfg"]], [])
        if not h:
            no_hit += 1
            continue
        best = pick(h)
        if best is None:
            unreviewed_only += 1
            continue
        kws = {k.strip() for k in best[3].split(";") if k.strip()}
        s = M84.stratum_of(kws)
        chosen[r["vfg"]] = {"uniprot": best[0], "organism": best[2], "stratum": s,
                            "keywords": sorted(kws), "category": r["category"],
                            "vfdb_organism": r["organism"]}
        strata[s] += 1

    res = {"built": time.strftime("%Y-%m-%d %H:%M:%S"),
           "n_admitted": len(admitted), "n_distinct_accessions": len(set(accs)),
           "n_mapped_reviewed": len(chosen), "n_no_hit": no_hit,
           "n_unreviewed_only": unreviewed_only,
           "strata": dict(strata.most_common()),
           "n_with_virulence_keyword": sum(1 for v in chosen.values()
                                           if "Virulence" in v["keywords"]),
           "n_with_toxin_keyword": sum(1 for v in chosen.values() if "Toxin" in v["keywords"]),
           "categories": dict(collections.Counter(v["category"] for v in chosen.values()).most_common()),
           "chosen": chosen}
    OUT.write_text(json.dumps(res, indent=2) + "\n")

    print(f"\n  mapped to a reviewed Swiss-Prot entry : {len(chosen)}")
    print(f"  no UniProt hit at all                 : {no_hit}")
    print(f"  hits but none reviewed                : {unreviewed_only}")
    print("\n  localization strata of the mapped set:")
    for s, n in strata.most_common():
        print(f"    {s:<16}{n:>6}")
    print(f"\n  carry the Virulence keyword: {res['n_with_virulence_keyword']}"
          f"  (the pool's build query excludes these, so the study-H bridge could not)")
    print(f"  carry the Toxin keyword    : {res['n_with_toxin_keyword']}")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
