#!/usr/bin/env python3
"""
100_independent_pool.py - a second benign pool the findings have never seen.

`docs/INDEPENDENT_POOL_PREREGISTRATION.md`. Eleven analysis scripts draw from the same admitted union
and the same test rows, so studies E through Q are re-analyses of ONE fit on ONE population of 8,259
Swiss-Prot proteins, and the "second arm" this project leans on is a different embedding of the SAME
proteins rather than different data.

🔑 The original build log makes a disjoint sample cheap: a per-organism cap of 30 rejected 28,728
proteins the identical query had already returned. This takes that tranche.

🔒 Every filter is IMPORTED from src/34 rather than copied -- the block lists, the length bounds, the
pagination. Four scripts sharing a regex by copy is how entry 58's defect happened, and a second pool
whose filters had drifted from the first would not be a replication of anything.

⚠️ src/34's documented peptidoglycan-hydrolase leak is inherited deliberately. Pool 2 must be drawn by
the same rule as pool 1, leak included, or the comparison is between two different populations.

Usage:
    python src/100_independent_pool.py --target 6000
    python src/100_independent_pool.py --compare
    python src/100_independent_pool.py --selftest
"""

import argparse
import collections
import importlib.util
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEQ = ROOT / "data" / "sequences"
POOL1 = SEQ / "_scaled_negative_pool.json"
OUT = SEQ / "_independent_pool.json"
FASTA = SEQ / "benign_pool_independent.fasta"
CAP = 30           # 🔒 pool 1's cap, not src/34's target-scaled one, so the rule is identical


def _load(stem):
    spec = importlib.util.spec_from_file_location(f"_{stem}", ROOT / "src" / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M34 = _load("34_scale_negative_set")


def selftest():
    p1 = json.loads(POOL1.read_text())
    assert p1["per_organism_cap"] == CAP, f"pool 1 used cap {p1['per_organism_cap']}, not {CAP}"
    for name in ("BLOCK_KEYWORDS", "KW_NAME_BLOCK", "NAME_BLOCK", "CLASS_BLOCK",
                 "LEN_MIN", "LEN_MAX"):
        assert hasattr(M34, name), f"src/34 no longer exposes {name}; filters would diverge"
    assert M34.LEN_MIN == 100 and M34.LEN_MAX == 1400
    # 🔒 the query this rebuilds must be the one pool 1 recorded, up to the kingdom clause
    excl = " ".join(f"NOT keyword:{k}" for k in M34.BLOCK_KEYWORDS)
    base = f"reviewed:true AND length:[{M34.LEN_MIN} TO {M34.LEN_MAX}] {excl}"
    assert base in p1["query"] or p1["query"].startswith(base[:60]), \
        f"rebuilt base query does not match pool 1's:\n  {base}\n  {p1['query']}"
    print(f"selftest PASS (cap {CAP}, filters imported, base query matches pool 1)")


def build(target):
    p1 = json.loads(POOL1.read_text())["proteins"]
    known_acc = {x["uniprot"] for x in p1}
    known_hash = {M34.H(x["sequence"]) for x in p1}
    print(f"excluding pool 1: {len(known_acc)} accessions, {len(known_hash)} sequence hashes")

    excl = " ".join(f"NOT keyword:{k}" for k in M34.BLOCK_KEYWORDS)
    base = f"reviewed:true AND length:[{M34.LEN_MIN} TO {M34.LEN_MAX}] {excl}"
    shares = {"Bacteria": 303421, "Eukaryota": 168417, "Archaea": 17822, "Viruses": 14253}
    tot = sum(shares.values())
    quota = {k: max(50, round(target * v / tot)) for k, v in shares.items()}
    print(f"stratified quotas (pool 1's proportions): {quota}; per-organism cap {CAP}")

    kept, rejected, per_org = [], collections.Counter(), collections.Counter()
    fetched = 0
    for kingdom, want_k in quota.items():
        got0 = len(kept)
        query = f"{base} AND taxonomy_name:{kingdom}"
        # 🔑 want is raised well above the quota because pool 1 already took the first 30 per
        # organism: most of what comes back will be rejected as already known.
        for batch in M34.pages(query, want=want_k * 40):
            fetched += len(batch)
            for rec in batch:
                if len(kept) - got0 >= want_k:
                    break
                acc = rec.get("primaryAccession", "")
                s = rec.get("sequence", {}).get("value", "")
                org = rec.get("organism", {}).get("scientificName", "?")
                name = M34.protein_name(rec)
                kws = [k.get("name", "") for k in rec.get("keywords", [])]
                if not s or acc in known_acc or M34.H(s) in known_hash:
                    rejected["already_known_or_empty"] += 1
                elif per_org[org] >= CAP:
                    rejected["per_organism_cap"] += 1
                elif any(b in k.lower() for k in kws for b in M34.KW_NAME_BLOCK):
                    rejected["blocked_keyword"] += 1
                elif any(b in name.lower() for b in M34.NAME_BLOCK):
                    rejected["blocked_name"] += 1
                elif any(b in name.lower() for b in M34.CLASS_BLOCK):
                    rejected["blocked_positive_class_term"] += 1
                else:
                    per_org[org] += 1
                    kept.append({"acc": f"sp|{acc}|{rec.get('uniProtkbId', acc)}",
                                 "uniprot": acc, "name": rec.get("uniProtkbId", acc),
                                 "protein_name": name, "organism": org, "kingdom": kingdom,
                                 "length": len(s), "sequence": s, "sha256": M34.H(s)})
            if len(kept) - got0 >= want_k:
                break
        print(f"  {kingdom}: {len(kept) - got0} kept (want {want_k})", flush=True)

    # 🔴 the guard the preregistration named: any overlap means the exclusion failed
    overlap = {x["uniprot"] for x in kept} & known_acc
    if overlap:
        raise SystemExit(f"{len(overlap)} accessions overlap pool 1; nothing is independent")

    res = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "query": base,
           "per_organism_cap": CAP, "target": target,
           "excluded_pool1_accessions": len(known_acc),
           "stratified_quotas": quota, "fetched": fetched, "kept": len(kept),
           "n_organisms": len(per_org), "rejections": dict(rejected),
           "overlap_with_pool1": 0,
           "kingdom_counts": dict(collections.Counter(x["kingdom"] for x in kept)),
           "note": "tranche 2 of the same query: proteins pool 1's per-organism cap rejected. "
                   "Filters imported from src/34, including its documented peptidoglycan leak, so "
                   "the two pools are drawn by one rule.",
           "proteins": kept}
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    FASTA.write_text("".join(f">{x['acc']}\n{x['sequence']}\n" for x in kept))
    print(f"\nkept {len(kept)} over {len(per_org)} organisms; fetched {fetched}")
    print(f"rejections: {dict(rejected)}")
    print(f"wrote {OUT.relative_to(ROOT)} and {FASTA.relative_to(ROOT)}")


def compare():
    """🔒 The composition check the preregistration requires, reported whatever it shows."""
    import numpy as np
    p1 = json.loads(POOL1.read_text())["proteins"]
    p2 = json.loads(OUT.read_text())["proteins"]
    l1 = np.array([x["length"] for x in p1], float)
    l2 = np.array([x["length"] for x in p2], float)
    k1 = collections.Counter(x["kingdom"] for x in p1)
    k2 = collections.Counter(x["kingdom"] for x in p2)
    o1 = {" ".join(x["organism"].replace("(", "").split()[:2]) for x in p1}
    o2 = {" ".join(x["organism"].replace("(", "").split()[:2]) for x in p2}
    print(f"{'':<22}{'pool 1':>12}{'pool 2':>12}")
    print(f"  {'n':<20}{len(p1):>12}{len(p2):>12}")
    print(f"  {'median length':<20}{np.median(l1):>12.0f}{np.median(l2):>12.0f}")
    print(f"  {'mean length':<20}{l1.mean():>12.0f}{l2.mean():>12.0f}")
    for k in ("Bacteria", "Eukaryota", "Archaea", "Viruses"):
        print(f"  {k:<20}{k1[k] / len(p1) * 100:>11.1f}%{k2[k] / len(p2) * 100:>11.1f}%")
    print(f"  {'distinct species':<20}{len(o1):>12}{len(o2):>12}")
    print(f"  {'species in both':<20}{'':>12}{len(o1 & o2):>12}"
          f"   ({len(o1 & o2) / len(o2) * 100:.0f}% of pool 2)")
    res = {"built": time.strftime("%Y-%m-%d %H:%M:%S"),
           "n": [len(p1), len(p2)],
           "median_length": [float(np.median(l1)), float(np.median(l2))],
           "kingdom_frac": {k: [k1[k] / len(p1), k2[k] / len(p2)]
                            for k in ("Bacteria", "Eukaryota", "Archaea", "Viruses")},
           "distinct_species": [len(o1), len(o2)], "species_shared": len(o1 & o2)}
    (ROOT / "results" / "independent_pool_composition.json").write_text(
        json.dumps(res, indent=2) + "\n")
    print("\nwrote results/independent_pool_composition.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=6000)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.compare:
        return compare()
    build(a.target)


if __name__ == "__main__":
    main()
