#!/usr/bin/env python3
"""Fetch a THIRD-PARTY hazard label for the 746 class-axis representatives.

Study AG left § 6 with a named hole: the only toxin-inclusive measurement in the paper is
an UNMATCHED per-category ranking, and the matched contrast `C/A` runs on a population
`src/74` built from "VFDB setA, non-Exotoxin records".  The local UniProt idmap inherits
that exclusion -- it covers 431 of the 746 representatives and **0 of the 102 Exotoxin** --
so no hazard axis can be built offline.

This pulls UniProtKB entries for all 746 RefSeq accessions in one ID-mapping job and
caches the TSV.  It fits nothing and measures nothing; the axis is defined in src/121.
The axis is Swiss-Prot keywords assigned by UniProt curators: not the author's judgment
(study Q), not VFDB's categories (study AG).

⚠️ Keyword presence requires a REVIEWED entry, which is entry 99's expensive rule.  What
fraction of the 746 survives it is the first thing src/121 has to report.
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAS = ROOT / "data/external/vfdb/VFDB_setA_pro.fas"
BUILD = ROOT / "results/external_class_axis_build.json"
OUT_TSV = ROOT / "data/external/uniprot_idmap/class_axis_idmap.tsv"
OUT_JSON = ROOT / "results/external_hazard_axis_fetch.json"
API = "https://rest.uniprot.org"
FIELDS = "accession,reviewed,protein_name,organism_name,keyword,length"
HDR = re.compile(r"^>(VFG\d+)\(gb\|([^)]+)\)")


def accessions():
    """VFG id -> RefSeq accession, for the 746 class-axis representatives."""
    reps = {r["vfg"]: r for r in json.loads(BUILD.read_text())["positives"]}
    acc = {}
    with FAS.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(">"):
                m = HDR.match(line)
                if m:
                    acc[m.group(1)] = m.group(2)
    return reps, {v: acc[v] for v in reps if v in acc}


def get(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "narrow-model-safety-eval"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def submit(ids):
    body = urllib.parse.urlencode(
        {"from": "RefSeq_Protein", "to": "UniProtKB", "ids": ",".join(ids)}
    ).encode()
    req = urllib.request.Request(
        f"{API}/idmapping/run", data=body,
        headers={"User-Agent": "narrow-model-safety-eval"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())["jobId"]


def wait(job, tries=60, pause=5):
    for i in range(tries):
        txt = get(f"{API}/idmapping/status/{job}")
        d = json.loads(txt)
        if d.get("jobStatus") in (None, "FINISHED"):
            return i
        if d.get("jobStatus") not in ("RUNNING", "NEW"):
            raise RuntimeError(f"job {job}: {d}")
        time.sleep(pause)
    raise TimeoutError(f"job {job} not finished after {tries * pause}s")


def selftest():
    ok = True
    reps, acc = accessions()
    if len(reps) != 746:
        print(f"  XX {len(reps)} representatives, expected 746")
        ok = False
    if len(acc) != 746:
        print(f"  XX {len(acc)} of {len(reps)} carry an accession, expected all 746")
        ok = False
    exo = [v for v in reps if reps[v]["cat"] == "Exotoxin"]
    if len(exo) != 102:
        print(f"  XX {len(exo)} Exotoxin representatives, expected 102")
        ok = False
    # the header regex must actually be doing work: a line it cannot parse must not match
    if HDR.match(">VFG000001 no parens here"):
        print("  XX header regex matches a line with no gb| field")
        ok = False
    if not HDR.match(">VFG000001(gb|WP_000000000) (x) foo"):
        print("  XX header regex fails on a well-formed header")
        ok = False
    print("selftest PASS" if ok else "selftest FAIL")
    return 0 if ok else 1


def main():
    reps, acc = accessions()
    ids = sorted(set(acc.values()))
    print(f"{len(reps)} representatives, {len(ids)} distinct accessions")

    if OUT_TSV.exists() and "--refetch" not in sys.argv:
        print(f"cached: {OUT_TSV.relative_to(ROOT)} (pass --refetch to replace)")
        tsv = OUT_TSV.read_text(encoding="utf-8", errors="replace")
        job, polls = "cached", -1
    else:
        job = submit(ids)
        print(f"job {job} submitted")
        polls = wait(job)
        url = (f"{API}/idmapping/uniprotkb/results/stream/{job}"
               f"?format=tsv&fields={FIELDS}")
        tsv = get(url, timeout=300)
        OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
        OUT_TSV.write_text(tsv)
        print(f"wrote {OUT_TSV.relative_to(ROOT)} after {polls} polls")

    lines = [ln for ln in tsv.splitlines() if ln.strip()]
    hdr, rows = lines[0].split("\t"), [ln.split("\t") for ln in lines[1:]]
    recs = [dict(zip(hdr, r)) for r in rows if len(r) == len(hdr)]
    OUT_JSON.write_text(json.dumps({
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "UniProt ID mapping, RefSeq_Protein -> UniProtKB",
        "fields": FIELDS,
        "job": job,
        "polls": polls,
        "n_representatives": len(reps),
        "n_accessions_submitted": len(ids),
        "n_rows_returned": len(recs),
        "n_distinct_from": len({r["From"] for r in recs}),
        "n_reviewed_rows": sum(1 for r in recs if r.get("Reviewed") == "reviewed"),
        "tsv": str(OUT_TSV.relative_to(ROOT)),
    }, indent=2))
    print(f"rows {len(recs)}  distinct From {len({r['From'] for r in recs})}  "
          f"reviewed {sum(1 for r in recs if r.get('Reviewed') == 'reviewed')}")
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else main())
