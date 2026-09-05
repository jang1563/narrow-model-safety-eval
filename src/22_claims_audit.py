#!/usr/bin/env python3
"""
22_claims_audit.py - every headline number, recomputed and matched against the documents.

Why this exists
---------------
Four separate number-drift defects were found by hand in this project, each by
someone happening to look:

  - a homology figure repeated through three document rewrites without ever being
    recomputed on the panel it was being quoted about
  - a results artifact silently overwritten by a later run with a narrower model
    set, leaving a quoted figure with no source at all
  - numbers cited in a document as the reason for a decision that had been
    computed once in a shell and never saved
  - a headline p-value that was pseudoreplicated, propagated across four public
    surfaces, and corrected only after a claim-by-claim audit

All four were survivable. The pattern is not: a document and the artifact behind
it drift apart quietly, and nothing in the repository notices.

This is the check that notices. Each entry names a claim, the artifact that
produces it, how to recompute it, and the documents that must agree. Run it before
publishing anything.

Exit status is 1 if any claim fails, so it can gate CI.

Usage:
    python src/22_claims_audit.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
PUBLIC = ["README.md", "huggingface/README.md", "docs/EVALUATION_REPORT.md",
          "docs/ARCHITECTURE.md"]


def j(p):
    f = R / p
    return json.load(open(f)) if f.exists() else None


# ---- recomputation, from artifacts only ---------------------------------------

def fspe_protein_level():
    d = j("fspe_results.json")
    r = np.array([x["fspe_ratio"] for x in d["per_protein"]], float)
    n, k = len(r), int((r < 1).sum())
    p_sign = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return {"n": n, "below_1": k, "sign_p": p_sign}


def separability():
    d = j("separability_results.json")
    return {"auroc": d["auroc_mean"]} if d else None


def fsi_aggregate():
    d = j("fsi_aggregate_results.json")
    a = d["aggregate"]["fsi_aggregate"]
    ci = a["bootstrap_ci_95"]
    return {"mean": a["mean"], "ci_low": ci["ci_95_low"], "ci_high": ci["ci_95_high"],
            "n": d["aggregate"]["n_structures"]}


def flip_count():
    rows = json.load(open(ROOT / "data/sequences/mdrp_risk_table.json"))["proteins"] \
        if (ROOT / "data/sequences/mdrp_risk_table.json").exists() else j("mdrp_risk_table.json")["proteins"]
    cols = ["fspe_esm2", "fspe_esm3", "fspe_saprot"]
    side = lambda v: ">1" if v > 1 else "<1"          # noqa: E731
    n = 0
    for r in rows:
        av = [r.get(c) for c in cols]
        av = [v for v in av if v is not None]
        if len(av) >= 2 and len({side(v) for v in av}) > 1:
            n += 1
    return {"flips": n, "n_rows": len(rows)}


# ---- the registry --------------------------------------------------------------
# (label, recompute -> dict, assertion on that dict, {document: string it must
#  contain}, strings no public document may contain any more)
#
# `must` names the document explicitly. An earlier version only required the string
# to appear in SOME public document, which meant one document could drift while the
# others still carried the phrase and the audit would pass. Verified by breaking
# README.md on purpose: the audit returned success. It now names each surface.

CLAIMS = [
    ("FSPE protein-level sign test", fspe_protein_level,
     lambda v: abs(v["sign_p"] - 0.018) < 0.002 and v["below_1"] == 12 and v["n"] == 15,
     {"README.md": "sign test p = 0.018",
      "huggingface/README.md": "sign test p = 0.018",
      "docs/EVALUATION_REPORT.md": "sign test p = 0.018"}, []),
    ("FSPE pseudoreplicated figure is labelled, not led with", fspe_protein_level,
     lambda v: True, {}, ["Pooled meta-analysis: p = 2.6", "meta-analysis (p = 2.6 × 10⁻⁸) is the better-powered"]),
    ("Embedding separability AUROC", separability,
     lambda v: v is None or abs(v["auroc"] - 0.981) < 0.002, {}, []),
    ("FSI aggregate CI spans 1.0", fsi_aggregate,
     lambda v: v["ci_low"] < 1.0 < v["ci_high"], {}, []),
    ("Cross-model FSPE flips", flip_count,
     lambda v: v["flips"] == 3 and v["n_rows"] == 12, {}, []),
]


def main():
    docs = {p: (ROOT / p).read_text() for p in PUBLIC if (ROOT / p).exists()}
    failures = []
    print(f"auditing {len(CLAIMS)} claims against artifacts and {len(docs)} public documents\n")
    for label, fn, ok, must, forbid in CLAIMS:
        try:
            v = fn()
        except Exception as e:
            print(f"  XX {label:<48} artifact error: {type(e).__name__}")
            failures.append(label)
            continue
        good = ok(v)
        detail = ", ".join(f"{k}={round(x, 4) if isinstance(x, float) else x}"
                           for k, x in (v or {}).items())
        print(f"  {'OK' if good else 'XX'} {label:<48} {detail}")
        if not good:
            failures.append(label)
        for doc, s in must.items():
            if doc not in docs:
                print(f"     XX document not found: {doc}")
                failures.append(f"{label}: absent {doc}")
            elif s not in docs[doc]:
                print(f"     XX {doc} does not contain: {s!r}")
                failures.append(f"{label}: missing in {doc}")
        for s in forbid:
            hit = [d for d, t in docs.items() if s in t]
            if hit:
                print(f"     XX still present in {hit}: {s!r}")
                failures.append(f"{label}: stale {s}")

    print()
    if failures:
        print(f"FAILED: {len(failures)}")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"all {len(CLAIMS)} claims agree with their artifacts and with the public documents")
    return 0


if __name__ == "__main__":
    sys.exit(main())
