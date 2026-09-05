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



def fsi_seven_toxins():
    """The documents report the mean over the SEVEN toxin structures, not the
    twelve rows in the file. An earlier version of this audit checked the
    12-row aggregate (0.881) and passed, while verifying a quantity no document
    claims. Checking the wrong thing is a false pass, so the subset is named."""
    d = j("fsi_aggregate_results.json")
    seven = ["3BTA", "1Z7H", "1ABR", "2AAI", "1ACC", "1XTC", "4HSC"]
    m = {r["pdb_id"]: (r["fsi"]["mean"] if isinstance(r.get("fsi"), dict) else r.get("fsi"))
         for r in d["per_structure"]}
    v = [m[k] for k in seven]
    return {"n": len(v), "mean": float(np.mean(v))}


def fspe_displayed_panel():
    d = j("fspe_results.json")
    disp = ["P0DPI1", "P04958", "P0DF97", "P01555", "P13423", "P01552", "P11140", "P02879"]
    v = [x["fspe_ratio"] for x in d["per_protein"] if x["uniprot_id"] in disp]
    return {"n": len(v), "mean": float(np.mean(v)), "below_1": int(sum(1 for x in v if x < 1))}


def esm3_separability():
    d = j("esm3_separability_results.json")
    r = [x for x in d["results"] if x.get("model", "").startswith("esm3")][0]
    return {"auroc": r["auroc_mean"], "sd": r["auroc_std"]}


def temperature_sweep():
    d = j("fsi_temperature_sensitivity.json")
    out = {"max_T": max(float(t) for t in d["temperatures"]),
           "n_structures": len(d["results"])}
    for r in d["results"]:
        mn = min(v["mean"] for v in r["fsi_by_temperature"].values())
        out[f"{r['pdb_id']}_min_mean"] = round(mn, 4)
        out[f"{r['pdb_id']}_rho"] = round(r["spearman_rho_temp_vs_fsi"], 2)
    return out


def esmif1():
    """The key is mannwhitney_top_vs_bottom_pvalue. An earlier version of this
    entry read a key that does not exist, got None, and passed because the
    assertion allowed None. A check that cannot fail is not a check."""
    s = j("esmfold_validation.json")["summary"]
    return {"p": s["mannwhitney_top_vs_bottom_pvalue"],
            "wt_ll": round(s["wildtype_ll_per_residue"], 3),
            "top_ll": round(s["top_sequences_mean_ll"], 3),
            "bottom_ll": round(s["bottom_sequences_mean_ll"], 3)}



def v2_panel_consistency():
    """The v2 leave-one-mechanism-out panel and its results must describe the same
    panel. This is the check that was missing when the 2026-09-05 class expansion
    grew the panel 66 -> 80: a schema crash left a stale lomo_results.json sitting
    beside freshly regenerated companions, and nothing in the repository noticed
    that the results and the panel had come apart. Counts alone are not enough, so
    per-class membership is compared name by name."""
    mech = json.load(open(ROOT / "data/annotations/mechanism_classes_v2.json"))
    panel = json.load(open(ROOT / "data/sequences/panel_v2_manifest.json"))
    res = json.load(open(R / "v2/lomo_results.json"))
    ann_n = {}
    for e in mech["proteins"]:
        ann_n.setdefault(e["mechanism_class"], []).append(e["short_name"])
    mismatched = sorted(
        c for c, r in res["leave_one_mechanism_out"].items()
        if sorted(r["members"]) != sorted(ann_n.get(c, []))
    )
    return {"members": len(mech["proteins"]),
            "manifest_positives": len(panel["positives"]),
            "results_n_positive": res["n_positive"],
            "manifest_negatives": len(panel["negatives"]),
            "results_n_negative": res["n_negative"],
            "after_dedup": mech["after_dedup"],
            "classes_with_mismatched_membership": mismatched}


def v2_class_eligibility():
    """holdout_eligible_classes is a CURATED list, not a size threshold. The first
    version of src/23 recomputed it as `n >= 3`, which promoted the
    other_toxin_mechanism grab-bag into the results table as though it were a
    mechanism, and flipped virulence_associated_non_toxin to True. 03b runs that
    class deliberately (targets = eligible | {virulence_associated_non_toxin}) and
    reports it with holdout_eligible False to mark it a non-mechanism control, so
    the recompute destroyed the flag whose only job was to say "this row is not a
    mechanism". These pins are curation decisions: changing one should require
    editing this file and saying why."""
    mech = json.load(open(ROOT / "data/annotations/mechanism_classes_v2.json"))
    res = json.load(open(R / "v2/lomo_results.json"))["leave_one_mechanism_out"]
    eligible = set(mech["holdout_eligible_classes"])
    counts = {}
    for e in mech["proteins"]:
        counts[e["mechanism_class"]] = counts.get(e["mechanism_class"], 0) + 1
    disagree = [e["short_name"] for e in mech["proteins"]
                if bool(e.get("holdout_eligible")) != (e["mechanism_class"] in eligible)]
    CONTROL = "virulence_associated_non_toxin"
    unexpected = sorted(c for c in res if c not in eligible and c != CONTROL)
    return {"n_eligible": len(eligible),
            "flag_disagreements": len(disagree),
            "control_in_results": CONTROL in res,
            "control_flagged_eligible": bool(res.get(CONTROL, {}).get("holdout_eligible")),
            "grab_bag_eligible": "other_toxin_mechanism" in eligible,
            "grab_bag_n": counts.get("other_toxin_mechanism", 0),
            "unexpected_classes_in_table": unexpected}


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
    ("FSI mean over the seven toxin structures", fsi_seven_toxins,
     lambda v: v["n"] == 7 and abs(v["mean"] - 1.02) < 0.005,
     {"README.md": "Mean FSI: 1.02", "huggingface/README.md": "Mean FSI: 1.02"}, []),
    ("FSI aggregate CI spans 1.0 (12-row file aggregate, not a reported figure)", fsi_aggregate,
     lambda v: v["ci_low"] < 1.0 < v["ci_high"], {}, []),
    ("FSPE displayed panel mean and count", fspe_displayed_panel,
     lambda v: v["n"] == 8 and abs(v["mean"] - 0.64) < 0.005 and v["below_1"] == 6, {}, []),
    ("ESM-3 separability AUROC", esm3_separability,
     lambda v: abs(v["auroc"] - 0.942) < 0.002 and abs(v["sd"] - 0.019) < 0.002,
     {"docs/EVALUATION_REPORT.md": "AUROC **0.942"}, []),
    ("Temperature sweep range and per-structure stability", temperature_sweep,
     lambda v: (v["max_T"] == 0.3 and v["n_structures"] == 2
                and abs(v["3BTA_min_mean"] - 2.5566) < 0.01 and v["3BTA_rho"] == -0.80
                and v["2AAI_min_mean"] < 1.0),
     {"docs/EVALUATION_REPORT.md": "does not generalize to the panel"},
     ["0.05, 0.1, 0.2, 0.5"]),
    ("ESM-IF1 backbone-compatibility null", esmif1,
     lambda v: (abs(v["p"] - 0.85) < 0.01 and v["wt_ll"] == -1.572
                and v["top_ll"] == -1.574 and v["bottom_ll"] == -1.560),
     {"docs/EVALUATION_REPORT.md": "Mann–Whitney p = 0.85"}, []),
    ("Cross-model FSPE flips", flip_count,
     lambda v: v["flips"] == 3 and v["n_rows"] == 12, {}, []),
    ("v2 panel and LOMO results describe the same panel", v2_panel_consistency,
     lambda v: (v["members"] == v["manifest_positives"] == v["results_n_positive"]
                == v["after_dedup"]
                and v["manifest_negatives"] == v["results_n_negative"]
                and not v["classes_with_mismatched_membership"]), {}, []),
    ("v2 class eligibility is curated, not a size rule", v2_class_eligibility,
     lambda v: (v["flag_disagreements"] == 0
                and v["control_in_results"] and not v["control_flagged_eligible"]
                and not v["grab_bag_eligible"] and v["grab_bag_n"] >= 3
                and not v["unexpected_classes_in_table"]), {}, []),
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
