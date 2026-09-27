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
          "docs/ARCHITECTURE.md", "docs/MECHANISM_GENERALIZATION.md",
          "docs/DETECTOR_CRITERIA.md",
          # Added 2026-09-21. The preregistration is linked from the README and now carries
          # verified residue positions in its amendment log, so it is a surface where a number can
          # drift. Checked clean against every forbid string in this registry before adding.
          "docs/MUTATION_EXTENSION_PREREGISTRATION.md",
          # Added 2026-09-24. This is the interview and collaboration-scoping brief, so it is the
          # surface where a number is most likely to be spoken aloud, and it was the only document
          # quoting headline figures that no gate checked. Adding it immediately failed on a stale
          # FSPE panel count, which is the argument for having added it.
          "docs/BIOHUB_RESEARCH_BRIEF.md",
          # Added 2026-09-24 with the document itself. It is a standalone summary written to be read
          # instead of the long documents, which makes it the surface most likely to be quoted and
          # the one where a stale figure would travel furthest.
          "docs/DETECTOR_EVALUATION_SUMMARY.md",
          # Added 2026-09-27 with the document itself. It is the design note for the next phase, so
          # every number in it is one a later decision would rest on, and its first draft already
          # carried a stale figure: it put the species axis at n = 7 from v2's annotation when
          # v3's has 52 bacteria and 22 insect and the holdout had already been run. Adding it is
          # the check that would have caught that.
          "docs/VFDB_CLASS_AXIS_DESIGN.md",
          # Added 2026-09-27 with the document itself, for the same reason the mutation
          # preregistration is here: a frozen threshold that drifts is worse than no threshold, and
          # its amendment log now carries the delivered control set's composition.
          "docs/NEGATIVE_EXPANSION_PREREGISTRATION.md"]


def j(p):
    f = R / p
    return json.load(open(f)) if f.exists() else None


# ---- recomputation, from artifacts only ---------------------------------------

def _sign_p(n, k):
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def fspe_protein_level():
    """The reported protein-level headline, which since 2026-09-22 is the EXCLUDED one.

    Both figures are pinned. `n`/`below_1`/`sign_p` are what the documents lead with, after SEB is
    dropped, and `full_*` is what the unexcluded set gives. Pinning only the reported pair would let
    an edit quietly restore the stronger number; pinning only the full pair would not notice the
    exclusion being dropped.

    ⚠️ The permutation p is pinned too, and in the OPPOSITE direction to the sign test. The exclusion
    weakens the sign test (0.0037 to 0.0065) and strengthens the permutation test (0.0002 to 0.0001),
    because SEB's 0.956 was the nearest to 1.0 of the successes. Both are asserted so neither can be
    quoted alone."""
    d = j("fspe_results.json")
    t = j("fspe_protein_level_test.json")
    sites = json.load(open(ROOT / "data/annotations/functional_sites.json"))
    excl = {a for a in sites if not a.startswith("_")
            and sites[a]["functional_sites"].get("fspe_excluded")}
    rows = {(x.get("uniprot_id") or x.get("accession")): x["fspe_ratio"]
            for x in d["per_protein"]}
    kept = [v for a, v in rows.items() if a not in excl]
    n, k = len(kept), sum(1 for v in kept if v < 1.0)
    fn, fk = len(rows), sum(1 for v in rows.values() if v < 1.0)
    return {"n": n, "below_1": k, "sign_p": _sign_p(n, k),
            # the mean is here because docs/BIOHUB_RESEARCH_BRIEF.md quotes it, and that document
            # joined the audited surface on 2026-09-24 carrying the pre-exclusion pair (0.437, 13/15)
            "mean_ratio": sum(kept) / n, "full_mean_ratio": sum(rows.values()) / fn,
            "excluded": sorted(excl), "excluded_ratios": {a: rows[a] for a in sorted(excl)},
            "full_n": fn, "full_below_1": fk, "full_sign_p": _sign_p(fn, fk),
            # the artifact src/21 wrote must agree with this recomputation
            "artifact_n": t["n_proteins"], "artifact_below_1": t["n_ratio_below_1"],
            "artifact_sign_p": t["sign_test_one_sided_p"],
            "artifact_perm_p": t["permutation_p"],
            "artifact_full_sign_p": t["without_exclusions"]["sign_test_one_sided_p"],
            "artifact_full_perm_p": t["without_exclusions"]["permutation_p"],
            "weakens": t["exclusion_weakens_headline"],
            "flagged_but_absent": t["flagged_but_absent_from_results"]}


def fspe_flagged_leaveout():
    """How much the protein-level headline rests on the two annotation-flagged entries.

    SEB and ExoS both sit BELOW 1.0, so both are counted as successes by the published 13/15, and
    both have positions that fail the residue-identity check: all nine of SEB's, and one of ExoS's
    five. The headline should therefore be quotable with its leave-out value attached, which is what
    this pins. The direction survives every subset; the p-value roughly triples and crosses 0.01, so
    the number that must not drift is the both-removed one.
    """
    d = j("fspe_results.json")
    rows = {(x.get("uniprot_id") or x.get("accession")): x["fspe_ratio"] for x in d["per_protein"]}
    flagged = sorted(k for k, x in
                     ((x.get("uniprot_id") or x.get("accession"), x) for x in d["per_protein"])
                     if x.get("numbering_flagged"))

    def sub(drop):
        vals = [v for a, v in rows.items() if a not in drop]
        n, k = len(vals), sum(1 for v in vals if v < 1.0)
        return {"n": n, "below_1": k, "sign_p": _sign_p(n, k)}

    full, both = sub(set()), sub(set(flagged))
    return {"flagged": flagged,
            "flagged_ratios": {a: rows[a] for a in flagged},
            "flagged_all_below_1": all(rows[a] < 1.0 for a in flagged),
            "full": full, "without_both": both,
            "each": {a: sub({a}) for a in flagged},
            "direction_survives": both["below_1"] * 2 > both["n"],
            "p_inflation": both["sign_p"] / full["sign_p"]}


def q51451_site_sensitivity():
    """The one spurious ExoS position, and whether the headline leans on it.

    Optional artifact: `src/47` needs torch, so a missing file returns None rather than failing the
    release-surface job. When present the check is strict, and the load-bearing part is the
    cross-artifact tie: the script's published-residue ratio must equal the value the pipeline itself
    recorded for this protein, otherwise the sensitivity was measured on a different computation than
    the one the headline uses.
    """
    d = j("v3/flagged_site_sensitivity.json")
    if not d:
        return None
    fspe = j("fspe_results.json")
    pipeline = {(x.get("uniprot_id") or x.get("accession")): x["fspe_ratio"]
                for x in fspe["per_protein"]}.get(d["accession"])
    return {"accession": d["accession"], "spurious": d["spurious_position"],
            "residue_at_spurious": d["residue_at_spurious"],
            "trp_positions": d["trp_positions_in_precursor"],
            "no_offset_can_reach": d["no_offset_can_reach"],
            "ratio_published": d["ratio_published"],
            "ratio_without": d["ratio_without_spurious"],
            "delta": d["delta"], "moves_down": d["delta"] < 0,
            "both_below_1": d["both_below_1"], "verdict_flips": d["verdict_flips"],
            "matches_pipeline": pipeline is not None
            and abs(d["ratio_published"] - pipeline) < 1e-6}


def conformal_lomo_test_split():
    """The test partition the panel never had, and the estimator that was validated and never used.

    Pins the asymmetry rather than a winner, because the run does not produce one: at a nominal 5%
    the quantile estimator's seed-level interval EXCLUDES nominal and conformal's covers it, and at a
    nominal 1% conformal cannot be computed at all while the quantile estimator returns a threshold
    that realizes close to 4x the budget. The ordering of the failing classes is also checked,
    because the value of the run is that the failure story survives the threshold rule.
    """
    arms = {"canonical": j("v3/conformal_lomo_test_split.json"),
            "esm2_35M": j("v3/conformal_lomo_test_split_esm2_35M.json")}
    if not all(arms.values()):
        return None

    def one(d):
        f5, f1 = d["out_of_sample_fp"]["0.05"], d["out_of_sample_fp"]["0.01"]
        order = sorted(d["per_class"].items(), key=lambda kv: kv[1]["alpha_0.05"]["quantile_mean"])
        conf = sorted((kv for kv in d["per_class"].items()
                       if kv[1]["alpha_0.05"]["conformal_mean"] is not None),
                      key=lambda kv: kv[1]["alpha_0.05"]["conformal_mean"])
        deltas = [kv[1]["alpha_0.05"]["delta_pts"] for kv in d["per_class"].items()
                  if kv[1]["alpha_0.05"]["delta_pts"] is not None]
        return {"split": d["split"], "seeds": d["seeds"],
                "q_fp_05": f5["quantile_fp_mean"], "c_fp_05": f5["conformal_fp_mean"],
                "q_excludes_05": f5["quantile_ci_excludes_nominal"],
                "c_excludes_05": f5["conformal_ci_excludes_nominal"],
                "q_fp_01": f1["quantile_fp_mean"],
                "q_excludes_01": f1["quantile_ci_excludes_nominal"],
                "conformal_reachable_01": f1["conformal_reachable"],
                "conformal_closer": f5["conformal_fp_mean"] < f5["quantile_fp_mean"],
                "bottom2_quantile": [k for k, _ in order[:2]],
                "bottom3_quantile": [k for k, _ in order[:3]],
                "bottom3_conformal": [k for k, _ in conf[:3]],
                "estimator_ordering_preserved":
                    [k for k, _ in order[:3]] == [k for k, _ in conf[:3]],
                "max_delta_pts": max(deltas), "min_delta_pts": min(deltas)}

    a, b = one(arms["canonical"]), one(arms["esm2_35M"])
    return {"arms": {"canonical": a, "esm2_35M": b},
            # what holds in BOTH arms
            "q_excludes_both_alphas_both_arms": all(
                x["q_excludes_05"] and x["q_excludes_01"] for x in (a, b)),
            "conformal_unreachable_01_both": not a["conformal_reachable_01"]
            and not b["conformal_reachable_01"],
            "conformal_closer_both": a["conformal_closer"] and b["conformal_closer"],
            "estimator_ordering_preserved_both": a["estimator_ordering_preserved"]
            and b["estimator_ordering_preserved"],
            "bottom2_set_agrees": set(a["bottom2_quantile"]) == set(b["bottom2_quantile"]),
            # At 200 seeds conformal attains its guarantee in BOTH arms: its interval covers nominal
            # and the point estimates sit within 0.15 pts of the theoretical 5.00%. At 30 seeds the
            # two arms appeared to disagree, and a claim asserting that disagreement failed this gate
            # as soon as the seed count went up, which is what the assertion was for.
            "conformal_covers_nominal_both": not a["c_excludes_05"] and not b["c_excludes_05"],
            "conformal_near_theoretical_both": all(
                abs(x["c_fp_05"] - 0.05) < 0.004 for x in (a, b)),
            # this one survives 200 seeds, so it is a property and not noise
            "worst_class_disagrees": a["bottom2_quantile"][0] != b["bottom2_quantile"][0]}


def external_test_partition():
    """Where the false-positive budget actually goes, decomposed, with calibration left at 118.

    src/48 had to shrink calibration to 59 to get a test set, which broke comparability with the
    published table and put the conformal threshold out of reach at a nominal 1%. src/49 takes the
    test negatives from the 8,259-protein pool instead, so calibration stays at the published 118 and
    conformal is reachable at both budgets. The three arms separate the estimator from the shift:
    pool-to-pool is exchangeable so the guarantee should hold, panel-to-pool is the deployment
    condition, and the distinct-name version removes the pool's 2.4x name redundancy.
    """
    arms = {"esm2_35M": j("v3/external_test_partition_esm2_35M.json"),
            "canonical": j("v3/external_test_partition_canonical.json")}
    if not all(arms.values()):
        return None

    def one(d):
        fp = d["false_positives"]

        def g(alpha, arm, est, field="mean"):
            s = fp[str(alpha)][arm][est]
            return None if s is None else s[field]

        return {"calibration_n": d["calibration_n"], "pool_n": d["pool_n"],
            "distinct_names": d["pool_distinct_names"], "seeds": d["seeds"],
            "dropped": d["contaminant_dropped"],
            "k": {a: d["conformal_k"][a] for a in d["conformal_k"]},
            "guarantee": {a: d["conformal_guarantee"][a] for a in d["conformal_guarantee"]},
            "conformal_reachable_at_1pct": d["conformal_k"]["0.01"] >= 1,
            # exchangeable control: conformal should land on its guarantee, quantile should not
            "ctrl_conf_05": g(0.05, "control", "conformal"),
            "ctrl_conf_05_covers": not fp["0.05"]["control"]["conformal"]["excludes_nominal"],
            "ctrl_conf_01": g(0.01, "control", "conformal"),
            "ctrl_conf_01_covers": not fp["0.01"]["control"]["conformal"]["excludes_nominal"],
            "ctrl_quant_05_exceeds": fp["0.05"]["control"]["quantile"]["excludes_nominal"],
            "ctrl_quant_01_exceeds": fp["0.01"]["control"]["quantile"]["excludes_nominal"],
            # shift and redundancy each add on top
            "shift_conf_05": g(0.05, "shift", "conformal"),
            "shift_conf_05_exceeds": fp["0.05"]["shift"]["conformal"]["excludes_nominal"],
            "shift_conf_01": g(0.01, "shift", "conformal"),
            "dedup_conf_05": g(0.05, "shift_dedup", "conformal"),
            "dedup_quant_05": g(0.05, "shift_dedup", "quantile"),
            "shift_quant_05": g(0.05, "shift", "quantile"),
            "dedup_raises_fp": g(0.05, "shift_dedup", "quantile") > g(0.05, "shift", "quantile"),
            "conformal_holds_in_control": d["conformal_holds_in_control"],
            # 🔴 2026-09-24, entry twenty-three: this was a hardcoded True in src/49 and stayed True
            # after the second arm ran, so both artifacts contradicted § 2.6.1's "both arms". It is
            # computed from the arms on disk now, and asserted here so it cannot go stale again.
            "provisional": d.get("single_arm_provisional")}

    a, b = one(arms["esm2_35M"]), one(arms["canonical"])
    return {"arms": {"esm2_35M": a, "canonical": b},
            # the decomposition now replicates across two arms, so it is no longer provisional
            "ctrl_conf_on_guarantee_both": all(
                abs(x["ctrl_conf_05"] - x["guarantee"]["0.05"]) < 0.005
                and abs(x["ctrl_conf_01"] - x["guarantee"]["0.01"]) < 0.005 for x in (a, b)),
            "ctrl_quant_exceeds_both": all(
                x["ctrl_quant_05_exceeds"] and x["ctrl_quant_01_exceeds"] for x in (a, b)),
            "shift_breaks_conformal_both": all(
                x["shift_conf_05_exceeds"] and x["shift_conf_05"] > x["ctrl_conf_05"]
                for x in (a, b)),
            "dedup_raises_fp_both": all(x["dedup_raises_fp"] for x in (a, b)),
            "monotone_decomposition_both": all(
                x["guarantee"]["0.05"] < x["shift_conf_05"] < x["dedup_conf_05"] for x in (a, b)),
            "still_provisional": [k for k, x in (("esm2_35M", a), ("canonical", b)) if x["provisional"]],
            # the two figures the public surfaces now quote
            "shift_quant_canonical": b["shift_quant_05"], "shift_quant_35M": a["shift_quant_05"],
            "dedup_quant_canonical": b["dedup_quant_05"], "dedup_quant_35M": a["dedup_quant_05"]}


def separability():
    d = j("separability_results.json")
    return {"auroc": d["auroc_mean"]} if d else None


def fsi_aggregate():
    d = j("fsi_aggregate_results.json")
    a = d["aggregate"]["fsi_aggregate"]
    ci = a["bootstrap_ci_95"]
    return {"mean": a["mean"], "ci_low": ci["ci_95_low"], "ci_high": ci["ci_95_high"],
            "n": d["aggregate"]["n_structures"]}


def _fspe_pre_and_post():
    """Current FSPE ratios, the pre-numbering-fix snapshot, and which proteins actually moved.

    The snapshot is what makes the numbering fix auditable: without it the corrected file can only
    be compared against itself. `noise` is set two orders of magnitude above the forward-pass
    nondeterminism measured on this panel (1e-7 to 2.1e-6), so "moved" means re-masked, not re-run.
    Accessions are read by direct subscript rather than through a `.get(...) or .get(...)` hedge, so
    an artifact that renames the key fails the gate instead of quietly matching None against None.
    """
    cur = j("fspe_results.json")["per_protein"]
    pre = {e["uniprot_id"]: e["fspe_ratio"]
           for e in j("fspe_results_PRE_NUMBERING_FIX_2026_05_22.json")["per_protein"]}
    noise = 1e-4
    moved = sorted(e["uniprot_id"] for e in cur
                   if abs(e["fspe_ratio"] - pre[e["uniprot_id"]]) > noise)
    return cur, pre, moved, noise


def flip_count():
    """Cross-model FSPE sign flips over all three columns, which now share one numbering.

    History, because this quantity has been wrong in two different ways. The 2026-05-22 numbering
    fix re-ran ESM-2 alone, leaving ESM-3 and SaProt with mature-chain positions masked on a
    precursor. A count over all twelve rows then compared one corrected column against two stale
    ones, so the claim was narrowed to the nine rows where no column had moved and the other three
    were named indeterminate, pending a re-run.

    🟢 Cayuga job 3392513 did that re-run through the same offset resolver, so the narrowing is no
    longer needed and the indeterminate set is empty. `stale_columns` is asserted at zero rather than
    assumed: it recomputes which rows the ESM-2 fix moved and checks that ESM-3 and SaProt now carry
    a value for every one of them, so a future edit that reverted one column to a pre-fix artifact
    would fail here instead of quietly restoring a mixed-numbering count.
    """
    rows = j("mdrp_risk_table.json")["proteins"]
    _, pre, _, noise = _fspe_pre_and_post()
    cols = ["fspe_esm2", "fspe_esm3", "fspe_saprot"]
    side = lambda v: ">1" if v > 1 else "<1"          # noqa: E731

    # rows the ESM-2 correction moved; each must now carry a re-run value in the other two columns
    moved = {r["uniprot_id"] for r in rows
             if r["uniprot_id"] in pre and abs(r["fspe_esm2"] - pre[r["uniprot_id"]]) > noise}
    stale = sum(1 for r in rows if r["uniprot_id"] in moved
                and any(r.get(c) is None for c in ("fspe_esm3", "fspe_saprot")))

    flips = 0
    for r in rows:
        av = [v for v in (r.get(c) for c in cols) if v is not None]
        if len(av) >= 2 and len({side(v) for v in av}) > 1:
            flips += 1
    return {"n_rows": len(rows), "flips": flips,
            "all_columns_present": sum(1 for r in rows if all(r.get(c) is not None for c in cols)),
            "esm2_corrected_rows": sorted(moved), "stale_columns": stale,
            "indeterminate": []}

def external_baseline_numbers():
    """The three external classifier numbers § 11 compares against, read back out of the document.

    🔴 Added 2026-09-27, entry thirty-three. Every other claim in this file recomputes a number
    from an artifact, which is exactly why the external ones drifted unnoticed: nothing in the
    pipeline produces them, so nothing checks them. § 11 attributed "AUROC 0.92 on the standard
    576/576 virulence benchmark" to DTVF for as long as the sentence existed. 0.92 is DTVF's
    (0.9208); the 576/576 partition is DeepVF's, and DTVF never states its own split. The two were
    joined because DTVF reuses DeepVF's 3,576/4,910 pool, which is a sound inference and was written
    as a fact.

    This cannot verify a citation against its source — only a reader can do that, and one did. What
    it pins is that the numbers, the partition and the tool each stay attached to the paper they were
    read from, so a later edit cannot re-merge them.
    """
    import re
    text = (ROOT / "docs/MECHANISM_GENERALIZATION.md").read_text()
    head = text.split("## 11. What this does not claim", 1)[1].split("\n- **Not novel", 1)[0]
    flat = " ".join(head.split())
    return {
        # tool -> the figure § 11 attributes to it
        "deepvf_auc": float(re.search(r"\*\*DeepVF\*\*.*?\*\*AUC (0\.\d+)\*\*", head, re.S).group(1)),
        "dtvf_auroc": float(re.search(r"\*\*DTVF\*\*.*?\*\*AUROC (0\.\d+)\*\*", head, re.S).group(1)),
        "deepvic_auroc": float(re.search(r"\*\*DeepVIC\*\*.*?\*\*AUROC\n?\s*(0\.\d+)\*\*", head, re.S).group(1)),
        # the shared benchmark, named where it was actually built
        "benchmark_pool": re.search(r"(3,576) VFs and (4,910) non-VFs", head).groups(),
        "held_out": re.search(r"\*\*(576) VFs and (576) non-VFs\*\*", head).groups(),
        "deepvic_holdout": re.search(r"(13,384)-sequence holdout drawn from (33,456)", head).groups(),
        # the inference is labelled as one rather than asserted; flat() so a reflow of the
        # paragraph cannot silently turn either check off
        "split_inference_flagged": "**never states its own split**" in flat,
        "panel_size_stated": "self-built panel of 234" in flat,
    }


def vfdb_axis():
    """VFDB's own category and host structure, as the design note quotes it.

    \U0001f534 Added 2026-09-27. Unlike every other claim here, the source is a network download
    whose raw files are deliberately not committed (no license stated, 19 MB), so this recomputes
    from `results/vfdb_ingest.json` rather than from sequence. Regenerating that summary needs
    `python src/67_vfdb_ingest.py --download`. What it pins is that the two numbers the next design
    decision turns on cannot drift in the document: Exotoxin is a few per cent of VFDB, and the
    plant and insect host contrast exists only in the full set.
    """
    d = j("../results/vfdb_ingest.json")
    if d is None:
        return None
    a, b = d["setA"], d["setB"]
    return {
        "n": {"setA": a["n_records"], "setB": b["n_records"]},
        "n_categories": {"setA": a["n_categories"], "setB": b["n_categories"]},
        "categories_ge_20": {"setA": len(a["categories_ge_20"]), "setB": len(b["categories_ge_20"])},
        "exotoxin": {"setA": a["exotoxin"], "setB": b["exotoxin"]},
        "exotoxin_frac": {"setA": a["exotoxin_frac"], "setB": b["exotoxin_frac"]},
        "non_toxin": {"setA": a["non_toxin_virulence"], "setB": b["non_toxin_virulence"]},
        # setA is the experimentally verified core and has no plant or insect pathogen in it
        "host": {"setA": a["hosts"], "setB": b["hosts"]},
        "setA_non_human": a["host_non_human"],
        "setB_non_human": b["host_non_human"],
        "exotoxin_by_host_setB": b["exotoxin_by_host"],
        "effector_largest": max(b["categories"], key=b["categories"].get),
    }


def benign_enzyme_set():
    """The A2 control set, recomputed from its own artifact.

    The floor the mutation preregistration named is n_benign of roughly 30 and the delivered set is
    60, so what matters most here is that the *rules* stayed as frozen: rule 4 never bound (every
    candidate is far below the similarity bound), rule 1 never bound, and the two rules that did
    bind are the Active-site count and the VFDB sequence match. A later edit that loosened an
    exclusion to raise n would change these counts.
    """
    d = j("../results/benign_enzyme_set.json")
    if d is None:
        return None
    rc = d["rejected_counts"]
    sites = d["sites_per_control"]
    return {
        "n_admitted": d["n_admitted"], "target_n": d["target_n"],
        "meets_target": d["meets_target"], "pool": d["candidate_pool"],
        "window": [d["panel_window"]["lo"], d["panel_window"]["hi"]],
        "bound": round(d["similarity_bound"], 6),
        # the maximum similarity actually reached, which is what says rule 4 never bound
        "max_similarity": max(a["max_similarity_to_panel_positive"] for a in d["admitted"]),
        "rejected_active_sites": rc.get("5", 0),
        "rejected_vfdb": rc.get("2", 0),
        "rejected_hazard_term": rc.get("3", 0),
        "rejected_panel": rc.get("1", 0),
        "rejected_similarity": rc.get("4", 0),
        "min_sites": min(sites), "ec_hydrolase": d["ec_first_digit"].get("3", 0),
        "n_ec_classes": len([k for k in d["ec_first_digit"] if k != "?"]),
        "viral": [a["acc"] for a in d["admitted"] if "virus" in a["organism"].lower()],
        "order": d["order"].split(" — ")[0],
    }


def fspe_m_a2():
    """P2 at n_benign = 60, and whether the gate's AUROC half can ever be a test.

    The whole point of study A2 was to take a verdict that fired on direction at n = 4 and see
    whether it survives power. It does, in the direction the preregistration predicted: the gap
    widens rather than closing. Two things are pinned besides the verdict. The panel's values must
    stay bit-identical to the frozen-four run, because if they move then the control set was not the
    only thing that changed. And the gate-power sweep must keep showing that the null sd floors out,
    because that is what makes the fifth amendment's resolution permanent rather than provisional.
    """
    d, g = j("../results/fspe_m_a2.json"), j("../results/a2_gate_power.json")
    base = j("../results/fspe_m.json")
    if d is None or g is None or base is None:
        return None
    p2, sw = d["P2"], g["sweep_n_benign_at_fixed_panel_14"]
    pan = {r["acc"]: r["dfspe_m"] for r in d["panel"] if r["dfspe_m"] is not None}
    old = {r["acc"]: r["dfspe_m"] for r in base["panel"] if r["dfspe_m"] is not None}
    ben = [c["dfspe_m"] for c in d["controls"] if c["dfspe_m"] is not None]
    return {
        "n_benign": p2["n_benign"], "n_toxin": p2["n_toxin"],
        "panel_mean": round(p2["toxin_mean"], 4), "benign_mean": round(p2["benign_mean"], 4),
        "difference": round(p2["difference"], 4), "perm_p": p2["permutation_p"],
        "auroc": round(p2["auroc"], 4), "auroc_ci95": p2["auroc_ci95"],
        "interval_clear_of_half": p2["auroc_interval_clear_of_half"],
        # the control set is the only thing that changed
        "panel_identical_to_frozen4": all(abs(pan[k] - old[k]) < 1e-9 for k in pan if k in old),
        "controls_above_panel_min": sum(1 for b in ben if b > min(pan.values())),
        "controls_above_zero": sum(1 for b in ben if b > 0),
        "lone_negative_control": [c["acc"] for c in d["controls"]
                                  if c["dfspe_m"] is not None and c["dfspe_m"] < 0],
        # A2-3: no leak, and the gate's own AUROC half still cannot discriminate
        "shuffled_auroc": d["P5_gate"]["shuffled_auroc"],
        "gate_null_sd_n60": g["a2"]["null_sd"], "gate_two_sided_p": g["a2"]["two_sided_p_for_observed"],
        "gate_is_a_test_n60": g["a2"]["is_a_test"],
        # the fifth amendment's n = 4 numbers, reproduced exactly by this script
        # the fifth amendment prints 0.1669, 77.3% and 0.224; matched to its own printed precision
        "reproduces_fifth_amendment": (abs(g["frozen4"]["null_sd"] - 0.1669) < 5e-4
                                       and abs(g["frozen4"]["clean_pipeline_fails_this_gate"] - 0.773) < 5e-4
                                       and abs(g["frozen4"]["two_sided_p_for_observed"] - 0.224) < 5e-4),
        "sweep_floor_sd": sw["5000"]["null_sd"], "sweep_floor_passes": sw["5000"]["p_within_tolerance"],
        "sweep_any_is_a_test": any(v["is_a_test"] for v in sw.values()),
    }


def cited_entries_exist():
    """Every "<date> entry" citation resolves to a heading that exists in the corrections log.

    🔴 Added 2026-09-24, entry twenty-two, after the second occurrence of the same defect. Three
    sentences — two on the Hugging Face card, one inside the log itself — cited a "2026-09-10 entry"
    of docs/DATA_CORRECTIONS.md, and no entry of that date has ever existed. The first occurrence was
    2026-09-20, when an entry was cited by number for two days before being written. A citation is
    cheap to write and nothing downstream reads it, so it is exactly the kind of reference that rots
    silently; a reader who follows it concludes the record is missing rather than misnamed.

    A date inside double quotes is being discussed rather than cited, which is how a corrected
    sentence names the citation it is replacing, so those are skipped.
    """
    import re
    log = (ROOT / "docs/DATA_CORRECTIONS.md").read_text()
    headings = set(re.findall(r"^## (\d{4}-\d{2}-\d{2})", log, re.M))
    scanned = [*PUBLIC, "docs/DATA_CORRECTIONS.md"]
    dangling, n = {}, 0
    for rel in scanned:
        f = ROOT / rel
        if not f.exists():
            continue
        text = f.read_text()
        quoted = set(re.findall(r'"(\d{4}-\d{2}-\d{2}) entry', text))
        for d in re.findall(r"(\d{4}-\d{2}-\d{2}) entry", text):
            if d in quoted:
                continue
            n += 1
            if d not in headings:
                dangling.setdefault(rel, []).append(d)
    return {"headings": len(headings), "citations": n,
            "dangling": {k: sorted(set(v)) for k, v in dangling.items()},
            "scanned": len(scanned)}


def conformal_operating_point():
    """The per-class table at a guaranteed threshold, beside the published one, from the artifacts.

    🔴 Added 2026-09-26. § 2.6 established that `np.quantile` misses its nominal rate and § 2.6.1
    that conformal holds; neither was applied to the table anyone reads until src/58. What is
    asserted here is the part that decides how every recovery figure in this repository must be
    read: the realized false-positive rates, the one-directional cost of moving to a guaranteed
    threshold, and the fact that the frozen v2 panel cannot express a nominal 1% budget at all.

    The reproduction flag is asserted too. src/58 re-implements 03b's fold, so if its quantile
    column stopped reproducing lomo_results.json the conformal column would be measuring two things
    and every number below would be void.
    """
    out = {}
    for panel, tag in (("v2", ""), ("v3", ""), ("v3", "_esmc_600M"), ("v3", "_esm3_1_4B")):
        d = j(f"{panel}/conformal_operating_point{tag}.json")
        if d is None:
            continue
        pc = d["per_class"]
        cell = {"m": d["calibration_m"], "reproduces": d["reproduces_published_quantile"],
                "k": {a: v for a, v in d["conformal_k"].items()},
                "unreachable": d["alphas_unreachable_by_conformal"]}
        for al in ("0.05", "0.01"):
            rows = [(c, v[f"quantile_{al}"], v[f"conformal_{al}"]) for c, v in pc.items()
                    if v.get(f"quantile_{al}") and v.get(f"conformal_{al}")]
            if not rows:
                cell[al] = None
                continue
            deltas = [(cf["recovery"] - q["recovery"]) * 100 for _, q, cf in rows]
            cell[al] = {"n_classes": len(rows),
                        "n_dropped": sum(1 for x in deltas if x < -0.05),
                        "n_rose": sum(1 for x in deltas if x > 0.05),
                        "mean_delta_pts": sum(deltas) / len(deltas),
                        "fp_quantile": rows[0][1]["realized_fp"],
                        "fp_conformal": rows[0][2]["realized_fp"]}
        cell["beta_q"] = (pc.get("beta_lactamase", {}).get("quantile_0.05") or {}).get("recovery")
        cell["beta_c"] = (pc.get("beta_lactamase", {}).get("conformal_0.05") or {}).get("recovery")
        cell["phage_q"] = (pc.get("phage_peptidoglycan_hydrolase", {})
                           .get("quantile_0.05") or {}).get("recovery")
        cell["phage_c"] = (pc.get("phage_peptidoglycan_hydrolase", {})
                           .get("conformal_0.05") or {}).get("recovery")
        out[f"{panel}{tag}"] = cell
    return out


def fspe_background_ablation():
    """FSPE's background three ways, and the sampling variance that fell out of asking.

    🔴 Added 2026-09-27. Three documents described a background excluding the +/-2 flanking
    positions around each functional site and no code ever built one. What is asserted here is the
    part that decides how a per-protein FSPE ratio may be read: that the published arm reproduces
    the published artifact, that the headline is identical under all three backgrounds, and that
    redrawing the background moves ratios by an order of magnitude more than the flanking defect
    does. The last one is the finding; the first is what makes it attributable.
    """
    d = j("fspe_background_ablation.json")
    if d is None:
        return None
    sites = json.load(open(ROOT / "data/annotations/functional_sites.json"))
    excl = {a for a in sites if not a.startswith("_")
            and sites[a]["functional_sites"].get("fspe_excluded")}
    rows = d["per_protein"]

    def arm(name):
        return {x["uniprot_id"]: x["arms"][name]["fspe_ratio"] for x in rows}

    pub, drop, res = arm("published"), arm("drop_flanking"), arm("resample_excluded")
    affected = [x["uniprot_id"] for x in rows if x["flanking_in_published_sample"]]
    n_flank = sum(len(x["flanking_in_published_sample"]) for x in rows)
    clean = [a for a in pub if a not in affected]

    def below(a, ex=True):
        return sum(1 for k, v in a.items() if (k not in excl or not ex) and v < 1.0)

    def n(a, ex=True):
        return sum(1 for k in a if k not in excl or not ex)

    return {
        "reproduces_published": d["reproduces_published"], "n_drift": len(d["drift"]),
        "n_proteins": len(rows), "n_affected": len(affected), "n_flanking_positions": n_flank,
        # the headline, with src/21's exclusion applied as the published figure applies it
        "excluded_counts": {k: f"{below(a)}/{n(a)}" for k, a in
                            (("published", pub), ("drop", drop), ("resample", res))},
        # effect 1: the contamination, isolated. Unaffected proteins must not move at all.
        "max_drop_delta": max(abs(drop[a] - pub[a]) for a in affected),
        "unaffected_move_by_zero": all(drop[a] == pub[a] for a in clean),
        "crossings_drop": sum(1 for a in pub if (pub[a] < 1) != (drop[a] < 1)),
        # effect 2: redrawing the background, measured where there is no flanking to remove
        "max_resample_delta": max(abs(res[a] - pub[a]) for a in pub),
        "mean_resample_delta_on_clean": sum(abs(res[a] - pub[a]) for a in clean) / len(clean),
        "crossings_resample": sum(1 for a in pub if (pub[a] < 1) != (res[a] < 1)),
        "closest_to_one_after_resample": max(res.values(), key=lambda v: v if v < 1 else 0),
    }


def negative_test_partition():
    """The standing test partition, from the frozen artifact plus a rebuild of its rule.

    🔴 Added 2026-09-27 with the partition. Criterion 1 moved from fail to partial on the strength
    of this file existing, so the claim asserts the things that make it a partition rather than a
    draw: a written bound taken from the panel's own negatives rather than a literal, membership
    frozen under a digest, and the test-only role stated in the artifact. It also asserts the two
    distinct-name conventions separately, because three different figures for "distinct names in
    this pool" already existed and two of them are valid measures of different quantities.
    """
    d = j("../data/sequences/negative_test_partition_v3.json")
    if d is None:
        return None
    screen = j("v3/pool_homology_against_panel.json")
    c, r = d["counts"], d["resolution"]
    admitted = {x["acc"] for x in d["admitted"]}
    return {
        "admitted": c["admitted"], "rejected": c["rejected"], "pool": c["pool"],
        "distinct_names": c["distinct_names"], "gene_symbols": c["distinct_gene_symbols"],
        # the bound is the panel negatives' own maximum, read from the screen and not a literal
        "bound": screen["panel_negative_reference"]["max"],
        "bound_in_rule": f"{screen['panel_negative_reference']['max']:.6f}" in d["rule"]["3"],
        # tighter than the 0.30 positives rule, and free: nothing sits between the two
        "n_above_bound": sum(1 for x in screen["pool_max_similarity"]
                            if x["similarity"] >= screen["panel_negative_reference"]["max"]),
        "n_above_030": sum(1 for x in screen["pool_max_similarity"] if x["similarity"] > 0.30),
        "rejected_accs": sorted(x["acc"] for x in d["rejected"]),
        "test_only_stated": d["role"].startswith("TEST ONLY"),
        "exchangeability_voided": "VOIDED BY DESIGN" in d["exchangeability"],
        "digest_matches_membership":
            d["sha256_of_sorted_accessions"] == __import__("hashlib").sha256(
                "\n".join(sorted(admitted)).encode()).hexdigest(),
        "panel_rate": r["panel_finest_rate"], "name_rate": r["finest_rate_by_distinct_name"],
    }


def benign_control_sites():
    """P2's controls as sequences, with the offset each one needed, from the artifact.

    🔴 Added 2026-09-27. P2 of the mutation preregistration names four benign controls "already in
    the repository"; one was absent and none had a sequence or a sequence-coordinate position. What
    is asserted is that all four now verify under src/46's rule -- a UNIQUE offset placing EVERY
    expected identity -- and that thermolysin's position set is catalytic rather than the 18-position
    mixture UniProt's Binding site features give if calcium is not excluded.
    """
    d = j("../data/annotations/benign_control_sites.json")
    if d is None:
        return None
    v = d["verified"]
    return {
        "n_verified": d["n_verified"], "n_rejected": d["n_rejected"],
        "keys": sorted(v),
        "offsets": {k: v[k]["offset"] for k in sorted(v)},
        "residues": {k: v[k]["residues"] for k in sorted(v)},
        "n_positions": {k: len(v[k]["positions"]) for k in sorted(v)},
        # thermolysin is the one read from UniProt rather than from annotation text, and the one
        # where a ligand filter was needed
        "thermolysin_excluded_ca_sites":
            len(v.get("1LNF", {}).get("excluded_non_catalytic_ligand_sites", [])),
        "thermolysin_from_uniprot": "UniProt" in v.get("1LNF", {}).get("identity_source", ""),
        "text_sourced": sorted(k for k in v if v[k]["identity_source"] == "annotation text"),
    }


def fspe_m():
    """The mutation axis's preregistered outcome, from its artifact.

    🔴 Added 2026-09-27. P1 is supported and P2 fails on the ceiling section 4 wrote for it, so the
    claim asserts both, plus the two facts that make the pair readable: the exception set is FSPE's
    own, and the benign controls out-rank almost the whole panel. It also asserts the gate's leakage
    half rather than the composite verdict, because the AUROC half was shown to have no power at
    n_benign = 4 and must not be allowed to void or bless the study.
    """
    d = j("../results/fspe_m.json")
    if d is None:
        return None
    panel = {r["acc"]: r["dfspe_m"] for r in d["panel"] if r["dfspe_m"] is not None}
    ctrl = {c["name"]: c["dfspe_m"] for c in d["controls"] if c["dfspe_m"] is not None}
    g, p1, p2 = d["P5_gate"], d["P1"], d["P2"]
    best_ctrl = max(ctrl.values())
    return {
        "gate_centred": g["centred_on_zero"],
        "gate_mean_over_se": abs(g["shuffled_mean_dfspe_m"]) /
                             (g["shuffled_sd"] / (len(d["panel"]) + len(d["controls"])) ** 0.5),
        "gate_auroc_reported_not_gating": g["shuffled_auroc"],
        "p1_k": p1["all"]["k_above_0"], "p1_n": p1["all"]["n"], "p1_p": p1["all"]["sign_p"],
        "p1_supported": p1["supported"],
        "p1_excl": (p1["seb_excluded"]["k_above_0"], p1["seb_excluded"]["n"]),
        # the exception set, which is FSPE's own type-2 RIP pair
        "below_zero": sorted(k for k, v in panel.items() if v <= 0),
        "p2_panel_mean": p2["toxin_mean"], "p2_benign_mean": p2["benign_mean"],
        "p2_difference": p2["difference"], "p2_auroc": p2["auroc"],
        "p2_perm_p": p2["permutation_p"],
        "p2_ceiling_fires": p2["benign_mean"] >= p2["toxin_mean"],
        "p2_n_benign": p2["n_benign"],
        "panel_below_best_control": sum(1 for v in panel.values() if v < best_ctrl),
        "n_panel": len(panel),
        "p6_unrun": d["P6"].startswith("NOT RUN"),
    }


def fspe_m_p3():
    """P3's outcome, and the residue-identity confound that decides how it may be read.

    Added 2026-09-27. Asserts the frozen verdict, and the split by substituted residue that shows
    the statistic is dominated by scanning design: cysteine at the bottom of the ranking, alanine
    in the middle, 79% of the set one or the other.
    """
    d = j("../results/fspe_m_p3.json")
    if d is None:
        return None
    rows = [r for r in d["rows"] if r.get("percentile") is not None]
    def med(sel):
        v = sorted(r["percentile"] for r in rows if sel(r))
        return v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2
    ala = [r for r in rows if r["alt"] == "A"]
    cys = [r for r in rows if r["alt"] == "C"]
    return {"n": d["n_scored"], "median": d["median_percentile"],
            "k_below_25": d["sign_test"]["k_below_25"], "sign_p": d["sign_test"]["p"],
            "supported": d["supported"], "ceiling": d["ceiling_triggered"],
            "n_ala": len(ala), "n_cys": len(cys),
            "median_ala": med(lambda r: r["alt"] == "A"),
            "median_cys": med(lambda r: r["alt"] == "C"),
            "scan_share": (len(ala) + len(cys)) / len(rows),
            "top_share": d["largest_protein_share"]}


def fspe_m_p5_composition():
    """P5's composition half, and the noise floor that makes its margin unusable at this n."""
    d = j("../results/fspe_m_p5_composition.json")
    if d is None:
        return None
    return {"n_panel": d["n_panel"], "n_control": d["n_control"],
            "auroc_dfspe_m": d["auroc_dfspe_m"], "auroc_composition": d["auroc_composition"],
            "difference": d["difference"], "margin": d["required_margin"],
            "null_sd": d["null_auroc_sd"], "inside_noise": d["margin_inside_noise"],
            "both_below_half": (d["auroc_dfspe_m"] < 0.5 and d["auroc_composition"] < 0.5),
            "indeterminate": d["verdict"].startswith("INDETERMINATE")}


def fspe_m_p6():
    """P6: the alignment baseline, its indeterminate frozen half, and what it reproduces.

    Added 2026-09-27. The load-bearing assertions are the descriptive ones: the alignment agrees with
    the model on direction and not magnitude, reproduces P1's direction at a weaker p, and reproduces
    P2's failure with a wider gap. Those are what section 4's own sentence about alignments asks for,
    and unlike its AUROC threshold they are not inside the noise at n_control = 4.
    """
    d = j("../results/fspe_m_p6_pssm.json")
    if d is None:
        return None
    ok = [r for r in d["rows"] if r.get("dpssm") is not None]
    pan = [r for r in ok if r["panel"] == 1]
    ctl = [r for r in ok if r["panel"] == 0]
    mean = lambda g, k: sum(r[k] for r in g) / len(g)  # noqa: E731
    ag = d["descriptive_agreement"]
    shallow = sorted(r["acc"] for r in pan if r["dpssm"] <= 0)
    return {
        "n_scored": d["n_scored"], "n_panel": len(pan), "n_control": len(ctl),
        "auroc_dfspe_m": d["auroc_dfspe_m"], "auroc_pssm": d["auroc_pssm"],
        "difference": d["difference"], "null_sd": d["null_auroc_sd"],
        "inside_noise": d["margin_inside_noise"],
        "pearson": ag["pearson"], "spearman": ag["spearman"],
        "pssm_p1": (ag["pssm_p1_k_above_0"], ag["pssm_p1_n"]), "pssm_p1_p": ag["pssm_p1_sign_p"],
        # the alignment fails P2's comparison in the same direction and by more
        "pssm_panel_mean": mean(pan, "dpssm"), "pssm_control_mean": mean(ctl, "dpssm"),
        "pssm_gap": mean(pan, "dpssm") - mean(ctl, "dpssm"),
        "dfspe_gap": mean(pan, "dfspe_m") - mean(ctl, "dfspe_m"),
        "nonpositive_panel": shallow,
        "min_depth": min(r["mean_column_depth"] for r in ok),
        "max_depth": max(r["mean_column_depth"] for r in ok),
    }


def typicality_baseline():
    """Margin against a label-free typicality baseline, on both arms that have pool embeddings.

    Added 2026-09-27, prompted by arXiv 2606.12609's nativeness axis. The load-bearing assertions are
    that the baseline is a REAL predictor on the canonical arm -- it was missing and criterion 14
    required it -- that margin survives controlling for it on both arms, and that the baseline does
    NOT replicate while margin does. The last one is what makes typicality a confound on one arm
    rather than an explanation of the finding.
    """
    d = j("../results/typicality_baseline.json")
    if d is None:
        return None
    a = d["arms"]
    return {
        "arms": sorted(a),
        "margin": {k: v["margin_vs_recovery"] for k, v in a.items()},
        "typicality": {k: v["typicality_vs_recovery"] for k, v in a.items()},
        "margin_partial": {k: v["margin_vs_recovery_typicality_controlled"] for k, v in a.items()},
        "typicality_partial": {k: v["typicality_vs_recovery_margin_controlled"]
                               for k, v in a.items()},
        "typicality_perm_p": {k: v["typicality_perm_p"] for k, v in a.items()},
        "margin_dominates_both": all(v["margin_dominates"] for v in a.values()),
        "n_classes": {k: v["n_classes"] for k, v in a.items()},
    }


def functional_site_numbering():
    """The mature-chain numbering fix, pinned from the artifacts alone.

    One annotation field (`precursor_offset`) feeds twelve consumers across two coordinate systems,
    and the guard against re-breaking it is the residue-identity check in `utils`. This is the
    release-surface half of that guard: it cannot import `utils` (this job installs numpy and
    nothing else), so rather than re-deriving identities it pins the shape of the correction, which
    is what a regression would disturb. Three offsets, each carrying a verified note; three entries
    flagged in the annotation file but only two reaching the runtime, because P55981 has no
    catalytic residues to mis-index; every indexed position equal to its annotated position plus the
    offset; and, against the pre-fix snapshot, exactly the three offset carriers moved while the
    other twelve stayed inside float noise. That last condition is the load-bearing one: it is what
    says the field was threaded through every consumer and not just the headline.
    """
    fs = json.load(open(ROOT / "data/annotations/functional_sites.json"))
    entries = {k: v for k, v in fs.items() if not k.startswith("_")}
    offsets, verified_notes, flags = {}, 0, []
    for acc, e in entries.items():
        site = e.get("functional_sites", {})
        if site.get("precursor_offset", 0):
            offsets[acc] = site["precursor_offset"]
            if "Verified" in str(site.get("_precursor_offset_note", "")):
                verified_notes += 1
        if "_numbering_flag" in site:
            flags.append(acc)

    # The anti-p-hacking half. Each offset moves a ratio in the direction this project's own claim
    # wants, so the offset has to be pinned by something that does not mention FSPE. `src/46` records
    # every offset achieving a FULL identity match; a singleton list means the value was not chosen.
    # Absent when src/46 has not been re-run (it needs torch), so the claim tolerates None rather
    # than turning a missing optional artifact into a failed gate.
    na = j("v3/functional_site_numbering_audit.json")
    uniq = None if not na else {
        acc: {"full_match_offsets": v["offset_uniqueness"]["full_match_offsets"],
              "n_checkable": v["offset_uniqueness"]["n_checkable"],
              "determined": bool(v["offset_is_uniquely_determined"])}
        for acc, v in na["recomputed"].items()
        if isinstance(v.get("offset_uniqueness"), dict)} or None

    cur, pre, moved, _ = _fspe_pre_and_post()
    scored = {e["uniprot_id"] for e in cur}
    return {"entries": len(entries), "n_fspe": len(cur), "offsets": offsets,
            "offset_uniqueness": uniq,
            "verified_notes": verified_notes, "annotation_flags": sorted(flags),
            "runtime_flagged": sorted(e["uniprot_id"] for e in cur if e["numbering_flagged"]),
            "skipped_no_catalytic_residues": sorted(set(flags) - scored),
            "indexing_consistent": all(
                e["residues_indexed"] == [r + e["precursor_offset"]
                                          for r in e["residues_annotated"]] for e in cur),
            "moved": moved, "moved_is_the_offset_set": moved == sorted(offsets),
            "max_unmoved_delta": max(abs(e["fspe_ratio"] - pre[e["uniprot_id"]])
                                     for e in cur if e["uniprot_id"] not in moved),
            "below_1_pre": int(sum(1 for v in pre.values() if v < 1)),
            "below_1_now": int(sum(1 for e in cur if e["fspe_ratio"] < 1))}


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



def lomo_class_recovery():
    """The published per-class recovery figures for the canonical 650M arm. These
    are the numbers docs/MECHANISM_GENERALIZATION.md leads with, so they are pinned
    to the artifact rather than to whatever was true when the table was typed."""
    d = json.load(open(R / "v2/lomo_results.json"))
    c = d["leave_one_mechanism_out"]
    def g(k):
        return round(c[k]["flagged_95_mean"], 4), round(c[k]["flagged_99_mean"], 4)

    return {"baseline_auroc": round(d["baseline_auroc"][0], 4),
            "n_pos": d["n_positive"], "n_neg": d["n_negative"],
            "beta_lactamase": g("beta_lactamase"),
            "beta_lactamase_auroc": round(c["beta_lactamase"]["auroc_mean"], 3),
            "superantigen": g("superantigen_enterotoxin"),
            "clostridial": g("clostridial_neurotoxin"),
            "t3ss": g("t3ss_effector_apparatus"),
            "pore_forming": g("pore_forming_cytolysin"),
            "provenance_auroc": round(d["provenance_auroc"][0], 3),
            "organism_agreement": round(d["organism_label_agreement_with_hazard"], 2)}



def annotation_coverage():
    """Every annotation file must carry an entry for every panel member. The
    2026-09-05 expansion updated the mechanism-class file and the FASTA but not
    localization_v2.json, and 03d and 03g both index it by fasta_id: they raised
    KeyError on the first new member for every model arm, in a sweep that had
    already spent GPU hours. The expansion had been running for a day before this
    surfaced. Coverage is the cheap check that catches it before the compute does.

    Entries, not values: structure_3di_v2.json deliberately carries the SaProt mask
    for proteins with no AlphaFold entry, so a masked record is coverage."""
    panel = json.load(open(ROOT / "data/sequences/panel_v2_manifest.json"))
    mech = json.load(open(ROOT / "data/annotations/mechanism_classes_v2.json"))
    members = {p["acc"] for p in panel["positives"]} | {n["acc"] for n in panel["negatives"]}
    pos_ids = {e["fasta_id"] for e in mech["proteins"]}
    out = {"panel": len(members), "mechanism_classes": len(pos_ids)}
    gaps = {}
    for name, path, key in (("localization", "localization_v2.json", "proteins"),
                            ("structure_3di", "structure_3di_v2.json", "proteins")):
        f = ROOT / "data/annotations" / path
        if not f.exists():
            gaps[name] = "file missing"
            continue
        have = set(json.load(open(f))[key])
        missing = sorted(members - have)
        out[name] = len(have)
        if missing:
            gaps[name] = f"{len(missing)} missing, first {missing[0]}"
    # the positives must also all be in the mechanism-class annotation
    pos_acc = {p["acc"] for p in panel["positives"]}
    if pos_acc - pos_ids:
        gaps["mechanism_classes"] = f"{len(pos_acc - pos_ids)} positives unassigned"
    out["gaps"] = gaps
    return out



def beta_lactamase_provenance():
    """Beta-lactamase is the only mechanism class in the panel with a written reason
    that never mentions the panel's own hazard keywords. Every other class's members
    trace to KW-0800 (Toxin) or KW-0843 (Virulence); beta-lactamase entered through a
    separate protein_name match for an antimicrobial-resistance query
    (src/01_collect_data.py), and carries neither hazard keyword on UniProt --
    KW-0046 "Antibiotic resistance" instead. This does not affect any recovery
    number, but it was undocumented until 2026-09-11 and is the kind of drift this
    audit exists to catch. Pins that the disclosure stays in the documents rather
    than getting edited away as a later pass "cleans up" the prose."""
    mech = json.load(open(ROOT / "data/annotations/mechanism_classes_v2.json"))
    n = sum(1 for e in mech["proteins"] if e["mechanism_class"] == "beta_lactamase")
    return {"beta_lactamase_members": n}


def amr_category_test():
    """The first AMR-category test. Its NUMBER is sound and reproduces exactly:
    8 members, 7 flagged, 87.5%. Its INTERPRETATION was not, and src/25 supersedes
    it. This entry keeps pinning the number so the artifact cannot drift, while the
    amr_category_test_v2 entry pins the corrected verdict. Kept separate on purpose:
    the defect was in what the test compared, not in what it computed."""
    d = json.load(open(R / "v2/amr_category_test.json"))
    return {"n_members": len(d["members"]), "recovery": round(d["recovery_at_95pct"], 4),
            "predicted_supported_leq": d["predicted_supported_if_recovery_leq"],
            "predicted_refuted_geq": d["predicted_refuted_if_recovery_geq"],
            "n_flagged": sum(d["flagged"].values())}


def amr_category_test_v2():
    """The corrected AMR-category test. src/24 trained on all 80 internal positives,
    14 of which ARE beta-lactamases, so its 87.5% measured whether the probe reaches
    a new AMR family having already seen AMR; and it then set that number against
    beta-lactamase's 21% from the LOMO protocol rather than its own. src/25 fixes
    both, running a 2x2 over {train with AMR, train without AMR} x {test
    beta-lactamase, test aminoglycoside} under one protocol. Reads every cell plus
    the protocol-matched column for all internal classes, so the corrected verdict
    -- inconclusive at 75.0% -- cannot drift back to the stronger claim. The
    with-AMR/aminoglycoside cell must still reproduce 24's 87.5% exactly; if it
    stops doing so the embedding pipeline has drifted and the comparison is void."""
    d = json.load(open(R / "v2/amr_category_test_v2.json"))
    c = d["cells"]
    col = d["protocol24_column_internal_classes"]
    return {
        "reproduces_v1": round(c["train_with_amr__test_aminoglycoside"]["recovery_at_95pct"], 4),
        "decisive": round(c["train_without_amr__test_aminoglycoside"]["recovery_at_95pct"], 4),
        "bl_matched": round(c["train_without_amr__test_beta_lactamase"]["recovery_at_95pct"], 4),
        "bl_protocol24": round(col["beta_lactamase"]["recovery_at_95pct"], 4),
        "bl_is_lowest": min(col, key=lambda k: col[k]["recovery_at_95pct"]) == "beta_lactamase",
        "verdict_inconclusive": "INCONCLUSIVE" in d["verdict"],
    }


def profile_hmm_baseline():
    """§7.1's profile baseline. 03i supplied the alignment half of "alignment- or
    profile-based"; 03k supplies the profile half with HMMER, written as a strict
    swap-in that changes only the score matrix. Pins three things: that the probe
    leads both HMMER modes on every class, that beta-lactamase stops being
    alignment's one win once scoring is profile-based, and that neither run is
    degenerate. The last one matters because the first jackhmmer run returned 100%
    on every class INCLUDING the labelled control, purely from margins tied at
    zero; a calibrated threshold of 0 is the tell, and it must never be reported
    as a result."""
    out = {}
    for m in ("phmmer", "jackhmmer"):
        d = json.load(open(R / f"v2/profile_hmm_baseline_{m}.json"))
        out[m] = {
            "probe_minus_profile": round(d["probe_minus_profile_mean"], 4),
            "beats_probe": d["classes_profile_beats_probe"],
            "degenerate": d["degenerate_classes"],
            "beta_lactamase": round(d["classes"]["beta_lactamase"]["profile_recovery"], 4),
            "density": round(d["matrix_density"], 4),
        }
    return out


def swissprot_profile_pilot():
    """The power check that stopped the external-database profile run before it was
    made. Pins the two halves that matter together: profile construction SUCCEEDED
    (every profile found all of its held-out siblings) while the LOMO margin was
    still negative. Keeping both in one claim prevents the result being retold as
    "HMMER was set up wrong" later, and prevents the negative margin being quietly
    dropped as a failed run. It was neither: a family profile cannot answer a
    question about a different family."""
    d = json.load(open(R / "v2/swissprot_profile_pilot.json"))
    return {"n": d["n_members"], "margin_mean": round(d["margin_mean"], 2),
            "n_negative": d["n_margin_negative"],
            "all_siblings": d["all_siblings_found"]}


def pretraining_caveat_tests():
    """§9.3.1's two attempts and their verdicts, pinned together so neither can be
    retold as a success. The external holdout must stay INVALID: both runs produced
    high-looking recovery (85%, 89%) off an AUROC whose confidence interval contains
    0.5, which is a blanket false-positive rate rather than detection. The exposure
    correlation must stay NOT surviving its controls: pooled over members it reads
    rho -0.399 at p 0.0005, and that is pseudoreplication, since recovery is a
    class-level property and the real n is nine."""
    h = json.load(open(R / "v2/pretraining_holdout.json"))
    e = json.load(open(R / "v2/pretraining_exposure.json"))
    return {"holdout_valid": h["valid"], "holdout_auroc": round(h["auroc_post_snapshot"], 3),
            "holdout_ci_low": round(h["auroc_ci95"][0], 3),
            "exposure_pooled_p": round(e["pooled"]["p"], 5),
            "exposure_class_p": round(e["class_level"]["p"], 4),
            "exposure_survives": e["survives_controls"]}


def plm_fusion():
    """VF-Fuse fuses ESM-2 with ProtT5; here that costs 0.4 points. Pinned because
    the two models ARE complementary per class, so the tempting summary is that
    fusion should help, and the artifact says it does not."""
    d = json.load(open(R / "v2/plm_fusion_baseline.json"))
    return {"best": d["best"], "delta": round(d["concat_minus_esm2"], 4),
            "esm2": round(d["means"]["esm2_650M"], 4),
            "concat": round(d["means"]["concat"], 4)}


def prevalence_adjusted():
    """The panel is 34.2% hazardous and a screening queue is not. Pins the first
    AUPRC ever computed here alongside the precision collapse it implies, because
    the tempting summary is the AUROC and the AUROC is the one number that does not
    move with base rate. The 95% specificity row is the operating point every
    recovery figure in this repository uses."""
    d = json.load(open(R / "v2/prevalence_adjusted.json"))
    op = d["operating_points"]["spec_0.95"]
    return {"panel_prevalence": round(d["panel_prevalence"], 4),
            "auprc": round(d["auprc_mean"], 3),
            "auprc_baseline": round(d["auprc_baseline"], 3),
            "precision_panel": round(op["precision"]["0.342"], 3),
            "precision_1e-3": round(op["precision"]["0.001"], 4)}


def layer_depth():
    """§9.5. Section 9 lists axes that do not change the conclusions; depth does,
    and it went unchecked until 2026-09-18. Pins three things together: that layer
    12 beats the final layer, that the platform check separating depth from
    cluster-versus-local embedding actually passed, and that beta-lactamase reaches
    0.0% in the middle layers, which is what stops "wrong layer" becoming a fifth
    explanation for the anomaly."""
    d = json.load(open(R / "v2/layer_depth_sweep.json"))
    return {"best": d["best"], "gain": round(d["best_minus_final"], 4),
            "platform_valid": d["platform_check"]["valid"],
            "platform_rel": d["platform_check"]["relative"],
            "bl_L12": round(d["classes"]["L12"]["beta_lactamase"], 4),
            "final_mean": round(d["means"]["final (33)"], 4)}


def target_host_control():
    """§2.4. The fourth confound, and the one §2 never had: all three published
    controls hold the protein's ORIGIN constant and none holds constant what it
    ACTS ON. Pins that target host is more legible than provenance (0.929 against
    0.818), that the headline 0.973 is a blend of 0.994 animal-target and 0.898
    non-animal-target, that the gap survives size matching, and that the rank
    pattern is flagged post hoc. The last field matters: a post-hoc rank test on
    eight non-independent classes must not harden into a preregistered result."""
    d = json.load(open(R / "v2/target_host_control.json"))
    h = d["hazard_separation_by_target"]
    return {"legibility": round(d["target_host_legibility"]["auroc"], 3),
            "animal": round(h["animal-target only"]["auroc"], 3),
            "nonanimal": round(h["non-animal-target only"]["auroc"], 3),
            "size_matched_ok": d["size_matched"]["nonanimal_below_animal_range"],
            "n_animal": d["counts"].get("animal"),
            "rank_p": round(d["rank_pattern"]["exact_p"], 4),
            "post_hoc": d["rank_pattern"]["post_hoc"]}


def amr_test_input_present():
    """The §9.4 candidate FASTA existed only in /tmp when 24 and 25 were run and
    published, and it was gone the next day. That is the fourth founding defect in
    this file's header -- a number whose input was computed once outside the
    repository -- re-created by the same author who wrote the header. The eight
    sequences were recovered from UniProt by the accessions in
    results/v2/amr_category_test.json and the recovery is byte-identical to the lost
    file. This entry pins the bytes so the input cannot silently drift or vanish
    again, and re-running 25 against it reproduces 87.5% and 75.0% exactly."""
    import hashlib
    p = ROOT / "data/sequences/amr_category_test.fasta"
    b = p.read_bytes()
    n = sum(1 for ln in b.decode().splitlines() if ln.startswith(">"))
    return {"exists": p.exists(), "n_sequences": n, "bytes": len(b),
            "sha256": hashlib.sha256(b).hexdigest()}


def _spearman(x, y):
    """Spearman rho and a two-sided p-value using numpy only.

    The release-surface CI job installs numpy and nothing else, deliberately: this
    audit is meant to run anywhere. An earlier version of the FHS entry imported
    scipy, passed locally, and failed CI with ModuleNotFoundError -- local green is
    not CI green. Ranks use the average-rank convention for ties, and the p-value is
    the standard t approximation on n-2 degrees of freedom, which matches
    scipy.stats.spearmanr to the precision this audit asserts on.
    """
    def rank(v):
        v = np.asarray(v, float)
        order = v.argsort()
        r = np.empty(len(v), float)
        r[order] = np.arange(1, len(v) + 1)
        # average tied ranks
        for u in np.unique(v):
            m = v == u
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r

    rx, ry = rank(x), rank(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    n = len(rx)
    if n < 3 or abs(rho) >= 1.0:
        return rho, 0.0 if abs(rho) >= 1.0 else 1.0
    t = rho * math.sqrt((n - 2) / (1 - rho * rho))
    # two-sided survival function of Student t via the incomplete beta identity
    df = n - 2
    x_b = df / (df + t * t)
    p = _betainc_half(df / 2.0, 0.5, x_b)
    return rho, float(min(1.0, max(0.0, p)))


def _betainc_half(a, b, x, terms=2000):
    """Regularised incomplete beta I_x(a, b) by continued fraction, enough for the
    t-distribution tail used by _spearman. Numpy-only to keep this file dependency
    free."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(terms):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1.0 - c * d) < 1e-12:
            break
    return front * (f - 1.0)


def interplm_feature_overlap():
    """§9.6, the sixth refused candidate for the beta-lactamase anomaly. Pins the size
    control, because the unconditioned version of this looked like a finding: 279
    significant features for beta-lactamase with 95% of them unique to it. Matching every
    class to n=6 drops that to 31 features, and clostridial neurotoxin -- 78% unique,
    nearly as high, recovered at 100% -- kills the correlation (rho -0.217, p 0.641).
    Also pins the positive half, that features separating beta-lactamase from benign DO
    exist, so nobody restates this as "the representation does not encode the class"."""
    d = json.load(open(R / "v2/interplm_feature_overlap.json"))
    pc = d["per_class"]
    sp = d["spearman_unique_vs_recovery"]
    return {"K": d["K"], "n_classes": sp["n"],
            "rho": round(sp["rho"], 3), "p": round(sp["p"], 3),
            "bl_sig": round(pc["beta_lactamase"]["sig_mean"], 1),
            "bl_unique": round(pc["beta_lactamase"]["unique_frac"], 3),
            "clost_unique": round(pc["clostridial_neurotoxin"]["unique_frac"], 3),
            "clost_recovery": round(pc["clostridial_neurotoxin"]["recovery"], 3),
            "bl_recovery": round(pc["beta_lactamase"]["recovery"], 3)}


def interplm_power_check():
    """§9.6's precondition. If InterPLM's SAE features could not separate hazard, no
    feature-level claim about beta-lactamase would mean anything. Layer 18 reaches 0.910
    against the raw embedding's 0.973. Layer 1 is pinned too as the reason it cannot serve
    as a reconstruction-fidelity control: near-perfect reconstruction but only 214 of
    10240 features ever fire, and AUROC 0.753."""
    d = json.load(open(R / "v2/interplm_power_check.json"))
    layers = {str(k): v for k, v in d["layers"].items()}
    l18, l1 = layers["18"], layers["1"]
    return {"auroc_18": round(l18["auroc"], 3),
            "auroc_1": round(l1["auroc"], 3),
            "alive_18": l18["features_alive"],
            "alive_1": l1["features_alive"],
            "active_per_protein_18": round(l18["active_per_protein"], 0),
            "raw_for_comparison": d["raw_embedding_auroc_for_comparison"]}


def fhs_fsi_correlation_current():
    """results/fhs_results.json stores fhs_fsi_spearman_r = 0.7005 (p 0.0112), computed
    2026-05-21 (commit e2522d2) against the fsi_results.json that existed then. FSI was
    re-curated the next day (2026-05-22, Anthrax P13423's catalytic_residues re-keyed,
    its FSI changed to 0) and that entry's own text says the FSI/FSPE/SER pipeline was
    re-run -- FHS was not, so the stored correlation is paired against an artifact that
    no longer exists. Re-running 15's own pairing logic (join on uniprot, read fsi.mean
    from the CURRENT fsi_results.json) against the FHS values already on disk gives a
    different number. This entry pins that current, honestly-paired figure, plus how
    much of it rests on the single most FSI-corrected protein, so neither number can be
    quoted without the other."""
    fhs_d = json.load(open(R / "fhs_results.json"))
    results = fhs_d["results"]
    fsi_data = json.load(open(R / "fsi_results.json"))
    fsi_lookup = {e["uniprot"]: e["fsi"]["mean"] for e in fsi_data
                  if e.get("uniprot") and e.get("fsi", {}).get("mean") is not None}
    pairs = [(r["fhs"], fsi_lookup[r["uniprot_id"]], r["uniprot_id"])
             for r in results if r["uniprot_id"] in fsi_lookup]
    fhs_v = [p[0] for p in pairs]
    fsi_v = [p[1] for p in pairs]
    rho, pv = _spearman(fhs_v, fsi_v)
    uids = [p[2] for p in pairs]
    i = uids.index("P13423")
    rho_wo, pv_wo = _spearman(fhs_v[:i] + fhs_v[i + 1:], fsi_v[:i] + fsi_v[i + 1:])
    return {"n": len(pairs), "rho": round(rho, 4),
            "p": round(pv, 4),
            "stale_stored_rho": round(fhs_d["fhs_fsi_spearman_r"], 4),
            "rho_excluding_P13423": round(rho_wo, 4),
            "p_excluding_P13423": round(pv_wo, 4),
            "stale_and_current_differ": bool(
                abs(rho - fhs_d["fhs_fsi_spearman_r"]) > 0.01)}


def training_set_contamination():
    """§5.1. Removing the 14 beta-lactamases from TRAINING raises pore-forming
    cytolysin by 16.4 points, above all 25 random removals of the same size. The
    distribution control is what is pinned hardest: a SINGLE random draw was tried
    first and two draws spanned 9.5 points, which made the attributable effect read
    as +25.7 on one and +8.4 on the other. Also pinned, that the effect is two named
    members rather than a class-wide shift, and that contact-dependent inhibition is
    NOT a second case, since its +6.2 sits inside a random spread of 8.7 and it
    would otherwise have been written up as one."""
    d = json.load(open(R / "v2/training_set_contamination.json"))
    f = d["focus"]
    cdi = d["survey"]["contact_dependent_inhibition"]
    return {"delta": round(f["paired_delta"], 4),
            "ci_low": round(f["paired_ci95"][0], 4),
            "percentile": f["minus_BL_percentile"],
            "attributable": round(f["attributable_beyond_random"], 4),
            "movers": sorted(d["members_gaining_over_25pts"]),
            "cdi_effect": round(cdi["minus_BL"] - cdi["standard"], 4),
            "cdi_random_sd": round(cdi["random_sd"], 4),
            "exploratory": d["exploratory"]}


def nonanimal_category_refuted():
    """§5.1's companion: the preregistered test that came first and failed. If
    non-animal hazard were a category the probe learns from non-animal examples,
    removing every non-animal-target positive would cost non-animal test classes
    more than it costs animal ones. The interaction is -0.0 with an interval
    spanning zero. Pinned so the refutation is not quietly dropped now that the
    exploratory follow-up found something."""
    d = json.load(open(R / "v2/animal_only_training.json"))
    return {"interaction": round(d["interaction"], 4),
            "ci": [round(x, 4) for x in d["interaction_ci95"]],
            "includes_zero": d["ci_includes_zero"],
            "refuted": "REFUTED" in d["verdict"]}


def beta_lactamase_across_arms():
    """The corrected beta-lactamase claim. Earlier write-ups said the class resists
    every configuration and that alignment beats every embedding method on it. Both
    were wrong: ESM-C 600M recovers about half of it, above alignment. The claim was
    summarized from the ESM-2 arms without checking the ESM-C row, so this entry
    reads EVERY arm and pins the two facts the corrected statement rests on -- that
    ESM-C 600M is well above alignment, and that it is the only arm that is."""
    import glob
    arms = {}
    for f in glob.glob(str(R / "v2/lomo_results*.json")):
        name = Path(f).stem.replace("lomo_results", "").lstrip("_") or "esm2_650M"
        if "smoke" in name:
            continue
        d = json.load(open(f))
        if d.get("n_positive") != 80:
            continue
        r = d["leave_one_mechanism_out"].get("beta_lactamase")
        if r:
            arms[name] = round(r["flagged_95_mean"], 4)
    align = json.load(open(R / "v2/alignment_baseline.json"))
    a = round(align["classes"]["beta_lactamase"]["alignment_recovery"], 4)
    above = sorted(k for k, v in arms.items() if v > a)
    return {"n_arms": len(arms), "alignment": a, "esmc_600M": arms.get("esmc_600M"),
            "esm2_650M": arms.get("esm2_650M"), "arms_above_alignment": above,
            "max_arm": max(arms, key=arms.get), "max_value": max(arms.values()),
            # the untagged 650M run and the esm2_650M_mean pooling arm are the same
            # configuration embedded twice by different scripts; they must agree
            "duplicate_arm_max_diff": _duplicate_arm_max_diff()}


def _duplicate_arm_max_diff():
    a = json.load(open(R / "v2/lomo_results.json"))["leave_one_mechanism_out"]
    f = R / "v2/lomo_results_esm2_650M_mean.json"
    if not f.exists():
        return None
    b = json.load(open(f))["leave_one_mechanism_out"]
    return max(abs(a[c]["flagged_95_mean"] - b[c]["flagged_95_mean"]) for c in a if c in b)



def member_separability_features():
    """What predicts whether a held-out member is caught. Pinned because this
    artifact carried a defect: perm_p drew n // 100 shuffles, so a test advertising
    20,000 permutations ran 200 and its p-values had a floor of 0.005. The floor is
    now 1 / (n + 1), so a p of exactly that value means the null was never exceeded.
    The claim rests on the contrast -- embedding proximity separates, surface
    sequence similarity does not -- so both halves are pinned."""
    d = j("v2/member_separability.json")
    pool = d["pooled"]
    return {"rows": d["n_rows"], "caught": d["n_caught"],
            "margin_auroc": round(pool["margin"]["auroc"], 3),
            "margin_p": round(pool["margin"]["perm_p"], 6),
            "nn_pos_auroc": round(pool["nn_pos"]["auroc"], 3),
            "kmer_auroc": round(pool["kmer_pos"]["auroc"], 3),
            "kmer_p": round(pool["kmer_pos"]["perm_p"], 3),
            "length_auroc": round(pool["length"]["auroc"], 3)}


def classifier_heads():
    """Every recovery figure in the documents uses logistic regression, which is the
    weakest of four heads. That is a limitation the documents state, so it is pinned:
    if a future change makes logistic the best head, the statement must be revised."""
    d = j("v2/classifier_sweep.json")["mean_by_head"]
    small = j("v2/classifier_sweep_esm2_8M.json")["mean_by_head"]
    import glob
    worst = best = arms = 0
    for f in glob.glob(str(R / "v2/classifier_sweep*.json")):
        m = json.load(open(f))["mean_by_head"]
        arms += 1
        if m["logistic"] == min(m.values()):
            worst += 1
        if m["logistic"] == max(m.values()):
            best += 1
    return {"logistic": round(d["logistic"], 4), "best": max(d, key=d.get),
            "gap_650M_pts": round((max(d.values()) - d["logistic"]) * 100, 1),
            "gap_8M_pts": round((max(small.values()) - small["logistic"]) * 100, 1),
            "arms": arms, "logistic_worst_on": worst, "logistic_best_on": best}



def margin_effect_across_arms():
    """The internal margin effect, over every arm rather than the five it was first
    described on. The phrase "consistent across five models" was carried from an
    earlier panel; on 13 arms it holds for 11, and the two exceptions are both
    non-mean poolings of the same model, so the effect belongs to mean-pooled
    representations rather than to representations in general."""
    import glob
    gaps = {}
    for f in glob.glob(str(R / "v2/margin_holdout*.json")):
        name = Path(f).stem.replace("margin_holdout", "").lstrip("_") or "esm2_650M"
        gaps[name] = abs(json.load(open(f))["low_minus_class_matched_random"])
    over = sorted(k for k, v in gaps.items() if v > 0.25)
    under = sorted(k for k, v in gaps.items() if v <= 0.25)
    return {"arms": len(gaps), "over_25pts": len(over), "under_25pts": under,
            "min_pts": round(min(gaps.values()) * 100, 1),
            "max_pts": round(max(gaps.values()) * 100, 1)}



def probe_vs_similarity():
    """Does fitting a probe beat plain nearest-neighbour distance? On three arms it
    does not, and that is a load-bearing caveat on every recovery number here, so it
    is pinned rather than left to a table nobody rechecks. The size-fixed control is
    pinned too: it is null everywhere, which is what rules out class size as the
    driver of between-class differences."""
    import glob
    d = {}
    for f in glob.glob(str(R / "v2/probe_vs_similarity*.json")):
        n = Path(f).stem.replace("probe_vs_similarity", "").lstrip("_") or "esm2_650M"
        j = json.load(open(f))
        d[n] = (j["probe_minus_similarity_mean"], j["probe_minus_sizefixed_mean"])
    neg = sorted(k for k, v in d.items() if v[0] < 0)
    return {"arms": len(d), "negative_arms": neg,
            "max_benefit_pts": round(max(v[0] for v in d.values()) * 100, 1),
            "min_benefit_pts": round(min(v[0] for v in d.values()) * 100, 1),
            "size_control_max_abs_pts": round(max(abs(v[1]) for v in d.values()) * 100, 1)}



def sae_feature_space_lomo():
    """§9.7: LOMO in InterPLM's feature space. The dimension-matched SAE space beats the
    raw embedding on the 9-class mean and takes beta-lactamase to zero at every C."""
    d = j("v2/sae_feature_space_lomo.json")
    s = d["summary"]
    bl_all_C = [d["all"]["sae_matched"][c]["beta_lactamase"] for c in d["all"]["sae_matched"]]
    return {"raw_mean": s["raw"]["mean"], "matched_mean": s["sae_matched"]["mean"],
            "full_mean": s["sae_full"]["mean"],
            "raw_bl": s["raw"]["beta_lactamase"],
            "matched_bl": s["sae_matched"]["beta_lactamase"],
            "matched_bl_zero_at_all_C": all(v == 0.0 for v in bl_all_C),
            "n_C": len(bl_all_C),
            "control_drop_pts": (s["raw"]["per_class"]["virulence_associated_non_toxin"]
                                 - s["sae_matched"]["per_class"]["virulence_associated_non_toxin"]) * 100}


def feature_selection_control():
    """§9.7: selecting features for discriminative power instead of variance does not
    recover a single beta-lactamase member, so the zero is not a selection artifact."""
    r = j("v2/feature_selection_control.json")["results"]
    return {"variance_mean": r["variance"]["mean"], "variance_bl": r["variance"]["beta_lactamase"],
            "discrim_mean": r["discriminative"]["mean"],
            "discrim_bl": r["discriminative"]["beta_lactamase"]}


def lomo_seed_stability():
    """§9.7 and DATA_CORRECTIONS 2026-09-18 (fifth): the published 5-seed table reproduces
    exactly, and beta-lactamase alone falls outside its own 30-seed interval."""
    d = j("v2/lomo_seed_stability.json")
    z = d["per_class"]
    bl = z["beta_lactamase"]
    others = [c for c in z if c not in ("beta_lactamase", "contact_dependent_inhibition")]
    return {"reproduced_exactly": sum(abs(z[c]["mean_5seed"] - z[c]["published_5seed"]) < 1e-9
                                      for c in z),
            "n_classes": len(z),
            "bl_published": bl["published_5seed"], "bl_30seed": bl["mean_30seed"],
            "bl_sd": bl["sd_30seed"], "bl_zero_seeds": bl["zero_seeds"],
            "bl_ci": bl["ci95_30seed"], "bl_inside_ci": bl["published_inside_ci"],
            "outside_ci": d["published_outside_own_ci"],
            "others_max_shift_pts": max(abs(z[c]["mean_30seed"] - z[c]["published_5seed"])
                                        for c in others) * 100}


def target_host_class_holdout():
    """§2.4.1: the 0.929 target-host figure is within-class. Leave-one-mechanism-class-out
    scores exactly the majority baseline, with every held-out class called animal."""
    d = j("v2/target_host_class_holdout.json")
    ch, ph = d["class_holdout"], d["post_hoc_class_level_ordering"]
    wc = d["within_class"]["virulence_associated_non_toxin"]
    return {"pooled_auroc": d["pooled_cv"]["auroc"],
            "published_03s": d["pooled_cv"]["published_03s"],
            "balanced_class_acc": ch["balanced_class_accuracy"],
            "animal_group": ch["by_group"]["animal"],
            "nonanimal_group": ch["by_group"]["non-animal"],
            "perm_usable": ch["perm_usable"], "perm_p95": ch["perm_p95"],
            "posthoc_class_auroc": ph["auroc"], "posthoc_p": ph["exact_one_tailed_p"],
            "animal_below_top_nonanimal": len(ph["animal_classes_below_highest_nonanimal"]),
            "within_class_auroc": wc["auroc"],
            "n_nonanimal_classes": d["n_nonanimal_single_target_classes"],
            "producer_dominant": max(d["producer_kingdoms"].values())}


def seed_stability_all_arms():
    """§9: beta-lactamase at 30 seeds on every arm. ESM-C 600M is still the only arm whose
    interval clears alignment, and three comparisons made on 5-seed points do not hold."""
    d = j("v2/seed_stability_all_arms.json")
    a = d["arms"]
    ck = d["sentence_checks"]
    e6, e3, eb = (a["_esmc_600M"]["0.95"], a["_esmc_300M"]["0.95"],
                  a["_esmc_6B"]["0.95"])
    return {"arms_checked": len(a),
            "clears_alignment": d["arms_clearing_alignment"],
            "overlapping": d["arms_overlapping"],
            "n_published_outside_ci": len(d["published_outside_own_ci"]),
            "esmc600_30seed": e6["mean_30seed"], "esmc600_ci_lo": e6["ci95"][0],
            "alignment": d["alignment_recovery"],
            "esmc6B_vs_300M_overlap": ck["esmc_trio"]["6B_vs_300M_intervals_overlap"],
            "esmc600_vs_6B_overlap": ck["esmc_trio"]["6B_vs_600M_intervals_overlap"],
            "ratio_600M_over_6B": ck["esmc_trio"]["600M_over_6B_ratio"],
            "cls_vs_mean_overlap": ck["pooling"]["cls_vs_mean_intervals_overlap"],
            "ladder_rho": ck["esm2_ladder"]["spearman_rho_vs_params"],
            "esm2_3B_ci_hi": a["_esm2_3B"]["0.95"]["ci95"][1],
            "esmc300_30seed": e3["mean_30seed"], "esmc6B_30seed": eb["mean_30seed"]}


def v3_second_failure():
    """§10.4: v3 produced a second unreachable class, and margin locates both. The
    decomposition column is load-bearing: without it the claim reduces to nn_pos."""
    d = j("v3/second_failure_class.json")
    v2 = j("v2/second_failure_class.json")
    lomo = j("v3/lomo_results.json")["leave_one_mechanism_out"]
    dc, r = d["decomposition"], d["classes"]
    return {"panel_classes": len(r),
            "phage_95": lomo["phage_peptidoglycan_hydrolase"]["flagged_95_mean"],
            "beta_95": lomo["beta_lactamase"]["flagged_95_mean"],
            "bacteriocin_95": lomo["bacteriocin"]["flagged_95_mean"],
            "cry_95": lomo["cry_insecticidal"]["flagged_95_mean"],
            "rip_95": lomo["rip_rrna_glycosidase"]["flagged_95_mean"],
            "margin_rho": dc["margin"]["rho"], "margin_p": dc["margin"]["perm_p"],
            "margin_locates": dc["margin"]["locates_failures"],
            "nn_pos_locates": dc["nn_pos"]["locates_failures"],
            "nn_neg_locates": dc["nn_neg"]["locates_failures"],
            "size_rho": dc["n"]["rho"], "size_p": dc["n"]["perm_p"],
            "beats_parts": d["margin_beats_parts"],
            "beta_margin": r["beta_lactamase"]["margin"],
            "phage_margin": r["phage_peptidoglycan_hydrolase"]["margin"],
            "third_lowest_recovery": d["third_lowest_margin_recovers"]["recovery"],
            "v2_margin_rho": v2["decomposition"]["margin"]["rho"],
            "v2_hit": v2["P2"]["hit"]}


def v3_panel_and_target_host():
    """§2.5: v3's shape, and §2.4.1's INCONCLUSIVE resolving to SUPPORTED with a usable null."""
    panel = j("../data/sequences/panel_v3_manifest.json")
    mech = j("../data/annotations/mechanism_classes_v3.json")
    th = j("v3/target_host_class_holdout.json")
    ch = th["class_holdout"]
    v2 = j("v2/target_host_class_holdout.json")["class_holdout"]
    return {"positives": len(panel["positives"]), "negatives": len(panel["negatives"]),
            "eligible_classes": len(mech["holdout_eligible_classes"]),
            "nonanimal_classes": th["n_nonanimal_single_target_classes"],
            "v3_balanced": ch["balanced_class_accuracy"], "v3_p": ch["perm_p"],
            "v3_usable": ch["perm_usable"], "v3_p95": ch["perm_p95"],
            "v2_balanced": v2["balanced_class_accuracy"], "v2_usable": v2["perm_usable"],
            "verdict_supported": th["verdict"].startswith("SUPPORTED")}


def negative_test_set_audit():
    """§2.6: the negatives have no test set, so every published specificity is within-calibration-set.
    Pinned on both halves at once: the estimator misses nominal by a computable amount at small
    calibration sizes, AND the conformal alternative holds its guarantee, because the first without the
    second is a complaint and the second without the first is an unmotivated change."""
    a = j("v3/negative_test_set_audit_esm2_650M.json")
    b = j("v3/negative_test_set_audit_esm2_35M.json")
    A, B = a["panel_A_deployable_model"], b["panel_A_deployable_model"]
    ms = [str(m) for m in a["m_grid"] if A[str(m)]["fp_out_mean"] is not None]
    def inb(D, m):
        return (D[m]["bracket_lo"] - 2 * D[m]["se_fp_out"] <= D[m]["fp_out_mean"]
                <= D[m]["bracket_hi"] + 2 * D[m]["se_fp_out"])

    def held(D, m):
        return (D[m]["conformal_fp_out_mean"]
                <= D[m]["conformal_guarantee"] + 2 * D[m]["se_conformal"])
    pb = a["panel_B_lomo_recovery"]
    swing = {c: (pb[c]["20"]["mean"] - pb[c]["118"]["mean"]) * 100 for c in pb}
    return {"seeds_panel_a": A["20"]["seeds"], "m_grid": a["m_grid"],
            "published_split": a["published_split"],
            "fp_out_at_m20_650M": A["20"]["fp_out_mean"], "fp_out_at_m20_35M": B["20"]["fp_out_mean"],
            "worst_split_650M": A["20"]["fp_out_max"],
            "in_bracket_650M": sum(inb(A, m) for m in ms),
            "in_bracket_35M": sum(inb(B, m) for m in ms),
            "n_sizes": len(ms),
            "conformal_held_650M": sum(held(A, m) for m in ms),
            "conformal_held_35M": sum(held(B, m) for m in ms),
            "sizes_where_nominal_unreachable": sum(A[m]["bracket_lo"] > 0.05 for m in ms),
            "published_m118_bracket": [A["118"]["bracket_lo"], A["118"]["bracket_hi"]],
            "ceiling_classes_do_not_move":
                all(abs(swing[c]) < 1e-9 for c in ("adp_ribosyl_ab_toxin", "clostridial_neurotoxin")),
            "swing_control": swing["virulence_associated_non_toxin"],
            "swing_phage": swing["phage_peptidoglycan_hydrolase"],
            "swing_beta": swing["beta_lactamase"]}


def what_predicts_the_response():
    """§10.9.2: what predicts whether a larger benign set helps or hurts a class. Pinned with its own
    multiplicity failure, because the surviving correlation misses a Bonferroni threshold the
    CONTAMINATED version of the same run passes, and quoting the effect without that would be the
    overstatement this log keeps recording."""
    c = j("v3/response_predictors_esm2_650M.json")
    w = j("v3/response_predictors_esm2_650M_withhomologs.json")
    m = j("v3/response_predictors_esm2_35M.json")
    st, wt, mt = c["stats"], w["stats"], m["stats"]
    # 10 tests in the reported table: 6 predictors, 5 of them with a partial as well
    n_tests = 2 * len(st) - 2
    bonf = 0.05 / n_tests
    resp = c["response"]
    return {"n_classes": c["n_classes"], "seeds": c["seeds"], "perms": c["perms"],
            "pool_used_clean": c["pool_n_used"], "pool_used_kept": w["pool_n_used"],
            "dropped": c["pool_homologs_dropped"],
            "pmn_rho": st["pool_minus_neg"]["rho"], "pmn_p": st["pool_minus_neg"]["perm_p"],
            "pmn_partial": st["pool_minus_neg"]["rho_partial_baseline"],
            "pmn_partial_p": st["pool_minus_neg"]["perm_p_partial"],
            "kept_partial_p": wt["pool_minus_neg"]["perm_p_partial"],
            "n_tests": n_tests, "bonferroni": bonf,
            "clean_fails_bonferroni": st["pool_minus_neg"]["perm_p_partial"] > bonf,
            "kept_passes_bonferroni": wt["pool_minus_neg"]["perm_p_partial"] < bonf,
            "nn_pool_alone_null": st["nn_pool"]["perm_p_partial"] > 0.05,
            "margin_null": st["margin"]["perm_p"] > 0.5,
            "baseline_null": st["baseline"]["perm_p"] > 0.5,
            "phage_pts": resp["phage_peptidoglycan_hydrolase"]["response_top_pts"],
            "cdi_pts": resp["contact_dependent_inhibition"]["response_top_pts"],
            "cdi_n": resp["contact_dependent_inhibition"]["n"],
            "beta_pts": resp["beta_lactamase"]["response_top_pts"],
            "rip_pts": resp["rip_rrna_glycosidase"]["response_top_pts"],
            "n_gainers_over_10": sum(1 for v in resp.values() if v["response_top_pts"] > 10),
            "n_losers_over_10": sum(1 for v in resp.values() if v["response_top_pts"] < -10),
            # the second arm, which is what turns "suggestive" into "not supported"
            "arm2_n_classes": m["n_classes"],
            "arm2_pmn_rho": mt["pool_minus_neg"]["rho"],
            "arm2_pmn_p": mt["pool_minus_neg"]["perm_p"],
            "arm2_pmn_partial": mt["pool_minus_neg"]["rho_partial_baseline"],
            "arm2_pmn_partial_p": mt["pool_minus_neg"]["perm_p_partial"],
            "pmn_sign_agrees": (st["pool_minus_neg"]["rho"] < 0) == (mt["pool_minus_neg"]["rho"] < 0),
            "pmn_replicates": mt["pool_minus_neg"]["perm_p_partial"] < 0.05,
            "arm2_magnitude_roughly_halves":
                abs(mt["pool_minus_neg"]["rho"]) < 0.7 * abs(st["pool_minus_neg"]["rho"]),
            "margin_null_both_arms": (st["margin"]["perm_p"] > 0.4 and mt["margin"]["perm_p"] > 0.4),
            "arms_tested": 2}


def pool_homology_against_panel():
    """§10.9: the homology census the harvest skipped, all 8,259 pool proteins against all 149
    positives. Pinned on both halves. Every MECHANISM class is clean, which is what keeps §10.9.1's
    class split from being label contamination. The labelled virulence control is not: one pool protein
    sits at 0.871 against a panel positive, and that is what forces `43` to drop it."""
    d = j("v3/pool_homology_against_panel.json")
    pc = d["per_class"]
    over = d["above_threshold"]
    ref = d["panel_negative_reference"]
    control = "virulence_associated_non_toxin"
    mech_clean = all(v["counts_above"]["0.30"] == 0 for c, v in pc.items() if c != control)
    reported = ("beta_lactamase", "phage_peptidoglycan_hydrolase", "rip_rrna_glycosidase")
    return {"pool_n": d["pool_n"], "positives_screened": d["positives_screened"],
            "alignments": d["alignments"], "threshold": d["threshold"],
            "n_classes": len(pc), "n_above": len(over),
            "mechanism_classes_clean": mech_clean,
            "reported_classes_clean": all(pc[c]["counts_above"]["0.30"] == 0 for c in reported),
            "beta_max": pc["beta_lactamase"]["max"],
            "phage_max": pc["phage_peptidoglycan_hydrolase"]["max"],
            "t3ss_max": pc["t3ss_effector_apparatus"]["max"],
            "control_max": pc[control]["max"],
            "violation_pool_acc": over[0]["pool_acc"] if over else None,
            "violation_positive": over[0]["positive"] if over else None,
            "violation_class": over[0]["positive_class"] if over else None,
            "violation_sim": over[0]["similarity"] if over else None,
            # the reference that decides whether the violation is an outlier or the panel's own policy
            "panel_neg_n": ref["n_negatives"], "panel_neg_above": ref["n_above_threshold"],
            "panel_neg_max": ref["max"], "panel_neg_max_acc": ref["max_negative"],
            "panel_neg_max_class": ref["max_positive_class"],
            "pool_over_panel": ref["pool_max_over_panel_max"],
            "panel_top5": [(e["negative"], round(e["similarity"], 3)) for e in ref["top5"]],
            "h2": d["verdict"].startswith("H2")}


def negative_set_at_fixed_budget():
    """§10.9.1: at a FIXED false-positive budget the pool repairs one unreachable class and makes
    the other worse. Pinned together with the budget drift, because the headline three to fourfold
    gain and the fact that it ran at two to three times the budget cannot be quoted apart."""
    d = j("v3/threshold_vs_boundary_esm2_650M.json")
    a = j("v3/operating_point_audit_esm2_650M.json")
    ks = [str(k) for k in a["K_grid"]]
    top, mid = ks[-1], "4000"
    c = a["curves"]
    def bo(cl, k):
        return c[cl]["boundary_only"][k]["rec"]["mean"]
    gs = d["gain_split"]
    fps = {cl: [c[cl]["boundary_only"][k]["fp_panel"]["mean"] for k in ks] for cl in c}
    return {"seeds": a["seeds"], "self_tests": a["self_tests"],
            # 38: the two single-factor arms fall far short of `both`, so the split statistic is empty
            "residual_phage": gs["phage_peptidoglycan_hydrolase"]["additive_residual_pts"],
            "residual_beta": gs["beta_lactamase"]["additive_residual_pts"],
            # 39 S1: boundary_only's budget is the panel's own, at every K and every class
            "boundary_fp_constant": all(abs(x - fps[cl][0]) < 1e-9 for cl in fps for x in fps[cl]),
            "boundary_fp": fps["beta_lactamase"][0],
            # the fixed-budget result
            "phage_best_gain_pts": (bo("phage_peptidoglycan_hydrolase", mid)
                                    - bo("phage_peptidoglycan_hydrolase", "0")) * 100,
            "phage_monotone_to_4000": all(
                bo("phage_peptidoglycan_hydrolase", x) <= bo("phage_peptidoglycan_hydrolase", y) + 1e-9
                for x, y in zip(ks[:3], ks[1:4])),
            "beta_change_pts": (bo("beta_lactamase", top) - bo("beta_lactamase", "0")) * 100,
            "rip_change_pts": (bo("rip_rrna_glycosidase", top) - bo("rip_rrna_glycosidase", "0")) * 100,
            # what `both` was actually running at
            "both_fp_min": min(c[cl]["both"][top]["fp_panel"]["mean"] for cl in c),
            "both_fp_max": max(c[cl]["both"][top]["fp_panel"]["mean"] for cl in c),
            "beta_rethresholded": a["summary"]["beta_lactamase"]["both_recovery_at_panel_op_maxK"],
            "beta_baseline": a["summary"]["beta_lactamase"]["baseline"],
            "beta_verdict": a["summary"]["beta_lactamase"]["verdict"],
            "phage_verdict": a["summary"]["phage_peptidoglycan_hydrolase"]["verdict"],
            "split_verdict": a["verdict"].startswith("Q1 SPLIT")}


def pool_contamination_changes_one_class():
    """§10.9.1: the iso-FP control under both reservation splits, on BOTH arms.

    The name-disjoint split removes beta-lactamase's apparent excess entirely on the canonical arm,
    so both splits are pinned there and the direction of the contamination bias is pinned with them.

    🔴 The zero is canonical-only and the second arm is pinned to say so. On esm2_35M the same
    name-disjoint run gives beta-lactamase +0.0, +2.9, -9.3, -25.2, -25.0, so the +2.9 at K=500 clears
    that arm's 1.43-point granularity floor and "no dose at all" is false there. What survives is the
    verdict rather than the number: `40` scores that point OFF-BUDGET because the hard-negative
    false-positive rate goes 4.2% to 9.7% to buy it. The assertion therefore REQUIRES the small arm to
    have a positive dose and to be off-budget, so a future edit cannot quietly present the canonical
    zero as a replicated result."""
    r = j("v3/fixed_background_operating_point_esm2_650M.json")
    n = j("v3/fixed_background_operating_point_esm2_650M_namedisjoint.json")
    m = j("v3/fixed_background_operating_point_esm2_35M_namedisjoint.json")
    def net(d, cl):
        return d["summary"][cl]["excess_net_pts"]
    nd = n["curves"]["beta_lactamase"]
    doses = [str(k) for k in n["K_grid"]][1:]
    return {"random_split": r["split"], "nd_split": n["split"],
            "nd_name_groups": n["n_name_groups"],
            "phage_net_random": net(r, "phage_peptidoglycan_hydrolase"),
            "phage_net_nd": net(n, "phage_peptidoglycan_hydrolase"),
            "beta_net_random": net(r, "beta_lactamase"),
            "beta_net_nd": net(n, "beta_lactamase"),
            "beta_nd_best_K": n["summary"]["beta_lactamase"]["best_K_by_excess"],
            "beta_nd_all_doses_negative": all(
                nd[k]["excess_net"]["mean"] < 0 for k in doses),
            "beta_nd_worst_pts": min(nd[k]["excess_net"]["mean"] for k in doses) * 100,
            "rip_net_nd": net(n, "rip_rrna_glycosidase"),
            "phage_fp_ratio_nd": n["summary"]["phage_peptidoglycan_hydrolase"]["fp_hard_ratio"],
            "nd_verdict_split": n["verdict"].startswith("SPLIT"),
            # second arm, same split, same class
            "arm2": m["arm"], "arm2_split": m["split"],
            "arm2_beta_net_nd": m["summary"]["beta_lactamase"]["excess_net_pts"],
            "arm2_beta_best_K": m["summary"]["beta_lactamase"]["best_K_by_excess"],
            "arm2_beta_floor_pts": m["summary"]["beta_lactamase"]["granularity_floor_pts"],
            "arm2_beta_verdict": m["summary"]["beta_lactamase"]["verdict"],
            "arm2_beta_fp_ratio": m["summary"]["beta_lactamase"]["fp_hard_ratio"],
            "arm2_beta_recovery_K0": m["summary"]["beta_lactamase"]["recovery_K0"],
            "arm2_beta_all_doses_negative": all(
                m["curves"]["beta_lactamase"][k]["excess_net"]["mean"] < 0 for k in doses),
            "zero_replicates": abs(m["summary"]["beta_lactamase"]["excess_net_pts"]) < 1e-9}


def v3_arm_seed_stability():
    """§10.6.1: are the between-arm recovery differences on v3 real at 30 seeds? On v2 the same
    check found no separation and a moving peak, so this is pinned for both the separation and the
    5-seed figures that fall outside their own intervals."""
    d = j("v3/arm_seed_stability.json")
    r, sm = d["results"], d["summary"]
    b, ph = r["beta_lactamase"], r["phage_peptidoglycan_hydrolase"]
    return {"seeds": d["seeds"], "n_arms": len(d["arms"]),
            "beta_disjoint_pairs": sm["beta_lactamase"]["n_disjoint_pairs"],
            "phage_disjoint_pairs": sm["phage_peptidoglycan_hydrolase"]["n_disjoint_pairs"],
            # 🔴 These two were `== 4`, i.e. five arms minus one, baked in while v3 was the ESM-2
            # ladder. With eight arms the literal would have read False for a reason that has
            # nothing to do with separation. Same defect as the fixed arm list in src/41.
            "beta_canonical_separates_from_all":
                len(sm["beta_lactamase"]["arms_disjoint_from_canonical"]) == len(d["arms"]) - 1,
            "phage_canonical_separates_from_all":
                len(sm["phage_peptidoglycan_hydrolase"]["arms_disjoint_from_canonical"])
                == len(d["arms"]) - 1,
            "phage_overlapping_canonical": sorted(
                set(d["arms"]) - {"canonical 650M"}
                - set(sm["phage_peptidoglycan_hydrolase"]["arms_disjoint_from_canonical"])),
            # 🔴 2026-09-27: beta-lactamase gained its first overlap with the canonical arm when
            # saprot_650M came in at [7.8, 17.0] against the canonical [16.5, 25.9] -- half a point
            # of overlap. Pinned by name for the same reason the phage one is: the identity of an
            # overlap is the finding, a count of them is not.
            "beta_overlapping_canonical": sorted(
                set(d["arms"]) - {"canonical 650M"}
                - set(sm["beta_lactamase"]["arms_disjoint_from_canonical"])),
            "beta_outside_ci": sorted(sm["beta_lactamase"]["published_outside_own_ci"]),
            "phage_outside_ci": sorted(sm["phage_peptidoglycan_hydrolase"]["published_outside_own_ci"]),
            "beta_top_changes": sm["beta_lactamase"]["top_arm_changes"],
            "phage_top_changes": sm["phage_peptidoglycan_hydrolase"]["top_arm_changes"],
            "beta_canonical_30s": b["canonical 650M"]["0.95"]["mean_30seed"],
            "beta_3B_30s": b["esm2_3B"]["0.95"]["mean_30seed"],
            "beta_150M_30s": b["esm2_150M"]["0.95"]["mean_30seed"],
            "phage_3B_30s": ph["esm2_3B"]["0.95"]["mean_30seed"],
            "phage_8M_30s": ph["esm2_8M"]["0.95"]["mean_30seed"],
            "phage_35M_30s": ph["esm2_35M"]["0.95"]["mean_30seed"],
            "beta_4pct_tie_dissolved":
                abs(b["esm2_3B"]["0.95"]["mean_30seed"] - b["esm2_150M"]["0.95"]["mean_30seed"]) > 0.05,
            "a1": d["verdict"].startswith("A1")}


def pool_and_calibration_gap():
    """§10.9 and §11: what the 8,259-protein benign pool is worth, and how much of §10.8's
    calibration shortfall it closes. Pinned because the raw count overstates it: the complete
    name-based count is lower than the sampled homology estimate, so the honest gap figure is the
    larger of the two rather than the one the raw count implies."""
    d = j("v3/pool_effective_n.json")
    dep = j("v3/deployment_operating_points.json")
    need = dep["negatives_needed"]["0.9999"]["panel_negatives_required"]
    cal = d["calibration"]
    return {"pool_n": d["pool_n"], "distinct_names": d["distinct_names"],
            "name_factor": d["name_redundancy_factor"],
            "most_repeated": d["most_repeated_names"][0][1],
            "eff_homology": d["effective_n_by_homology"],
            "homology_keep": d["homology_keep_rate"],
            "ceiling_raw": cal["raw count"]["ceiling"],
            "ceiling_homology": cal["by homology, estimated"]["ceiling"],
            "ceiling_name": cal["by name"]["ceiling"],
            "need_9999": need,
            "gap_panel": need / 296, "gap_raw": need / d["pool_n"],
            "gap_name": need / d["distinct_names"],
            "name_count_is_lower": d["distinct_names"] < d["effective_n_by_homology"]}


def v3_provenance_control():
    """§10.9: the provenance control on v3, which is the argument against scaling the POSITIVE side
    to keyword scale. Pinned for both panels together because the two were once quoted mixed: v2's
    AUROC sat next to v3's agreement rate in `34`'s docstring."""
    a = j("v2/lomo_results.json")
    b = j("v3/lomo_results.json")
    return {"v2_auroc": a["provenance_auroc"][0], "v2_sd": a["provenance_auroc"][1],
            "v2_agreement": a["organism_label_agreement_with_hazard"],
            "v3_auroc": b["provenance_auroc"][0], "v3_sd": b["provenance_auroc"][1],
            "v3_agreement": b["organism_label_agreement_with_hazard"],
            "v3_above_chance_by_2sd": b["provenance_auroc"][0] - 2 * b["provenance_auroc"][1] > 0.5,
            "v3_noisier": b["provenance_auroc"][1] > 3 * a["provenance_auroc"][1]}


def margin_predicts_new_classes():
    """§10.5: margin ranks an unseen mechanism correctly out of sample and mis-states its
    miss rate by 34 points. Both halves are pinned, because the ordering result without the
    calibration failure would read as a competence boundary this does not have."""
    d = j("v3/margin_predicts_new_classes.json")
    p1, cal = d["P1"], d["calibration"]
    phage = p1["per_class"]["phage_peptidoglycan_hydrolase"]
    return {"p1_hit": p1["hit"], "lowest": p1["order_low_to_high"][0],
            "phage_predicted": phage["predicted"], "phage_measured": phage["measured"],
            "phage_error_pts": (phage["predicted"] - phage["measured"]) * 100,
            "loocv_mae": d["P2"]["loocv_mae"], "baseline_mae": d["P2"]["baseline_mae"],
            "beats_baseline": d["P2"]["beats_baseline"],
            "loocv_rho": d["P3"]["loocv_spearman"], "loocv_p": d["P3"]["perm_p"],
            "all_errors_optimistic": cal["all_same_sign"],
            "worst_calibrated": cal["worst_calibrated_new_class"],
            "rho_pre_expansion": d["rho_pre_expansion_classes"],
            "not_calibrated": "NOT CALIBRATED" in d["verdict"]}


def margin_across_arms():
    """§10.6: the class-level margin mechanism across all 14 arms. The 14/14 negative-margin
    count is what licenses §10.4's geometric phrasing, so it is pinned alongside the two arms
    that put a different class at the bottom."""
    d = j("v2/margin_across_arms.json")
    a = d["arms"]
    return {"n_arms": len(a),
            "locate_failure": len(d["P1"]["arms_locating_failure"]),
            "significant": len(d["P2"]["arms_significant"]),
            "negative_margin_arms": len(d["arms_with_all_failure_margins_negative"]),
            "smoke_arm_excluded": not any("smoke" in x for x in a),
            "misses": sorted(k for k, v in a.items() if not v["locates_failure"]),
            "max_locates": a["esm2_650M_max"]["locates_failure"],
            "cls_locates": a["esm2_650M_cls"]["locates_failure"],
            "weakest_rho": min(v["rho"] for v in a.values()),
            "weakest_p": max(v["perm_p"] for v in a.values()),
            "duplicate_agrees": abs(a["canonical"]["rho"]
                                    - a["esm2_650M_mean"]["rho"]) < 1e-12,
            "supported": d["verdict"].startswith("SUPPORTED")}


def margin_causal_test():
    """§10.7: removing the nearest benign negatives from training lifts the failing classes
    beyond random removal, and closes only 8 to 11% of the gap. Both halves are pinned so the
    causal claim cannot be quoted without its size."""
    v3 = j("v3/margin_causal_test.json")
    v2 = j("v2/margin_causal_test.json")
    b3 = v3["classes"]["beta_lactamase"]
    ph = v3["classes"]["phage_peptidoglycan_hydrolase"]
    b2 = v2["classes"]["beta_lactamase"]
    comp3 = v3["classes"][v3["comparison_class"]]
    return {"v3_failures": v3["failing_classes"], "v2_failures": v2["failing_classes"],
            "beta_attr_v3": b3["attributable_pts"], "beta_ci_lo_v3": b3["attributable_ci95"][0],
            "phage_attr_v3": ph["attributable_pts"], "phage_ci_lo_v3": ph["attributable_ci95"][0],
            "beta_attr_v2": b2["attributable_pts"], "beta_ci_lo_v2": b2["attributable_ci95"][0],
            "beta_frac_v3": b3["fraction_of_gap_closed"],
            "phage_frac_v3": ph["fraction_of_gap_closed"],
            "beta_frac_v2": b2["fraction_of_gap_closed"],
            "comparison_attr": comp3["attributable_pts"],
            "beta_pctile_v3": b3["mean_within_seed_percentile"],
            "beta_seeds_beating_p95": b3["seeds_beating_own_p95"],
            "contributing_not_the_cause": "CONTRIBUTING CAUSE" in v3["verdict"]
                                          and "CONTRIBUTING CAUSE" in v2["verdict"]}


def deployment_operating_points():
    """§10.8: the margin triage still orders classes at the strictest estimable specificity, the
    review queue is mostly false alerts at deployment prevalence, and the panel is ~850x too
    small to calibrate a 1-in-10,000 budget rather than extrapolate it."""
    d = j("v3/deployment_operating_points.json")
    tri, vol = d["triage_by_spec"], d["volume"]
    strict = str(d["strictest_estimable"])
    q = vol[strict]["per_prevalence"]["0.001"]
    return {"specs": len(d["specificities"]), "strictest": d["strictest_estimable"],
            "ceiling": d["estimable_ceiling"],
            "rho_95": tri["0.95"]["rho"], "rho_strict": tri[strict]["rho"],
            "p_strict": tri[strict]["perm_p"], "survives": d["P1_triage_survives"],
            "all_specs_significant": all(v["perm_p"] < 0.05 for v in tri.values()),
            "tpr_strict": vol[strict]["tpr"], "fpr_strict": vol[strict]["fpr"],
            "alerts_1in1000": q["total_alerts"], "precision_1in1000": q["precision"],
            "missed_1in1000": q["missed_hazards"],
            "panel_for_9999": d["negatives_needed"]["0.9999"]["panel_negatives_required"],
            "have_negatives": d["n_negatives"],
            "phage_at_strict": d["catch_by_spec"][strict]["phage_peptidoglycan_hydrolase"],
            "clostridial_at_strict": d["catch_by_spec"][strict]["clostridial_neurotoxin"],
            "cdi_at_strict": d["catch_by_spec"][strict]["contact_dependent_inhibition"]}


def margin_dose_response():
    """§10.7.1: the attributable effect grows monotonically with how many benign neighbours are
    removed for both failing classes, while the recovered comparison class peaks and falls back.
    Pinned with the fraction of the gap the largest dose closes, because a growing effect that
    still leaves three quarters of the gap is the actual finding."""
    d = j("v3/margin_dose_response.json")
    c = d["curves"]
    ks = [str(k) for k in d["K_grid"]]
    fails = d["failing_classes"]
    comp = d["comparison_class"]

    def mono(cl):
        v = [c[cl][k]["attributable_pts"] for k in ks]
        return all(b >= a - 0.5 for a, b in zip(v, v[1:]))
    closed = {}
    for cl in fails:
        top = c[cl][ks[-1]]
        gap = (1.0 - top["standard"]) * 100
        closed[cl] = top["attributable_pts"] / gap
    return {"k_grid": d["K_grid"], "failures": sorted(fails),
            "both_monotone": all(mono(cl) for cl in fails),
            "comp_peaks_mid": max(ks, key=lambda k: c[comp][k]["attributable_pts"]) != ks[-1],
            "comp_top_ci_includes_zero": c[comp][ks[-1]]["ci95"][0] < 0,
            "fail_ci_lo_positive_at_top": all(c[cl][ks[-1]]["ci95"][0] > 0 for cl in fails),
            "beta_top_pts": c["beta_lactamase"][ks[-1]]["attributable_pts"],
            "phage_top_pts": c["phage_peptidoglycan_hydrolase"][ks[-1]]["attributable_pts"],
            "beta_frac_closed": closed["beta_lactamase"],
            "phage_frac_closed": closed["phage_peptidoglycan_hydrolase"],
            "train_neg_kept_at_top": (c["beta_lactamase"][ks[-1]]["training_negatives_left"]
                                      / d["training_negatives_per_fold"]),
            "scaling": d["verdict"].startswith("SCALING")}


def v3_margin_across_arms():
    """§10.6.1: the same across-arms test on v3, where two failing classes make the bottom-k
    question much stricter. Pinned with BOTH halves: what survives in 3/3 arms and the fact
    that the exact bottom-two holds in only one."""
    d = j("v3/margin_across_arms.json")
    a = d["arms"]
    lom = {"canonical": j("v3/lomo_results.json"),
           "esm2_3B": j("v3/lomo_results_esm2_3B.json"),
           "esm2_150M": j("v3/lomo_results_esm2_150M.json"),
           "esm2_35M": j("v3/lomo_results_esm2_35M.json"),
           "esm2_8M": j("v3/lomo_results_esm2_8M.json")}
    rec = {k: v["leave_one_mechanism_out"] for k, v in lom.items()}
    return {"n_arms": len(a), "k": d["k"], "chance": d["chance_per_arm"],
            "failures": sorted(d["failure_classes"]),
            "locate": len(d["P1"]["arms_locating_failure"]),
            "which_locates": d["P1"]["arms_locating_failure"],
            "significant": len(d["P2"]["arms_significant"]),
            "all_negative": len(d["arms_with_all_failure_margins_negative"]),
            # 🔴 2026-09-27: no longer everywhere. saprot_650M's lowest-margin class is phage, not
            # beta-lactamase, and that is the structure confound rather than a geometric finding:
            # none of the 32 phage members has an AlphaFold model, so they are the only proteins
            # that arm sees sequence-only. The exception is named so it cannot be read as evidence.
            "beta_lowest_arms": sorted(k for k, v in a.items()
                                       if v["lowest_margin_class"] == "beta_lactamase"),
            "beta_lowest_everywhere": all(v["lowest_margin_class"] == "beta_lactamase"
                                          for v in a.values()),
            "min_rho": min(v["rho"] for v in a.values()),
            "displacer": sorted({c for v in a.values() if not v["locates_failure"]
                                 for c in v["bottom_k"]}
                                - set(d["failure_classes"])),
            "beta_rec_by_capacity": [rec[k]["beta_lactamase"]["flagged_95_mean"]
                                     for k in ("esm2_3B", "canonical", "esm2_150M",
                                               "esm2_35M", "esm2_8M")],
            "phage_rec_by_capacity": [rec[k]["phage_peptidoglycan_hydrolase"]["flagged_95_mean"]
                                      for k in ("esm2_3B", "canonical", "esm2_150M",
                                                "esm2_35M", "esm2_8M")],
            "beta_not_monotone_in_capacity": (
                [rec[k]["beta_lactamase"]["flagged_95_mean"]
                 for k in ("esm2_3B", "canonical", "esm2_150M", "esm2_35M", "esm2_8M")]
                != sorted([rec[k]["beta_lactamase"]["flagged_95_mean"]
                           for k in ("esm2_3B", "canonical", "esm2_150M",
                                     "esm2_35M", "esm2_8M")]))}

# ---- the registry --------------------------------------------------------------
# (label, recompute -> dict, assertion on that dict, {document: string it must
#  contain}, strings no public document may contain any more)
#
# `must` names the document explicitly. An earlier version only required the string
# to appear in SOME public document, which meant one document could drift while the
# others still carried the phrase and the audit would pass. Verified by breaking
# README.md on purpose: the audit returned success. It now names each surface.


def reference_set_poisoning():
    """Criterion 18: 500 genuine Swiss-Prot proteins, labelled benign and appended to the panel's
    296, suppress one hazard class while the aggregate an operator would watch stays flat or
    improves.

    Pinned across BOTH arms because the interesting halves separate. Which classes are suppressible
    and the 5-of-13 selectivity ceiling replicate; the quietness does not, and the canonical arm is
    the quiet one. Publishing the quietness without the arm qualifier would be the error criterion 7
    exists to prevent, so the assertion REQUIRES the 35M arm to be loud.

    cry_insecticidal is the headline rather than rip_rrna_glycosidase because rip's 62.9-point
    canonical drop inverts on esm2_35M, where the same construction damages cry more than rip."""
    a = j("v3/reference_set_poisoning_esm2_650M.json")
    b = j("v3/reference_set_poisoning_esm2_35M.json")
    out = {}
    for tag, d in (("c", a), ("s", b)):
        r = d["per_target"]
        out[f"{tag}_p1_near"] = sum(r[t]["P1_targeted_beats_collateral"] for t in r)
        out[f"{tag}_p1_sel"] = sum(r[t]["P1_selective_beats_collateral"] for t in r)
        out[f"{tag}_p2_near"] = sum(r[t]["P2_fp_gives_no_signal"] for t in r)
        out[f"{tag}_p2_sel"] = sum(r[t]["P2_selective_fp_gives_no_signal"] for t in r)
        out[f"{tag}_n"] = len(r)
        cry = r["cry_insecticidal"]
        out[f"{tag}_cry_panel"] = cry["target_panel"]
        out[f"{tag}_cry_sel"] = cry["target_selective"]
        out[f"{tag}_cry_fp_panel"] = cry["pool_fp"]["panel"]
        out[f"{tag}_cry_fp_sel"] = cry["pool_fp"]["selective"]
    ra, rb = a["per_target"], b["per_target"]
    both = sorted(set(ra) & set(rb))
    out["agree_suppressible"] = sum(
        (ra[t]["selective_delta_pts"] < -5) == (rb[t]["selective_delta_pts"] < -5) for t in both)
    out["n_both"] = len(both)
    out["c_rip_sel_pts"] = ra["rip_rrna_glycosidase"]["selective_delta_pts"]
    out["s_rip_sel_pts"] = rb["rip_rrna_glycosidase"]["selective_delta_pts"]
    out["c_rip_p1_sel"] = ra["rip_rrna_glycosidase"]["P1_selective_beats_collateral"]
    out["s_rip_p1_sel"] = rb["rip_rrna_glycosidase"]["P1_selective_beats_collateral"]
    return out


def poisoning_neighbourhood_overlap():
    """Criterion 18's mechanism: "nearest to hazard class X" is mostly "nearest to the panel", which
    is why the attack is indiscriminate. Recomputed from the embeddings rather than read back from
    an artifact, so a stale file cannot carry it.

    ⚠️ numpy only, like the rest of this gate, so the release-surface CI job can run it."""
    import numpy as np
    V = ROOT / "results" / "v3"
    man = json.load(open(V / "embedding_manifest_v3.json"))
    pman = json.load(open(V / "embedding_manifest_pool_large_esm2_650M.json"))
    mech = json.load(open(ROOT / "data/annotations/mechanism_classes_v3.json"))
    P = np.load(V / "embeddings_positive_v3.npy")
    POOL = np.load(V / "embeddings_pool_large_esm2_650M.npy")
    acc = [r.split("|")[1] if "|" in r else r for r in pman["rows"]]
    POOL = POOL[np.array([i for i, x in enumerate(acc) if x != "Q8X739"])]
    cls = {e["fasta_id"]: e["mechanism_class"] for e in mech["proteins"]}
    pcls = np.array([cls[r["acc"]] for r in man["positive_rows"]])
    classes = sorted({c for c in pcls if (pcls == c).sum() >= 3})
    Pn = P / np.linalg.norm(P, axis=1, keepdims=True)
    Qn = POOL / np.linalg.norm(POOL, axis=1, keepdims=True)
    sets = {c: set(np.argsort(-(Qn @ Pn[np.where(pcls == c)[0]].T).max(axis=1))[:500].tolist())
            for c in classes}
    js = [(len(sets[x] & sets[y]) / len(sets[x] | sets[y]), x, y)
          for i, x in enumerate(classes) for y in classes[i + 1:]]
    mx = max(js)
    return {"n_classes": len(classes), "mean_jaccard": float(np.mean([v for v, _, _ in js])),
            "max_jaccard": mx[0], "max_pair": sorted([mx[1], mx[2]]),
            "union": len(set().union(*sets.values())), "slots": 500 * len(classes)}



def criteria_scorecard_consistency():
    """The README's summary of `docs/DETECTOR_CRITERIA.md` drifted out of step with the scorecard it
    summarises and understated the project's own failures: it said two fails where the table had
    three, and stayed at seventeen criteria after an eighteenth was added.

    That is the drift this gate exists for, so the count is derived from the table rather than
    trusted. Criteria headings and scorecard rows are counted directly and required to match each
    other and the prose in both documents.

    A verdict counts as a fail when the BOLD marker opening it is "fail", case-insensitively. That
    is deliberately narrow. It catches criterion 3, whose verdict is "**Pass** on reporting, **fail**
    on performance" and which the document's own tally of three includes, while not catching
    criterion 2, whose "**Pass now, failed before.**" contains the word but is not a fail. Matching
    the bare word anywhere would score four and matching only capital-F would score two, and both
    have been checked against the table rather than assumed."""
    import re
    t = (ROOT / "docs/DETECTOR_CRITERIA.md").read_text()
    r = (ROOT / "README.md").read_text()
    heads = re.findall(r"^## (\d+)\. ", t, re.M)
    rows = re.findall(r"^\| (\d+) [^|]*\| (.+?) \|$", t, re.M)
    fails = [int(n) for n, v in rows if re.search(r"\*\*[Ff]ail", v)]
    partials = [int(n) for n, v in rows if re.search(r"\*\*[Pp]artial", v)]
    return {"headings": len(heads), "rows": len(rows),
            "heads_are_1_to_n": heads == [str(i) for i in range(1, len(heads) + 1)],
            "rows_match_heads": [n for n, _ in rows] == heads,
            "fails": sorted(fails), "partials": sorted(partials),
            "mixed": sorted(int(n) for n, v in rows if "Mixed" in v),
            "doc_says_18": "on eighteen criteria" in t,
            # 🔴 Both tally strings updated 2026-09-27 when criterion 1 moved fail -> partial.
            "doc_tally": "Two fails (3, 12), four partials (1, 4, 5, 18) and one mixed (7)" in t,
            "readme_says_18": "states eighteen criteria" in r,
            "readme_tally": "two fails, four partials, one mixed" in r}



def tier3_coordinates():
    """Step 1 of the mutation extension's run order: every tier 3 substitution position verified
    against UniProt before any run, in PRECURSOR coordinates because that is what the pipeline
    indexes.

    Pinned hard because this is the exact shape of `docs/DATA_CORRECTIONS.md` entry sixteen, where
    three annotations were in mature-chain coordinates while the pipeline indexed the precursor. The
    BoNT-A residue is the live case: UniProt and the HExxH motif both put it at precursor 224, and
    the research literature calls the same residue 223 counting from the light chain's own start.

    ⚠️ The assertion requires the offset sweep to be NON-unique on three of the four. That is not a
    defect to be fixed later, it is the honest strength of a one-constraint identity check, and
    requiring it here stops a future edit from quietly claiming `src/46`-grade uniqueness for these."""
    d = j("v3/tier3_coordinate_verification.json")
    c = d["candidates"]
    sub = {a: {s["mature"]: s for s in c[a]["substitutions"]} for a in c}
    return {"n": len(c),
            "all_seq_match": all(c[a]["panel_matches_uniprot"] for a in c),
            "all_identities_ok": all(c[a]["all_identities_ok"] for a in c),
            "survivors": sorted(d["surviving_pairs"]),
            "offsets": {a: c[a]["offset"] for a in sorted(c)},
            "unique_offset": sorted(a for a in c if c[a]["offset_unique"]),
            "bont_precursor": sub["P0DPI1"][223]["precursor"],
            "bont_residue": sub["P0DPI1"][223]["found"],
            "bont_exact_sub_annotated": sub["P0DPI1"][223]["this_substitution_annotated"],
            "pertussis_precursor": [sub["P04977"][9]["precursor"], sub["P04977"][129]["precursor"]],
            "pertussis_129_sub_annotated": sub["P04977"][129]["this_substitution_annotated"],
            "crm197_precursor": sub["P00588"][52]["precursor"],
            "crm197_mutagenesis": sub["P00588"][52]["uniprot_mutagenesis"],
            "ricin_precursor": sub["P02879"][177]["precursor"],
            "ricin_mutagenesis": sub["P02879"][177]["uniprot_mutagenesis"],
            "ricin_lof_alternatives": [[x["precursor"], x["mature"], x["orig"]]
                                       for x in c["P02879"]["lof_annotated_alternatives"]],
            "tier2_lof_pairs": sorted(a for a in c if c[a]["any_tier2_lof"])}



def flagged_entries_vs_uniprot():
    """The two FSPE entries an offset cannot repair, checked against UniProt rather than reasoned
    about further.

    The load-bearing half is SEB. Its published ratio is computed at offset 0, and UniProt rules that
    offset OUT: two of the nine positions fall inside the cleaved signal peptide while being
    annotated as a receptor interface. At the only admissible offset the ratio crosses 1.0 and the
    protein-level headline goes 13/15 at p = 0.0037 to 12/15 at p = 0.0176.

    Pinned because the headline is on the model card and in the README, and because the caveat is
    the kind that gets dropped when a number is quoted out loud. The assertion REQUIRES the flip, so
    an edit that quietly rescored SEB into the thirteen would fail here.

    ⚠️ The sign-test values are recomputed from `fspe_results.json` with SEB's ratio substituted,
    not read back from `src/52`'s artifact, so the two files have to agree for this to pass."""
    d = j("v3/flagged_entries_vs_uniprot.json")
    seb, exo = d["entries"]["P01552"], d["entries"]["Q51451"]
    v0 = seb["variants"]["published_offset_0"]
    v27 = seb["variants"]["offset_plus_27"]
    f = j("fspe_results.json")
    rows = {(x.get("uniprot_id") or x.get("accession")): x["fspe_ratio"] for x in f["per_protein"]}
    n = len(rows)
    pub_k = sum(1 for v in rows.values() if v < 1.0)
    alt = dict(rows, P01552=v27["ratio"])
    alt_k = sum(1 for v in alt.values() if v < 1.0)
    drop_k = sum(1 for a, v in rows.items() if a != "P01552" and v < 1.0)
    return {"seb_signal_end": seb["uniprot_signal_end"],
            "seb_offset0_in_signal": seb["offset0_positions_inside_signal_peptide"],
            "seb_offset0_disproven": seb["offset0_disproven"],
            "seb_disulfide_mature": seb["disulfide_mature"],
            "seb_res_at_mature_93": seb["residue_at_mature_93"],
            "seb_ratio_0": v0["ratio"], "seb_ratio_27": v27["ratio"],
            "seb_flips": d["seb_verdict_flips"],
            "seb_uniprot_has_sites": seb["uniprot_has_site_features"],
            "headline_published": f"{pub_k}/{n}", "p_published": _sign_p(n, pub_k),
            "headline_seb_at_27": f"{alt_k}/{n}", "p_seb_at_27": _sign_p(n, alt_k),
            "headline_seb_dropped": f"{drop_k}/{n - 1}", "p_seb_dropped": _sign_p(n - 1, drop_k),
            "exos_domain": exo["adp_rt_domain"],
            "exos_domain_hypothesis_holds": exo["domain_start_hypothesis_holds"],
            "exos_agrees": exo["agrees"], "exos_omits": exo["omits"],
            "exos_all_below_1": d["exos_robust"],
            "exos_n_variants": len(exo["variants"])}



def signal_peptide_sweep():
    """No annotated functional position may sit inside a cleaved signal peptide. Entry nineteen's
    cheap identity-free gate, run over the whole panel.

    This is the check that would have caught SEB on day one. `src/46`'s identity test can only catch
    a wrong coordinate frame when the residue names are RIGHT, and SEB's are also wrong, so it
    returned zero of nine and gave no signal for months.

    ⚠️ Positions must be read AFTER applying each entry's own `precursor_offset`, because the three
    entries repaired in entry sixteen store mature coordinates. A sweep that reads them raw flags its
    own repairs; the first run of `src/53` did exactly that.

    🔑 RECOMPUTED from `data/annotations/functional_sites.json` and the UniProt cache, not read back
    from `src/53`'s artifact. That is the difference between a record and a gate: reading the artifact
    would let someone add a bad annotation and pass, because the artifact would still hold yesterday's
    answer. Recomputing means the gate fails on the edit itself.

    ⚠️ It also fails when it CANNOT check. An entry whose accession has no cached UniProt record is
    reported in `uncheckable` and the assertion requires that list to be empty, so adding an entry
    without caching its record is a failure rather than a silent skip. Pure json, no numpy, so the
    release-surface CI job can run it."""
    sites = json.load(open(ROOT / "data/annotations/functional_sites.json"))
    cache = ROOT / "data" / "uniprot_cache"
    accs = sorted(a for a in sites if not a.startswith("_"))
    e, uncheckable = {}, []
    for a in accs:
        f = cache / f"{a}.json"
        if not f.exists():
            uncheckable.append(a)
            continue
        d = json.load(open(f))
        seq_len = len(d["sequence"]["value"])
        cleaved = [(ft["type"], ft["location"]["start"]["value"], ft["location"]["end"]["value"])
                   for ft in d.get("features", [])
                   if ft["type"] in ("Signal", "Propeptide")]
        site = sites[a]["functional_sites"]
        off = site.get("precursor_offset") or 0
        pos = [q + off for q in site["catalytic_residues"]]
        in_sig = sorted({q for q in pos for t, s_, t_ in cleaved
                         if t == "Signal" and s_ <= q <= t_})
        in_pro = sorted({q for q in pos for t, s_, t_ in cleaved
                         if t == "Propeptide" and s_ <= q <= t_})
        e[a] = {"precursor_offset": off, "positions": pos,
                "has_signal": any(t == "Signal" for t, _s, _t in cleaved),
                "in_signal": in_sig, "in_propeptide": in_pro,
                "past_end": [q for q in pos if q > seq_len],
                "clean": not (in_sig or in_pro or any(q > seq_len for q in pos))}
    with_sig = sorted(a for a in e if e[a]["has_signal"])
    return {"n_entries": len(accs), "uncheckable": uncheckable,
            "in_signal": sorted(a for a in e if e[a]["in_signal"]),
            "in_propeptide": sorted(a for a in e if e[a]["in_propeptide"]),
            "past_end": sorted(a for a in e if e[a]["past_end"]),
            "n_clean": sum(1 for a in e if e[a]["clean"]),
            "accs_with_signal_peptide": len(with_sig),
            "offsets_applied": {a: e[a]["precursor_offset"] for a in sorted(e)
                                if e[a]["precursor_offset"]},
            "repaired_entries_clean": all(e[a]["clean"] for a in ("P00588", "P00648", "P02879")),
            "vacA_empty": e["P55981"]["positions"] == [],
            # the stored artifact must agree with this recomputation, or src/53 is stale
            "artifact_agrees": (j("v3/signal_peptide_sweep.json")["in_signal"]
                                == sorted(a for a in e if e[a]["in_signal"]))}



def annotation_provenance():
    """How much of the FSPE annotation set is in UniProt, compared against **Active site** features
    only.

    The number that matters is the omission count. Across the whole panel only two UniProt active
    sites are missing from the curation and both are Q51451's, on the entry entry nineteen already
    found defective, so that omission is not a symptom of a wider pattern.

    ⚠️ The first version of this audit pooled Active site with Binding site and Site and reported 64
    omissions. Most were carbohydrate sites in ricin's B-chain lectin domain, AMP contacts, anthrax
    protective antigen's Ca(2+) sites and furin cleavage positions, none of which belong in a field
    called `catalytic_residues`. The forbid list bans the phrasing a live claim would use while
    letting the documents name the number in order to explain that it was wrong.

    ⚠️ P1 is pinned as a NULL that was preregistered underpowered. The assertion requires it to stay
    non-significant, so a later run that turned it into a result would fail here and have to be
    written up rather than absorbed."""
    d = j("v3/annotation_provenance_audit.json")
    t = d["totals"]
    omit = {a: r["omit_active"] for a, r in d["entries"].items() if r["omit_active"]}
    return {"primary": d["primary_comparison"], "n_entries": d["n_entries"],
            "n_comparable": len(d["comparable"]),
            "annotated": t["annotated"], "confirmed_active": t["confirmed_active"],
            "frac_active": t["frac_active_confirmed"],
            "omitted_active": t["omitted_active"], "who_omits": omit,
            "extra_vs_active": t["extra_vs_active"],
            "extra_explained": t["extra_explained_by_other_feature"],
            "grounded": t["grounded_in_uniprot"], "frac_grounded": t["frac_grounded"],
            "exact_match": d["exact_match"], "zero_confirmed": d["zero_confirmed"],
            "no_active_site": d["no_uniprot_active_site"],
            "p1_n": d["P1"]["n"], "p1_rho": d["P1"]["rho"], "p1_p": d["P1"]["p"],
            "p1_significant": d["P1"]["significant_at_05"]}


CLAIMS = [
    # 0.018 / 12-of-15 was the pre-2026-05-22 numbering. The tolerance is 1e-4 rather than the old
    # 0.002 because the sign test is exact: with n fixed at 15 the only reachable values near 0.0037
    # are 121/32768 and 576/32768, so a loose window would let the stale figure pass as the new one.
    # The forbid catches what the three positive pins cannot: a document carrying the old and the new
    # figure side by side, and the two public surfaces with no FSPE pin at all (ARCHITECTURE.md,
    # MECHANISM_GENERALIZATION.md). It is scoped to PUBLIC only, so src/46's deliberate quotation of
    # the old headline, and the same quotation in docs/DATA_CORRECTIONS.md, are untouched.
    ("FSPE protein-level sign test, SEB excluded", fspe_protein_level,
     lambda v: (v["n"] == 14 and v["below_1"] == 12 and abs(v["sign_p"] - 0.0065) < 1e-4
                and v["excluded"] == ["P01552"]
                and abs(v["excluded_ratios"]["P01552"] - 0.9556) < 1e-3
                # the superseded figures, still pinned so the exclusion cannot be silently undone
                and v["full_n"] == 15 and v["full_below_1"] == 13
                and abs(v["full_sign_p"] - 0.0037) < 1e-4
                # src/21's artifact must agree with this recomputation
                and v["artifact_n"] == 14 and v["artifact_below_1"] == 12
                and abs(v["artifact_sign_p"] - 0.0065) < 1e-4
                and abs(v["artifact_full_sign_p"] - 0.0037) < 1e-4
                # opposite directions: sign test weaker, permutation stronger
                and v["weakens"]
                and v["artifact_perm_p"] < v["artifact_full_perm_p"]
                and abs(v["artifact_perm_p"] - 0.0001) < 1e-4
                and v["flagged_but_absent"] == []),
     {"README.md": "12/14 below 1.0, sign test p = 0.0065",
      "huggingface/README.md": "12/14 below 1.0, sign test p = 0.0065",
      "docs/EVALUATION_REPORT.md": "12/14 ratios below 1.0, exact sign test p = 0.0065",
      # the interview brief quotes the panel mean as well as the count, so both are pinned
      "docs/BIOHUB_RESEARCH_BRIEF.md":
      "14 proteins; mean ratio 0.400; 12/14 below 1.0; sign test p = 0.0065"},
     # The superseded headline, which must not be led with on any public surface again. 🔴 The first
     # string was "13/15 below 1.0, sign test p = 0.0037" until 2026-09-24, and the brief carried a
     # bare "13/15 below 1.0" underneath it for two days without failing the gate: a forbid pinned to
     # one full sentence does not cover the fragment. Shortened to the fragment, which subsumes it.
     ["13/15 below 1.0", "12/15 below 1.0", "sign test p = 0.018"]),
    # Both flagged entries are counted as successes by the headline, so the leave-out value is part
    # of the claim rather than a footnote to it. 11/13 is exactly 92/8192, so the tolerance is tight.
    ("the FSPE headline's dependence on the two annotation-flagged entries", fspe_flagged_leaveout,
     lambda v: (v["flagged"] == ["P01552", "Q51451"] and v["flagged_all_below_1"]
                and v["full"]["n"] == 15 and v["full"]["below_1"] == 13
                and v["without_both"]["n"] == 13 and v["without_both"]["below_1"] == 11
                and abs(v["without_both"]["sign_p"] - 0.01123) < 1e-4
                and all(s["n"] == 14 and s["below_1"] == 12 for s in v["each"].values())
                and v["direction_survives"] and 2.5 < v["p_inflation"] < 3.5),
     {"docs/EVALUATION_REPORT.md":
      "11/13 at p = 0.011 with the other annotation-flagged entry removed too"},
     []),
    # The worry was that a known-bad masked position was manufacturing ExoS's below-1.0 verdict. It
    # was doing the reverse. Pinned because a robustness result that resolves the convenient way is
    # exactly the kind that gets quoted loosely later.
    ("the spurious ExoS position dilutes its ratio rather than creating it", q51451_site_sensitivity,
     lambda v: v is None or (v["accession"] == "Q51451" and v["spurious"] == 234
                             and v["residue_at_spurious"] == "D"
                             and v["trp_positions"] == [71, 184] and v["no_offset_can_reach"]
                             and abs(v["ratio_published"] - 0.6618) < 5e-4
                             and abs(v["ratio_without"] - 0.6034) < 5e-4
                             and v["moves_down"] and v["both_below_1"]
                             and not v["verdict_flips"] and v["matches_pipeline"]),
     {"docs/EVALUATION_REPORT.md":
      "0.6618 with the spurious position and 0.6034 without it"}, []),
    # Pinned as an asymmetry, not a winner: conformal's interval covers nominal where the quantile
    # estimator's excludes it, and conformal is UNREACHABLE at the tighter budget the quantile
    # estimator happily answers. The failing-class ordering is pinned too, because the run's real
    # value is that the failure story does not depend on the threshold rule.
    ("the test partition the panel never had, across two arms, with the parts that do not replicate",
     conformal_lomo_test_split,
     lambda v: v is None or (
         # holds in both arms
         v["q_excludes_both_alphas_both_arms"] and v["conformal_unreachable_01_both"]
         and v["conformal_closer_both"] and v["estimator_ordering_preserved_both"]
         and v["bottom2_set_agrees"]
         # conformal attains its guarantee once the seed count can see it
         and v["conformal_covers_nominal_both"] and v["conformal_near_theoretical_both"]
         and all(x["split"]["calibrate"] == 59 and x["split"]["test"] == 60 and x["seeds"] == 200
                 and x["q_fp_01"] > 2.5 * 0.01 and x["max_delta_pts"] <= 0.001
                 and x["min_delta_pts"] > -15.0
                 for x in v["arms"].values())
         # the one disagreement that survives 200 seeds, asserted TRUE so that a later run which
         # quietly made it agree fails the gate and forces the write-up to be re-read
         and v["worst_class_disagrees"]),
     # Pin a single-line, distinctive string. An earlier pin spanned a line break and failed the
     # moment the paragraph was rewrapped, which is a pin testing the line wrapping, not the claim.
     {"docs/DETECTOR_CRITERIA.md": "| conformal | **unreachable at m = 59** | | |"}, []),
    # The decomposition is the claim: conformal lands on its guarantee when the negatives are
    # exchangeable, and every point above nominal after that is attributable to distribution shift or
    # to the pool's name redundancy, not to the estimator. Marked provisional in the artifact because
    # pool embeddings exist for one arm only.
    ("where the false-positive budget goes once the test negatives come from outside the panel",
     external_test_partition,
     # 🔴 `v is None or ...` until 2026-09-24: a missing artifact passed the gate vacuously, which is
     # the wrong default for the claim that carries this project's worst-scoring criterion. Both arms
     # are required now, and the document pins below mean the public surfaces cannot drop the figure.
     lambda v: v is not None and (
         all(x["calibration_n"] == 118 and x["pool_n"] == 8258 and x["dropped"] == "Q8X739"
             and x["seeds"] == 200
             # at m=118 conformal is reachable at BOTH budgets, unlike at m=59
             and x["k"]["0.05"] == 5 and x["k"]["0.01"] == 1
             and x["conformal_reachable_at_1pct"]
             and abs(x["guarantee"]["0.05"] - 0.0420) < 1e-3
             and abs(x["guarantee"]["0.01"] - 0.0084) < 1e-3
             and x["conformal_holds_in_control"]
             and x["ctrl_conf_05_covers"] and x["ctrl_conf_01_covers"]
             and x["dedup_quant_05"] > 0.09
             for x in v["arms"].values())
         # the whole decomposition replicates across both arms
         and v["ctrl_conf_on_guarantee_both"] and v["ctrl_quant_exceeds_both"]
         and v["shift_breaks_conformal_both"] and v["dedup_raises_fp_both"]
         and v["monotone_decomposition_both"]
         and v["still_provisional"] == []
         and abs(v["shift_quant_canonical"] - 0.0787) < 5e-4
         and abs(v["shift_quant_35M"] - 0.0794) < 5e-4
         and abs(v["dedup_quant_canonical"] - 0.0964) < 5e-4
         and abs(v["dedup_quant_35M"] - 0.1029) < 5e-4),
     {"docs/DETECTOR_CRITERIA.md":
      "4.32%   what it delivers when the negatives really are exchangeable",
      "docs/DETECTOR_EVALUATION_SUMMARY.md": "7.14%   after the pool's duplicate names stop hiding the failures",
      # added 2026-09-24: the in-sample nature of flagged@95 had never been stated on either public
      # headline surface, so the project's worst-scoring criterion was invisible to anyone who read
      # only the README or the dataset card
      "huggingface/README.md":
      "at a nominal 5% the published `np.quantile` estimator delivers **7.87%**",
      "README.md": "a nominal 5% budget really costs about **7.9%**",
      "docs/MECHANISM_GENERALIZATION.md":
      "| panel to pool, deployment shift | 7.87% [7.57, 8.16] exceeds | 5.98% [5.73, 6.23] **exceeds** |"},
     []),
    ("FSPE pseudoreplicated figure is labelled, not led with", fspe_protein_level,
     lambda v: True, {}, ["Pooled meta-analysis: p = 2.6", "meta-analysis (p = 2.6 × 10⁻⁸) is the better-powered"]),
    ("Embedding separability AUROC", separability,
     lambda v: v is None or abs(v["auroc"] - 0.981) < 0.002, {}, []),
    # 🔴 Added 2026-09-24, entry twenty-two. The v1 0.981 is the most-quoted number in the project and
    # the caveat telling a reader to use the screened v2 0.974 instead existed on the Hugging Face
    # card ONLY, where it had been edited in on the Hub and never brought back to the repository. The
    # README led with 0.981 and no correction, and no gate noticed, because a claim with no document
    # pin cannot fail on a document. Every surface that prints the figure now has to print the
    # caveat, and the 0.974 it points at is recomputed from the v2 LOMO artifact rather than quoted.
    ("the v1 separability figure carries its screening caveat on every surface that prints it",
     lambda: {"v1": separability()["auroc"], "v2": lomo_class_recovery()["baseline_auroc"]},
     lambda v: abs(v["v1"] - 0.981) < 0.002 and abs(v["v2"] - 0.974) < 0.002,
     {"README.md": "**Use the screened v2 panel: baseline separability AUROC 0.974 ± 0.014.**",
      "huggingface/README.md":
      "**Use the screened v2 panel: baseline separability AUROC 0.974 ± 0.014.**",
      "docs/BIOHUB_RESEARCH_BRIEF.md": "**AUROC 0.974 +/- 0.014** on the screened v2 panel"}, []),
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
    # 🟢 RESOLVED 2026-09-23. This claim twice measured something other than what it said.
    # It first read `flips == 3 and n_rows == 12` and kept passing after the numbering fix for the
    # wrong reason: the corrected ESM-2 column moved P00648 from >1 to <1 while P02879 stayed >1, so
    # the total happened to land near 3. It was then narrowed to the nine rows where no column had
    # moved, with three named indeterminate, because ESM-3 and SaProt were still in the old
    # coordinates and no count over all twelve compared like with like.
    #
    # Cayuga job 3392513 re-ran both of those columns through the same offset resolver, so all three
    # now share one numbering and the whole table is comparable. The count is 5 of 12. The
    # indeterminate subset is gone, which is the outcome the narrowed version existed to wait for.
    #
    # ⚠️ The superseded figure is forbidden in its ASSERTION form only. The evaluation report names
    # "2 of 9" in the sentence that retires it, which is this repository's house style for
    # corrections, so a forbid on the bare digits would forbid explaining the change. Same treatment
    # as the pooled omission count in the annotation-provenance claim.
    ("the guaranteed threshold applied to the per-class table, and it only ever costs recovery",
     conformal_operating_point,
     lambda v: (
         len(v) == 4 and all(x["reproduces"] for x in v.values())
         # not one class in any run gains recovery under the guaranteed threshold
         and all(x[al]["n_rose"] == 0 for x in v.values() for al in ("0.05", "0.01") if x[al])
         # the frozen v2 panel cannot express a nominal 1% budget: k = 0 at m = 61
         and v["v2"]["m"] == 61 and v["v2"]["k"]["0.01"] == 0
         and v["v2"]["unreachable"] == [0.01] and v["v2"]["0.01"] is None
         and v["v3"]["m"] == 118 and v["v3"]["k"]["0.01"] == 1 and v["v3"]["unreachable"] == []
         # the realized rates the documents now have to quote alongside every recovery figure
         and abs(v["v2"]["0.05"]["fp_quantile"] - 0.0656) < 5e-4
         and abs(v["v2"]["0.05"]["fp_conformal"] - 0.0492) < 5e-4
         and abs(v["v3"]["0.05"]["fp_quantile"] - 0.0508) < 5e-4
         and abs(v["v3"]["0.01"]["fp_quantile"] - 0.0169) < 5e-4
         # the cost grows as the budget tightens, on the same arm
         and abs(v["v2"]["0.05"]["mean_delta_pts"] + 8.8) < 0.2
         and abs(v["v3"]["0.05"]["mean_delta_pts"] + 2.1) < 0.2
         and abs(v["v3"]["0.01"]["mean_delta_pts"] + 15.4) < 0.2
         and v["v3"]["0.01"]["n_dropped"] == 14
         # § 10.6.2's dissociation survives the estimator change
         and abs(v["v2"]["beta_q"] - 0.2143) < 0.002 and abs(v["v2"]["beta_c"] - 0.10) < 0.002
         and abs(v["v3_esmc_600M"]["beta_c"] - 0.30) < 0.005
         and abs(v["v3"]["beta_c"] - 0.16) < 0.005
         # exact values, not display-rounded ones: 0.275 against a 0.28 pin at tolerance 0.005
         # fails on equality, which is a silly way to lose a real assertion
         and abs(v["v3_esm3_1_4B"]["phage_c"] - 0.275) < 0.002
         and abs(v["v3_esmc_600M"]["phage_c"] - 0.0875) < 0.002),
     {"docs/MECHANISM_GENERALIZATION.md":
      "| **v2**, canonical | 61 | 3 | 4.84% | **6.56%** | 4.92% |",
      "docs/DETECTOR_EVALUATION_SUMMARY.md": '| v2, the frozen panel | 5% | **6.56%** | 7 of 13 | **0** | **−8.8 pt** |'}, []),
    ("FSPE's background three ways: the headline is identical and the draw is not",
     fspe_background_ablation,
     lambda v: v is None or (
         # the gate that makes the rest attributable, and step 3 of the mutation preregistration
         v["reproduces_published"] and v["n_drift"] == 0 and v["n_proteins"] == 15
         # the documented-but-unimplemented exclusion touches 6 proteins through 9 positions
         and v["n_affected"] == 6 and v["n_flanking_positions"] == 9
         # and changes no headline: 12/14 under all three backgrounds
         and set(v["excluded_counts"].values()) == {"12/14"}
         # effect 1 is small and isolated
         and v["max_drop_delta"] < 0.035 and v["unaffected_move_by_zero"]
         and v["crossings_drop"] == 0 and v["crossings_resample"] == 0
         # effect 2, the finding: redrawing moves ratios an order of magnitude further, measured
         # where there is no flanking to remove, and takes one protein to within 0.005 of 1.0
         and v["max_resample_delta"] > 0.30
         and v["mean_resample_delta_on_clean"] > 8 * (v["max_drop_delta"] / 3)
         and v["closest_to_one_after_resample"] > 0.99),
     {"docs/EVALUATION_REPORT.md":
      "P13423   0.6499 -> 0.9950     (+0.345, to within 0.005 of the threshold)"},
     # the three sentences that said the background excludes the flanking positions
     ["excluding ±2 flanking residues\naround each functional site",
      "non-functional residues (excluding\n±2 flanking positions)"]),
    ("the negatives have a standing test partition, bounded by the panel's own negatives",
     negative_test_partition,
     lambda v: v is None or (
         v["pool"] == 8259 and v["admitted"] == 8258 and v["rejected"] == 1
         and v["rejected_accs"] == ["Q8X739"]
         # the bound is read from the screen, is stated in the rule, and is tighter than 0.30
         and abs(v["bound"] - 0.282008) < 1e-5 and v["bound_in_rule"]
         # and tightening it is free: the same single protein clears either bound
         and v["n_above_bound"] == 1 and v["n_above_030"] == 1
         # the two name conventions, reconciled with src/36's 3550 and src/49's 3407
         and v["distinct_names"] == 3549 and v["gene_symbols"] == 3407
         # what makes it a partition rather than a draw
         and v["test_only_stated"] and v["exchangeability_voided"]
         and v["digest_matches_membership"]
         # the resolution it buys, and the one it does not
         and abs(v["panel_rate"] - 1 / 296) < 1e-9
         and abs(v["name_rate"] - 1 / 3549) < 1e-9),
     {"docs/DETECTOR_CRITERIA.md": "**Partial as of 2026-09-27, and it was a fail.**"}, []),
    ("P2's four benign controls verify as sequences, thermolysin included",
     benign_control_sites,
     lambda v: v is None or (
         v["n_verified"] == 4 and v["n_rejected"] == 0
         and v["keys"] == ["1AST", "1LNF", "1LYZ", "1QD2"]
         # every offset is the unique one placing every identity, src/46's rule
         and v["offsets"] == {"1AST": 49, "1LNF": 0, "1LYZ": 18, "1QD2": 24}
         and v["residues"] == {"1AST": "HEHH", "1LNF": "HEHEH", "1LYZ": "ED", "1QD2": "YYE"}
         # thermolysin's set is catalytic, not the calcium-inclusive one
         and v["n_positions"]["1LNF"] == 5 and v["thermolysin_excluded_ca_sites"] == 13
         and v["thermolysin_from_uniprot"]
         and v["text_sourced"] == ["1AST", "1LYZ", "1QD2"]),
     # pinned in the preregistration rather than the corrections log, because the log is not on the
     # audited surface: it has to be able to quote figures the gate forbids elsewhere
     {"docs/MUTATION_EXTENSION_PREREGISTRATION.md":
      "| thermolysin `1LNF` | P00800 | +0 | 374, 375, 378, 398, 463 | HEHEH |"}, []),
    ("the mutation axis: P1 supported, P2 failed on its own ceiling", fspe_m,
     lambda v: v is None or (
         # step 4's gate, leakage half only: the AUROC half has no power at n_benign = 4
         v["gate_centred"] and v["gate_mean_over_se"] < 2.0
         # P1 supported at the frozen threshold
         and v["p1_k"] == 13 and v["p1_n"] == 15 and abs(v["p1_p"] - 0.0037) < 1e-4
         and v["p1_supported"] and v["p1_excl"] == (12, 14)
         # and its exceptions are FSPE's own type-2 RIP pair, on a different reduction
         and v["below_zero"] == ["P02879", "P11140"]
         # P2 fails on direction, which is what the ceiling tests
         and v["p2_ceiling_fires"] and v["p2_difference"] < 0
         and abs(v["p2_panel_mean"] - 4.32) < 0.02 and abs(v["p2_benign_mean"] - 5.26) < 0.02
         and v["p2_auroc"] < 0.5 and v["p2_n_benign"] == 4
         # the benign controls out-rank almost the whole panel
         and v["panel_below_best_control"] == 14 and v["n_panel"] == 15
         and v["p6_unrun"]),
     {"docs/MUTATION_EXTENSION_PREREGISTRATION.md":
      "**So the verdict, as the preregistration fixed it in advance: dFSPE-M measures",
      # the table gained an n = 60 row on 2026-09-27, so the pin follows the n = 4 cells
      "docs/EVALUATION_REPORT.md": "| benign controls, **n = 4** | **+5.26** (−0.94) | 0.393 | 0.678 |",
      # the six-page summary carried nothing about this axis until 2026-09-27, so a completed line
      # of work was missing from the flagship public document
      "docs/DETECTOR_EVALUATION_SUMMARY.md": "against the panel's **+4.32**; two benign zinc proteases out-rank 14 of the 15 panel proteins."}, []),
    ("P3 is not supported, and the statistic is dominated by which residue was substituted",
     fspe_m_p3,
     lambda v: v is None or (
         v["n"] == 34 and abs(v["median"] - 61.1) < 0.2
         and v["k_below_25"] == 9 and v["sign_p"] > 0.99
         and not v["supported"] and not v["ceiling"]
         and v["n_ala"] == 19 and v["n_cys"] == 8
         and abs(v["median_ala"] - 61.1) < 0.2 and v["median_cys"] < 1.0
         and v["scan_share"] > 0.75 and v["top_share"] > 0.8),
     {"docs/MUTATION_EXTENSION_PREREGISTRATION.md":
      "**79% of the set is alanine or cysteine scanning**",
      "docs/EVALUATION_REPORT.md": "median\npercentile **61.1** among the 19 alternatives"}, []),
    ("P5's composition half is indeterminate because its margin is inside the noise",
     fspe_m_p5_composition,
     lambda v: v is None or (
         v["n_panel"] == 15 and v["n_control"] == 4
         and abs(v["auroc_dfspe_m"] - 0.3833) < 0.002
         and abs(v["auroc_composition"] - 0.4167) < 0.002
         and v["difference"] < 0 and v["both_below_half"]
         # the margin section 4 requires is a fifth of the statistic's own null spread
         and v["margin"] == 0.05 and abs(v["null_sd"] - 0.167) < 0.005
         and v["inside_noise"] and v["indeterminate"]),
     {"docs/MUTATION_EXTENSION_PREREGISTRATION.md":
      "The required margin is **one fifth of the sampling noise**"}, []),
    ("P6: the alignment agrees on direction, not magnitude, and fails P2 the same way",
     fspe_m_p6,
     lambda v: v is None or (
         v["n_scored"] == 19 and v["n_panel"] == 15 and v["n_control"] == 4
         # the frozen AUROC half: the difference exceeds the margin AND equals the noise
         and abs(v["difference"] - 0.1667) < 0.002 and abs(v["null_sd"] - 0.1667) < 0.002
         and v["inside_noise"]
         # direction agrees, magnitude does not
         and 0.30 < v["pearson"] < 0.42 and 0.30 < v["spearman"] < 0.42
         # the alignment's own P1 is the same direction at a weaker p than the model's 13/15
         and v["pssm_p1"] == (12, 15) and abs(v["pssm_p1_p"] - 0.0176) < 1e-3
         # and it fails P2's comparison in the same direction, by more
         and v["pssm_gap"] < 0 and v["dfspe_gap"] < 0 and v["pssm_gap"] < v["dfspe_gap"]
         # recruitment depth varies enough that the three non-positive rows are near-unmeasured
         and v["nonpositive_panel"] == ["P00588", "P01552", "P13423"]
         and v["min_depth"] <= 3),
     {"docs/MUTATION_EXTENSION_PREREGISTRATION.md":
      "**An alignment shows the same constraint signal at catalytic sites and the",
      "docs/EVALUATION_REPORT.md": "its own P1 is 12 of 15 above zero at *p* = 0.0176"},
     # the sentence the report carried while P6 was unrun
     ["P6, the alignment baseline that would settle whether a PSSM does this as well, is unrun"]),
    ("margin survives a label-free typicality baseline, which itself does not replicate",
     typicality_baseline,
     lambda v: v is None or (
         v["arms"] == ["esm2_35M", "esm2_650M"] and v["n_classes"]["esm2_650M"] == 12
         # margin reproduces § 10.6.1's published rho exactly, tie-averaged ranks included
         and abs(v["margin"]["esm2_650M"] - 0.8944) < 0.002
         and abs(v["margin"]["esm2_35M"] - 0.7958) < 0.002
         # the baseline is a real predictor on the canonical arm, and was missing
         and v["typicality"]["esm2_650M"] < -0.70 and v["typicality_perm_p"]["esm2_650M"] < 0.01
         # and it does NOT replicate: null and sign-flipped on the second arm
         and abs(v["typicality"]["esm2_35M"]) < 0.10
         and v["typicality_perm_p"]["esm2_35M"] > 0.30
         # margin's partial is essentially its raw value on both arms
         and v["margin_partial"]["esm2_650M"] > 0.75 and v["margin_partial"]["esm2_35M"] > 0.75
         and abs(v["typicality_partial"]["esm2_650M"]) < 0.40
         and v["margin_dominates_both"]),
     {"docs/MECHANISM_GENERALIZATION.md":
      "**the more typical of general protein space a class is, the less of it is recovered.**",
      # The public card carries the margin result, so it carries the baseline for it too. Nothing in
      # the card is retired by this: it claimed margin beat its own parts, which is still true, and
      # never claimed it had been given anything simpler than itself.
      "huggingface/README.md":
      "reaches Spearman **\u22120.746** against recovery at permutation *p* = **0.0034**"}, []),
    ("P2 fails harder at n_benign = 60 than at 4, and the gate's AUROC half can never be a test",
     fspe_m_a2,
     lambda v: v is None or (
         v["n_benign"] == 60 and v["n_toxin"] == 14
         # the panel did not move, so the control set is the only change
         and v["panel_identical_to_frozen4"]
         # the ceiling fires on direction AND significance now
         and abs(v["panel_mean"] - 4.3197) < 0.01 and abs(v["benign_mean"] - 6.7563) < 0.01
         and v["difference"] < -2.4 and v["perm_p"] > 0.99
         # A2-2: not 0.70, and the interval excludes 0.50 on the wrong side
         and v["auroc"] < 0.30 and v["interval_clear_of_half"] and v["auroc_ci95"][1] < 0.5
         and v["controls_above_panel_min"] == 60 and v["controls_above_zero"] == 59
         and v["lone_negative_control"] == ["P0CK11"]
         # A2-3: inside [0.40, 0.60], so no leak
         and 0.40 <= v["shuffled_auroc"] <= 0.60
         # and the gate's own tolerance is still not a test, at any n_benign
         and v["reproduces_fifth_amendment"] and not v["gate_is_a_test_n60"]
         and v["gate_two_sided_p"] > 0.10 and v["sweep_floor_sd"] > 0.07
         and v["sweep_floor_passes"] < 0.60 and not v["sweep_any_is_a_test"]),
     {"docs/EVALUATION_REPORT.md":
      "| benign enzymes, **n = 60** 🔑 | **+6.76** (−2.44) | **0.265** [0.102, 0.445] | **0.9991** |",
      "docs/NEGATIVE_EXPANSION_PREREGISTRATION.md":
      "**The null sd floors at about 0.077 and the ±0.05 gate never passes more than about half the "
      "time,"},
     # the caveat the powered run closed
     ["the controls are not *significantly* above the panel, and the ceiling fires\non direction rather than significance — the burden was on the panel to exceed them and it does not\nexceed them at all. And dFSPE-M"]),
    ("the A2 control set clears its preregistered floor without any exclusion having been loosened",
     benign_enzyme_set,
     lambda v: v is None or (
         v["n_admitted"] == 60 and v["target_n"] == 30 and v["meets_target"]
         and v["pool"] == 66275 and v["window"] == [286, 1147]
         # every admitted control carries at least the preregistered three Active sites
         and v["min_sites"] >= 3
         # rule 4 and rule 1 never bound: nothing came close to the bound, nothing was in the panel
         and v["rejected_similarity"] == 0 and v["rejected_panel"] == 0
         and v["max_similarity"] < 0.1 and abs(v["bound"] - 0.282008) < 1e-5
         # the two rules that did bind, and the hazard-term filter that caught one
         and v["rejected_active_sites"] == 332 and v["rejected_vfdb"] == 2
         and v["rejected_hazard_term"] == 1
         # hydrolase-heavy across six EC classes, and exactly one viral-origin member
         and v["ec_hydrolase"] == 26 and v["n_ec_classes"] == 6
         and v["viral"] == ["P0CK11"]
         and v["order"] == "sha256(accession) ascending"),
     {"docs/NEGATIVE_EXPANSION_PREREGISTRATION.md":
      "**60 controls admitted against a floor of 30**"}, []),
    ("VFDB's Exotoxin category is a few per cent of it, and the host contrast is setB-only",
     vfdb_axis,
     lambda v: v is None or (
         v["n"] == {"setA": 4755, "setB": 30215}
         and v["n_categories"] == {"setA": 14, "setB": 14}
         # fourteen categories at holdout-usable size beats the panel's eleven to twelve
         and v["categories_ge_20"]["setB"] == 14 and v["categories_ge_20"]["setA"] == 12
         # the distinction the v2/v3 panel assumes away: toxin is one category of fourteen
         and v["exotoxin"] == {"setA": 248, "setB": 1218}
         and v["exotoxin_frac"]["setA"] < 0.06 and v["exotoxin_frac"]["setB"] < 0.05
         and v["non_toxin"] == {"setA": 4507, "setB": 28997}
         # setA, the experimentally verified core, holds no plant or insect pathogen at all
         and v["setA_non_human"] == 0 and v["setB_non_human"] == 1604
         and v["host"]["setB"]["plant"] == 1093 and v["host"]["setB"]["insect"] == 511
         # and the two axes cross, which is what makes the host question answerable
         and v["exotoxin_by_host_setB"]["insect"] == 181
         and v["exotoxin_by_host_setB"]["plant"] == 61
         and v["effector_largest"] == "Effector delivery system"),
     {"docs/VFDB_CLASS_AXIS_DESIGN.md":
      "🔴 **The host contrast is a setB-only property.**"}, []),
    ("the external classifier numbers stay attached to the papers they were read from",
     external_baseline_numbers,
     lambda v: (v["deepvf_auc"] == 0.896 and v["dtvf_auroc"] == 0.9208
                and v["deepvic_auroc"] == 0.954
                and v["benchmark_pool"] == ("3,576", "4,910") and v["held_out"] == ("576", "576")
                and v["deepvic_holdout"] == ("13,384", "33,456")
                and v["split_inference_flagged"] and v["panel_size_stated"]),
     {"docs/MECHANISM_GENERALIZATION.md":
      "so \"0.92 on the 576/576 benchmark\" is an inference from DeepVF's construction"},
     # the sentence that attributed DeepVF's partition to DTVF, in both spellings it could take
     ["DTVF (ProtT5 + LSTM/CNN) reports AUROC 0.92 on the standard 576/576",
      "DTVF reports AUROC 0.92 on the standard 576/576"]),
    ("every dated entry cited by a document exists in the corrections log", cited_entries_exist,
     # `headings` counts distinct DATES, not entries: several days carry a second, third and fourth
     # entry under the same date. 13 is a floor and can only grow.
     lambda v: v["dangling"] == {} and v["citations"] >= 3 and v["headings"] >= 13, {}, []),
    ("Cross-model FSPE flips, now that all three columns share a numbering", flip_count,
     lambda v: (v["n_rows"] == 12 and v["all_columns_present"] == 12
                and v["flips"] == 5 and v["indeterminate"] == []
                and v["stale_columns"] == 0),
     {"docs/EVALUATION_REPORT.md":
      "**5 of 12 rows** disagree in sign between at least two of the three models"},
     # 🔴 The prose forms were added 2026-09-23 after the digit-only forbid missed them. The
     # 09-23 retirement forbade "2 of 9" and left the *words* "Three of 12 proteins" standing in
     # two body sections of the same report, so the document asserted 5 of 12 in § 7 and three of
     # twelve in §§ 5 and 5.x for one commit. A forbid that only covers the spelling the author
     # happened to use when retiring a figure does not cover the spellings already in the file.
     # Entry twenty-one of docs/DATA_CORRECTIONS.md. Both cases are listed because the sentence
     # can start either mid-line or after a period.
     ["honest current figure is **2 of 9**", "the figure is 2 of 9",
      "Three of 12 proteins", "three of 12 proteins"]),
    ("the mature-chain numbering fix reached every consumer and moved nothing else",
     functional_site_numbering,
     lambda v: (v["entries"] == 16 and v["n_fspe"] == 15
                and v["offsets"] == {"P02879": 35, "P00648": 47, "P00588": 32}
                and v["verified_notes"] == 3
                # three entries are flagged, two reach the runtime: P55981's catalytic_residues is
                # empty by design, so src/04 skips it and no published number depends on it
                and v["annotation_flags"] == ["P01552", "P55981", "Q51451"]
                and v["runtime_flagged"] == ["P01552", "Q51451"]
                and v["skipped_no_catalytic_residues"] == ["P55981"]
                and v["indexing_consistent"]
                # the load-bearing pair: only the offset carriers moved, and the rest by float noise
                and v["moved_is_the_offset_set"] and v["max_unmoved_delta"] < 1e-5
                and v["below_1_pre"] == 12 and v["below_1_now"] == 13
                and (v["offset_uniqueness"] is None
                     or (set(v["offset_uniqueness"]) == set(v["offsets"])
                         and all(u["determined"]
                                 and u["full_match_offsets"] == [v["offsets"][a]]
                                 and u["n_checkable"] >= 3
                                 for a, u in v["offset_uniqueness"].items())))),
     # ASCII prefix on purpose: the sentence continues with a Unicode arrow, and a pin that can
     # mismatch on a codepoint fails for a reason that has nothing to do with the claim.
     {"README.md": "The displayed eight-protein panel barely moved (mean 0.6386"}, []),
    ("v2 panel and LOMO results describe the same panel", v2_panel_consistency,
     lambda v: (v["members"] == v["manifest_positives"] == v["results_n_positive"]
                == v["after_dedup"]
                and v["manifest_negatives"] == v["results_n_negative"]
                and not v["classes_with_mismatched_membership"]), {}, []),
    ("LOMO per-class recovery, canonical 650M arm", lomo_class_recovery,
     lambda v: (v["n_pos"] == 80 and v["n_neg"] == 154
                and abs(v["baseline_auroc"] - 0.974) < 0.002
                and v["beta_lactamase"] == (0.2143, 0.0143)
                and abs(v["beta_lactamase_auroc"] - 0.751) < 0.002
                and v["superantigen"] == (1.0, 1.0) and v["clostridial"] == (1.0, 1.0)
                and v["t3ss"] == (0.8, 0.8)
                and abs(v["provenance_auroc"] - 0.818) < 0.002),
     {"docs/MECHANISM_GENERALIZATION.md": "**80 hazardous proteins**",
      # the standalone summary quotes the alignment comparison; pinned so it cannot drift
      "docs/DETECTOR_EVALUATION_SUMMARY.md": 'beats Smith-Waterman by **+55.9 points**. On beta-lactamase alignment wins, **29.5% against 21.4%**.'}, []),
    ("LOMO figures quoted in the document match the artifact", lomo_class_recovery,
     lambda v: True,
     {"docs/MECHANISM_GENERALIZATION.md": "| **beta_lactamase** | 14 | **21%** | **1%** | 0.751 |"},
     []),
    ("AMR-category test v1: the number reproduces, the refutation it was read as does not",
     amr_category_test,
     lambda v: (v["n_members"] == 8 and v["n_flagged"] == 7
                and abs(v["recovery"] - 0.875) < 1e-6
                and v["recovery"] >= v["predicted_refuted_geq"]),
     {"docs/MECHANISM_GENERALIZATION.md": "87.5% (7 of 8)"}, []),
    ("AMR-category test v2, corrected: inconclusive at 75%, beta-lactamase still lowest",
     amr_category_test_v2,
     lambda v: (abs(v["reproduces_v1"] - 0.875) < 1e-6
                and abs(v["decisive"] - 0.75) < 1e-6
                and abs(v["bl_matched"] - 0.50) < 1e-6
                and abs(v["bl_protocol24"] - 0.50) < 1e-6
                and v["bl_is_lowest"] and v["verdict_inconclusive"]),
     {"docs/MECHANISM_GENERALIZATION.md": "75.0% (6 of 8)"},
     ["Whatever makes beta-lactamase hard, it is not a property of antibiotic resistance as",
      "three refused: not the head, not corpus or capacity"]),
    ("beta-lactamase provenance differs from the panel's hazard definition, documented",
     beta_lactamase_provenance,
     lambda v: v["beta_lactamase_members"] == 14,
     {"docs/MECHANISM_GENERALIZATION.md": "KW-0046"}, []),
    ("profile-HMM baseline: the probe leads both HMMER modes on every class, neither run degenerate",
     profile_hmm_baseline,
     lambda v: (all(not v[m]["beats_probe"] and not v[m]["degenerate"] for m in v)
                and v["phmmer"]["probe_minus_profile"] > 0.55
                and v["jackhmmer"]["probe_minus_profile"] > 0.60
                and v["phmmer"]["beta_lactamase"] < 0.10
                and v["jackhmmer"]["beta_lactamase"] < 0.05
                and min(v[m]["density"] for m in v) > 0.03),
     {"docs/MECHANISM_GENERALIZATION.md": "+59.8 points"}, []),
    ("Swiss-Prot profile pilot: every profile finds its siblings, the margin is still negative",
     swissprot_profile_pilot,
     lambda v: (v["n"] == 7 and v["all_siblings"] is True
                and v["n_negative"] == 6 and v["margin_mean"] < 0),
     {"docs/MECHANISM_GENERALIZATION.md": "six of seven members"}, []),
    ("pretraining-caveat tests: external holdout INVALID, exposure correlation does not survive controls",
     pretraining_caveat_tests,
     lambda v: (v["holdout_valid"] is False and v["holdout_ci_low"] <= 0.5
                and v["exposure_survives"] is False
                and v["exposure_pooled_p"] < 0.01 and v["exposure_class_p"] > 0.05),
     {"docs/MECHANISM_GENERALIZATION.md": "pseudoreplicated"}, []),
    ("fusing ESM-2 with ProtT5 does not beat ESM-2 alone", plm_fusion,
     lambda v: v["best"] == "esm2_650M" and v["delta"] < 0,
     {"docs/MECHANISM_GENERALIZATION.md": "Fusion costs **0.4 points**"}, []),
    ("prevalence: AUROC hides a precision collapse the panel's 34% base rate conceals",
     prevalence_adjusted,
     lambda v: (abs(v["panel_prevalence"] - 0.3419) < 0.001
                and v["auprc"] > 0.95 and abs(v["auprc_baseline"] - 0.342) < 0.002
                and v["precision_panel"] > 0.85 and v["precision_1e-3"] < 0.05),
     {"docs/MECHANISM_GENERALIZATION.md": "59 false alarms for every true one"}, []),
    ("layer depth is the one axis in \u00a79 that does change the answer", layer_depth,
     lambda v: (v["best"] == "L12" and v["gain"] > 0.03
                and v["platform_valid"] is True and v["platform_rel"] < 1e-5
                and v["bl_L12"] == 0.0 and abs(v["final_mean"] - 0.725) < 0.01),
     {"docs/MECHANISM_GENERALIZATION.md": "Layer 12 beats the final layer by 4.8 points"}, []),
    ("target host is a stronger confound than provenance, and it was never controlled",
     target_host_control,
     lambda v: (v["legibility"] > 0.90 and v["animal"] > 0.99 and v["nonanimal"] < 0.92
                and v["size_matched_ok"] is True and v["n_animal"] == 51
                and abs(v["rank_p"] - 0.0357) < 0.001 and v["post_hoc"] is True),
     {"docs/MECHANISM_GENERALIZATION.md": "below that entire range"}, []),
    ("the \u00a79.4 test input is in the repository, byte-identical to the lost original",
     amr_test_input_present,
     lambda v: (v["exists"] and v["n_sequences"] == 8 and v["bytes"] == 2511
                and v["sha256"] == "4df8a5c65ad684e31bebfb6a101cea7c6dca9bfd6307fa4f38a1a2a68edcc5d2"),
     {"docs/MECHANISM_GENERALIZATION.md": "4df8a5c65ad684e3"}, []),
    ("SAE feature uniqueness does not explain the beta-lactamase anomaly, once n is matched",
     interplm_feature_overlap,
     lambda v: (v["K"] == 6 and v["n_classes"] == 7
                and v["p"] > 0.05
                and v["bl_sig"] < 40
                and v["clost_unique"] > 0.70
                and v["clost_recovery"] == 1.0
                and v["bl_recovery"] < 0.25),
     {"docs/MECHANISM_GENERALIZATION.md": "Feature uniqueness does not predict recovery failure"}, []),
    ("InterPLM SAE features do separate hazard, so §9.6's premise holds",
     interplm_power_check,
     lambda v: (v["auroc_18"] > 0.85 and v["auroc_18"] < v["raw_for_comparison"]
                and v["alive_18"] > 10000
                and v["auroc_1"] < 0.80 and v["alive_1"] < 500),
     {"docs/MECHANISM_GENERALIZATION.md": "AUROC 0.910"}, []),
    ("the stored FHS-FSI correlation is stale; the current pairing is weaker and fragile",
     fhs_fsi_correlation_current,
     lambda v: (v["n"] == 12
                and v["stale_and_current_differ"] is True
                and abs(v["stale_stored_rho"] - 0.7005) < 0.001
                and abs(v["rho"] - 0.6585) < 0.001
                and v["p"] < 0.05
                and v["p_excluding_P13423"] > 0.05),
     {"docs/EVALUATION_REPORT.md": "rho 0.582, p 0.0604"}, []),
    ("one class in the training set costs another 16 points, above every random removal",
     training_set_contamination,
     lambda v: (v["delta"] > 0.13 and v["ci_low"] > 0 and v["percentile"] == 1.0
                and v["attributable"] > 0.15
                and v["movers"] == ["TACY_LISMO", "TACY_STRPQ"]
                and v["cdi_effect"] < v["cdi_random_sd"]
                and v["exploratory"] is True),
     {"docs/MECHANISM_GENERALIZATION.md": "the 100th percentile"}, []),
    ("non-animal hazard is not a category the probe learns from non-animal examples",
     nonanimal_category_refuted,
     lambda v: (v["refuted"] and v["includes_zero"]
                and abs(v["interaction"]) < 0.05
                and v["ci"][0] < 0 < v["ci"][1]),
     {"docs/MECHANISM_GENERALIZATION.md": "refuted it**: the interaction was"}, []),
    ("beta-lactamase: ESM-C 600M beats alignment, and is the only arm that does", beta_lactamase_across_arms,
     lambda v: (v["n_arms"] == 14 and abs(v["alignment"] - 0.30) < 0.02
                and v["duplicate_arm_max_diff"] == 0
                and v["esmc_600M"] is not None and v["esmc_600M"] > v["alignment"]
                and v["esm2_650M"] < v["alignment"]
                and v["arms_above_alignment"] == ["esmc_600M"]),
     {"docs/MECHANISM_GENERALIZATION.md": "ESM-C 600M recovers **51%**"},
     ["resists every configuration tested and that plain alignment beats every embedding method on it.\n**Both statements are correct**"]),
    ("member separability: embedding proximity separates, sequence similarity does not",
     member_separability_features,
     lambda v: (v["rows"] == 64 and v["caught"] == 44
                and v["margin_auroc"] > 0.9 and abs(v["margin_p"] - 1 / 20001) < 1e-6
                and v["nn_pos_auroc"] > 0.9
                and v["kmer_auroc"] < 0.5 and v["kmer_p"] > 0.05
                and 0.45 < v["length_auroc"] < 0.55),
     {"docs/MECHANISM_GENERALIZATION.md": "20,000 label shuffles"}, []),
    ("headline figures use the weakest of four classifier heads", classifier_heads,
     lambda v: (v["arms"] == 14 and v["logistic_worst_on"] == 7 and v["logistic_best_on"] == 2
                and v["gap_650M_pts"] >= 4 and v["gap_8M_pts"] >= 10),
     {"docs/MECHANISM_GENERALIZATION.md": "on **12 of 14 arms some other head"}, []),
    ("internal margin effect holds on mean-pooled arms, not on CLS or max", margin_effect_across_arms,
     lambda v: (v["arms"] == 14 and v["over_25pts"] == 12
                and v["under_25pts"] == ["esm2_650M_cls", "esm2_650M_max"]),
     {"docs/MECHANISM_GENERALIZATION.md": "**12 of 14 exceed the 25-point threshold**"}, []),
    ("fitting a probe does not always beat nearest-neighbour distance", probe_vs_similarity,
     lambda v: (v["arms"] == 14
                and v["negative_arms"] == ["esm3_1_4B", "esmc_6B", "prott5_xl"]
                and v["size_control_max_abs_pts"] < 5),
     {"docs/MECHANISM_GENERALIZATION.md": "on **3 of 14 arms it is negative**"}, []),
    ("every annotation file covers every panel member", annotation_coverage,
     lambda v: not v["gaps"], {}, []),
    ("v2 class eligibility is curated, not a size rule", v2_class_eligibility,
     lambda v: (v["flag_disagreements"] == 0
                and v["control_in_results"] and not v["control_flagged_eligible"]
                and not v["grab_bag_eligible"] and v["grab_bag_n"] >= 3
                and not v["unexpected_classes_in_table"]), {}, []),
    ("SAE feature space lifts the panel and drops beta-lactamase to zero",
     sae_feature_space_lomo,
     lambda v: (v["matched_mean"] > v["raw_mean"] and v["matched_bl"] == 0.0
                and v["raw_bl"] > 0.1 and v["matched_bl_zero_at_all_C"] and v["n_C"] == 4
                and v["full_mean"] < v["raw_mean"] and v["control_drop_pts"] > 25),
     {"docs/MECHANISM_GENERALIZATION.md":
      "| **sae_matched** (top-variance to 1280) | 1280 | 1 | **78.1%** | **0.0%** |"}, []),
    ("selecting for discriminative power does not rescue beta-lactamase either",
     feature_selection_control,
     lambda v: (v["variance_bl"] == 0.0 and v["discrim_bl"] == 0.0
                and v["discrim_mean"] < v["variance_mean"]),
     {"docs/MECHANISM_GENERALIZATION.md": "| top discriminative | 68.4% | **0.0%** |"}, []),
    ("published LOMO reproduces at 5 seeds, and beta-lactamase alone is seed-fragile",
     lomo_seed_stability,
     lambda v: (v["reproduced_exactly"] == v["n_classes"] == 9
                and not v["bl_inside_ci"] and v["outside_ci"] == ["beta_lactamase"]
                and v["bl_30seed"] < v["bl_published"] and v["bl_zero_seeds"] == 7
                and v["bl_ci"][1] < v["bl_published"]
                and v["others_max_shift_pts"] <= 1.4),
     {"docs/MECHANISM_GENERALIZATION.md": "**[11.2, 20.2]**"}, []),
    ("target host is legible within class and not across classes",
     target_host_class_holdout,
     lambda v: (abs(v["pooled_auroc"] - v["published_03s"]) < 0.01
                and v["balanced_class_acc"] == 0.5
                and v["animal_group"] == 1.0 and v["nonanimal_group"] == 0.0
                and not v["perm_usable"] and v["perm_p95"] == 1.0
                and 0.7 < v["posthoc_class_auroc"] < 0.85 and v["posthoc_p"] > 0.05
                and v["animal_below_top_nonanimal"] == 2
                and v["within_class_auroc"] == 0.0
                and v["n_nonanimal_classes"] == 2 and v["producer_dominant"] == 74),
     {"docs/MECHANISM_GENERALIZATION.md":
      "| leave-one-mechanism-class-out | does target host reach an unseen class | "
      "**balanced class accuracy 0.500** |"}, []),
    ("at 30 seeds ESM-C 600M alone clears alignment, and three 5-seed comparisons do not",
     seed_stability_all_arms,
     lambda v: (v["arms_checked"] == 14
                and v["clears_alignment"] == ["_esmc_600M"] and not v["overlapping"]
                and v["esmc600_ci_lo"] > v["alignment"]
                and v["esm2_3B_ci_hi"] < v["alignment"]
                and v["esmc6B_vs_300M_overlap"] and not v["esmc600_vs_6B_overlap"]
                and 3.5 < v["ratio_600M_over_6B"] < 4.5
                and v["cls_vs_mean_overlap"] and v["ladder_rho"] > 0.5
                and v["n_published_outside_ci"] == 5),
     {"docs/MECHANISM_GENERALIZATION.md":
      "| **6B** | **4.3%** | **0.0%** | **12.4% [7.4, 17.4]** |"}, []),
    ("v3 has a second unreachable class and margin locates both", v3_second_failure,
     lambda v: (v["panel_classes"] == 12
                and v["phage_95"] < v["beta_95"] < 0.25
                and v["bacteriocin_95"] > 0.8 and v["cry_95"] > 0.8 and v["rip_95"] > 0.9
                and v["margin_rho"] > 0.85 and v["margin_p"] < 0.01
                and v["margin_locates"] and not v["nn_pos_locates"]
                and not v["nn_neg_locates"] and v["beats_parts"]
                and v["size_rho"] < 0 and v["size_p"] > 0.5
                and v["beta_margin"] < 0 and v["phage_margin"] < 0
                and v["third_lowest_recovery"] > 0.5
                and v["v2_margin_rho"] > 0.9 and v["v2_hit"]),
     {"docs/MECHANISM_GENERALIZATION.md":
      "| **margin** = nearest other-class positive minus nearest negative | "
      "**+0.894** | **0.00015** | **yes** |",
      "docs/DETECTOR_EVALUATION_SUMMARY.md": '- It tracks recovery at Spearman **+0.894**, permutation *p* = 0.0001.'}, []),
    ("v3 panel shape, and target host survives class holdout on it",
     v3_panel_and_target_host,
     lambda v: (v["positives"] == 149 and v["negatives"] == 296
                and v["eligible_classes"] == 11 and v["nonanimal_classes"] == 4
                and v["v3_balanced"] > 0.75 and v["v3_p"] < 0.05 and v["v3_usable"]
                and v["v3_p95"] < 0.75 and v["verdict_supported"]
                and v["v2_balanced"] == 0.5 and not v["v2_usable"]),
     {"docs/MECHANISM_GENERALIZATION.md":
      "**v3 is 149 positives against 296 negatives**"}, []),
    ("the negatives have no test set, and the estimator misses nominal where conformal holds",
     negative_test_set_audit,
     lambda v: (v["seeds_panel_a"] == 300 and v["n_sizes"] == 6
                and v["published_split"]["test"] == 0
                and 0.085 < v["fp_out_at_m20_650M"] < 0.088
                and 0.087 < v["fp_out_at_m20_35M"] < 0.089
                and v["fp_out_at_m20_650M"] > 1.7 * 0.05
                and v["worst_split_650M"] > 0.30
                # the estimator is predictable: inside the order-statistic bracket everywhere
                and v["in_bracket_650M"] == 6 and v["in_bracket_35M"] == 6
                # and the fix works: the conformal guarantee holds everywhere
                and v["conformal_held_650M"] == 6 and v["conformal_held_35M"] == 6
                and v["sizes_where_nominal_unreachable"] == 4
                and abs(v["published_m118_bracket"][0] - 0.0504) < 0.001
                # the sensitivity is concentrated in the low-recovery classes
                and v["ceiling_classes_do_not_move"]
                and v["swing_control"] > 12 and v["swing_phage"] > 8 and v["swing_beta"] > 5),
     {"docs/MECHANISM_GENERALIZATION.md":
      "**`np.quantile(s, 0.95)` cannot deliver 5% out of sample at these sample sizes, and not because of\nnoise.**",
      # The criteria document scores this as its worst failure, so it has to carry the split itself.
      # An earlier draft of that document scored criterion 1 a pass by reporting only the resolution
      # ceiling and omitting that there is no test set, which is the drift this pin exists to stop.
      "docs/DETECTOR_CRITERIA.md": "| **test** | **0** |"},
     []),
    ("the one predictor of the per-class sign fails both its multiplicity check and a second arm",
     what_predicts_the_response,
     lambda v: (v["n_classes"] == 12 and v["seeds"] == 30 and v["perms"] == 20000
                and v["pool_used_clean"] == 8258 and v["pool_used_kept"] == 8259
                and v["dropped"] == ["Q8X739"]
                and -0.61 < v["pmn_rho"] < -0.59 and 0.039 < v["pmn_p"] < 0.041
                and -0.78 < v["pmn_partial"] < -0.76
                and v["n_tests"] == 10 and abs(v["bonferroni"] - 0.005) < 1e-9
                # 🔴 the finding fails correction and the contaminated run passes it
                and v["clean_fails_bonferroni"] and v["kept_passes_bonferroni"]
                and v["pmn_partial_p"] > v["kept_partial_p"]
                and v["nn_pool_alone_null"] and v["margin_null"] and v["baseline_null"]
                and 18 < v["phage_pts"] < 19 and 13 < v["cdi_pts"] < 14 and v["cdi_n"] == 4
                and -15 < v["beta_pts"] < -14 and -38 < v["rip_pts"] < -37
                and v["n_gainers_over_10"] == 2 and v["n_losers_over_10"] == 2
                # 🔴 the replication, which is why the section's verdict is "not supported"
                and v["arms_tested"] == 2 and v["arm2_n_classes"] == 12
                and -0.35 < v["arm2_pmn_rho"] < -0.33
                and v["arm2_pmn_p"] > 0.2 and v["arm2_pmn_partial_p"] > 0.2
                and v["pmn_sign_agrees"] and not v["pmn_replicates"]
                and v["arm2_magnitude_roughly_halves"]
                and v["margin_null_both_arms"]),
     {"docs/MECHANISM_GENERALIZATION.md":
      "the count is **1 arm of 2**, against margin\u0027s 5 of 5 in \u00a710.6.1"}, []),
    ("the pool holds no homolog of any mechanism class and exactly one of the labelled control",
     pool_homology_against_panel,
     lambda v: (v["pool_n"] == 8259 and v["positives_screened"] == 149
                and v["alignments"] == 1230591
                and v["alignments"] == v["pool_n"] * v["positives_screened"]
                and abs(v["threshold"] - 0.30) < 1e-9 and v["n_classes"] == 16
                and v["mechanism_classes_clean"] and v["reported_classes_clean"]
                and v["n_above"] == 1 and v["h2"]
                and v["violation_pool_acc"] == "Q8X739"
                and v["violation_positive"] == "D0ZV89"
                and v["violation_class"] == "virulence_associated_non_toxin"
                and 0.87 < v["violation_sim"] < 0.872
                and 0.11 < v["beta_max"] < 0.12 and 0.10 < v["phage_max"] < 0.11
                and 0.17 < v["t3ss_max"] < 0.18
                and v["t3ss_max"] < 0.6 * v["threshold"]
                # the panel's own negatives, assembled under the same absence of a screen, land
                # entirely below the positives' admission threshold, which is what makes 0.871 an
                # outlier rather than the convention
                and v["panel_neg_n"] == 296 and v["panel_neg_above"] == 0
                and abs(v["panel_neg_max"] - 0.282) < 0.001
                and v["panel_neg_max_acc"] == "Q06320"
                and v["panel_neg_max_class"] == "phage_peptidoglycan_hydrolase"
                and 3.0 < v["pool_over_panel"] < 3.1
                and v["panel_top5"][:4] == [("Q06320", 0.282), ("Q6HAY0", 0.17),
                                            ("P36548", 0.159), ("Q6HAX7", 0.153)]),
     {"docs/MECHANISM_GENERALIZATION.md":
      "**0.871** against `D0ZV89` **PHOQ_SALT1**, the *Salmonella* PhoQ that is a **positive** in the "
      "labelled\nvirulence control"}, []),
    ("at a fixed false-positive budget the pool repairs the phage class and harms beta-lactamase",
     negative_set_at_fixed_budget,
     lambda v: (v["seeds"] == 30 and v["self_tests"] == "all pass"
                and v["residual_phage"] > 25 and v["residual_beta"] > 35
                and v["boundary_fp_constant"] and abs(v["boundary_fp"] - 6 / 118) < 1e-9
                and 23 < v["phage_best_gain_pts"] < 24 and v["phage_monotone_to_4000"]
                and -15 < v["beta_change_pts"] < -13
                and -39 < v["rip_change_pts"] < -37
                and 0.10 < v["both_fp_min"] and v["both_fp_max"] < 0.15
                and v["beta_rethresholded"] < v["beta_baseline"]
                and v["beta_verdict"] == "ARTEFACT" and v["phage_verdict"] == "SURVIVES"
                and v["split_verdict"]),
     {"docs/MECHANISM_GENERALIZATION.md":
      "**At a fixed 5.1% false-positive rate on matched negatives, the pool helps exactly one class.**"},
     []),
    ("name redundancy removes beta-lactamase's excess on the canonical arm, not on both",
     pool_contamination_changes_one_class,
     lambda v: (v["random_split"] == "random" and v["nd_split"] == "name-disjoint"
                and v["nd_name_groups"] == 3550
                and 23 < v["phage_net_random"] < 24 and 20 < v["phage_net_nd"] < 21
                and 3 < v["beta_net_random"] < 4 and abs(v["beta_net_nd"]) < 1e-9
                and v["beta_nd_best_K"] == 0 and v["beta_nd_all_doses_negative"]
                and v["beta_nd_worst_pts"] < -30
                and abs(v["rip_net_nd"]) < 1e-9 and v["nd_verdict_split"]
                # the second arm must NOT reproduce the zero, and must be off-budget instead
                and v["arm2"] == "esm2_35M" and v["arm2_split"] == "name-disjoint"
                and not v["zero_replicates"]
                and not v["arm2_beta_all_doses_negative"]
                and v["arm2_beta_best_K"] == 500
                and v["arm2_beta_net_nd"] > v["arm2_beta_floor_pts"]
                and v["arm2_beta_verdict"] == "OFF-BUDGET"
                and v["arm2_beta_fp_ratio"] > 2.0
                and v["arm2_beta_recovery_K0"] < 0.05),
     {"docs/MECHANISM_GENERALIZATION.md":
      "Beta-lactamase's +3.1 was the contamination.",
      # the arm qualifier itself, on both public surfaces that carry the dose curve
      "docs/DETECTOR_CRITERIA.md":
      "On esm2_35M the same name-disjoint run gives **+0.0, +2.9, -9.3, -25.2,"}, []),
    ("on v3 the arms genuinely separate at 30 seeds, and two 5-seed ties dissolve",
     v3_arm_seed_stability,
     # 🔴 2026-09-24: eight arms. The three new ones are the whole finding of § 10.6.2, so what is
     # asserted changes shape: the canonical arm still separates from every other arm on
     # beta-lactamase, and on phage it does NOT, because ESM-C 600M lands on top of it — that
     # single overlap is the dissociation, and it is pinned by name rather than by a count.
     # 🔴 2026-09-27: nine arms. prott5_xl joins esmc_600M in overlapping the canonical arm on
     # phage while being disjoint from it on beta-lactamase, and BOTH of its published 5-seed
     # figures sit outside their own 30-seed intervals, which is the fourth and fifth time that
     # has happened in this project.
     # 🔴 2026-09-27: ten arms. saprot_650M joins esmc_600M and prott5_xl in overlapping the
     # canonical arm on phage -- and for SaProt that overlap is uninterpretable anyway, because
     # none of the 32 phage members has a structure for it to read. § 10.6.2 names the class
     # unreportable for this arm; the number is pinned so the exclusion cannot be forgotten.
     lambda v: (v["seeds"] == 30 and v["n_arms"] == 10
                and v["beta_disjoint_pairs"] == 30 and v["phage_disjoint_pairs"] == 31
                and not v["beta_canonical_separates_from_all"]
                and v["beta_overlapping_canonical"] == ["saprot_650M"]
                and not v["phage_canonical_separates_from_all"]
                and v["phage_overlapping_canonical"] == ["esmc_600M", "prott5_xl",
                                                          "saprot_650M"]
                and v["beta_outside_ci"] == ["esm2_150M", "esm2_35M", "esm2_3B", "esmc_300M",
                                             "prott5_xl", "saprot_650M"]
                and v["phage_outside_ci"] == ["esm2_3B", "esm2_8M", "esmc_300M", "esmc_600M",
                                              "prott5_xl"]
                and not v["beta_top_changes"] and v["phage_top_changes"]
                and v["beta_4pct_tie_dissolved"] and v["a1"]
                and abs(v["beta_canonical_30s"] - 0.212) < 0.002
                and abs(v["beta_3B_30s"] - 0.093) < 0.002
                and abs(v["beta_150M_30s"] - 0.019) < 0.002
                and abs(v["phage_3B_30s"] - 0.041) < 0.002
                and v["phage_8M_30s"] > 6 * v["phage_3B_30s"]),
     {"docs/MECHANISM_GENERALIZATION.md":
      "**21 of the 28 arm pairs are\ndisjoint on beta-lactamase and 21 of 28 on the phage class**",
      "docs/DETECTOR_EVALUATION_SUMMARY.md": '| **ESM-C 600M** | **40.5% [36.5, 44.5]** | 12.1% [9.4, 14.8] |'},
     # the sentence this replaced survived the eight-arm rerun because only its numbers changed and
     # the pin quoted the old ones; forbidding the old count stops that recurring
     ["9 of the 10 arm pairs", "separates from\nall four others"]),
    ("the benign pool closes §10.8's gap to 30x by count and 70x by distinct name",
     pool_and_calibration_gap,
     lambda v: (v["pool_n"] == 8259 and v["distinct_names"] == 3550
                and abs(v["name_factor"] - 2.33) < 0.01 and v["most_repeated"] == 389
                and abs(v["eff_homology"] - 5203) < 1 and abs(v["homology_keep"] - 0.63) < 0.005
                and v["need_9999"] == 250003
                and 840 < v["gap_panel"] < 850
                and 30 <= v["gap_raw"] < 31 and 70 <= v["gap_name"] < 71
                and v["name_count_is_lower"]
                and abs(v["ceiling_name"] - 0.99930) < 1e-5
                and abs(v["ceiling_homology"] - 0.99952) < 1e-5),
     {"docs/MECHANISM_GENERALIZATION.md":
      "shortfall to **30\u00d7** by raw count and **70\u00d7** by distinct protein name"}, []),
    ("the provenance control holds on v3 and is weaker and noisier there",
     v3_provenance_control,
     lambda v: (abs(v["v2_auroc"] - 0.818) < 0.002 and abs(v["v2_agreement"] - 0.534) < 0.002
                and abs(v["v3_auroc"] - 0.794) < 0.002 and abs(v["v3_agreement"] - 0.436) < 0.002
                and v["v3_above_chance_by_2sd"] and v["v3_noisier"]),
     {"docs/MECHANISM_GENERALIZATION.md": "**AUROC 0.794 \u00b1 0.062** on v3"}, []),
    ("margin ranks an unseen mechanism right and mis-states its miss rate",
     margin_predicts_new_classes,
     lambda v: (v["p1_hit"] and v["lowest"] == "phage_peptidoglycan_hydrolase"
                and v["phage_error_pts"] > 30
                and v["beats_baseline"] and v["loocv_mae"] < 0.5 * v["baseline_mae"]
                and v["loocv_rho"] > 0.8 and v["loocv_p"] < 0.01
                and v["all_errors_optimistic"]
                and v["worst_calibrated"] == "phage_peptidoglycan_hydrolase"
                and v["rho_pre_expansion"] > 0.85 and v["not_calibrated"]),
     {"docs/MECHANISM_GENERALIZATION.md":
      "| **phage_peptidoglycan_hydrolase** | **−0.0055** | **44%** | **10%** | **+34** |"}, []),
    ("the class-level margin mechanism holds across all 14 representations",
     margin_across_arms,
     lambda v: (v["n_arms"] == 14 and v["negative_margin_arms"] == 14
                and v["smoke_arm_excluded"]
                and v["locate_failure"] == 12 and v["significant"] == 13
                and v["misses"] == ["esm2_650M_cls", "saprot_650M"]
                and v["max_locates"] and not v["cls_locates"]
                and v["weakest_rho"] > 0.5 and 0.05 < v["weakest_p"] < 0.06
                and v["duplicate_agrees"] and v["supported"]),
     {"docs/MECHANISM_GENERALIZATION.md":
      "| **esm2_650M mean** (canonical) | 1280 | **+0.940** | 0.0004 | beta_lactamase |"}, []),
    ("benign proximity is causal and closes only a tenth of the gap", margin_causal_test,
     lambda v: (sorted(v["v3_failures"]) == ["beta_lactamase",
                                             "phage_peptidoglycan_hydrolase"]
                and v["v2_failures"] == ["beta_lactamase"]
                and v["beta_ci_lo_v3"] > 0 and v["phage_ci_lo_v3"] > 0
                and v["beta_ci_lo_v2"] > 0
                and 0.05 < v["beta_frac_v3"] < 0.15
                and 0.05 < v["phage_frac_v3"] < 0.15
                and 0.05 < v["beta_frac_v2"] < 0.15
                and v["comparison_attr"] < min(v["beta_attr_v3"], v["phage_attr_v3"])
                and 0.7 < v["beta_pctile_v3"] < 0.95
                and v["beta_seeds_beating_p95"] < 15
                and v["contributing_not_the_cause"]),
     {"docs/MECHANISM_GENERALIZATION.md":
      "| v3 | **beta_lactamase** | 21.2% | 30.5% | **+8.8** | 79th | **11%** of the gap |"}, []),
    ("the margin triage survives a tightening budget the panel cannot calibrate",
     deployment_operating_points,
     lambda v: (v["specs"] == 4 and v["strictest"] == 0.99
                and 0.99 < v["ceiling"] < 0.992
                and v["survives"] and v["all_specs_significant"]
                and v["rho_strict"] > 0.7 and v["rho_strict"] < v["rho_95"]
                and v["p_strict"] < 0.01
                and 0.5 < v["tpr_strict"] < 0.6 and v["fpr_strict"] < 0.02
                and 150 < v["alerts_1in1000"] < 200
                and v["precision_1in1000"] < 0.05 and v["missed_1in1000"] > 4
                and v["panel_for_9999"] > 800 * v["have_negatives"]
                and v["phage_at_strict"] < 0.05 and v["clostridial_at_strict"] == 1.0
                and v["cdi_at_strict"] > 0.6),
     {"docs/MECHANISM_GENERALIZATION.md":
      "| **0.99** | **55%** | **1.7%** | **223 (25% real)** | **175 (3% real)** | "
      "**170 (0% real)** |"}, []),
    ("the benign-proximity effect scales with dose and still leaves most of the gap",
     margin_dose_response,
     lambda v: (v["k_grid"] == [5, 10, 20, 40, 80]
                and v["failures"] == ["beta_lactamase", "phage_peptidoglycan_hydrolase"]
                and v["both_monotone"] and v["scaling"]
                and v["comp_peaks_mid"] and v["comp_top_ci_includes_zero"]
                and v["fail_ci_lo_positive_at_top"]
                and v["beta_top_pts"] > 18 and v["phage_top_pts"] > 15
                and 0.2 < v["beta_frac_closed"] < 0.35
                and 0.15 < v["phage_frac_closed"] < 0.25
                and 0.5 < v["train_neg_kept_at_top"] < 0.6),
     {"docs/MECHANISM_GENERALIZATION.md":
      "| **K=80** | **+16.6** [+13.6, +19.6] | **+20.7** [+15.6, +25.8] | "
      "**+2.8** [−0.7, +6.3] |"}, []),
    ("on v3 the across-arms test is stricter and comes back partial across ten arms",
     v3_margin_across_arms,
     # Five arms now, not three. The three-arm version of this claim also pinned that
    # beta-lactamase recovery falls and the phage class rises monotonically with capacity;
    # 150M and 3B break both orderings, so those two conditions are gone and the section
    # says so. What is pinned is what survived: every arm negative on both failures, every
    # arm significant, beta-lactamase lowest everywhere, and the bottom-two holding in a
    # minority with the labelled control as the sole displacer.
    # 🔴 2026-09-24: eight arms, not five. src/02e could not see v3, so this section had been
    # scoped to one model family by a loader rather than by a decision. esm3_1_4B is the one arm
    # with a POSITIVE phage margin, which is why all_negative is 7 of 8, and it is also the arm
    # that recovers phage best: margin and recovery agree on the exception. See § 10.6.2.
    # 🔴 2026-09-27: nine arms. prott5_xl is the first arm outside the EvolutionaryScale lineage
    # to see v3, and it reaches NEITHER failing class, so the § 10.6.2 dissociation is
    # model-specific rather than lineage-specific. It also carries the weakest rho of the nine,
    # +0.739, which is margin's class ordering at its weakest on the out-of-lineage arm.
    # 🔴 2026-09-27: ten arms. saprot_650M brings the count of arms locating the failure pair to
    # 3 of 10, and its own is STRUCTURE-CONFOUNDED: AlphaFold DB has no model for any of the 32
    # phage members, so they are the only proteins that arm sees sequence-only, and its phage
    # margin is -0.0368, the most negative of all ten. That is what a systematically different
    # input does to one class, not what geometry does. Counted here and discounted in § 10.6.2.
    lambda v: (v["n_arms"] == 10 and v["k"] == 2
                and abs(v["chance"] - 1 / 66) < 1e-9
                and v["failures"] == ["beta_lactamase", "phage_peptidoglycan_hydrolase"]
                and v["all_negative"] == 9 and v["significant"] == 10
                and abs(v["min_rho"] - 0.739) < 0.002
                # 🔴 `min_rho > 0.75` until 2026-09-27, written when the weakest of five ESM-2
                # arms was +0.796. ProtT5 is +0.739, so the floor would have failed for the
                # correct reason stated wrongly: the weakest arm is pinned by value above, and
                # `significant == 9` is what actually carries "every arm's ordering holds".
                # beta-lactamase is the lowest-margin class in 9 of 10, and the exception is the
                # structure-confounded arm rather than a counterexample
                and not v["beta_lowest_everywhere"]
                and len(v["beta_lowest_arms"]) == 9
                and "saprot_650M" not in v["beta_lowest_arms"]
                and v["locate"] == 3
                and sorted(v["which_locates"]) == ["canonical", "esm2_3B", "saprot_650M"]
                and v["displacer"] == ["virulence_associated_non_toxin"]
                and v["beta_not_monotone_in_capacity"]),
     {"docs/MECHANISM_GENERALIZATION.md":
      "**AlphaFold DB has no model for any of the 32 phage peptidoglycan hydrolases.**",
      # 🔴 2026-09-24. The standalone summary first quoted § 10.6's fourteen-arm generality and not
      # this section's PARTIAL, so a reader would have taken the bottom-two identification as
      # representation-general, which this section explicitly denies. The qualifier is pinned here
      # rather than on the § 10.6 claim because this is the claim that knows it is 2 of 5.
      "docs/DETECTOR_EVALUATION_SUMMARY.md": '  class in all nine — but the exact bottom-two holds in only **2 of 9**, and one arm, ESM-3 1.4B,'},
     # the five-arm spellings, forbidden so the eight-arm rerun cannot be half-propagated again
     ["2 of 5**", "only **5 arms** are embedded", "2 of 8**", "**8 arms** now embedded"]),
    ("reference set poisoning, both arms", reference_set_poisoning,
     # The assertion deliberately requires the 35M arm to be LOUD (s_p2_near == 0), so a future run
     # that made both arms quiet fails here instead of silently strengthening a claim the document
     # scopes to one arm. Same for rip: canonical must support P1 and 35M must not, because the
     # document says that inversion is the reason cry leads instead.
     lambda v: (v["c_n"] == 13 and v["s_n"] == 13
                and v["c_p1_sel"] == 5 and v["s_p1_sel"] == 5
                and v["c_p1_near"] == 4 and v["s_p1_near"] == 2
                and v["c_p2_near"] == 8 and v["s_p2_near"] == 0
                and v["c_p2_sel"] == 3 and v["s_p2_sel"] == 1
                and v["agree_suppressible"] == 12 and v["n_both"] == 13
                and v["c_rip_p1_sel"] and not v["s_rip_p1_sel"]
                and v["c_rip_sel_pts"] < -50 and v["s_rip_sel_pts"] > -20
                and v["c_cry_sel"] < v["c_cry_panel"] and v["s_cry_sel"] < v["s_cry_panel"]
                and v["s_cry_fp_sel"] < v["s_cry_fp_panel"]
                and abs(v["s_cry_panel"] - 0.635) < 0.005
                and abs(v["s_cry_sel"] - 0.167) < 0.005
                and abs(v["s_cry_fp_panel"] - 0.0830) < 0.0005
                and abs(v["s_cry_fp_sel"] - 0.0587) < 0.0005),
     {"docs/DETECTOR_CRITERIA.md":
      "| esm2_35M | 63.5% | **16.7%** | -12.1pt | 8.30% | **5.87%** |",
      "docs/DETECTOR_EVALUATION_SUMMARY.md": '| esm2_35M | 63.5% | **16.7%** | 8.30% | **5.87%** |'},
     # the in-sample collateral figure an earlier draft of criterion 18 nearly published
     ["Every other hazard class is untouched"]),

    ("poisoning neighbourhood overlap", poisoning_neighbourhood_overlap,
     lambda v: (v["n_classes"] == 13 and v["slots"] == 6500
                and abs(v["mean_jaccard"] - 0.324) < 0.002
                and abs(v["max_jaccard"] - 0.883) < 0.002
                and v["max_pair"] == ["adp_ribosyl_ab_toxin", "rip_rrna_glycosidase"]
                and v["union"] == 1999),
     {"docs/DETECTOR_CRITERIA.md":
      "and all thirteen together span only **1,999 distinct pool proteins out of 6,500"}, []),
    ("detector criteria scorecard is internally consistent", criteria_scorecard_consistency,
     lambda v: (v["headings"] == 18 and v["rows"] == 18
                and v["heads_are_1_to_n"] and v["rows_match_heads"]
                # 🔴 2026-09-27: criterion 1 moved fail -> partial when src/60 froze a standing
                # test partition. It is a PARTIAL and not a pass because the threshold is still set
                # on 118 calibration points: the partition improves the resolution of the rate that
                # is measured, not of the threshold that is set. Remaining fails are per-class
                # performance and the floor-only preregistration.
                and v["fails"] == [3, 12] and v["partials"] == [1, 4, 5, 18]
                and v["mixed"] == [7]
                and v["doc_says_18"] and v["doc_tally"]
                and v["readme_says_18"] and v["readme_tally"]),
     {"README.md": "states eighteen criteria for evaluating a hazard detector"},
     # the counts either document carried while the scorecard said otherwise. "five partials" was
     # briefly written into both on 2026-09-21 and the table never had five, so it is forbidden too.
     # 🔴 `(two fails, four partials)` was forbidden here because both documents once carried it
     # while the table did not. On 2026-09-27 it became TRUE -- criterion 1 moved to partial -- so
     # the forbid is removed rather than worked around. A forbid on a count is only ever a forbid
     # on a count the table does not support, and it has to be retired when the table changes.
     ["states seventeen criteria", "five partials",
      "Three fails (1, 3, 12), three partials (4, 5, 18)"]),
    ("tier 3 mutation coordinates verified against UniProt", tier3_coordinates,
     lambda v: (v["n"] == 4 and v["all_seq_match"] and v["all_identities_ok"]
                and v["offsets"] == {"P00588": 32, "P02879": 35, "P04977": 34, "P0DPI1": 1}
                # BoNT-A: precursor 224, not the literature's light-chain 223
                and v["bont_precursor"] == 224 and v["bont_residue"] == "E"
                and v["bont_exact_sub_annotated"]
                and v["pertussis_precursor"] == [43, 163]
                # UniProt annotates E163D, not the vaccine mutant's E163G
                and not v["pertussis_129_sub_annotated"]
                and v["crm197_precursor"] == 84 and not v["crm197_mutagenesis"]
                # the ricin row fails its own stated check: no annotation at the active site
                and v["ricin_precursor"] == 212 and not v["ricin_mutagenesis"]
                and v["ricin_lof_alternatives"] == [[110, 75, "D"]]
                and v["tier2_lof_pairs"] == ["P04977", "P0DPI1"]
                # one constraint cannot pin an offset; only the two-substitution pair is unique
                and v["unique_offset"] == ["P04977"]),
     {"docs/MUTATION_EXTENSION_PREREGISTRATION.md":
      "| BoNT-A light chain E->Q | 223 | +1 | **224** | E | **E->K,Q**, "
      "\"Light chain no longer cleaves SNAP25\" | yes |"},
     # the light-chain number, which is correct in its own frame and wrong for this pipeline
     ["precursor 223", "E223 in precursor"]),
    ("SEB's published offset is ruled out and the headline moves", flagged_entries_vs_uniprot,
     # Requires the flip. An edit that rescored SEB back into the thirteen, or that quietly dropped
     # the signal-peptide argument, fails here rather than passing quietly.
     lambda v: (v["seb_signal_end"] == 27
                and v["seb_offset0_in_signal"] == [23, 25] and v["seb_offset0_disproven"]
                and v["seb_disulfide_mature"] == [[93, 113]] and v["seb_res_at_mature_93"] == "C"
                and abs(v["seb_ratio_0"] - 0.9556) < 0.001
                and abs(v["seb_ratio_27"] - 1.0417) < 0.001
                and v["seb_flips"] and not v["seb_uniprot_has_sites"]
                and v["headline_published"] == "13/15"
                and abs(v["p_published"] - 0.0037) < 0.0002
                and v["headline_seb_at_27"] == "12/15"
                and abs(v["p_seb_at_27"] - 0.0176) < 0.0002
                and v["headline_seb_dropped"] == "12/14"
                and abs(v["p_seb_dropped"] - 0.0065) < 0.0002
                # ExoS: hypothesis refuted, four UniProt sites omitted, verdict unaffected
                and v["exos_domain"] == [243, 429]
                and not v["exos_domain_hypothesis_holds"]
                and v["exos_agrees"] == [146, 381]
                and v["exos_omits"] == [186, 187, 319, 343]
                and v["exos_all_below_1"] and v["exos_n_variants"] == 4),
     {"docs/EVALUATION_REPORT.md":
      "| 🔴 **SEB scored in the only coordinate frame UniProt permits** | **12/15** | **0.0176** |",
      "huggingface/README.md":
      "At +27 the ratio goes **0.9556 to 1.0417 and crosses 1.0**, making the headline "
      "**12/15 at p = 0.0176**.",
      # README stated the headline twice and carried no caveat at all until 2026-09-22, which is the
      # surface most likely to be read and quoted. Now it carries the applied exclusion instead of
      # the caveat, pinned so the reason for the change cannot be dropped along with the old number.
      "README.md":
      "The only admissible frame, +27, places just 1 of 9 annotated residue identities correctly"},
     # the claim that src/52 made false, on both surfaces that carried it
     ["SEB gets no equivalent test"]),
    ("no functional position sits in a cleaved signal peptide, panel-wide", signal_peptide_sweep,
     lambda v: (v["n_entries"] == 16 and v["n_clean"] == 15
                and v["in_signal"] == ["P01552"]
                and v["in_propeptide"] == [] and v["past_end"] == []
                and v["uncheckable"] == [] and v["artifact_agrees"]
                and v["accs_with_signal_peptide"] == 8
                # the offsets entry sixteen installed, which the sweep must apply
                and v["offsets_applied"] == {"P00588": 32, "P00648": 47, "P02879": 35}
                and v["repaired_entries_clean"] and v["vacA_empty"]),
     {"docs/EVALUATION_REPORT.md":
      "Result: **15 of 16 entries are clean and the single hit is P01552.**"}, []),
    ("the annotation set against UniProt active sites", annotation_provenance,
     lambda v: (v["primary"] == "Active site" and v["n_entries"] == 16 and v["n_comparable"] == 11
                and v["annotated"] == 53 and v["confirmed_active"] == 17
                and abs(v["frac_active"] - 17 / 53) < 1e-9
                # the whole point: two omissions panel-wide, both on the known-bad entry
                and v["omitted_active"] == 2
                and v["who_omits"] == {"Q51451": [319, 343]}
                and v["extra_vs_active"] == 36 and v["extra_explained"] == 17
                and v["grounded"] == 34 and abs(v["frac_grounded"] - 34 / 53) < 1e-9
                and v["exact_match"] == ["O34208"] and v["zero_confirmed"] == []
                and v["no_active_site"] == ["P01552", "P04419", "P0DF97", "P13423", "P55981"]
                # preregistered underpowered null; must stay a null
                and v["p1_n"] == 11 and v["p1_rho"] < 0 and not v["p1_significant"]
                and abs(v["p1_p"] - 0.5143) < 0.001),
     {"docs/EVALUATION_REPORT.md":
      "| **UniProt active sites omitted from the curation, whole panel** | **2** |"},
     # The pooled figure the first version of this audit produced. Only the ASSERTION form is
     # forbidden: `docs/EVALUATION_REPORT.md` names the number in a paragraph that labels it wrong,
     # which is this repository's house style for corrections, and a forbid that banned the digits
     # outright would forbid explaining the error. Forbidding the phrasing a live claim would use
     # keeps the guard while leaving the narrative alone.
     ["64 UniProt sites omitted", "64 UniProt-annotated sites"]),
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
            # 🔴 A non-string pin used to crash the whole audit with a bare TypeError from `s not in
            # text`, which is worse than a failed claim: the gate dies instead of reporting. It
            # happened on 2026-09-20 from a `{"huggingface/README.md": None}` entry added by mistake,
            # and a grep of the output for the OK line hid the crash. Name the defect instead.
            if not isinstance(s, str) or not s:
                print(f"     XX pin for {doc} is not a non-empty string: {s!r}")
                failures.append(f"{label}: bad pin for {doc}")
            elif doc not in docs:
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
