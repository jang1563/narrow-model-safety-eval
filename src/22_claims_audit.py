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
          "docs/DETECTOR_CRITERIA.md"]


def j(p):
    f = R / p
    return json.load(open(f)) if f.exists() else None


# ---- recomputation, from artifacts only ---------------------------------------

def _sign_p(n, k):
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def fspe_protein_level():
    d = j("fspe_results.json")
    r = np.array([x["fspe_ratio"] for x in d["per_protein"]], float)
    n, k = len(r), int((r < 1).sum())
    return {"n": n, "below_1": k, "sign_p": _sign_p(n, k)}


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
    d = j("v3/external_test_partition_esm2_35M.json")
    if not d:
        return None
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
            "single_arm_provisional": d["single_arm_provisional"]}


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
    """Cross-model FSPE sign flips, counted only on the rows where the comparison is still legitimate.

    The 2026-05-22 numbering fix re-ran ESM-2 alone. ESM-3 and SaProt still hold values computed
    with mature-chain positions masked on a precursor, so on the three proteins carrying a
    `precursor_offset` this table now compares one corrected column against two stale ones. The
    published "3 of 12" was true of the pre-fix table, and the "4 of 12" the current file would
    produce is a mixed-numbering artifact: neither is a statement anyone can make today. So the
    count is reported on the nine rows where no column moved, with the other three named as
    indeterminate until ESM-3 and SaProt are re-run. The moved set is derived from the FSPE
    snapshots and cross-checked against the annotation offsets, so re-running one model cannot
    quietly shrink the indeterminate set while leaving this claim passing.
    """
    rows = j("mdrp_risk_table.json")["proteins"]
    _, pre, _, noise = _fspe_pre_and_post()
    moved = {r["uniprot_id"] for r in rows
             if abs(r["fspe_esm2"] - pre[r["uniprot_id"]]) > noise}
    cols = ["fspe_esm2", "fspe_esm3", "fspe_saprot"]
    side = lambda v: ">1" if v > 1 else "<1"          # noqa: E731

    def flips(subset):
        n = 0
        for r in rows:
            if r["uniprot_id"] not in subset:
                continue
            av = [v for v in (r.get(c) for c in cols) if v is not None]
            if len(av) >= 2 and len({side(v) for v in av}) > 1:
                n += 1
        return n

    ids = {r["uniprot_id"] for r in rows}
    return {"n_rows": len(rows), "indeterminate": sorted(moved),
            "comparable_rows": len(ids - moved), "comparable_flips": flips(ids - moved),
            "all_rows_flips_do_not_quote": flips(ids),
            "max_comparable_delta": max(abs(r["fspe_esm2"] - pre[r["uniprot_id"]])
                                        for r in rows if r["uniprot_id"] not in moved)}


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
    """§10.9.1: the iso-FP control under both reservation splits. The name-disjoint split is what
    removes beta-lactamase's apparent excess entirely, so both splits are pinned and the direction of
    the contamination bias is pinned with them."""
    r = j("v3/fixed_background_operating_point_esm2_650M.json")
    n = j("v3/fixed_background_operating_point_esm2_650M_namedisjoint.json")
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
            "nd_verdict_split": n["verdict"].startswith("SPLIT")}


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
            "beta_canonical_separates_from_all":
                len(sm["beta_lactamase"]["arms_disjoint_from_canonical"]) == 4,
            "phage_canonical_separates_from_all":
                len(sm["phage_peptidoglycan_hydrolase"]["arms_disjoint_from_canonical"]) == 4,
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

CLAIMS = [
    # 0.018 / 12-of-15 was the pre-2026-05-22 numbering. The tolerance is 1e-4 rather than the old
    # 0.002 because the sign test is exact: with n fixed at 15 the only reachable values near 0.0037
    # are 121/32768 and 576/32768, so a loose window would let the stale figure pass as the new one.
    # The forbid catches what the three positive pins cannot: a document carrying the old and the new
    # figure side by side, and the two public surfaces with no FSPE pin at all (ARCHITECTURE.md,
    # MECHANISM_GENERALIZATION.md). It is scoped to PUBLIC only, so src/46's deliberate quotation of
    # the old headline, and the same quotation in docs/DATA_CORRECTIONS.md, are untouched.
    ("FSPE protein-level sign test", fspe_protein_level,
     lambda v: abs(v["sign_p"] - 0.0037) < 1e-4 and v["below_1"] == 13 and v["n"] == 15,
     {"README.md": "13/15 below 1.0, sign test p = 0.0037",
      "huggingface/README.md": "sign test p = 0.0037",
      "docs/EVALUATION_REPORT.md": "sign test p = 0.0037"},
     ["12/15 below 1.0", "sign test p = 0.018"]),
    # Both flagged entries are counted as successes by the headline, so the leave-out value is part
    # of the claim rather than a footnote to it. 11/13 is exactly 92/8192, so the tolerance is tight.
    ("the FSPE headline's dependence on the two annotation-flagged entries", fspe_flagged_leaveout,
     lambda v: (v["flagged"] == ["P01552", "Q51451"] and v["flagged_all_below_1"]
                and v["full"]["n"] == 15 and v["full"]["below_1"] == 13
                and v["without_both"]["n"] == 13 and v["without_both"]["below_1"] == 11
                and abs(v["without_both"]["sign_p"] - 0.01123) < 1e-4
                and all(s["n"] == 14 and s["below_1"] == 12 for s in v["each"].values())
                and v["direction_survives"] and 2.5 < v["p_inflation"] < 3.5),
     {"docs/EVALUATION_REPORT.md": "11/13 at p = 0.011 with both annotation-flagged entries removed"},
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
     lambda v: v is None or (
         v["calibration_n"] == 118 and v["pool_n"] == 8258 and v["dropped"] == "Q8X739"
         and v["seeds"] == 200 and v["single_arm_provisional"]
         # at m=118 conformal is reachable at BOTH budgets, unlike at m=59
         and v["k"]["0.05"] == 5 and v["k"]["0.01"] == 1 and v["conformal_reachable_at_1pct"]
         and abs(v["guarantee"]["0.05"] - 0.0420) < 1e-3
         and abs(v["guarantee"]["0.01"] - 0.0084) < 1e-3
         # exchangeable control: conformal sits on its guarantee, the quantile estimator does not
         and v["conformal_holds_in_control"] and v["ctrl_conf_05_covers"]
         and v["ctrl_conf_01_covers"]
         and abs(v["ctrl_conf_05"] - v["guarantee"]["0.05"]) < 0.005
         and abs(v["ctrl_conf_01"] - v["guarantee"]["0.01"]) < 0.005
         and v["ctrl_quant_05_exceeds"] and v["ctrl_quant_01_exceeds"]
         # shift costs conformal its guarantee, and redundancy costs more on top
         and v["shift_conf_05_exceeds"] and v["shift_conf_05"] > v["ctrl_conf_05"]
         and v["dedup_conf_05"] > v["shift_conf_05"] and v["dedup_raises_fp"]
         and v["dedup_quant_05"] > 0.09),
     {"docs/DETECTOR_CRITERIA.md":
      "4.23%   what it delivers when the negatives really are exchangeable"}, []),
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
    # 🔴 This claim used to read `flips == 3 and n_rows == 12`, and it kept passing after the
    # numbering fix for the wrong reason: the corrected ESM-2 column moved P00648 from >1 to <1
    # while P02879 stayed >1, so the total happened to stay near 3. The quantity was never
    # recomputable, because two of the three columns are still in the old coordinates. What is
    # pinned now is the subset where the comparison is legitimate, plus the three rows that are not.
    ("Cross-model FSPE flips, on the rows where the models share a numbering", flip_count,
     lambda v: (v["n_rows"] == 12 and v["comparable_rows"] == 9 and v["comparable_flips"] == 2
                and v["indeterminate"] == ["P00588", "P00648", "P02879"]
                and v["max_comparable_delta"] < 1e-5),
     {}, []),
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
     {"docs/MECHANISM_GENERALIZATION.md": "**80 hazardous proteins**"}, []),
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
      "**+0.894** | **0.00015** | **yes** |"}, []),
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
    ("removing the pool's name redundancy removes beta-lactamase's apparent excess entirely",
     pool_contamination_changes_one_class,
     lambda v: (v["random_split"] == "random" and v["nd_split"] == "name-disjoint"
                and v["nd_name_groups"] == 3550
                and 23 < v["phage_net_random"] < 24 and 20 < v["phage_net_nd"] < 21
                and 3 < v["beta_net_random"] < 4 and abs(v["beta_net_nd"]) < 1e-9
                and v["beta_nd_best_K"] == 0 and v["beta_nd_all_doses_negative"]
                and v["beta_nd_worst_pts"] < -30
                and abs(v["rip_net_nd"]) < 1e-9 and v["nd_verdict_split"]),
     {"docs/MECHANISM_GENERALIZATION.md":
      "Beta-lactamase's +3.1 was the contamination."}, []),
    ("on v3 the arms genuinely separate at 30 seeds, and two 5-seed ties dissolve",
     v3_arm_seed_stability,
     lambda v: (v["seeds"] == 30 and v["n_arms"] == 5
                and v["beta_disjoint_pairs"] == 9 and v["phage_disjoint_pairs"] == 8
                and v["beta_canonical_separates_from_all"]
                and v["phage_canonical_separates_from_all"]
                and v["beta_outside_ci"] == ["esm2_150M", "esm2_35M", "esm2_3B"]
                and v["phage_outside_ci"] == ["esm2_3B", "esm2_8M"]
                and not v["beta_top_changes"] and v["phage_top_changes"]
                and v["beta_4pct_tie_dissolved"] and v["a1"]
                and abs(v["beta_canonical_30s"] - 0.212) < 0.002
                and abs(v["beta_3B_30s"] - 0.093) < 0.002
                and abs(v["beta_150M_30s"] - 0.019) < 0.002
                and abs(v["phage_3B_30s"] - 0.041) < 0.002
                and v["phage_8M_30s"] > 6 * v["phage_3B_30s"]),
     {"docs/MECHANISM_GENERALIZATION.md":
      "**9 of the 10 arm pairs are\ndisjoint on beta-lactamase and 8 of 10 on the phage class**"}, []),
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
    ("on v3 the across-arms test is stricter and comes back partial across five arms",
     v3_margin_across_arms,
     # Five arms now, not three. The three-arm version of this claim also pinned that
    # beta-lactamase recovery falls and the phage class rises monotonically with capacity;
    # 150M and 3B break both orderings, so those two conditions are gone and the section
    # says so. What is pinned is what survived: every arm negative on both failures, every
    # arm significant, beta-lactamase lowest everywhere, and the bottom-two holding in a
    # minority with the labelled control as the sole displacer.
    lambda v: (v["n_arms"] == 5 and v["k"] == 2
                and abs(v["chance"] - 1 / 66) < 1e-9
                and v["failures"] == ["beta_lactamase", "phage_peptidoglycan_hydrolase"]
                and v["all_negative"] == 5 and v["significant"] == 5
                and v["beta_lowest_everywhere"] and v["min_rho"] > 0.75
                and v["locate"] == 2
                and sorted(v["which_locates"]) == ["canonical", "esm2_3B"]
                and v["displacer"] == ["virulence_associated_non_toxin"]
                and v["beta_not_monotone_in_capacity"]),
     {"docs/MECHANISM_GENERALIZATION.md":
      "| esm2_3B | 2560 | +0.831 | 0.0006 | −0.0030 | 4.3% → **9.3%** [5.6, 13.0] "
      "| −0.0014 | 6.9% → **4.1%** [2.4, 5.7] | **yes** |"}, []),
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
