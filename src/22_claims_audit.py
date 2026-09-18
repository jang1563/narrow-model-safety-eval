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
          "docs/ARCHITECTURE.md", "docs/MECHANISM_GENERALIZATION.md"]


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
      "**+0.894** | **0.0001** | **yes** |"}, []),
    ("v3 panel shape, and target host survives class holdout on it",
     v3_panel_and_target_host,
     lambda v: (v["positives"] == 149 and v["negatives"] == 296
                and v["eligible_classes"] == 11 and v["nonanimal_classes"] == 4
                and v["v3_balanced"] > 0.75 and v["v3_p"] < 0.05 and v["v3_usable"]
                and v["v3_p95"] < 0.75 and v["verdict_supported"]
                and v["v2_balanced"] == 0.5 and not v["v2_usable"]),
     {"docs/MECHANISM_GENERALIZATION.md":
      "**v3 is 149 positives against 296 negatives**"}, []),
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
