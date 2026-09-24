#!/usr/bin/env python3
"""
55_tier2_mutagenesis_set.py - step 2 of the mutation-axis preregistration: build the tier 2
                              substitution set, logging every exclusion with its reason.

What this is and is not
-----------------------
This is set CONSTRUCTION only. No model is loaded, no score is computed and no hypothesis is
tested here. `docs/MUTATION_EXTENSION_PREREGISTRATION.md` § 7 fixes the order: tier 3 coordinates
first (`src/51`, done 2026-09-21), then this, then the FSPE-M forward pass, then P5's
shuffled-label arm, then the tests. Running any of those before this one finishes would let the
ground-truth set be chosen after seeing a score.

The rule, quoted from § 3 rather than restated
----------------------------------------------
A substitution enters the set only if all four hold:

  1. it is a UniProt `Mutagenesis` feature on a panel member,
  2. the phenotype text is unambiguous in direction — "loss of activity", "abolishes", "no
     detectable activity" are loss; "no effect" is tolerated; a quantitative reduction is loss
     ONLY when the text states a fold change of 100 or more; anything hedged, conditional, or
     describing a change in specificity rather than magnitude is excluded and logged,
  3. the position passes the residue-identity check against the panel FASTA,
  4. its protein counts once toward effective n.

⚠️ The classifier below is deliberately strict and deliberately dumb. It matches the phrases the
preregistration named and excludes everything else, including verbs that a human reader would
probably accept. "Suppresses the toxic activity" is excluded: it is a real reduction with no
stated magnitude, and the rule admits unquantified reductions only through the named phrases.
Loosening it after seeing the count is the exact failure the preregistration exists to prevent,
so the strict pass is the one that runs and every borderline call is in the log for a reader to
disagree with.

The stopping rule, also fixed in advance
----------------------------------------
§ 7 step 2: **if fewer than 12 substitutions survive across at least 4 distinct proteins, P3 is
dropped rather than run underpowered**, and the remaining primary tests drop to five with the
multiplicity threshold recomputed to 0.05 / 5 = 0.01. This script prints that verdict; it does
not decide it.

Usage:
    python src/55_tier2_mutagenesis_set.py
    python src/55_tier2_mutagenesis_set.py --selftest
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "uniprot_cache"
OUT = ROOT / "results" / "tier2_mutagenesis_set.json"

# --- the phrase lists, from § 3 rule 2. -------------------------------------------------------
#
# 🔴 Corrected 2026-09-24, before the second run, with the first run's numbers already written
# into the preregistration's amendment log so the correction cannot be read as reaching for the
# 12-substitution threshold. The first version required `loss of ... activity` and therefore
# threw away "Loss of toxicity.", "Loss of ability to bind to LF and completely non-toxic." and
# 19 other unambiguous losses as "no phrase matched", while accepting "Abolished interaction
# with LF" — strict about the wording of a loss and permissive about which endpoint was lost.
# § 3's test is DIRECTION, not endpoint. It also checked hedges before losses, so "Loss of
# toxicity due to decrease in cell binding" was excluded for a hedge qualifying the mechanism
# rather than the loss.

# a hedge on the loss ITSELF, which is what § 3 means by hedged
HEDGE_ON_LOSS = [r"\bpartial", r"\bslight", r"\bmoderate", r"\bsomewhat",
                 r"\bmay\b", r"\bmight\b", r"\bappears?\b", r"\blikely\b", r"\bprobabl"]

# a change in what it acts on, not in how much: excluded by name in the rule
SPECIFICITY = [r"specificit", r"alters? the substrate", r"changes? the preference",
               r"nickase", r"shifts? the"]

# conditional: the phenotype is stated only under a further condition
CONDITIONAL = [r"\bwhen\b", r"\bin the presence of\b", r"\bin combination with\b",
               r"\bonly in\b", r"\bonly when\b"]

# unambiguous loss, of any endpoint
LOSS = [r"\bloss of\b", r"\babolish", r"\bno detectable\b", r"\bno longer\b",
        r"\bnon-?toxic\b", r"\binactive\b", r"\bdoes not cleave\b", r"\bis not cleaved\b",
        r"\blacks?\b", r"\bdevoid of\b", r"\bno activit", r"\bloss of function\b",
        r"\bcompletely (non|in)", r"\beliminat"]

# an unquantified reduction: real, directional, and not admitted by § 3 without a magnitude
REDUCTION = [r"\breduce[sd]?\b", r"\breduction\b", r"\bdecrease", r"\bsuppress",
             r"\bimpair", r"\bdiminish", r"\blower", r"\bweak", r"\bless\b"]

TOLERATED = [r"no effect", r"no significant effect", r"does not affect", r"no change",
             r"without (any )?effect", r"retains? (full |wild-type )?activit",
             r"\bnormal activit", r"\bwild-type activit"]

# a stated fold change of 100 or more is loss; anything smaller is not
FOLD = re.compile(r"(\d[\d,\.]*)\s*[-\s]?fold", re.I)
ORDERS = re.compile(r"(several|\d+|two|three|four|five)\s+orders?\s+of\s+magnitude", re.I)
_WORDNUM = {"two": 2, "three": 3, "four": 4, "five": 5, "several": 3}


def _any(pats, text):
    for p in pats:
        if re.search(p, text, re.I):
            return p
    return None


def classify(text):
    """Return (verdict, reason). Verdicts: loss | tolerated | excluded.

    Order is the rule: a hedge on the loss, a specificity change or a condition disqualifies
    before anything else is read; then a stated magnitude decides in both directions; then an
    unambiguous loss; then an unambiguous tolerated; then an unquantified reduction excludes.
    """
    t = (text or "").strip()
    if not t:
        return "excluded", "no phenotype text"

    hit = _any(HEDGE_ON_LOSS, t)
    if hit:
        return "excluded", f"hedged ({hit})"
    hit = _any(SPECIFICITY, t)
    if hit:
        return "excluded", f"describes a change in specificity, not magnitude ({hit})"
    hit = _any(CONDITIONAL, t)
    if hit:
        return "excluded", f"conditional phenotype ({hit})"

    m = FOLD.search(t)
    if m:
        try:
            fold = float(m.group(1).replace(",", ""))
        except ValueError:
            fold = None
        if fold is not None:
            return ("loss", f"stated {fold:g}-fold reduction, >= 100") if fold >= 100 else (
                "excluded", f"stated {fold:g}-fold reduction, below the 100-fold rule")
    m = ORDERS.search(t)
    if m:
        n = _WORDNUM.get(m.group(1).lower())
        if n is None:
            try:
                n = int(m.group(1))
            except ValueError:
                n = 0
        return ("loss", f"{m.group(0)}, >= 100-fold") if n >= 2 else (
            "excluded", f"{m.group(0)}, below the 100-fold rule")

    hit = _any(LOSS, t)
    if hit:
        return "loss", f"unambiguous loss ({hit})"
    hit = _any(TOLERATED, t)
    if hit:
        return "tolerated", f"unambiguous tolerated ({hit})"
    hit = _any(REDUCTION, t)
    if hit:
        return "excluded", f"reduction with no stated magnitude ({hit})"
    return "excluded", "no phrase from the preregistered list matched"


def read_fasta(path):
    recs, acc, seq = {}, None, []
    for line in open(path):
        if line.startswith(">"):
            if acc:
                recs[acc] = "".join(seq)
            acc, seq = line[1:].split()[0], []
        elif acc:
            seq.append(line.strip())
    if acc:
        recs[acc] = "".join(seq)
    return recs


def panel_sequences():
    """Accession -> precursor sequence, from the FASTA files the FSPE pipeline indexes."""
    out = {}
    for f in sorted((ROOT / "data" / "sequences").glob("*.fasta")) + \
            sorted((ROOT / "data" / "sequences").glob("*.fa")):
        for header, seq in read_fasta(f).items():
            acc = header.split("|")[1] if "|" in header else header
            out.setdefault(acc, seq)
    return out


def selftest():
    cases = [
        ("Loss of catalytic activity.", "loss"),
        ("Abolishes toxin activity.", "loss"),
        ("No detectable activity.", "loss"),
        ("Light chain no longer cleaves SNAP25.", "loss"),
        ("Reduction of several orders of magnitude.", "loss"),
        ("1000-fold reduction in activity.", "loss"),
        ("10-fold reduction in activity.", "excluded"),
        ("Suppresses the toxic activity.", "excluded"),
        ("Partial loss of activity.", "excluded"),
        ("Alters the substrate specificity.", "excluded"),
        ("No effect on the toxic activity.", "tolerated"),
        ("No effect on the toxic activity but suppresses the vascular leak syndrome.", "tolerated"),
        ("", "excluded"),
        # the five cases the 2026-09-24 correction is about
        ("Loss of toxicity.", "loss"),
        ("Loss of ability to bind to LF and completely non-toxic.", "loss"),
        ("Loss of toxicity due to decrease in cell binding.", "loss"),
        ("Abolished interaction with LF.", "loss"),
        ("Target DNA is not cleaved; nickase activity.", "excluded"),
    ]
    bad = [(t, want, classify(t)[0]) for t, want in cases if classify(t)[0] != want]
    for t, want, got in bad:
        print(f"  FAIL want={want:9} got={got:9} {t!r}")
    print(f"selftest: {len(cases) - len(bad)}/{len(cases)} pass")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    sites = json.load(open(ROOT / "data/annotations/functional_sites.json"))
    panel = [k for k in sites if not k.startswith("_")]
    seqs = panel_sequences()

    kept, excluded, identity_fail = [], [], []
    for acc in sorted(panel):
        f = CACHE / f"{acc}.json"
        if not f.exists():
            excluded.append({"acc": acc, "reason": "no cached UniProt record"})
            continue
        rec = json.load(open(f))
        seq = seqs.get(acc)
        for ft in rec.get("features", []):
            if ft["type"] != "Mutagenesis":
                continue
            loc = ft["location"]
            if loc["start"]["value"] != loc["end"]["value"]:
                excluded.append({"acc": acc, "pos": [loc["start"]["value"], loc["end"]["value"]],
                                 "text": ft.get("description", ""),
                                 "reason": "multi-residue feature, not a single substitution"})
                continue
            pos = loc["start"]["value"]
            alt = ft.get("alternativeSequence", {}) or {}
            wt = alt.get("originalSequence")
            verdict, reason = classify(ft.get("description", ""))
            row = {"acc": acc, "pos": pos, "wt": wt,
                   "alts": alt.get("alternativeSequences", []),
                   "text": ft.get("description", ""), "verdict": verdict, "reason": reason}

            # rule 3: the identity check is not optional, and it runs on EVERY feature so the
            # log records coordinate failures separately from phenotype failures
            if seq is None:
                row["identity"] = "no panel sequence"
            elif wt is None:
                row["identity"] = "feature carries no original residue"
            elif not (1 <= pos <= len(seq)):
                row["identity"] = f"position {pos} outside the {len(seq)}-residue panel sequence"
            elif seq[pos - 1] != wt:
                row["identity"] = f"panel has {seq[pos - 1]} at {pos}, UniProt says {wt}"
            else:
                row["identity"] = "ok"

            if row["identity"] != "ok":
                identity_fail.append(row)
            elif verdict == "excluded":
                excluded.append(row)
            else:
                kept.append(row)

    loss = [r for r in kept if r["verdict"] == "loss"]
    tol = [r for r in kept if r["verdict"] == "tolerated"]
    # a substitution is one (position, alternative) pair, which is what P3 scores
    n_subs = sum(max(1, len(r["alts"])) for r in loss)
    prots = sorted({r["acc"] for r in loss})
    p3_runs = n_subs >= 12 and len(prots) >= 4

    per_prot = {}
    for r in loss:
        per_prot[r["acc"]] = per_prot.get(r["acc"], 0) + max(1, len(r["alts"]))
    dominant = max(per_prot.values()) / n_subs if n_subs else 0.0

    res = {
        "step": "preregistration section 7 step 2, tier 2 set construction",
        "panel_entries": len(panel),
        "features_seen": len(kept) + len(excluded) + len(identity_fail),
        "loss_features": len(loss), "tolerated_features": len(tol),
        "loss_substitutions": n_subs, "loss_proteins": prots,
        "substitutions_per_protein": per_prot,
        "largest_protein_share": round(dominant, 3),
        "excluded_n": len(excluded), "identity_failures_n": len(identity_fail),
        "p3_threshold": "12 substitutions across at least 4 proteins",
        "p3_runs": bool(p3_runs),
        "multiplicity_alpha": 0.05 / 6 if p3_runs else 0.05 / 5,
        "n_primary_tests": 6 if p3_runs else 5,
        "kept": kept, "excluded": excluded, "identity_failures": identity_fail,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)

    print(f"panel entries           {len(panel)}")
    print(f"Mutagenesis features    {res['features_seen']}")
    print(f"  loss                  {len(loss)} features -> {n_subs} substitutions")
    print(f"  tolerated             {len(tol)} features")
    print(f"  excluded (phenotype)  {len(excluded)}")
    print(f"  excluded (identity)   {len(identity_fail)}")
    print(f"loss proteins           {len(prots)}  {prots}")
    print(f"largest single share    {dominant:.0%}")
    print()
    print("P3 threshold: 12 substitutions across >= 4 proteins")
    print(f"P3 {'RUNS' if p3_runs else 'IS DROPPED'} -> "
          f"{res['n_primary_tests']} primary tests, alpha = {res['multiplicity_alpha']:.4f}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
