#!/usr/bin/env python3
"""
25_amr_category_test_v2.py -- corrects two defects in src/24_amr_category_test.py
and runs the comparison that test was supposed to make.

What was wrong with 24
----------------------
24 asked whether beta-lactamase's poor recovery is a property of antibiotic
resistance as a category. It embedded a second AMR family (aminoglycoside-
modifying enzymes), scored it against a probe, got 87.5%, and called the
hypothesis refuted. Two things make that reading unsupported.

  (i) Training-set contamination. 24 trains on `X = vstack([P, N])` -- the FULL
      internal positive set. 14 of those 80 positives (17.5%) ARE beta-lactamases.
      So the probe had already seen antibiotic-resistance enzymes when it was
      asked to reach a new antibiotic-resistance family. That measures
      within-category generalization, not whether a toxin-trained representation
      reaches AMR at all, which is what the hypothesis was about.

 (ii) Protocol mismatch. 24's 87.5% was compared against beta-lactamase's 21%
      from the LOMO table. Those are computed differently: LOMO holds out 40% of
      the negatives and calibrates the threshold on the held-out negatives, while
      24 trains on every negative and calibrates in-sample. Under 24's own
      protocol, beta-lactamase with AMR removed from training recovers 50.0%, not
      21%. The preregistered bounds in 24 ("<=40%, in the same range as
      beta-lactamase's 21%") were therefore anchored to a number from the other
      protocol.

Holding protocol fixed at 24's own (train on all negatives, threshold at the
95th percentile of in-sample negatives, train on all positives except the class
under test) gives this column, computed in this script and needing no new
embeddings:

    adp_ribosyl_ab_toxin            100.0%      rip_rrna_glycosidase    100.0%
    clostridial_neurotoxin          100.0%      superantigen_enterotoxin 100.0%
    contact_dependent_inhibition    100.0%      t3ss_effector_apparatus   90.0%
    pore_forming_cytolysin          100.0%      virulence (control)       70.0%
    beta_lactamase                   50.0%      <- still the clear outlier

So beta-lactamase remains uniquely hard under a matched protocol. But the
aminoglycoside number it was compared against still came from a probe that had
beta-lactamase in training, and beta-lactamase's 50% came from one that did not.
That difference alone could produce the whole gap.

The missing cell
----------------
The 2x2 is: {train with AMR, train without AMR} x {test beta-lactamase, test
aminoglycoside}. Three cells are known. This script computes the fourth.

  PREREGISTERED, written 2026-09-11 before the aminoglycosides were re-embedded.
  The test class has 8 members, so recovery moves in steps of 12.5%.

    CONTAMINATION DROVE IT   aminoglycoside recovery <= 62.5% (<=5/8) with AMR
                             removed from training. The 87.5% in 24 was carried
                             by having beta-lactamase in the training set, the
                             category hypothesis is live again, and 24's
                             refutation does not stand as written.

    REFUTATION STANDS        recovery >= 87.5% (>=7/8), unchanged from 24. Having
                             AMR in training was not what carried it; a
                             toxin-trained probe reaches this AMR family on its
                             own, and beta-lactamase's difficulty is specific to
                             beta-lactamase.

    INCONCLUSIVE             75% (6/8), one member between the bounds.

Nothing about the internal panel changes: same frozen embeddings, same members,
same annotations. Only the training mask and the reporting differ.

Usage:
    python src/25_amr_category_test_v2.py --candidates /path/to/aminoglycoside_final.fasta
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "results" / "v2"
ANNOT = ROOT / "data" / "annotations" / "mechanism_classes_v2.json"


def read_fasta(path):
    seqs, order, cur = {}, [], None
    for ln in open(path):
        if ln.startswith(">"):
            cur = ln[1:].strip().split()[0]
            seqs[cur] = ""
            order.append(cur)
        else:
            seqs[cur] += ln.strip()
    return order, seqs


def embed_esm2(seqs, model_name="facebook/esm2_t33_650M_UR50D", max_len=1022):
    """Byte-for-byte the pooling used by src/02b_esm2_embed_v2.py: attention-mask
    weighted mean over every unmasked token, BOS/EOS included."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    truncated = [s[:max_len] for _, s in seqs]
    enc = tok(truncated, return_tensors="pt", padding=True, truncation=True,
              max_length=max_len + 2)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        h = model(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1)
    return ((h * mask).sum(1) / mask.sum(1)).float().cpu().numpy()


def fit_probe(P, N, train_mask):
    """24's protocol exactly: train on the selected positives plus EVERY negative,
    then take the threshold from the 95th percentile of those same negatives."""
    X = np.vstack([P[train_mask], N])
    y = np.r_[np.ones(int(train_mask.sum())), np.zeros(len(N))]
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0)).fit(X, y)
    t95 = float(np.quantile(pipe.predict_proba(N)[:, 1], 0.95))
    return pipe, t95


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    a = ap.parse_args()

    manifest = json.load(open(V2 / "embedding_manifest_v2.json"))
    mech = json.load(open(ANNOT))
    P = np.load(V2 / "embeddings_positive_v2.npy")
    N = np.load(V2 / "embeddings_negative_v2.npy")

    cls = {e["fasta_id"]: e["mechanism_class"] for e in mech["proteins"]}
    pcls = np.array([cls[r["acc"]] for r in manifest["positive_rows"]])
    assert P.shape[0] == len(pcls) == 80 and N.shape[0] == 154

    bl = pcls == "beta_lactamase"
    print(f"internal panel: {P.shape[0]} positives ({bl.sum()} beta-lactamase), "
          f"{N.shape[0]} negatives, dim {P.shape[1]}\n")

    # --- the protocol-24 column for every internal class, no new embeddings ----
    print("protocol-24 applied to every internal class (class removed from training)")
    print(f"{'class':<34}{'n':>4}{'recovery@95':>13}")
    print("-" * 51)
    column = {}
    for C in sorted(set(pcls)):
        hi = np.where(pcls == C)[0]
        if len(hi) < 4:
            continue
        pipe, t95 = fit_probe(P, N, pcls != C)
        r = float((pipe.predict_proba(P[hi])[:, 1] >= t95).mean())
        column[C] = {"n": int(len(hi)), "recovery_at_95pct": r}
        print(f"{C:<34}{len(hi):>4}{r * 100:>12.1f}%")
    print()

    # --- the 2x2 --------------------------------------------------------------
    order, seqs = read_fasta(a.candidates)
    print(f"test class: {len(order)} members -- "
          f"{', '.join(o.split('|')[2] for o in order)}")
    print("embedding with frozen ESM-2 650M ...")
    t0 = time.time()
    X_test = embed_esm2([(o, seqs[o]) for o in order])
    print(f"done in {time.time() - t0:.0f}s, shape {X_test.shape}\n")
    assert X_test.shape[1] == P.shape[1], "embedding dimension mismatch, pipeline drifted"

    cells, scores_out = {}, {}
    for train_label, mask in (("with_amr", np.ones(len(P), bool)), ("without_amr", ~bl)):
        pipe, t95 = fit_probe(P, N, mask)
        for test_label, Xt, names in (
            ("aminoglycoside", X_test, [o.split("|")[2] for o in order]),
            ("beta_lactamase", P[bl], list(np.array(
                [r["name"] for r in manifest["positive_rows"]])[bl])),
        ):
            s = pipe.predict_proba(Xt)[:, 1]
            key = f"train_{train_label}__test_{test_label}"
            cells[key] = {
                "n_train_positive": int(mask.sum()),
                "threshold_95pct": t95,
                "n_test": len(s),
                "n_flagged": int((s >= t95).sum()),
                "recovery_at_95pct": float((s >= t95).mean()),
                "in_sample": (train_label == "with_amr" and test_label == "beta_lactamase"),
            }
            scores_out[key] = {n: round(float(v), 4) for n, v in zip(names, s)}

    print(f"{'':<16}{'test: beta-lactamase':>24}{'test: aminoglycoside':>24}")
    print("-" * 64)
    for tl, lab in (("with_amr", "train WITH AMR"), ("without_amr", "train WITHOUT AMR")):
        b = cells[f"train_{tl}__test_beta_lactamase"]
        g = cells[f"train_{tl}__test_aminoglycoside"]
        bs = f"{b['recovery_at_95pct'] * 100:.1f}%" + (" (in-sample)" if b["in_sample"] else "")
        print(f"{lab:<16}{bs:>24}{g['recovery_at_95pct'] * 100:>23.1f}%")
    print()

    key = "train_without_amr__test_aminoglycoside"
    rec = cells[key]["recovery_at_95pct"]
    if rec <= 0.625:
        verdict = ("CONTAMINATION DROVE IT (<=62.5%): the 87.5% in 24 rested on having "
                   "beta-lactamase in training; the AMR-category hypothesis is live again")
    elif rec >= 0.875:
        verdict = ("REFUTATION STANDS (>=87.5%): removing AMR from training does not change "
                   "the result; beta-lactamase's difficulty is specific to beta-lactamase")
    else:
        verdict = "INCONCLUSIVE by the preregistered bounds (75%, one member between them)"

    print(f"decisive cell -- aminoglycoside with AMR removed from training: {rec:.1%} "
          f"({cells[key]['n_flagged']}/{cells[key]['n_test']})")
    print(f"verdict: {verdict}")

    out = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test": "amr_category_hypothesis_v2_corrected",
        "corrects": "src/24_amr_category_test.py",
        "defects_corrected": [
            "24 trained on all 80 positives, which include the 14 beta-lactamases, so the "
            "test class's own category was in the training set",
            "24 compared its 87.5% against the LOMO protocol's 21%; under 24's own protocol "
            "beta-lactamase recovers 50.0%",
        ],
        "preregistration": "src/25_amr_category_test_v2.py docstring, written before re-embedding",
        "contamination_drove_it_if_leq": 0.625,
        "refutation_stands_if_geq": 0.875,
        "model": "facebook/esm2_t33_650M_UR50D",
        "protocol": "train on selected positives + all negatives; threshold = 95th pct of those negatives",
        "protocol24_column_internal_classes": column,
        "cells": cells,
        "scores": scores_out,
        "decisive_cell": key,
        "decisive_recovery": rec,
        "verdict": verdict,
    }
    outpath = V2 / "amr_category_test_v2.json"
    json.dump(out, open(outpath, "w"), indent=2)
    print(f"\nwrote {outpath}")


if __name__ == "__main__":
    main()
