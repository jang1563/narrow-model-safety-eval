#!/usr/bin/env python3
"""
03n_pretraining_holdout.py - the first test of §9.3's caveat rather than a restatement of it.

The caveat, as §9.3 currently states it
---------------------------------------
"Leave-one-mechanism-out removes the class from the probe's training set, not from
the foundation model's. Every model here has seen ricin, Shiga toxin, botulinum
neurotoxin and thousands of beta-lactamases during pretraining." That is written
as a limitation with no number attached. This attaches one.

Construction
------------
ESM-2 was pretrained on UniRef50 release 2021_04 (Lin et al., Science 2023). A
sequence whose own first version postdates that release cannot have been in it.

  ⚠️ The obvious filter is wrong. UniProt's date_created is the date an entry was
  promoted into Swiss-Prot, NOT the date its sequence first appeared. Q9JXM7 has
  date_created 2022-12-14 and a sequence last updated 2000-10-01; it sat in TrEMBL
  for two decades and is certainly in UniRef50 2021_04. Filtering on date_created
  selects recently REVIEWED proteins, not recently DISCOVERED ones, and it inflated
  the bacterial virulence pool from 13 to 119.

  The filter used here is date_sequence_modified >= 2021-10-01 together with
  sequence version 1, so the sequence itself is new rather than the annotation.

  Homology screening is still required on top of that, and catching three cases
  proves it: the E. coli OspC3 ortholog scores 0.955 normalized Smith-Waterman
  against the panel's Shigella OspC3, and two Bacillus alveolysins score 0.567
  against streptolysin O. All three are dropped under the panel's own <=0.30 rule.

Panel: 40 positives (30 KW-0800 toxin, 10 KW-0843 virulence; 12 bacterial, 28
animal or plant), median max normalized SW to any panel positive 0.041, and 40
length-matched benign bacterial proteins from the same post-snapshot window.

⚠️ The confound this cannot remove
----------------------------------
Bacterial hazards whose SEQUENCES postdate the snapshot are scarce: 12 survive
screening. The rest are animal venom and plant toxins, so the positive set differs
from the panel in taxon and in length distribution as well as in novelty. A drop
in recovery is therefore not attributable to pretraining novelty alone. The
bacterial subset is reported separately for that reason, and it is small.

What this does and does not claim
---------------------------------
UniRef50 clusters at 50% identity, so "this sequence was not in pretraining" is not
"nothing resembling it was in pretraining". This measures novelty of the sequence,
not novelty of the function.

PREREGISTERED, written before embedding
---------------------------------------
    CAVEAT IS SMALL     recovery >= 70% at the panel's 95th-percentile operating
                        point. The probe reaches hazards it provably never saw, and
                        §9.3's warning, while true, bounds a small effect.
    CAVEAT IS MATERIAL  recovery <= 40%. Panel performance depends substantially on
                        pretraining exposure to those specific sequences.
    PARTIAL             between, reported as such rather than rounded to a story.

Usage:
    python src/03n_pretraining_holdout.py --panel /path/to/postsnap_final.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "results" / "v2"


def embed_esm2(seqs, model_name="facebook/esm2_t33_650M_UR50D", max_len=1022):
    """Byte-for-byte the pooling of src/02b_esm2_embed_v2.py: attention-mask
    weighted mean over every unmasked token, BOS/EOS included."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    out = []
    B = 8
    for i in range(0, len(seqs), B):
        chunk = [s[:max_len] for s in seqs[i:i + B]]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len + 2)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1)
        out.append(((h * m).sum(1) / m.sum(1)).float().cpu().numpy())
        print(f"  embedded {min(i + B, len(seqs))}/{len(seqs)}", flush=True)
    return np.vstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    a = ap.parse_args()

    d = json.load(open(a.panel))
    pos, neg = d["pos"], d["neg"]
    print(f"post-snapshot set: {len(pos)} positives, {len(neg)} length-matched negatives")

    P = np.load(V2 / "embeddings_positive_v2.npy")
    N = np.load(V2 / "embeddings_negative_v2.npy")
    print(f"internal panel: {P.shape[0]} positives, {N.shape[0]} negatives")

    t0 = time.time()
    X = embed_esm2([x["seq"] for x in pos] + [x["seq"] for x in neg])
    print(f"embedded in {time.time() - t0:.0f}s, shape {X.shape}")
    assert X.shape[1] == P.shape[1], "embedding dimension mismatch, pipeline drifted"
    Xp, Xn = X[:len(pos)], X[len(pos):]

    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
    pipe.fit(np.vstack([P, N]), np.r_[np.ones(len(P)), np.zeros(len(N))])

    sp, sn = pipe.predict_proba(Xp)[:, 1], pipe.predict_proba(Xn)[:, 1]
    t_panel = float(np.quantile(pipe.predict_proba(N)[:, 1], 0.95))
    t_era = float(np.quantile(sn, 0.95))
    auroc = float(roc_auc_score(np.r_[np.ones(len(sp)), np.zeros(len(sn))], np.r_[sp, sn]))

    def rec(mask, t):
        return float((sp[mask] >= t).mean()) if mask.sum() else float("nan")

    BACT = ("Escherichia", "Salmonella", "Bacillus", "Pseudomonas", "Xanthomonas",
            "Neisseria", "Staphylococcus", "Acinetobacter", "Borreliella",
            "Wolbachia", "Mesomycoplasma", "Shigella", "Vibrio")
    is_b = np.array([any(k in x["org"] for k in BACT) for x in pos])
    allm = np.ones(len(pos), bool)

    print(f"\nthreshold on panel negatives      {t_panel:.4f}")
    print(f"threshold on post-snapshot negatives {t_era:.4f}")
    print(f"AUROC, post-snapshot positives vs matched negatives: {auroc:.3f}")
    print(f"\n{'subset':<28}{'n':>4}{'rec@panel-thr':>15}{'rec@era-thr':>13}")
    print("-" * 60)
    rows = {}
    for lab, m in (("all", allm), ("bacterial", is_b), ("animal/plant", ~is_b)):
        rows[lab] = {"n": int(m.sum()), "recovery_panel_threshold": rec(m, t_panel),
                     "recovery_era_threshold": rec(m, t_era)}
        print(f"{lab:<28}{int(m.sum()):>4}{rec(m, t_panel):>14.0%}{rec(m, t_era):>13.0%}")

    # ---- validity gate, checked BEFORE the recovery verdict -----------------
    # Recovery on its own cannot distinguish detection from a blanket false
    # positive rate. If the probe scores the matched negatives as high as the
    # positives, a high recovery number means it is flagging everything in this
    # distribution, not finding hazards in it. AUROC at or below chance is the
    # tell, and the first version of this script omitted the check and printed
    # "CAVEAT IS SMALL" off 85% recovery while AUROC was 0.424. That is the third
    # preregistration in this project to specify a floor and no power check.
    # A FIXED AUROC threshold is the wrong gate and this project has now been bitten
    # by one twice: 03k's degeneracy guard fired at >0.50 and the real case measured
    # 0.49; this script's first guard fired at <=0.55 and the real case measured
    # 0.556. Both let through a result that carried no information. The gate that
    # survives small n is a confidence interval: bootstrap the AUROC and refuse the
    # result whenever the interval still contains 0.5.
    rng_b = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        i = rng_b.integers(0, len(sp), len(sp))
        j = rng_b.integers(0, len(sn), len(sn))
        yb = np.r_[np.ones(len(i)), np.zeros(len(j))]
        boots.append(roc_auc_score(yb, np.r_[sp[i], sn[j]]))
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    print(f"AUROC bootstrap 95% CI: [{lo:.3f}, {hi:.3f}]  (2000 resamples)")

    r = rows["all"]["recovery_panel_threshold"]
    if lo <= 0.5:
        verdict = (f"INVALID, NOT A RESULT: AUROC {auroc:.3f}, 95% CI [{lo:.3f}, {hi:.3f}] "
                   f"includes 0.5. The probe does not separate this external set, so the "
                   f"{r:.0%} recovery is a false positive rate rather than detection.")
        print(f"\n\U0001F534 {verdict}")
        print("   Do not quote the recovery number. Diagnose the negative set first.")
    else:
        verdict = ("CAVEAT IS SMALL (>=70%)" if r >= 0.70 else
                   "CAVEAT IS MATERIAL (<=40%)" if r <= 0.40 else
                   "PARTIAL: between the preregistered bounds")
        print(f"\nverdict: {verdict}")

    out = {"built": time.strftime("%Y-%m-%d"), "model": "facebook/esm2_t33_650M_UR50D",
           "pretraining_release": "UniRef50 2021_04 (Lin et al., Science 2023)",
           "filter": "date_sequence_modified >= 2021-10-01 AND sequence version 1, "
                     "then normalized Smith-Waterman <= 0.30 against every panel positive",
           "n_positive": len(pos), "n_negative": len(neg),
           "median_max_sw_to_panel": float(np.median([x["max_sw_to_panel"] for x in pos])),
           "threshold_panel_negatives": t_panel, "threshold_era_negatives": t_era,
           "auroc_post_snapshot": auroc, "subsets": rows,
           "preregistered_small_if_geq": 0.70, "preregistered_material_if_leq": 0.40,
           "verdict": verdict, "valid": bool(lo > 0.5),
           "auroc_ci95": [lo, hi],
           "scores": {x["acc"]: round(float(s), 4) for x, s in zip(pos, sp)},
           "negative_scores": {x["acc"]: round(float(s), 4) for x, s in zip(neg, sn)},
           "members": [{"acc": x["acc"], "org": x["org"], "len": x["len"],
                        "label": x["label"], "max_sw_to_panel": x["max_sw_to_panel"],
                        "score": round(float(s), 4)} for x, s in zip(pos, sp)]}
    p = V2 / "pretraining_holdout.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
