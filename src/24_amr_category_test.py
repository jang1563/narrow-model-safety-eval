#!/usr/bin/env python3
"""
24_amr_category_test.py -- preregistered external test of the §9.4 host-interaction
hypothesis for the beta-lactamase anomaly.

Prediction, written 2026-09-11 before any embedding was computed
--------------------------------------------------------------
docs/MECHANISM_GENERALIZATION.md §9.4 proposes that beta-lactamase resists this
probe not because of the classifier head, corpus, or scale (all three tested and
refused in §9.1/§9.3), but because it is not the same *kind* of hazard as the
other eight classes: it requires no host interaction at all, where every other
class is defined by one (a ribosome inactivated, a membrane perforated, a
receptor bridged). If that is the operative property, a second, sequence- and
fold-distinct AMR family that ALSO requires no host interaction should show the
same failure, not because it resembles beta-lactamase in sequence, but because it
shares the same absence of a host-interaction signature.

  PREDICTED  if the hypothesis holds: recovery at the 95th-percentile threshold
             <= 40%, in the same range as beta-lactamase's 21%, well below the
             ~90-100% seen for genuinely host-interacting classes.
  REFUTED    if recovery >= 70%, comparable to the classes that DO require host
             interaction. That would mean beta-lactamase's difficulty is
             something else -- specific to its own fold diversity or its own
             training-data properties -- not a property of "AMR" as a category.

This is treated as an EXTERNAL validation in the sense of the frozen
preregistration: the internal 80/154 panel is not touched, not re-embedded, not
re-split. The new class is embedded once with the same frozen ESM-2 650M
pipeline, scored against a probe trained on the full internal positive set (no
internal holdout), at the same 95th-percentile-of-negatives threshold used
throughout. Panel membership, mechanism_classes_v2.json, and every existing
result file are unchanged by running this script.

Test class: aminoglycoside-modifying enzymes (AAC, ANT, APH, and one AAC/APH
fusion), 8 members spanning three distinct fold families (GNAT acetyltransferase,
nucleotidyltransferase, protein-kinase-like) plus a divergent Eis-family GNAT and
a bifunctional fusion, from eight different organisms. Screened at normalized
Smith-Waterman <= 0.30 against each other and against all 234 existing panel
members (see /tmp/aminoglycoside_final.fasta build log). All carry UniProt
KW-0046 (Antibiotic resistance); none carry KW-0800 (Toxin) or KW-0843
(Virulence), the same keyword profile as beta-lactamase.

Usage:
    python src/24_amr_category_test.py --candidates /path/to/aminoglycoside_final.fasta
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


def read_fasta(path):
    seqs, order = {}, []
    cur = None
    for ln in open(path):
        if ln.startswith(">"):
            cur = ln[1:].strip().split()[0]
            seqs[cur] = ""
            order.append(cur)
        else:
            seqs[cur] += ln.strip()
    return order, seqs


def embed_esm2(seqs, model_name="facebook/esm2_t33_650M_UR50D", max_len=1022):
    """Byte-for-byte the same pooling as src/02b_esm2_embed_v2.py: attention-mask
    weighted mean over every unmasked token, BOS/EOS included, because that is
    what the internal panel's embeddings were computed with. Dropping BOS/EOS
    here would make this a different representation and invalidate the
    comparison before the test even runs."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    accs = [a for a, _ in seqs]
    truncated = [s[:max_len] for _, s in seqs]
    enc = tok(truncated, return_tensors="pt", padding=True, truncation=True,
              max_length=max_len + 2)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        h = model(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1)
    pooled = ((h * mask).sum(1) / mask.sum(1)).float().cpu().numpy()
    for a in accs:
        print(f"  embedded: {a}", flush=True)
    return pooled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True,
                    help="FASTA of the screened, homology-checked candidate class")
    a = ap.parse_args()

    order, seqs = read_fasta(a.candidates)
    print(f"test class: {len(order)} members -- {', '.join(o.split('|')[2] for o in order)}\n")

    print("embedding test class with frozen ESM-2 650M ...")
    t0 = time.time()
    X_test = embed_esm2([(o, seqs[o]) for o in order])
    print(f"done in {time.time() - t0:.0f}s, shape {X_test.shape}\n")

    print("loading the frozen internal panel's canonical 650M embeddings ...")
    manifest = json.load(open(V2 / "embedding_manifest_v2.json"))
    P = np.load(V2 / "embeddings_positive_v2.npy")
    N = np.load(V2 / "embeddings_negative_v2.npy")
    pos_acc = [r["acc"] for r in manifest["positive_rows"]]
    neg_acc = [r["acc"] for r in manifest["negative_rows"]]
    assert P.shape[0] == len(pos_acc) and N.shape[0] == len(neg_acc)
    print(f"internal panel: {P.shape[0]} positives, {N.shape[0]} negatives, dim {P.shape[1]}")
    assert X_test.shape[1] == P.shape[1], "embedding dimension mismatch -- pipeline drifted"

    # train on the FULL internal panel, no holdout -- this is an external test,
    # not a leave-one-mechanism-out arm. calibrate the 95th-percentile threshold
    # on the internal negatives, exactly as 03b does for every other class.
    # identical pipeline to src/03b_leave_one_mechanism_out.py's clf(): a fresh
    # StandardScaler fit on this training data, then LogisticRegression(C=1.0).
    # Using the same object (not just the same hyperparameters) means the scaler
    # that transforms the new class was fit on the internal panel, exactly as it
    # would be for any held-out internal class.
    X = np.vstack([P, N])
    y = np.r_[np.ones(len(P)), np.zeros(len(N))]
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
    pipe.fit(X, y)

    neg_scores = pipe.predict_proba(N)[:, 1]
    t95 = np.quantile(neg_scores, 0.95)

    test_scores = pipe.predict_proba(X_test)[:, 1]
    flagged = test_scores >= t95
    recovery = float(flagged.mean())

    print(f"\nthreshold (95th pct of internal negatives): {t95:.4f}\n")
    print(f"{'member':<16}{'score':>8}{'flagged':>9}")
    for o, s, f in zip(order, test_scores, flagged):
        print(f"{o.split('|')[2]:<16}{s:>8.3f}{str(bool(f)):>9}")

    print(f"\nrecovery @ 95th-pct threshold: {recovery:.1%}  ({int(flagged.sum())}/{len(order)})")
    if recovery <= 0.40:
        verdict = "PREDICTED range (<=40%): hypothesis SUPPORTED"
    elif recovery >= 0.70:
        verdict = "REFUTED range (>=70%): hypothesis NOT SUPPORTED"
    else:
        verdict = "between the stated bounds: INCONCLUSIVE by the preregistered criteria"
    print(f"verdict: {verdict}")

    out = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test": "amr_category_host_interaction_hypothesis",
        "preregistration": "src/24_amr_category_test.py docstring, written before embedding",
        "predicted_supported_if_recovery_leq": 0.40,
        "predicted_refuted_if_recovery_geq": 0.70,
        "model": "facebook/esm2_t33_650M_UR50D",
        "n_internal_positive": int(P.shape[0]),
        "n_internal_negative": int(N.shape[0]),
        "threshold_95pct_of_negatives": float(t95),
        "test_class": "aminoglycoside_modifying_enzymes",
        "members": [o.split("|")[2] for o in order],
        "scores": {o.split("|")[2]: float(s) for o, s in zip(order, test_scores)},
        "flagged": {o.split("|")[2]: bool(f) for o, f in zip(order, flagged)},
        "recovery_at_95pct": recovery,
        "verdict": verdict,
    }
    outpath = V2 / "amr_category_test.json"
    json.dump(out, open(outpath, "w"), indent=2)
    print(f"\nwrote {outpath}")


if __name__ == "__main__":
    main()
