#!/usr/bin/env python3
"""Study AI-1: what does DeepVIC's benchmark score with NO sequence semantics at all?

`docs/DEEPVIC_BENCHMARK_PREREGISTRATION.md` § 3, bands in § 4, predictions P1/P2 in § 5.

DeepVIC reports **AUROC 0.954** on a 13,384-sequence holdout using ProtBert-BFD embeddings.
`src/122` measured, before any outcome existed, that its positives' median length is **549**
against the negatives' **268** -- a 2.05x ratio. This asks what a classifier that knows only
the length, or only the length and the amino-acid frequencies, scores on the same task.

This is the question this project has already asked of its own panel, where two metadata
fields with no sequence information reached AUROC 0.8096 against the embedding's 0.9323.
Nothing here is fitted on an embedding; no protein language model is loaded.
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DV = ROOT / "data/external/deepvic"
FAA = DV / "DLDB_33456.faa"
META = DV / "metadata.txt"
BUILD = ROOT / "results/deepvic_build.json"
OURS = ROOT / "results/external_class_axis_build.json"
OUT = ROOT / "results/deepvic_metadata_baseline.json"

AA = "ACDEFGHIKLMNPQRSTVWY"
SEEDS = 5
TEST_FRAC = 0.20
VFG_RE = re.compile(r"VFG\d+")
# frozen in section 4
BAND_LEN_CONFOUND, BAND_LEN_CLEAN = 0.75, 0.60
BAND_COMP_CONFOUND, BAND_COMP_CLEAN = 0.85, 0.70
DEEPVIC_PUBLISHED = 0.954


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))


def auroc(pos, neg):
    a = np.concatenate([np.asarray(pos, float), np.asarray(neg, float)])
    o = a.argsort()
    r = np.empty(len(a), float)
    r[o] = np.arange(1, len(a) + 1)
    for u in np.unique(a):
        m = a == u
        if m.sum() > 1:
            r[m] = r[m].mean()
    n1 = len(pos)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(neg)))


def read_fasta(p):
    """id -> sequence. Sequences are kept here; src/122 kept only lengths."""
    out, key, buf = {}, None, []
    with p.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(">"):
                if key is not None:
                    out[key] = "".join(buf)
                key = re.split(r"[\t ]", line[1:].rstrip("\n"), maxsplit=1)[0].strip()
                buf = []
            else:
                buf.append(line.strip())
    if key is not None:
        out[key] = "".join(buf)
    return out


def features(seqs):
    """log length, and the 20 amino-acid frequencies. No position, no order, no model."""
    n = np.array([len(s) for s in seqs], float)
    loglen = np.log(np.maximum(n, 1.0))[:, None]
    comp = np.array([[s.count(c) for c in AA] for s in seqs], float)
    comp /= np.maximum(n[:, None], 1.0)
    return loglen, np.hstack([loglen, comp])


def evaluate(X, y, seeds=SEEDS):
    """80:20 at each seed; DeepVIC states the ratio but does not publish the partition."""
    aus = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y))
        cut = int(len(y) * (1 - TEST_FRAC))
        tr, te = idx[:cut], idx[cut:]
        m = clf().fit(X[tr], y[tr])
        s = m.predict_proba(X[te])[:, 1]
        aus.append(auroc(s[y[te] == 1], s[y[te] == 0]))
    return float(np.mean(aus)), float(np.std(aus, ddof=1)), aus


def verdict(val, confound, clean):
    if val >= confound:
        return "confound band"
    if val <= clean:
        return "clean band"
    return "between bands"


def selftest():
    ok = True
    # auroc against hand-computable cases
    if auroc([3, 4, 5], [0, 1, 2]) != 1.0 or auroc([0, 1, 2], [3, 4, 5]) != 0.0:
        print("  XX auroc fails on a perfect separation")
        ok = False
    if abs(auroc([1, 2], [1, 2]) - 0.5) > 1e-12:
        print("  XX auroc on identical samples != 0.5")
        ok = False

    # features: a constant-composition pair must differ ONLY in the length column
    a, b = features(["AAAA", "AAAAAAAA"])
    if not (abs(a[1, 0] - a[0, 0] - np.log(2)) < 1e-12):
        print("  XX log length column is wrong")
        ok = False
    if not np.allclose(b[0, 1:], b[1, 1:]):
        print("  XX composition columns are not length-normalized")
        ok = False
    if b.shape[1] != 21:
        print(f"  XX feature width {b.shape[1]}, expected 21")
        ok = False
    # and a composition difference must be visible when lengths are equal
    c = features(["AAAA", "CCCC"])[1]
    if np.allclose(c[0, 1:], c[1, 1:]):
        print("  XX composition does not distinguish AAAA from CCCC")
        ok = False

    # evaluate must separate a planted signal and must NOT separate noise
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(400), np.zeros(400)]
    Xsig = np.r_[rng.normal(3, 1, (400, 1)), rng.normal(0, 1, (400, 1))]
    mu, _, _ = evaluate(Xsig, y, seeds=3)
    if mu < 0.9:
        print(f"  XX planted signal scored {mu:.3f}, expected > 0.9")
        ok = False
    Xnoise = rng.normal(0, 1, (800, 1))
    mu, _, _ = evaluate(Xnoise, y, seeds=3)
    if not (0.35 < mu < 0.65):
        print(f"  XX pure noise scored {mu:.3f}, expected near 0.5")
        ok = False

    # the bands must discriminate in all three directions
    for v, want in ((0.80, "confound band"), (0.55, "clean band"),
                    (0.68, "between bands")):
        if verdict(v, BAND_LEN_CONFOUND, BAND_LEN_CLEAN) != want:
            print(f"  XX {v} is not in the {want}")
            ok = False

    # and the composition measured by src/122 must still hold, or the data changed
    b122 = json.loads(BUILD.read_text())
    if b122["length"]["positives_median"] != 549 or \
       b122["length"]["negatives_median"] != 268:
        print("  XX the frozen median lengths no longer match the build")
        ok = False

    print("selftest PASS" if ok else "selftest FAIL")
    return 0 if ok else 1


def main():
    t0 = time.time()
    fa = read_fasta(FAA)
    b122 = json.loads(BUILD.read_text())
    with META.open(encoding="utf-8", errors="replace") as fh:
        head = fh.readline().rstrip("\n").split("\t")
        rows = [dict(zip(head, ln.rstrip("\n").split("\t"))) for ln in fh]
    pos_ids = {r["DLDB header/Internal Reference"] for r in rows}
    vfg_of = {}
    for r in rows:
        m = VFG_RE.search(r["Original Database header"] or "")
        if m:
            vfg_of[r["DLDB header/Internal Reference"]] = m.group(0)
    our_reps = {q["vfg"] for q in json.loads(OURS.read_text())["positives"]}
    shared = {k for k, v in vfg_of.items() if v in our_reps}

    res = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "study": "AI-1 -- DeepVIC's benchmark scored with no sequence semantics",
        "preregistration": "docs/DEEPVIC_BENCHMARK_PREREGISTRATION.md",
        "deepvic_published_auroc": DEEPVIC_PUBLISHED,
        "bands": {"length_confound_floor": BAND_LEN_CONFOUND,
                  "length_clean_ceiling": BAND_LEN_CLEAN,
                  "composition_confound_floor": BAND_COMP_CONFOUND,
                  "composition_clean_ceiling": BAND_COMP_CLEAN},
        "seeds": SEEDS,
        "test_fraction": TEST_FRAC,
        "n_shared_with_our_panel": len(shared),
        "median_length": b122["length"],
        "variants": {},
    }

    for variant in ("full", "shared_removed"):
        keys = [k for k in fa if not (variant == "shared_removed" and k in shared)]
        y = np.array([1.0 if k in pos_ids else 0.0 for k in keys])
        seqs = [fa[k] for k in keys]
        Xlen, Xcomp = features(seqs)
        mu_l, sd_l, all_l = evaluate(Xlen, y)
        mu_c, sd_c, all_c = evaluate(Xcomp, y)
        res["variants"][variant] = {
            "n": len(keys),
            "n_positives": int(y.sum()),
            "n_negatives": int((y == 0).sum()),
            "auroc_length_only": mu_l, "sd_length_only": sd_l,
            "auroc_length_plus_composition": mu_c, "sd_length_plus_composition": sd_c,
            "per_seed_length_only": all_l,
            "per_seed_length_plus_composition": all_c,
            "verdict_length_only": verdict(mu_l, BAND_LEN_CONFOUND, BAND_LEN_CLEAN),
            "verdict_composition": verdict(mu_c, BAND_COMP_CONFOUND, BAND_COMP_CLEAN),
            "gap_to_published": DEEPVIC_PUBLISHED - mu_c,
            "frac_of_published": mu_c / DEEPVIC_PUBLISHED,
        }
        v = res["variants"][variant]
        print(f"  {variant:<15} n={v['n']:,}  "
              f"length-only AUROC {mu_l:.4f} (sd {sd_l:.4f}) -> {v['verdict_length_only']}")
        print(f"  {'':<15} length+composition {mu_c:.4f} (sd {sd_c:.4f}) -> "
              f"{v['verdict_composition']}")
        print(f"  {'':<15} published 0.954; gap {v['gap_to_published']:+.4f}, "
              f"{v['frac_of_published']:.1%} of it", flush=True)

    F, S = res["variants"]["full"], res["variants"]["shared_removed"]
    res["ai4_shared_removal_moves"] = {
        "length_only_pp": (S["auroc_length_only"] - F["auroc_length_only"]) * 100,
        "composition_pp": (S["auroc_length_plus_composition"]
                           - F["auroc_length_plus_composition"]) * 100,
        "under_2pp": bool(
            abs(S["auroc_length_only"] - F["auroc_length_only"]) < 0.02
            and abs(S["auroc_length_plus_composition"]
                    - F["auroc_length_plus_composition"]) < 0.02),
    }
    res["predictions"] = {
        "P1_length_only_clears_0.75": {
            "kind": "derived", "value": F["auroc_length_only"],
            "held": bool(F["auroc_length_only"] >= BAND_LEN_CONFOUND)},
        "P2_composition_clears_0.85": {
            "kind": "derived", "value": F["auroc_length_plus_composition"],
            "held": bool(F["auroc_length_plus_composition"] >= BAND_COMP_CONFOUND)},
        "P5_shared_removal_under_2pp": {
            "kind": "reasoned", "value": res["ai4_shared_removal_moves"],
            "held": res["ai4_shared_removal_moves"]["under_2pp"]},
    }
    res["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print(f"\nAI-4 removal moves: length {res['ai4_shared_removal_moves']['length_only_pp']:+.3f}pp, "
          f"composition {res['ai4_shared_removal_moves']['composition_pp']:+.3f}pp")
    for k, v in res["predictions"].items():
        print(f"  {k:<36} {v['kind']:<9} held={v['held']}")
    print(f"wrote {OUT.relative_to(ROOT)} in {res['seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else main())
