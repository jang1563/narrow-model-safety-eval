#!/usr/bin/env python3
"""
95_pooling_artifact.py - is the short arm of the length-U a special-token artifact?

`docs/POOLING_ARTIFACT_PREREGISTRATION.md`. Study M found the flag rate is U-shaped in protein length
inside both strata -- intracellular 14.15% -> 1.69% -> 12.18%. The short arm has a mechanical
candidate: every embedding here is src/02b's mean over the FULL attention mask, so <cls> and <eos> are
averaged in and are a large share of the mean for a short protein. The displacement falls as 1/(L+2),
which the pool's two published arrays confirm at r = 0.977.

🔴 The comparison is within-pooling, not across: a residue-only embedding is a different
representation, so absolute rates move for reasons unrelated to the U. What is compared is the SHAPE --
U-index = rate(0-250) / rate(450-550) inside the intracellular stratum, 8.37 under include-specials.

🔒 The long arm is the control: the displacement shrinks with length, so a special-token mechanism
cannot explain it. If the long-arm index moves as much as the short-arm index, N-2 is uninterpretable.

⚠️ 650M only -- the pool has no residue-only array at 35M -- so this is indicative, not a verdict.

Usage:
    python src/95_pooling_artifact.py
    python src/95_pooling_artifact.py --selftest
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "v3"
POOL_JSON = ROOT / "data" / "sequences" / "_scaled_negative_pool.json"
BUILD = ROOT / "results" / "external_class_axis_build.json"
SCREEN = ROOT / "results" / "external_class_axis_screen.json"
OUT = ROOT / "results" / "pooling_artifact.json"
SEEDS, SPEC = 30, 0.95
SHORT, MID, LONG = (0, 250), (450, 550), (700, 10 ** 9)
BAND_FLOOR, U_LO, U_HI = 100, 2.0, 5.0
ARMS = {"include_specials": ("embeddings_class_axis_positives.npy",
                             "embeddings_pool_large_esm2_650M.npy"),
        "residues_only": ("embeddings_class_axis_positives_mean_res.npy",
                          "embeddings_pool_large_esm2_650M_mean_res.npy")}


def _load(stem):
    spec = importlib.util.spec_from_file_location(f"_{stem}", ROOT / "src" / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M92 = _load("92_localization_coverage")
M84 = M92.M84
M83 = _load("83_provenance_control")


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))


def selftest():
    for name, (p, q) in ARMS.items():
        for f in (p, q):
            if not (RES / f).exists():
                raise SystemExit(f"{name}: {f} is missing; run src/94 first")
    a = np.load(RES / ARMS["include_specials"][1], mmap_mode="r")
    b = np.load(RES / ARMS["residues_only"][1], mmap_mode="r")
    assert a.shape == b.shape, "the two pool arrays must be the same shape and row order"
    p = np.load(RES / ARMS["include_specials"][0], mmap_mode="r")
    q = np.load(RES / ARMS["residues_only"][0], mmap_mode="r")
    assert p.shape == q.shape, "the two positive arrays must be the same shape and row order"
    # 🔒 the two poolings must actually differ, or the study compares an array with itself
    d = float(np.abs(np.asarray(a[:200]) - np.asarray(b[:200])).max())
    assert d > 1e-4, f"the two pool arrays are identical to {d:.1e}; nothing to compare"
    print(f"selftest PASS (pool {a.shape}, positives {p.shape}, max |Δ| on 200 rows {d:.3e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    pool = json.loads(POOL_JSON.read_text())["proteins"]
    test = json.loads(BUILD.read_text())["pool_partition"]["test_rows"]
    accs = [pool[i]["uniprot"] for i in test]
    go, kw = M92.fetch_go(accs), M84.fetch_keywords(accs)
    elig = [j for j, i in enumerate(test) if pool[i]["kingdom"] in M84.KINGDOMS]
    strat = {s: [] for s in M92.STRATA}
    for j in elig:
        g = go[accs[j]]
        strat[M92.stratum_of(g["go"], g["sig"], g["tm"], kw[accs[j]])].append(j)
    ex, ins = strat["extracellular"], strat["intracellular"]
    lengths = np.array([pool[test[j]]["length"] for j in range(len(test))], float)
    band = {n: [j for j in ins if b[0] <= lengths[j] < b[1]]
            for n, b in (("short", SHORT), ("mid", MID), ("long", LONG))}
    print("intracellular length bands: " + ", ".join(f"{k} {len(v)}" for k, v in band.items()))
    floors_met = all(len(v) >= BAND_FLOOR for v in band.values())
    if not floors_met:
        print(f"  ⚠️  a band holds under {BAND_FLOOR} proteins")

    vf = M83.vfdb_sequences()
    union = [i for i in json.loads(SCREEN.read_text())["admitted_rows"]
             if pool[i]["sequence"] not in vf]
    n_tr, n_ca = M92.fold_sizes(len(union))
    rows = np.array(test)

    out = {}
    for name, (pf, qf) in ARMS.items():
        P, POOL = np.load(RES / pf), np.load(RES / qf)
        y = np.r_[np.ones(P.shape[0]), np.zeros(n_tr)]
        acc = {k: [] for k in ("short", "mid", "long", "ex", "in")}
        t0 = time.time()
        for seed in range(a.seeds):
            r = np.random.default_rng(seed)
            perm = r.permutation(len(union))
            tr = [union[i] for i in perm[:n_tr]]
            ca = [union[i] for i in perm[n_tr:n_tr + n_ca]]
            model = clf().fit(np.vstack([P, POOL[tr]]), y)
            t = float(np.quantile(model.predict_proba(POOL[ca])[:, 1], SPEC))
            flag = model.predict_proba(POOL[rows])[:, 1] >= t
            for k, v in band.items():
                acc[k].append(float(flag[v].mean()))
            acc["ex"].append(float(flag[ex].mean()))
            acc["in"].append(float(flag[ins].mean()))
        m = {k: float(np.mean(v)) for k, v in acc.items()}
        out[name] = {
            "rates": m,
            "U_index": m["short"] / m["mid"] if m["mid"] > 0 else float("nan"),
            "long_index": m["long"] / m["mid"] if m["mid"] > 0 else float("nan"),
            "R": m["ex"] / m["in"] if m["in"] > 0 else float("nan"),
            "seconds": round(time.time() - t0),
        }
        print(f"  {name:<18} short {m['short'] * 100:6.2f}%  mid {m['mid'] * 100:5.2f}%  "
              f"long {m['long'] * 100:6.2f}%   U {out[name]['U_index']:6.2f}  "
              f"long-idx {out[name]['long_index']:6.2f}  R {out[name]['R']:.2f}")

    # ---- 🔑 why it did or did not move: the displacement's size and its angle to the probe --------
    A = np.load(RES / ARMS["include_specials"][1])
    B = np.load(RES / ARMS["residues_only"][1])
    disp = np.linalg.norm(A - B, axis=1)
    norm = np.linalg.norm(A, axis=1)
    P0 = np.load(RES / ARMS["include_specials"][0])
    y0 = np.r_[np.ones(P0.shape[0]), np.zeros(n_tr)]
    r0 = np.random.default_rng(0)
    tr0 = [union[i] for i in r0.permutation(len(union))[:n_tr]]
    pipe = clf().fit(np.vstack([P0, A[tr0]]), y0)
    w = pipe.named_steps["logisticregression"].coef_[0] / pipe.named_steps["standardscaler"].scale_
    w = w / np.linalg.norm(w)
    proj = np.abs((A - B) @ w)
    scores = A @ w
    iqr = float(np.percentile(scores, 75) - np.percentile(scores, 25))
    geometry = {
        "median_embedding_norm": float(np.median(norm)),
        "median_displacement": float(np.median(disp)),
        "displacement_pct_of_norm": float(np.median(disp / norm) * 100),
        "displacement_pct_of_norm_short": float(
            np.median((disp / norm)[np.array([min(q["length"], 1022) for q in pool]) < 250]) * 100),
        # 🔑 the displacement is nearly ORTHOGONAL to what the probe reads, which is the answer
        "projection_pct_of_displacement": float(np.median(proj) / np.median(disp) * 100),
        "score_shift_pct_of_iqr": float(np.median(proj) / iqr * 100),
    }
    print(f"\n  displacement is {geometry['displacement_pct_of_norm']:.2f}% of the embedding norm "
          f"({geometry['displacement_pct_of_norm_short']:.2f}% for proteins under 250), and only "
          f"{geometry['projection_pct_of_displacement']:.1f}% of it lies along the probe's decision "
          f"direction -> a score shift of {geometry['score_shift_pct_of_iqr']:.2f}% of an IQR")

    inc, res = out["include_specials"], out["residues_only"]
    verdict = ("the short arm was substantially a special-token artifact" if res["U_index"] <= U_LO
               else "not a pooling artifact" if res["U_index"] >= U_HI else "partial")
    u_move = abs(res["U_index"] - inc["U_index"]) / inc["U_index"]
    l_move = abs(res["long_index"] - inc["long_index"]) / inc["long_index"]
    interpretable = l_move < u_move
    result = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "arm": "esm2_650M",
              "single_arm_indicative": True, "seeds": a.seeds,
              "band_n": {k: len(v) for k, v in band.items()}, "floors_met": bool(floors_met),
              "arms": out, "band_thresholds": [U_LO, U_HI], "verdict": verdict,
              "U_index_relative_move": u_move, "long_index_relative_move": l_move,
              "geometry": geometry,
              # 🔴 the control can only discriminate if something MOVED. Neither index did, so it is
              # uninformative here rather than passed, and saying "passed" would be a false comfort.
              "control_informative": bool(max(u_move, l_move) > 0.05),
              "control_holds": bool(interpretable),
              "R_supported": bool(res["R"] >= 1.5)}
    OUT.write_text(json.dumps(result, indent=2) + "\n")

    print(f"\nN-2  U-index {inc['U_index']:.2f} -> {res['U_index']:.2f}  ->  {verdict}")
    ctrl = ("UNINFORMATIVE — neither index moved, so the control has nothing to discriminate"
            if max(u_move, l_move) <= 0.05
            else "interpretable" if interpretable else "UNINTERPRETABLE")
    print(f"     control: long-arm index {inc['long_index']:.2f} -> {res['long_index']:.2f}  "
          f"(moved {l_move * 100:.1f}% against the short arm's {u_move * 100:.1f}%) -> {ctrl}")
    print(f"     R under residue-only pooling: {res['R']:.2f} "
          f"({'supports the probe' if result['R_supported'] else 'COLLAPSED'})")
    print("\n⚠️  650M only, so this is indicative and not a verdict.")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
