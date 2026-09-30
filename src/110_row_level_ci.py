#!/usr/bin/env python3
"""
110_row_level_ci.py - is the interval on R a seed interval or a row interval?

A review check, not a new study. `stat()` in this repository returns mean +/- 1.96 * SE over probe
SEEDS on a FIXED row set. That answers "where is the mean of the seed distribution", not "how uncertain
is R". Studies Y and Z cross-fit the localization axis, which cut their measurement strata to 73 and 98
extracellular rows, so row sampling dominates there and the published interval understates it.

🔒 This resamples hierarchically: a seed, then rows WITH replacement within each stratum. The comparison
to the seed-only interval is the point, so both are reported side by side.

⚠️ Earlier studies are much less affected and this does not retract them: study E measures R on the full
652-1,666 strata, where row uncertainty is a few percent. The acute case is the cross-fitted one.

Usage:
    python src/110_row_level_ci.py --arm esm2_650M
    python src/110_row_level_ci.py --arm esm2_35M
    python src/110_row_level_ci.py --selftest
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_STEM = ROOT / "results" / "row_level_ci"
B = 4000
LO, HI = 2.5, 97.5


def _load(stem):
    spec = importlib.util.spec_from_file_location(f"_{stem}", ROOT / "src" / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M109 = _load("109_leace_erasure")
M107, M99, M92, M84, M83 = M109.M107, M109.M99, M109.M92, M109.M84, M109.M83


def hier_boot(cells, rng, b=B):
    """Resample a (seed, fold) cell, then rows with replacement inside each stratum."""
    out = []
    for _ in range(b):
        fe, fi = cells[rng.integers(len(cells))]
        a = fe[rng.integers(0, len(fe), len(fe))].mean()
        d = fi[rng.integers(0, len(fi), len(fi))].mean()
        if d > 0:
            out.append(a / d)
    return np.array(out)


def pct(v):
    return {"mean": float(np.mean(v)), "ci": [float(np.percentile(v, LO)),
                                              float(np.percentile(v, HI))]}


def selftest():
    rng = np.random.default_rng(0)
    # 🔒 the property under test: with FEW rows the row interval must be materially wider than the
    # seed interval, and with MANY rows the two must converge. A bootstrap that ignored rows would
    # pass the second and fail the first.
    # 🔴 The first version of this test drew NEW rows for every seed, so its "seed interval" already
    # contained the row variation it was supposed to lack, and the comparison measured nothing. The
    # real design fixes the rows once and varies only the probe, so the synthetic must too.
    def design(n_ex, n_in, seeds=60):
        """Fixed rows with fixed latent scores; each seed moves the threshold slightly."""
        s_ex = rng.normal(1.0, 1.0, n_ex)
        s_in = rng.normal(-0.4, 1.0, n_in)
        cells = []
        for _ in range(seeds):
            t = 1.40 + rng.normal(0, 0.02)             # the only thing a seed changes
            cells.append(((s_ex >= t).astype(float), (s_in >= t).astype(float)))
        return cells

    widths = {}
    for n_ex, n_in in ((75, 300), (3000, 12000)):
        cells = design(n_ex, n_in)
        seed_R = np.array([fe.mean() / fi.mean() for fe, fi in cells if fi.mean() > 0])
        row_R = hier_boot(cells, np.random.default_rng(1), 3000)
        widths[(n_ex)] = (np.percentile(seed_R, HI) - np.percentile(seed_R, LO),
                          np.percentile(row_R, HI) - np.percentile(row_R, LO))
    # 🔴 "the two converge at large n" was my second wrong guess here, and measuring it killed that
    # too: the seed interval is set by threshold jitter and is BLIND to row count, while the row
    # interval scales as 1/sqrt(n). They do not converge; the ratio just falls from ~18 to ~2. The
    # three properties below are what is actually true, and each is checked.
    # (a) at few rows the row interval dominates -- the defect, in the regime studies Y and Z are in
    assert widths[75][1] > 2.0 * widths[75][0], widths[75]
    # (b) the row interval responds to row count, so the bootstrap is measuring rows and not noise
    assert widths[3000][1] < 0.3 * widths[75][1], (widths[75][1], widths[3000][1])
    # (c) and the seed interval does NOT respond to it, which is why it cannot stand in for one
    assert 0.4 < widths[3000][0] / widths[75][0] < 2.5, (widths[75][0], widths[3000][0])
    # 🔴 and `stat()`'s SE-of-the-mean must be narrower than the seed SPREAD, which is the defect
    cells = design(75, 300)
    sr = np.array([fe.mean() / fi.mean() for fe, fi in cells if fi.mean() > 0])
    se = M99.stat(sr)
    assert (se["ci"][1] - se["ci"][0]) < 0.6 * (np.percentile(sr, HI) - np.percentile(sr, LO)), \
        "stat()'s interval must be demonstrably narrower than the seed spread"
    print("selftest PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="esm2_650M", choices=["esm2_650M", "esm2_35M"])
    ap.add_argument("--seeds", type=int, default=M99.SEEDS)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    pool = json.loads(M107.POOL_JSON.read_text())["proteins"]
    test = json.loads(M107.BUILD.read_text())["pool_partition"]["test_rows"]
    accs = [pool[i]["uniprot"] for i in test]
    kw, go = M84.fetch_keywords(accs), M92.fetch_go(accs)
    vf, sp_ok = M83.vfdb_sequences(), M83.vfdb_species()
    bcell = {"extracellular": [], "intracellular": []}
    for k_, i in enumerate(test):
        sp = " ".join(pool[i]["organism"].replace("(", "").split()[:2])
        if pool[i]["kingdom"] != "Bacteria" or sp not in sp_ok or pool[i]["sequence"] in vf:
            continue
        st = M84.stratum_of(kw[pool[i]["uniprot"]])
        if st in bcell:
            bcell[st].append(k_)
    axis = {"extracellular": [], "intracellular": []}
    for k_, i in enumerate(test):
        if pool[i]["kingdom"] not in M84.KINGDOMS or pool[i]["sequence"] in vf:
            continue
        g = go[accs[k_]]
        st = M92.stratum_of(g["go"], g["sig"], g["tm"], kw[accs[k_]])
        if st in axis:
            axis[st].append(k_)

    sfx = "" if a.arm == "esm2_650M" else f"_{a.arm}"
    P = np.load(M107.RES / f"embeddings_class_axis_positives{sfx}.npy")
    POOL = np.load(M107.RES / f"embeddings_pool_large_{a.arm}.npy")
    union = [i for i in json.loads(M107.SCREEN.read_text())["admitted_rows"]
             if pool[i]["sequence"] not in vf]
    n_tr, n_ca = M92.fold_sizes(len(union))
    y = np.r_[np.ones(P.shape[0]), np.zeros(n_tr)]
    T = POOL[np.array(test)]
    lab_rows = np.array(axis["extracellular"] + axis["intracellular"])
    lab = np.r_[np.ones(len(axis["extracellular"])),
                np.zeros(len(axis["intracellular"]))].astype(int)
    order = np.random.default_rng(M107.FOLD_SEED).permutation(len(lab_rows))
    halves = [order[:len(order) // 2], order[len(order) // 2:]]
    key_ex, key_in = set(bcell["extracellular"]), set(bcell["intracellular"])

    cells, strata = [], []
    t0 = time.time()
    for f in (0, 1):
        fit = halves[f]
        Xf, yf = T[lab_rows[fit]], lab[fit]
        A, mu = M109.leace(POOL, Xf, yf)
        fitted = set(lab_rows[fit].tolist())
        m_ex = np.array(sorted(key_ex - fitted))
        m_in = np.array(sorted(key_in - fitted))
        strata.append([len(m_ex), len(m_in)])
        Pe, POOLe, Te = (M109.erase(x, A, mu) for x in (P, POOL, T))
        for seed in range(a.seeds):
            r = np.random.default_rng(seed)
            perm = r.permutation(len(union))
            tr = [union[i] for i in perm[:n_tr]]
            ca = [union[i] for i in perm[n_tr:n_tr + n_ca]]
            mdl = M99.clf().fit(np.vstack([Pe, POOLe[tr]]), y)
            t = float(np.quantile(mdl.predict_proba(POOLe[ca])[:, 1], M107.SPEC))
            fb = mdl.predict_proba(Te)[:, 1] >= t
            cells.append((fb[m_ex].astype(float), fb[m_in].astype(float)))
    print(f"  {time.time() - t0:.0f}s   measurement strata per fold: {strata}")

    seed_R = np.array([fe.mean() / fi.mean() for fe, fi in cells if fi.mean() > 0])
    se = M99.stat(seed_R)
    sp = pct(seed_R)
    row = pct(hier_boot(cells, np.random.default_rng(0)))
    w = {k: round(v["ci"][1] - v["ci"][0], 4) for k, v in
         (("stat_se", se), ("seed_pct", sp), ("row_seed", row))}
    res = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "arm": a.arm, "seeds": a.seeds,
           "B": B, "strata_per_fold": strata,
           "n_extracellular_measured": sum(s[0] for s in strata),
           "published_stat_interval": {"mean": round(se["mean"], 4),
                                       "ci": [round(x, 4) for x in se["ci"]]},
           "seed_percentile_interval": {"mean": round(sp["mean"], 4),
                                        "ci": [round(x, 4) for x in sp["ci"]]},
           "row_and_seed_interval": {"mean": round(row["mean"], 4),
                                     "ci": [round(x, 4) for x in row["ci"]]},
           "widths": w, "widening_factor": round(w["row_seed"] / w["stat_se"], 3),
           "excludes_1_5": bool(row["ci"][0] > 1.5),
           "excludes_3_0": bool(row["ci"][0] > 3.0)}
    dest = Path(f"{OUT_STEM}{sfx}.json")
    dest.write_text(json.dumps(res, indent=2) + "\n")
    print(f"\n  stat() as published   {se['mean']:.3f} [{se['ci'][0]:.3f}, {se['ci'][1]:.3f}]"
          f"   width {w['stat_se']:.3f}")
    print(f"  seed percentiles      {sp['mean']:.3f} [{sp['ci'][0]:.3f}, {sp['ci'][1]:.3f}]"
          f"   width {w['seed_pct']:.3f}")
    print(f"  row + seed (honest)   {row['mean']:.3f} [{row['ci'][0]:.3f}, {row['ci'][1]:.3f}]"
          f"   width {w['row_seed']:.3f}   ({res['widening_factor']}x)")
    print(f"  excludes 1.5: {res['excludes_1_5']}    excludes 3.0: {res['excludes_3_0']}")
    print(f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
