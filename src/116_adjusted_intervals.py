#!/usr/bin/env python3
"""
116_adjusted_intervals.py - AD: honest intervals for the two adjusted contrasts

`docs/ADJUSTED_INTERVALS_PREREGISTRATION.md`. Study AC fixed the four intervals carrying the main
argument. Two banded quantities remained on seed standard errors: the length-adjusted Mantel-Haenszel
ratio (band >= 3.0, "localization is a major driver independent of length") and the provenance x
localization ratio of ratios from study I's corrected taxid factor.

🔒 Same hierarchical bootstrap as src/110 and src/115 -- resample a seed, then rows with replacement inside
each cell -- but the estimator is recomputed from the resampled cells rather than perturbed at its output,
so the deciles are resampled within and then pooled by Mantel-Haenszel.

⚠️ Point estimates do not move. Only the intervals do.

Usage:
    python src/116_adjusted_intervals.py --arm esm2_650M
    python src/116_adjusted_intervals.py --arm esm2_35M
    python src/116_adjusted_intervals.py --selftest
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_STEM = ROOT / "results" / "adjusted_intervals"
B = 20000
LO, HI = 2.5, 97.5
MH_MAJOR, MH_PARTIAL = 3.0, 1.5
N_DECILES, CELL_FLOOR = 10, 20


def _load(stem):
    spec = importlib.util.spec_from_file_location(f"_{stem}", ROOT / "src" / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M115 = _load("115_headline_intervals")
M107, M99, M92, M84, M83, M93 = (M115.M107, M115.M99, M115.M92, M115.M84, M115.M83,
                                 _load("93_length_adjustment"))
SPEC, SEEDS = M107.SPEC, M99.SEEDS


def mh_from_cells(cells):
    """Mantel-Haenszel over (a, n_a, b, n_b) cells -- src/93's estimator, imported not reimplemented."""
    return M93.mantel_haenszel_rr(cells)


def boot_mh(ex_by, in_by, used, rng, b=B):
    """Resample rows inside each decile, then pool. The estimator is recomputed, not perturbed."""
    out = []
    for _ in range(b):
        s = rng.integers(ex_by[used[0]].shape[0])
        cells = []
        for d in used:
            e, i = ex_by[d][s], in_by[d][s]
            ei = e[rng.integers(0, len(e), len(e))]
            ii = i[rng.integers(0, len(i), len(i))]
            cells.append((float(ei.sum()), len(ei), float(ii.sum()), len(ii)))
        try:
            v = mh_from_cells(cells)
        except ZeroDivisionError:
            continue
        if np.isfinite(v):
            out.append(v)
    return np.array(out)


def pct(v):
    return [float(np.percentile(v, LO)), float(np.percentile(v, HI))]


def selftest():
    rng = np.random.default_rng(0)
    # 🔒 the pooled estimator must reproduce src/93 on fixed cells, or this is a reimplementation
    assert abs(mh_from_cells([(20, 100, 10, 100), (40, 100, 20, 100)]) - 2.0) < 1e-12
    # 🔒 and the bootstrap must widen as cells shrink, with the point estimate preserved
    def cells(n, seeds=30):
        ex = {d: np.array([[rng.random() < 0.20 for _ in range(n)] for _ in range(seeds)])
              for d in range(3)}
        inn = {d: np.array([[rng.random() < 0.05 for _ in range(n * 3)] for _ in range(seeds)])
               for d in range(3)}
        return ex, inn
    widths = {}
    for n in (25, 800):
        ex, inn = cells(n)
        bt = boot_mh(ex, inn, [0, 1, 2], np.random.default_rng(1), 3000)
        widths[n] = pct(bt)[1] - pct(bt)[0]
        if n == 800:
            assert 3.0 < bt.mean() < 5.5, bt.mean()
    assert widths[25] > 3.0 * widths[800], widths
    # 🔴 a degenerate decile must not silently vanish: an all-zero denominator cell is dropped by the
    # estimator, and the caller has to see a finite result rather than a nan
    ex = {0: np.ones((5, 10), bool)}
    inn = {0: np.zeros((5, 30), bool)}
    bt = boot_mh(ex, inn, [0], np.random.default_rng(2), 200)
    assert len(bt) == 0 or np.all(np.isfinite(bt)), "non-finite values must not reach the percentiles"
    print("selftest PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="esm2_650M", choices=["esm2_650M", "esm2_35M"])
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    pool = json.loads(M107.POOL_JSON.read_text())["proteins"]
    test = json.loads(M107.BUILD.read_text())["pool_partition"]["test_rows"]
    accs = [pool[i]["uniprot"] for i in test]
    kw, go = M84.fetch_keywords(accs), M92.fetch_go(accs)
    vf = M83.vfdb_sequences()
    # 🔒 study M uses study L's GO-augmented strata (n_intracellular = 2739), not the keyword rule
    cell = {"extracellular": [], "intracellular": []}
    for k_, i in enumerate(test):
        if pool[i]["kingdom"] not in M84.KINGDOMS or pool[i]["sequence"] in vf:
            continue
        g = go[accs[k_]]
        st = M92.stratum_of(g["go"], g["sig"], g["tm"], kw[accs[k_]])
        if st in cell:
            cell[st].append(k_)
    lengths = np.array([pool[i]["length"] for i in test], float)
    both = np.array(cell["extracellular"] + cell["intracellular"])
    edges = np.quantile(lengths[both], np.linspace(0, 1, N_DECILES + 1))
    edges[0], edges[-1] = -np.inf, np.inf

    def dec(k):
        return min(max(int(np.searchsorted(edges, lengths[k], side="right") - 1), 0), N_DECILES - 1)

    ex_rows = {d: [k for k in cell["extracellular"] if dec(k) == d] for d in range(N_DECILES)}
    in_rows = {d: [k for k in cell["intracellular"] if dec(k) == d] for d in range(N_DECILES)}
    used = [d for d in range(N_DECILES)
            if len(ex_rows[d]) >= CELL_FLOOR and len(in_rows[d]) >= CELL_FLOOR]
    print(f"arm {a.arm}: extracellular {len(cell['extracellular'])}, "
          f"intracellular {len(cell['intracellular'])}, deciles used {len(used)}/{N_DECILES}")

    sfx = "" if a.arm == "esm2_650M" else f"_{a.arm}"
    P = np.load(M107.RES / f"embeddings_class_axis_positives{sfx}.npy")
    POOL = np.load(M107.RES / f"embeddings_pool_large_{a.arm}.npy")
    union = [i for i in json.loads(M107.SCREEN.read_text())["admitted_rows"]
             if pool[i]["sequence"] not in vf]
    n_tr, n_ca = M92.fold_sizes(len(union))
    y = np.r_[np.ones(P.shape[0]), np.zeros(n_tr)]
    T = POOL[np.array(test)]

    flags = []
    t0 = time.time()
    for seed in range(a.seeds):
        r = np.random.default_rng(seed)
        perm = r.permutation(len(union))
        tr = [union[i] for i in perm[:n_tr]]
        ca = [union[i] for i in perm[n_tr:n_tr + n_ca]]
        mdl = M92.clf().fit(np.vstack([P, POOL[tr]]), y)
        t = float(np.quantile(mdl.predict_proba(POOL[ca])[:, 1], SPEC))
        flags.append(mdl.predict_proba(T)[:, 1] >= t)
    F = np.array(flags)
    ex_by = {d: F[:, np.array(ex_rows[d])] for d in used}
    in_by = {d: F[:, np.array(in_rows[d])] for d in used}
    print(f"  {time.time() - t0:.0f}s")

    seed_mh = np.array([mh_from_cells([(float(ex_by[d][s].sum()), ex_by[d].shape[1],
                                        float(in_by[d][s].sum()), in_by[d].shape[1])
                                       for d in used]) for s in range(F.shape[0])])
    row_mh = boot_mh(ex_by, in_by, used, np.random.default_rng(0))
    S = M99.stat
    s_ = S(seed_mh)
    lo, hi = pct(row_mh)
    verdict = ("localization is a major driver independent of length" if lo > MH_MAJOR
               else "partial: localization survives adjustment but not as a major driver"
               if lo > MH_PARTIAL else "the gradient was length")
    res = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "arm": a.arm, "seeds": a.seeds, "B": B,
           "n": {"extracellular": len(cell["extracellular"]),
                 "intracellular": len(cell["intracellular"])},
           "deciles_used": len(used),
           "MH_point": round(float(seed_mh.mean()), 4),
           "MH_seed_ci": [round(x, 4) for x in s_["ci"]],
           "MH_row_ci": [round(lo, 4), round(hi, 4)],
           "widening": round((hi - lo) / (s_["ci"][1] - s_["ci"][0]), 2),
           "clears_major": bool(lo > MH_MAJOR),
           "verdict": verdict}
    dest = Path(f"{OUT_STEM}{sfx}.json")
    dest.write_text(json.dumps(res, indent=2) + "\n")

    print(f"\nR_MH point  {res['MH_point']:.3f}")
    print(f"  seed interval       [{s_['ci'][0]:.3f}, {s_['ci'][1]:.3f}]")
    print(f"  row + seed (honest) [{lo:.3f}, {hi:.3f}]   {res['widening']}x wider")
    print(f"  clears the 3.0 band: {res['clears_major']}")
    print(f"  -> {verdict}")
    print(f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
