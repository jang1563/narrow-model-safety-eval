#!/usr/bin/env python3
"""Study AW: does the geometry let you do anything?

`docs/MARGIN_ROUTING_PREREGISTRATION.md`, frozen before this file existed.

🔑 Every result in this project so far MEASURES. This one asks whether any of it is usable:
cluster the positives without labels, score each cluster with the same margin formula, give a
fixed budget of experts to the LOWEST-margin clusters, and see whether that beats giving the
same budget to random clusters.

🔴 The project's own evidence predicts failure. Removing the benign neighbours the geometry
implicates repairs 8-11% of a failure it predicts at rho = +0.894, so prediction and repair
dissociate -- and a margin-routed mixture is a repair built on a prediction. P1 says AW-2
lands in the "does not" band.

🔒 Nothing here rests on a per-query HAZARD claim; the member-level margin failed externally
and may not be relied on. Routing asks where a query sits, not what it is, and the
allocation decision is made once, offline, before any query exists.

🔒 The comparison is against study AP's TUNED single probe. An unexamined C = 1.0 cost up to
24 points per category, so beating the untuned probe would be beating a corrected bug.
"""
import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results/v3"
OUT = ROOT / "results/margin_routing.json"
ARMS = {"esm2_650M": "", "esm2_35M": "_esm2_35M"}
STRATEGIES = ("margin", "random", "size", "none")
K_CLUSTERS, M_EXPERTS, SEEDS, FLOOR = 7, 3, 30, 7
AW1_OK, AW1_NOT = 0.05, 0.01
AW2_OK, AW2_NOT = 0.03, 0.01


def load(mod):
    spec = importlib.util.spec_from_file_location(mod[:4], ROOT / "src" / mod)
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod[:4]] = m
    spec.loader.exec_module(m)
    return m


def cluster_margins(X, pos_rows, labels, union, m30):
    """margin(cluster), the same formula with clusters in place of curator categories.

    🔒 Computed on the TRAINED positives only, so the held-out category contributes nothing
    to the allocation decision.
    """
    P = X[pos_rows]
    mo, mn = np.empty(len(P)), np.empty(len(P))
    CH = 1024
    NEGU = X[union]
    for i in range(0, len(P), CH):
        sl = slice(i, min(i + CH, len(P)))
        spp = m30.cos(P[sl], P)
        for r in range(spp.shape[0]):
            spp[r, i + r] = -np.inf
        spp[labels[sl][:, None] == labels[None, :]] = -np.inf
        mo[sl] = spp.max(1)
        mn[sl] = m30.cos(P[sl], NEGU).max(1)
    return {int(c): float((mo[labels == c] - mn[labels == c]).mean())
            for c in sorted(set(labels.tolist()))}


def allocate(strategy, margins, sizes, m, rng):
    """Which clusters get their own expert. 🔒 `none` is the single-probe baseline."""
    cl = sorted(margins)
    if strategy == "none":
        return []
    if strategy == "margin":
        return sorted(cl, key=lambda c: margins[c])[:m]
    if strategy == "size":
        return sorted(cl, key=lambda c: -sizes[c])[:m]
    if strategy == "random":
        return sorted(rng.choice(cl, size=min(m, len(cl)), replace=False).tolist())
    raise SystemExit(f"unknown strategy {strategy!r}")


def mixture_fit(X, experts, global_rows_pos, global_neg, expert_rows, m133, cv):
    """Fit the global probe and one probe per expert cluster. 🔒 ONE fit per probe per seed.

    🔴 The first version refitted every probe for each of the three row sets it had to score
    -- calibrate, held-out and test -- which tripled the cost of the most expensive study in
    the project for nothing. Fit once, score three times.
    """
    gm = m133.clf(cv).fit(np.vstack([X[global_rows_pos], X[global_neg]]),
                          np.r_[np.ones(len(global_rows_pos)), np.zeros(len(global_neg))])
    ems = {}
    for c in experts:
        tr_pos = expert_rows[c]
        if len(tr_pos) < FLOOR:
            continue
        ems[c] = m133.clf(cv).fit(
            np.vstack([X[tr_pos], X[global_neg]]),
            np.r_[np.ones(len(tr_pos)), np.zeros(len(global_neg))])
    return gm, ems, (1 + len(ems)) * X.shape[1]


def mixture_score(X, gm, ems, assign, score_rows):
    """Route every row to the expert of its own cluster; otherwise the global probe.

    🔒 The scores are assembled into one vector so the threshold is taken on the MIXTURE's
    own output, which is what keeps the operating point matched to every other study here.
    """
    out = np.empty(len(score_rows))
    routed = assign[score_rows]
    todo = np.ones(len(score_rows), bool)
    for c, em in ems.items():
        sel = routed == c
        if sel.any():
            out[sel] = em.predict_proba(X[score_rows[sel]])[:, 1]
            todo &= ~sel
    if todo.any():
        out[todo] = gm.predict_proba(X[score_rows[todo]])[:, 1]
    return out


def run_arm(arm, seeds, jobs, m30, m133):
    from sklearn.cluster import KMeans
    sfx = ARMS[arm]
    E = np.load(RES / f"embeddings_deepvic{sfx}.npy").astype(np.float64)
    man = json.loads((RES / f"embedding_manifest_deepvic{sfx}.json").read_text())
    lab, cls = np.array(man["label_of_row"]), np.array(man["class_of_row"])
    pos, neg = np.where(lab == 1)[0], np.where(lab == 0)[0]
    perm = np.random.default_rng(12345).permutation(len(neg))
    cut = 2 * len(neg) // 3
    union, test = neg[perm[:cut]], neg[perm[cut:]]
    counts = Counter(cls[pos])
    cats = sorted(c for c, n in counts.items() if n >= FLOOR)
    print(f"  {arm}: {len(pos)} positives, {len(cats)} categories, "
          f"{len(union)} union / {len(test)} test", flush=True)

    rec = {s: {} for s in STRATEGIES}
    fpr = {s: [] for s in STRATEGIES}
    params = {s: [] for s in STRATEGIES}
    purity, picked = [], {s: [] for s in STRATEGIES}
    for ci, cat in enumerate(cats):
        hi = pos[cls[pos] == cat]
        tri = pos[cls[pos] != cat]
        # 🔒 clustering and allocation see only the TRAINED positives
        km = KMeans(n_clusters=K_CLUSTERS, n_init=10, random_state=0).fit(E[tri])
        labels = km.labels_
        margins = cluster_margins(E, tri, labels, union, m30)
        sizes = {int(c): int((labels == c).sum()) for c in margins}
        expert_rows = {int(c): tri[labels == c] for c in margins}
        # cluster-to-category purity, for P3
        for c in margins:
            mem = cls[tri[labels == c]]
            purity.append(Counter(mem).most_common(1)[0][1] / max(1, len(mem)))
        # every row's cluster, for routing
        assign = np.full(len(E), -1)
        assign[tri] = labels
        assign[hi] = km.predict(E[hi])
        assign[neg] = km.predict(E[neg])
        cv, _ = m133.select_C(E, tri, union, ci)
        for strat in STRATEGIES:
            rng = np.random.default_rng(1000 + ci)
            experts = allocate(strat, margins, sizes, M_EXPERTS, rng)
            picked[strat].append([int(c) for c in experts])
            rr, ff, pp = [], [], []
            for seed in range(seeds):
                r = np.random.default_rng(seed)
                q = r.permutation(len(union))
                tr = union[q[:m133.N_TRAIN]]
                ca = union[q[m133.N_TRAIN:m133.N_TRAIN + m133.N_CAL]]
                gm, ems, npar = mixture_fit(E, experts, tri, tr, expert_rows, m133, cv)
                t = float(np.quantile(mixture_score(E, gm, ems, assign, ca), m133.SPEC))
                rr.append(float((mixture_score(E, gm, ems, assign, hi) >= t).mean()))
                ff.append(float((mixture_score(E, gm, ems, assign, test) >= t).mean()))
                pp.append(npar)
            rec[strat][cat] = {"n": int(len(hi)), "mean": float(np.mean(rr)),
                               "sd": float(np.std(rr, ddof=1))}
            fpr[strat].append(float(np.mean(ff)))
            params[strat].append(int(np.mean(pp)))
        print(f"    {cat:<46}" + "  ".join(
            f"{s} {rec[s][cat]['mean']*100:5.1f}%" for s in STRATEGIES), flush=True)

    means = {s: float(np.mean([rec[s][c]["mean"] for c in cats])) for s in STRATEGIES}
    return {
        "arm": arm, "n_categories": len(cats), "k_clusters": K_CLUSTERS,
        "m_experts": M_EXPERTS, "seeds": seeds,
        "recovery": rec, "mean_recovery": means,
        "test_fpr": {s: {"min": float(min(fpr[s])), "max": float(max(fpr[s]))}
                     for s in STRATEGIES},
        "n_params": {s: int(np.mean(params[s])) for s in STRATEGIES},
        "clusters_picked": picked,
        "mean_cluster_purity": float(np.mean(purity)),
        "AW1_margin_minus_none": means["margin"] - means["none"],
        "AW2_margin_minus_random": means["margin"] - means["random"],
        "AW3_margin_minus_size": means["margin"] - means["size"],
    }


def band(d, ok, not_):
    return "works" if d >= ok else "does not" if d <= not_ else "between bands"


def verdict(arms):
    if len(arms) < 2:
        return {"verdict": f"INCOMPLETE -- {len(arms)} of 2 arms; both must agree"}
    b = {k: {"AW1": band(v["AW1_margin_minus_none"], AW1_OK, AW1_NOT),
             "AW2": band(v["AW2_margin_minus_random"], AW2_OK, AW2_NOT),
             "AW3": band(v["AW3_margin_minus_size"], AW2_OK, AW2_NOT)}
         for k, v in arms.items()}
    agree = {t: len({b[a][t] for a in b}) == 1 for t in ("AW1", "AW2", "AW3")}
    out = {"per_arm": b, "arms_agree": agree,
           "AW2_is_the_one_that_matters": (
               "AW-1 can be won by having more parameters; only AW-2 isolates whether "
               "margin picked the right clusters")}
    if not all(agree.values()):
        out["verdict"] = ("ARMS DISAGREE -- " + ", ".join(
            f"{t}: " + "/".join(b[a][t] for a in sorted(b))
            for t in ("AW1", "AW2", "AW3") if not agree[t]))
    else:
        aw2 = next(iter(b.values()))["AW2"]
        out["verdict"] = (
            "routing works" if aw2 == "works"
            else "the geometry predicts and does not repair" if aw2 == "does not"
            else "AW-2 between bands")
    return out


def selftest():
    ok = True
    m30, m133 = load("30_margin_across_arms.py"), load("133_estimation_repair.py")
    for fn in ("cos", "_spearman"):
        if not hasattr(m30, fn):
            print(f"  XX src/30 exposes no {fn}")
            ok = False
    for fn in ("select_C", "clf", "N_TRAIN", "N_CAL", "SPEC"):
        if not hasattr(m133, fn):
            print(f"  XX src/133 exposes no {fn}")
            ok = False
    if (K_CLUSTERS, M_EXPERTS) != (7, 3):
        print(f"  XX k/m are {(K_CLUSTERS, M_EXPERTS)}, frozen at (7, 3)")
        ok = False
    if K_CLUSTERS >= 14:
        print("  XX k must be below the 14 categories or clusters could align with them")
        ok = False
    # 🔒 allocation must do what each name says
    marg = {0: 0.5, 1: -0.2, 2: 0.1, 3: -0.9, 4: 0.3, 5: 0.0, 6: 0.8}
    sz = {0: 10, 1: 20, 2: 300, 3: 40, 4: 500, 5: 60, 6: 700}
    rng = np.random.default_rng(0)
    if allocate("margin", marg, sz, 3, rng) != [3, 1, 5]:
        print(f"  XX margin allocation is wrong: {allocate('margin', marg, sz, 3, rng)}")
        ok = False
    if allocate("size", marg, sz, 3, rng) != [6, 4, 2]:
        print(f"  XX size allocation is wrong: {allocate('size', marg, sz, 3, rng)}")
        ok = False
    if allocate("none", marg, sz, 3, rng) != []:
        print("  XX `none` allocated an expert")
        ok = False
    r = allocate("random", marg, sz, 3, np.random.default_rng(1))
    if len(r) != 3 or len(set(r)) != 3:
        print(f"  XX random allocation is not 3 distinct clusters: {r}")
        ok = False
    # 🔒 the bands and their reading
    if band(0.06, AW1_OK, AW1_NOT) != "works":
        print("  XX a +6pp difference did not read as works")
        ok = False
    if band(0.005, AW1_OK, AW1_NOT) != "does not":
        print("  XX a +0.5pp difference did not read as does-not")
        ok = False
    if band(0.03, AW1_OK, AW1_NOT) != "between bands":
        print("  XX a +3pp difference against AW-1's bands is not between them")
        ok = False
    # 🔒 one arm must be able to veto -- that is the second-arm rule, and section 5 of the
    # preregistration accepts the reachability warning about it with a reason
    a = {"esm2_650M": {"AW1_margin_minus_none": 0.06, "AW2_margin_minus_random": 0.04,
                       "AW3_margin_minus_size": 0.04},
         "esm2_35M": {"AW1_margin_minus_none": 0.06, "AW2_margin_minus_random": 0.000,
                      "AW3_margin_minus_size": 0.04}}
    v = verdict(a)
    if not v["verdict"].startswith("ARMS DISAGREE") or "AW2" not in v["verdict"]:
        print(f"  XX one arm failed to veto: {v['verdict']!r}")
        ok = False
    both_fail = {k: {"AW1_margin_minus_none": 0.004, "AW2_margin_minus_random": 0.002,
                     "AW3_margin_minus_size": 0.001} for k in ARMS}
    if verdict(both_fail)["verdict"] != "the geometry predicts and does not repair":
        print(f"  XX a both-arms failure read {verdict(both_fail)['verdict']!r}")
        ok = False
    both_win = {k: {"AW1_margin_minus_none": 0.08, "AW2_margin_minus_random": 0.05,
                    "AW3_margin_minus_size": 0.05} for k in ARMS}
    if verdict(both_win)["verdict"] != "routing works":
        print(f"  XX a both-arms win read {verdict(both_win)['verdict']!r}")
        ok = False
    if not verdict({"esm2_650M": both_win["esm2_650M"]})["verdict"].startswith("INCOMPLETE"):
        print("  XX one arm returned a verdict")
        ok = False
    # 🔒 routing must send a row to its own cluster's expert and everything else to the
    # global probe -- checked on a planted assignment rather than assumed
    class _P:
        def __init__(self, v):
            self.v = v

        def predict_proba(self, Z):
            return np.column_stack([np.zeros(len(Z)), np.full(len(Z), self.v)])

    X = np.zeros((10, 3))
    assign = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    got = mixture_score(X, _P(0.9), {0: _P(0.1), 2: _P(0.2)}, assign, np.arange(10))
    want = [0.1, 0.1, 0.9, 0.9, 0.2, 0.2, 0.9, 0.9, 0.9, 0.9]
    if not np.allclose(got, want):
        print(f"  XX routing is wrong: {got.tolist()} vs {want}")
        ok = False
    # an empty expert set must route everything to the global probe
    if not np.allclose(mixture_score(X, _P(0.7), {}, assign, np.arange(10)), 0.7):
        print("  XX an empty mixture did not fall back to the global probe")
        ok = False
    print("selftest PASS" if ok else "selftest FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--arm", action="append", default=[])
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    t0 = time.time()
    m30, m133 = load("30_margin_across_arms.py"), load("133_estimation_repair.py")
    out = json.loads(OUT.read_text()) if OUT.exists() else {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "study": "AW -- does the geometry let you do anything?",
        "preregistration": "docs/MARGIN_ROUTING_PREREGISTRATION.md",
        "expects_to_fail": ("prediction and repair dissociate: removing the benign "
                            "neighbours the geometry implicates repairs 8-11% of a failure "
                            "it predicts at rho +0.894. P1 says AW-2 lands in 'does not'"),
        "not_claimed": ("no per-query hazard claim; the member-level margin failed on two "
                        "external panels. Routing asks where a query sits, and the "
                        "allocation is made once, offline"),
        "bands": {"AW1_ok": AW1_OK, "AW1_not": AW1_NOT,
                  "AW2_ok": AW2_OK, "AW2_not": AW2_NOT},
        "arms": {}}
    for arm in (a.arm or list(ARMS)):
        if arm in out["arms"]:
            continue
        out["arms"][arm] = run_arm(arm, a.seeds, a.jobs, m30, m133)
        disk = json.loads(OUT.read_text()) if OUT.exists() else {"arms": {}}
        disk_arms = dict(disk.get("arms", {}))
        disk_arms.update(out["arms"])
        out["arms"] = disk_arms
        out["verdict_block"] = verdict(out["arms"])
        OUT.write_text(json.dumps(out, indent=2) + "\n")
        print(f"  wrote {OUT.name} after {arm}", flush=True)

    v = verdict(out["arms"])
    out["verdict_block"] = v
    if len(out["arms"]) == 2:
        pur = float(np.mean([x["mean_cluster_purity"] for x in out["arms"].values()]))
        out["predictions"] = {
            "P1_AW2_lands_in_does_not": {
                "kind": "reasoned",
                "held": bool(all(x["AW2_margin_minus_random"] <= AW2_NOT
                                 for x in out["arms"].values())),
                "values": {k: x["AW2_margin_minus_random"]
                           for k, x in out["arms"].items()}},
            "P2_AW1_positive_but_under_5pp": {
                "kind": "derived",
                "held": bool(all(0 < x["AW1_margin_minus_none"] < AW1_OK
                                 for x in out["arms"].values())),
                "values": {k: x["AW1_margin_minus_none"]
                           for k, x in out["arms"].items()}},
            "P3_cluster_purity_below_0.50": {
                "kind": "derived", "held": bool(pur < 0.50),
                "values": {"mean_purity": pur}},
        }
    out["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    for k, x in out["arms"].items():
        print(f"\n  {k}: " + "  ".join(
            f"{s} {x['mean_recovery'][s]*100:.2f}%" for s in STRATEGIES))
        print(f"      AW-1 {x['AW1_margin_minus_none']*100:+.2f}pp  "
              f"AW-2 {x['AW2_margin_minus_random']*100:+.2f}pp  "
              f"AW-3 {x['AW3_margin_minus_size']*100:+.2f}pp  "
              f"purity {x['mean_cluster_purity']:.3f}  "
              f"params {x['n_params']['margin']} vs {x['n_params']['none']}")
    print(f"\nverdict: {v['verdict']}")
    print(f"wrote {OUT.relative_to(ROOT)} in {out['seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
