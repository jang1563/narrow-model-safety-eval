#!/usr/bin/env python3
"""
92_localization_coverage.py - study E's contrast on a localization source that clears its own ceiling.

`docs/LOCALIZATION_COVERAGE_PREREGISTRATION.md`. Study E returned this project's main adverse finding
-- the probe is strongly graded by localization on BENIGN proteins -- and breached its own frozen
ceiling doing it: 55.0% of the eligible pool had no localization keyword at all, against a ceiling of
50%, and the availability check failed at 0.41. Both failures have one cause: UniProt's curated
keywords are sparse, so the contrast was measured on the 45% a curator had reached.

GO cellular component, signal-peptide and transmembrane features take coverage to 66.1%, so the
unannotated stratum falls to 33.9% and the ceiling clears.

🔒 Study E's keyword rule is EMBEDDED, not replaced: the new sources are unions with it, so every
protein study E classified keeps its class and the change is purely additive. That is what makes this
comparable to study E rather than a different experiment.

Usage:
    python src/92_localization_coverage.py --fetch
    python src/92_localization_coverage.py --arm esm2_650M
    python src/92_localization_coverage.py --arm esm2_35M
    python src/92_localization_coverage.py --selftest
"""

import argparse
import hashlib
import importlib.util
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "v3"
POOL_JSON = ROOT / "data" / "sequences" / "_scaled_negative_pool.json"
CACHE = ROOT / "data" / "external" / "uniprot_localization_go"
BUILD = ROOT / "results" / "external_class_axis_build.json"
SCREEN = ROOT / "results" / "external_class_axis_screen.json"
OUT_STEM = ROOT / "results" / "localization_coverage"
SEEDS, N_TRAIN, N_CAL, SPEC = 30, 1000, 500, 0.95
STRATA = ("extracellular", "membrane", "intracellular", "unannotated")
FLOOR_N, CEIL_UNANNOT, AVAIL_RATIO, LEN_AUROC = 300, 0.50, 1.5, 0.65
BAND_LO, BAND_HI = 1.5, 3.0

# ---- frozen GO cellular-component terms (preregistration § 1.1), lower-cased substrings ----------
GO_EXTRA = ("extracellular", "cell outer membrane", "cell surface", "cell wall", "fimbri",
            "pilus", "flagell", "capsule", "s-layer")
GO_MEMB = ("membrane",)
GO_INTRA = ("cytoplasm", "cytosol", "periplasm", "nucleoid", "ribosome", "chromosome")


def _load(stem):
    spec = importlib.util.spec_from_file_location(f"_{stem}", ROOT / "src" / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M84 = _load("84_localization_control")


def stratum_of(go, sig, tm, keywords):
    """Study E's class where it has one; otherwise the GO / signal / transmembrane rule.

    🔴 Amendment 1. § 1.1 was written as a first-match-wins rule over the union of all sources and in
    the same breath claimed the change was "purely additive". Those are different rules and the
    script's own guard caught it: 49 proteins that study E classified would have been reclassified,
    because a SIGNAL feature or an extracellular GO term outranks a `Cytoplasm` keyword under
    first-match-wins. This study exists to test whether study E's COVERAGE breach invalidates its
    verdict, so classification is held fixed and only coverage changes; `reclassified_under_union`
    reports what the other reading would have moved.
    """
    kw = M84.stratum_of(keywords)
    if kw != "unannotated":
        return kw
    return union_stratum(go, sig, tm, keywords)


def union_stratum(go, sig, tm, keywords):
    """§ 1.1 as literally written: first match wins across all sources. The sensitivity arm."""
    g = (go or "").lower()
    kw = M84.stratum_of(keywords)
    if any(t in g for t in GO_EXTRA) or (sig or "").strip() or kw == "extracellular":
        return "extracellular"
    if any(t in g for t in GO_MEMB) or (tm or "").strip() or kw == "membrane":
        return "membrane"
    if any(t in g for t in GO_INTRA) or kw == "intracellular":
        return "intracellular"
    return "unannotated"


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))


def fold_sizes(n):
    return (N_TRAIN, N_CAL) if n >= N_TRAIN + N_CAL else (2 * n // 3, n - 2 * n // 3)


def fetch_go(accs, batch=500, pause=0.34):
    """GO cellular component, signal-peptide and transmembrane features. Content-addressed cache."""
    CACHE.mkdir(parents=True, exist_ok=True)
    got = {}
    for i in range(0, len(accs), batch):
        chunk = accs[i:i + batch]
        tag = hashlib.sha256("\n".join(chunk).encode()).hexdigest()[:16]
        shard = CACHE / f"go_{i // batch:04d}_{tag}.tsv"
        if not shard.exists():
            q = urllib.parse.urlencode({"accessions": ",".join(chunk), "format": "tsv",
                                        "fields": "accession,go_c,ft_signal,ft_transmem"})
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(
                            f"https://rest.uniprot.org/uniprotkb/accessions?{q}", timeout=180) as r:
                        shard.write_text(r.read().decode())
                    break
                except Exception as e:                                 # noqa: BLE001
                    if attempt == 4:
                        raise SystemExit(f"GO fetch failed on shard {i // batch}: {e}")
                    time.sleep(2 ** attempt)
            time.sleep(pause)
            print(f"  fetched {i + len(chunk):>6}/{len(accs)}", flush=True)
        rows = {}
        for line in shard.read_text().splitlines()[1:]:
            f = line.split("\t")
            if f and f[0]:
                rows[f[0]] = {"go": f[1] if len(f) > 1 else "", "sig": f[2] if len(f) > 2 else "",
                              "tm": f[3] if len(f) > 3 else ""}
        stray = set(rows) - set(chunk)
        if stray:
            raise SystemExit(f"cache shard {shard.name} holds unrequested accessions; delete it")
        got.update(rows)
    return got


def selftest():
    # 🔒 the union with study E's rule must be additive: anything it classified keeps its class
    assert stratum_of("", "", "", {"Secreted"}) == "extracellular"
    assert stratum_of("", "", "", {"Cytoplasm"}) == "intracellular"
    assert stratum_of("", "", "", {"Membrane"}) == "membrane"
    # the new sources classify what the keywords could not
    assert stratum_of("cytosol [GO:0005829]", "", "", set()) == "intracellular"
    assert stratum_of("extracellular region [GO:0005576]", "", "", set()) == "extracellular"
    assert stratum_of("", "SIGNAL 1..34", "", set()) == "extracellular"
    assert stratum_of("", "", "TRANSMEM 5..25", set()) == "membrane"
    assert stratum_of("", "", "", set()) == "unannotated"
    # 🔴 first match wins: "cell outer membrane" is extracellular, not membrane
    assert stratum_of("cell outer membrane [GO:0009279]", "", "", set()) == "extracellular"
    # and a signal peptide outranks a cytoplasmic GO term, exactly as study E's Signal did
    assert union_stratum("cytoplasm [GO:0005737]", "SIGNAL 1..20", "", set()) == "extracellular"
    # 🔴 and the primary rule does NOT override a class study E already assigned
    assert stratum_of("extracellular region [GO:0005576]", "SIGNAL 1..9", "",
                      {"Cytoplasm"}) == "intracellular", "study E's class must be held fixed"
    assert union_stratum("extracellular region [GO:0005576]", "", "",
                         {"Cytoplasm"}) == "extracellular", "the sensitivity arm may override"
    assert fold_sizes(1450) == (966, 484)
    assert abs(M84.auroc([2, 3], [0, 1]) - 1.0) < 1e-9
    print("selftest PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="esm2_650M", choices=["esm2_650M", "esm2_35M"])
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    pool = json.loads(POOL_JSON.read_text())["proteins"]
    test = json.loads(BUILD.read_text())["pool_partition"]["test_rows"]
    accs = [pool[i]["uniprot"] for i in test]
    go = fetch_go(accs)
    if a.fetch:
        print(f"  {len(go)} of {len(accs)} resolved")
        return
    kw = M84.fetch_keywords(accs)
    missing = [x for x in accs if x not in go]
    if missing:
        raise SystemExit(f"{len(missing)} accessions have no GO record; rerun --fetch")

    elig = [j for j, i in enumerate(test) if pool[i]["kingdom"] in M84.KINGDOMS]
    strat = {s: [] for s in STRATA}
    old = {s: [] for s in STRATA}
    reclassified = 0
    for j in elig:
        g = go[accs[j]]
        strat[stratum_of(g["go"], g["sig"], g["tm"], kw[accs[j]])].append(j)
        old[M84.stratum_of(kw[accs[j]])].append(j)
        u = union_stratum(g["go"], g["sig"], g["tm"], kw[accs[j]])
        if M84.stratum_of(kw[accs[j]]) not in ("unannotated", u):
            reclassified += 1
    lengths = np.array([pool[test[j]]["length"] for j in range(len(test))], float)

    print(f"arm {a.arm}: {len(elig)} eligible")
    print(f"{'stratum':<16}{'study E':>10}{'this study':>13}{'delta':>9}")
    for s in STRATA:
        print(f"  {s:<14}{len(old[s]):>10}{len(strat[s]):>13}{len(strat[s]) - len(old[s]):>+9}")
    # 🔒 the union must be additive: nothing study E classified may change class
    moved = [j for s in ("extracellular", "membrane", "intracellular")
             for j in old[s] if j not in set(strat[s])]
    if moved:
        raise SystemExit(f"{len(moved)} proteins changed class; the rule is not additive")
    print(f"  🔒 additive: 0 of the {sum(len(old[s]) for s in STRATA if s != 'unannotated')} "
          f"proteins study E classified changed class")
    print(f"  ⚠️  § 1.1 as literally written would have reclassified {reclassified} of them "
          f"(Amendment 1); reported, not used")

    sfx = "" if a.arm == "esm2_650M" else f"_{a.arm}"
    vf = _load("83_provenance_control").vfdb_sequences()
    P = np.load(RES / f"embeddings_class_axis_positives{sfx}.npy")
    POOL = np.load(RES / f"embeddings_pool_large_{a.arm}.npy")
    union = [i for i in json.loads(SCREEN.read_text())["admitted_rows"]
             if pool[i]["sequence"] not in vf]
    n_tr, n_ca = fold_sizes(len(union))
    y = np.r_[np.ones(P.shape[0]), np.zeros(n_tr)]
    rows = np.array(test)
    ann = [j for s in ("extracellular", "membrane", "intracellular") for j in strat[s]]
    rng = np.random.default_rng(0)
    mi_x, mi_i = M84.length_matched(lengths[strat["extracellular"]],
                                    lengths[strat["intracellular"]], rng)
    mx = [strat["extracellular"][k] for k in mi_x]
    mn = [strat["intracellular"][k] for k in mi_i]

    # 🔒 § 1 froze "the same pool rows", which is study E's evaluated population and therefore still
    # contains the VFDB contaminants study G found are differentially distributed across exactly this
    # contrast (5.67% of extracellular against 1.20% of intracellular). The frozen version governs;
    # the decontaminated one is reported beside it because it is what compares to study G's 5.074.
    clean_idx = {s: [j for j in v if pool[test[j]]["sequence"] not in vf] for s, v in strat.items()}
    per = {s: [] for s in STRATA}
    per.update({f"clean_{s}": [] for s in ("extracellular", "intracellular")})
    per["all"], per["annotated"] = [], []
    matched = []
    t0 = time.time()
    for seed in range(a.seeds):
        r = np.random.default_rng(seed)
        perm = r.permutation(len(union))
        tr = [union[i] for i in perm[:n_tr]]
        ca = [union[i] for i in perm[n_tr:n_tr + n_ca]]
        model = clf().fit(np.vstack([P, POOL[tr]]), y)
        t = float(np.quantile(model.predict_proba(POOL[ca])[:, 1], SPEC))
        flag = model.predict_proba(POOL[rows])[:, 1] >= t
        for s in STRATA:
            per[s].append(float(flag[strat[s]].mean()) if strat[s] else float("nan"))
        for s in ("extracellular", "intracellular"):
            per[f"clean_{s}"].append(float(flag[clean_idx[s]].mean()))
        per["all"].append(float(flag[elig].mean()))
        per["annotated"].append(float(flag[ann].mean()))
        matched.append((float(flag[mx].mean()), float(flag[mn].mean())))
    print(f"  {time.time() - t0:.0f}s")

    def stat(v):
        v = np.asarray(v, float)
        return {"mean": float(v.mean()), "ci": [float(v.mean() - 1.96 * v.std(ddof=1) / len(v) ** 0.5),
                                                float(v.mean() + 1.96 * v.std(ddof=1) / len(v) ** 0.5)]}

    ratios = np.array(per["extracellular"], float) / np.array(per["intracellular"], float)
    R = stat(ratios)
    band = ("localization is not a material driver" if R["mean"] <= BAND_LO
            else "localization is a major driver" if R["mean"] >= BAND_HI else "partial")
    unannot_frac = len(strat["unannotated"]) / len(elig)
    avail = float(np.mean(per["unannotated"]) / np.mean(per["annotated"]))
    len_auroc = M84.auroc(lengths[strat["extracellular"]], lengths[strat["intracellular"]])
    mr = np.array([m[0] / m[1] if m[1] else np.nan for m in matched], float)

    clean_R = stat(np.array(per["clean_extracellular"], float)
                   / np.array(per["clean_intracellular"], float))
    res = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "arm": a.arm, "seeds": a.seeds,
           "n_eligible": len(elig),
           "strata_n": {s: len(strat[s]) for s in STRATA},
           "strata_n_study_e": {s: len(old[s]) for s in STRATA},
           "additive": True, "reclassified_under_union": reclassified,
           "rates": {k: stat(v) for k, v in per.items() if not np.isnan(np.mean(v))},
           "L1_ratio": R, "band": band,
           "L1_ratio_decontaminated": clean_R,
           "clean_strata_n": {s: len(v) for s, v in clean_idx.items()},
           "L3_unannotated_frac": unannot_frac,
           "L3_ceiling_met": bool(unannot_frac <= CEIL_UNANNOT),
           "L4_availability_ratio": avail,
           "L4_availability_ok": bool(1 / AVAIL_RATIO <= avail <= AVAIL_RATIO),
           "floors_met": bool(min(len(strat["extracellular"]), len(strat["intracellular"]))
                              >= FLOOR_N),
           "length_auroc": len_auroc,
           "length_confounded": bool(abs(len_auroc - 0.5) > (LEN_AUROC - 0.5)),
           "length_matched_ratio": float(np.nanmean(mr)), "n_matched_pairs": len(mx)}
    dest = Path(f"{OUT_STEM}{sfx}.json")
    dest.write_text(json.dumps(res, indent=2) + "\n")

    print(f"\n{'stratum':<20}{'n':>7}{'flag rate at a nominal 5%':>30}")
    print("-" * 60)
    for s in STRATA + ("annotated", "all"):
        if s in res["rates"]:
            n = len(strat[s]) if s in strat else (len(ann) if s == "annotated" else len(elig))
            st = res["rates"][s]
            print(f"  {s:<18}{n:>7}{st['mean'] * 100:>16.2f}% [{st['ci'][0] * 100:.2f}, {st['ci'][1] * 100:.2f}]")
    print(f"\nL-1  R = {clean_R['mean']:.3f} [{clean_R['ci'][0]:.3f}, {clean_R['ci'][1]:.3f}] "
          f"with contaminants removed from the strata (study G's clean condition gave 5.074)")
    print(f"L-1  R = {R['mean']:.3f} [{R['ci'][0]:.3f}, {R['ci'][1]:.3f}]  ->  {band}")
    print(f"L-3  unannotated {unannot_frac * 100:.1f}% (ceiling {CEIL_UNANNOT * 100:.0f}%) -> "
          f"{'CLEARED' if res['L3_ceiling_met'] else 'BREACHED'}   [study E: 55.0%, breached]")
    print(f"L-4  availability {avail:.2f} -> {'ok' if res['L4_availability_ok'] else 'FAILS'}"
          f"   [study E: 0.41, failed]")
    print(f"     length AUROC {len_auroc:.3f} ({'CONFOUNDED' if res['length_confounded'] else 'ok'}), "
          f"matched R = {np.nanmean(mr):.3f} on {len(mx)} pairs; floors "
          f"{'met' if res['floors_met'] else 'NOT met'}")
    print(f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
