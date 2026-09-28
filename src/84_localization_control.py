#!/usr/bin/env python3
"""
84_localization_control.py - is it hazard, or is it just being on the outside of the cell?

`docs/LOCALIZATION_CONTROL_PREREGISTRATION.md`. Study D removed pathogen origin as the explanation
for about a quarter of the benign-to-virulence gap and left localization as the candidate for the
rest. A probe that separates secreted and surface-exposed proteins from cytoplasmic ones would
reproduce most of this project's results without containing any notion of hazard.

The test is on the benign side, where the hypothesis is sharp: if the probe is a localization
detector, benign extracellular proteins from non-pathogens must be flagged far above benign
cytoplasmic ones.

⚠️ No new probe and no new inference. Study B's fold is used exactly as src/83 uses it, the pool is
already embedded on both arms, and the only new thing is an annotation join.

Usage:
    python src/84_localization_control.py --fetch          # cache UniProt keywords (once)
    python src/84_localization_control.py --arm esm2_650M
    python src/84_localization_control.py --arm esm2_35M
    python src/84_localization_control.py --selftest
"""

import argparse
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
CACHE = ROOT / "data" / "external" / "uniprot_localization"
BUILD = ROOT / "results" / "external_class_axis_build.json"
SCREEN = ROOT / "results" / "external_class_axis_screen.json"
OUT_STEM = ROOT / "results" / "localization_control"
SEEDS, N_TRAIN, N_CAL, SPEC = 30, 1000, 500, 0.95

# ---- frozen strata rule (preregistration § 1.1), evaluated in this order --------------------------
EXTRACELLULAR = ("Secreted", "Signal", "Cell wall", "Cell outer membrane", "Fimbrium", "Flagellum")
MEMBRANE = ("Cell membrane", "Membrane")
INTRACELLULAR = ("Cytoplasm", "Periplasm", "Nucleus", "Cytoplasmic vesicle")
STRATA = ("extracellular", "membrane", "intracellular", "unannotated")
KINGDOMS = ("Bacteria", "Archaea")          # frozen: the kingdoms VFDB draws from
FLOOR_N, CEIL_UNANNOT, AVAIL_RATIO, LEN_AUROC = 300, 0.50, 1.5, 0.65
BAND_LO, BAND_HI = 1.5, 3.0


def stratum_of(keywords):
    """Frozen first-match-wins rule. `keywords` is the set of UniProt keyword strings."""
    for name, terms in (("extracellular", EXTRACELLULAR), ("membrane", MEMBRANE),
                        ("intracellular", INTRACELLULAR)):
        if any(t in keywords for t in terms):
            return name
    return "unannotated"


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))


def auroc(pos, neg):
    """Mann-Whitney AUROC of `pos` over `neg`, ties at 0.5."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = np.empty(len(allv))
    order = np.argsort(allv, kind="mergesort")
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def length_matched(len_a, len_b, rng):
    """1:1 nearest-neighbour match on length, without replacement. Returns index arrays."""
    order = np.argsort(len_b, kind="mergesort")
    pool_idx, pool_len = list(order), list(np.asarray(len_b)[order])
    keep_a, keep_b = [], []
    for ia in rng.permutation(len(len_a)):
        if not pool_idx:
            break
        k = int(np.argmin([abs(x - len_a[ia]) for x in pool_len]))
        keep_a.append(int(ia))
        keep_b.append(int(pool_idx.pop(k)))
        pool_len.pop(k)
    return np.array(keep_a, int), np.array(keep_b, int)


def fetch_keywords(accs, batch=500, pause=0.34):
    """UniProt keywords + CC subcellular location, cached per batch so it resumes."""
    CACHE.mkdir(parents=True, exist_ok=True)
    got = {}
    for i in range(0, len(accs), batch):
        chunk = accs[i:i + batch]
        shard = CACHE / f"kw_{i // batch:04d}.tsv"
        if not shard.exists():
            q = urllib.parse.urlencode({"accessions": ",".join(chunk), "format": "tsv",
                                        "fields": "accession,keyword,cc_subcellular_location"})
            url = f"https://rest.uniprot.org/uniprotkb/accessions?{q}"
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(url, timeout=120) as r:
                        shard.write_text(r.read().decode())
                    break
                except Exception as e:                                  # noqa: BLE001
                    if attempt == 4:
                        raise SystemExit(f"UniProt fetch failed on shard {i // batch}: {e}")
                    time.sleep(2 ** attempt)
            time.sleep(pause)
            print(f"  fetched {i + len(chunk):>6}/{len(accs)}", flush=True)
        for line in shard.read_text().splitlines()[1:]:
            f = line.split("\t")
            if f and f[0]:
                got[f[0]] = {k.strip() for k in (f[1] if len(f) > 1 else "").split(";") if k.strip()}
    return got


def selftest():
    assert stratum_of({"Secreted", "Cytoplasm"}) == "extracellular", "first match must win"
    assert stratum_of({"Cytoplasm", "Membrane"}) == "membrane", "membrane outranks intracellular"
    assert stratum_of({"Cytoplasm"}) == "intracellular"
    assert stratum_of({"Lyase", "Decarboxylase"}) == "unannotated", "non-localization keywords only"
    assert stratum_of(set()) == "unannotated"
    assert abs(auroc([2, 3, 4], [0, 1]) - 1.0) < 1e-9
    assert abs(auroc([0, 1], [0, 1]) - 0.5) < 1e-9
    rng = np.random.default_rng(0)
    a = np.array([100.0, 200.0, 300.0])
    b = np.array([1000.0, 305.0, 95.0, 210.0])
    ka, kb = length_matched(a, b, rng)
    assert len(ka) == len(kb) == 3 and len(set(kb.tolist())) == 3, "matching is without replacement"
    d = float(np.abs(a[ka] - b[kb]).mean())
    assert d < 30, f"length matching should pair close lengths, mean gap {d:.1f}"
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
    man = json.loads((RES / "embedding_manifest_pool_large_esm2_650M.json").read_text())
    if len(pool) != len(man["rows"]):
        raise SystemExit(f"pool has {len(pool)} entries, manifest {len(man['rows'])}")
    for p, row in zip(pool, man["rows"]):
        if p["acc"] != row:
            raise SystemExit(f"pool/manifest row order differs: {p['acc']} vs {row}")

    build = json.loads(BUILD.read_text())
    test = build["pool_partition"]["test_rows"]
    accs = [pool[i]["uniprot"] for i in test]

    if a.fetch:
        print(f"fetching UniProt keywords for {len(accs)} test-partition accessions")
        kw = fetch_keywords(accs)
        print(f"  {len(kw)} of {len(accs)} resolved")
        return

    kw = fetch_keywords(accs)
    missing = [x for x in accs if x not in kw]
    if missing:
        raise SystemExit(f"{len(missing)} accessions unresolved (e.g. {missing[:3]}); rerun --fetch")

    # ---- frozen strata, Bacteria + Archaea -------------------------------------------------------
    elig = [j for j, i in enumerate(test) if pool[i]["kingdom"] in KINGDOMS]
    strat = {s: [] for s in STRATA}
    for j in elig:
        strat[stratum_of(kw[accs[j]])].append(j)
    lengths = np.array([pool[test[j]]["length"] for j in range(len(test))], float)

    print(f"arm {a.arm}: {len(elig)} of {len(test)} test proteins are Bacteria/Archaea")
    for s in STRATA:
        print(f"  {s:<16}{len(strat[s]):>6}  ({len(strat[s]) / max(len(elig), 1) * 100:5.1f}%)")

    # ---- study B's fold, unchanged ---------------------------------------------------------------
    sfx = "" if a.arm == "esm2_650M" else f"_{a.arm}"
    P = np.load(RES / f"embeddings_class_axis_positives{sfx}.npy")
    POOL = np.load(RES / f"embeddings_pool_large_{a.arm}.npy")
    union = json.loads(SCREEN.read_text())["admitted_rows"]
    # 🔴 src/83 takes a fallback branch here and src/84 and src/85 did not, so for one commit the
    # three studies used different folds while all three asserted they used the same one. The screen
    # rejects 32 of the 1,500 partition rows for similarity to positives, leaving 1,468 -- FEWER than
    # N_TRAIN + N_CAL -- so a fixed 1000/500 slice silently calibrated on 468. Entry 55.
    n_tr, n_ca = (N_TRAIN, N_CAL) if len(union) >= N_TRAIN + N_CAL else \
        (2 * len(union) // 3, len(union) - 2 * len(union) // 3)
    y = np.r_[np.ones(P.shape[0]), np.zeros(n_tr)]
    rows = np.array(test)

    per = {s: [] for s in STRATA}
    per["all"], per["annotated"] = [], []
    matched = []
    ann = [j for s in ("extracellular", "membrane", "intracellular") for j in strat[s]]
    rng = np.random.default_rng(0)
    mi_x, mi_i = length_matched(lengths[strat["extracellular"]], lengths[strat["intracellular"]], rng)
    mx = [strat["extracellular"][k] for k in mi_x]
    mn = [strat["intracellular"][k] for k in mi_i]

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
        per["all"].append(float(flag[elig].mean()))
        per["annotated"].append(float(flag[ann].mean()))
        matched.append((float(flag[mx].mean()), float(flag[mn].mean())))
    print(f"  {time.time() - t0:.0f}s")

    def stat(v):
        v = np.asarray(v, float)
        return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                "ci": [float(v.mean() - 1.96 * v.std(ddof=1) / len(v) ** 0.5),
                       float(v.mean() + 1.96 * v.std(ddof=1) / len(v) ** 0.5)]}

    ratios = np.array(per["extracellular"], float) / np.array(per["intracellular"], float)
    R = stat(ratios)
    band = ("localization is not a material driver" if R["mean"] <= BAND_LO
            else "localization is a major driver" if R["mean"] >= BAND_HI else "partial")

    len_auroc = auroc(lengths[strat["extracellular"]], lengths[strat["intracellular"]])
    mr = np.array([m[0] / m[1] if m[1] else np.nan for m in matched], float)
    unannot_frac = len(strat["unannotated"]) / max(len(elig), 1)
    avail = (np.mean(per["unannotated"]) / np.mean(per["annotated"])
             if np.mean(per["annotated"]) else float("nan"))

    gate = {
        "floor_n_met": min(len(strat["extracellular"]), len(strat["intracellular"])) >= FLOOR_N,
        "unannotated_frac": unannot_frac,
        "ceiling_unannotated_met": unannot_frac <= CEIL_UNANNOT,
        "availability_ratio": float(avail),
        "availability_ok": bool(1 / AVAIL_RATIO <= avail <= AVAIL_RATIO),
        "length_auroc": len_auroc,
        "length_confounded": bool(abs(len_auroc - 0.5) > (LEN_AUROC - 0.5)),
    }

    # Two share-of-gap figures, because § 3 of the preregistration asked for one thing and called it
    # another. `contrast` is § 3 as written (extracellular over intracellular). `study_d_form` is the
    # arithmetic study D actually used -- (control - pool) / (vfdb - pool) -- which is the one that is
    # commensurable with its 22.7%. Both are reported; neither replaces the other.
    vfdb = 0.7348585427532796 if a.arm == "esm2_650M" else None
    e3 = None
    if vfdb:
        gap = vfdb - np.mean(per["all"])
        e3 = {"contrast": float((np.mean(per["extracellular"]) - np.mean(per["intracellular"])) / gap),
              "study_d_form": float((np.mean(per["extracellular"]) - np.mean(per["all"])) / gap),
              "vfdb_rate": vfdb, "pool_rate": float(np.mean(per["all"]))}

    res = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "arm": a.arm, "seeds": a.seeds,
           "nominal": 1 - SPEC, "kingdoms": list(KINGDOMS),
           "n_test": len(test), "n_eligible": len(elig),
           "strata_n": {s: len(strat[s]) for s in STRATA},
           "rates": {k: stat(v) for k, v in per.items() if not np.isnan(np.mean(v))},
           "E1_ratio": R, "band": band, "gate": gate,
           "length_matched": {"n_pairs": len(mx), "ratio": stat(mr[~np.isnan(mr)]),
                              "extracellular": stat([m[0] for m in matched]),
                              "intracellular": stat([m[1] for m in matched])},
           "E3_fraction_of_gap": e3}
    dest = Path(f"{OUT_STEM}{sfx}.json")
    dest.write_text(json.dumps(res, indent=2) + "\n")

    print(f"\n{'stratum':<20}{'n':>7}{'flag rate at a nominal 5%':>30}")
    print("-" * 60)
    for s in STRATA + ("annotated", "all"):
        if s in res["rates"]:
            st, n = res["rates"][s], len(strat[s]) if s in strat else (len(ann) if s == "annotated" else len(elig))
            print(f"  {s:<18}{n:>7}{st['mean'] * 100:>16.2f}% [{st['ci'][0] * 100:.2f}, {st['ci'][1] * 100:.2f}]")
    print(f"\nE-1  R = {R['mean']:.3f} [{R['ci'][0]:.3f}, {R['ci'][1]:.3f}]  ->  {band}")
    print(f"     length AUROC {len_auroc:.3f} ({'CONFOUNDED' if gate['length_confounded'] else 'ok'}), "
          f"matched R = {np.nanmean(mr):.3f} on {len(mx)} pairs")
    print(f"     unannotated {unannot_frac * 100:.1f}% (ceiling {CEIL_UNANNOT * 100:.0f}%), "
          f"availability ratio {avail:.2f} ({'ok' if gate['availability_ok'] else 'CONFOUNDED'})")
    if e3 is not None:
        print(f"E-3  localization spans {e3['contrast'] * 100:.1f}% of the pool->VFDB gap "
              f"(§ 3 as written), {e3['study_d_form'] * 100:.1f}% in study D's arithmetic "
              f"(its provenance figure was 22.7%)")
    print(f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
