#!/usr/bin/env python3
"""
88_taxid_provenance.py - the provenance factor matched on names, and names change.

`docs/TAXID_PROVENANCE_PREREGISTRATION.md`. Studies F and G decide "pathogen-derived" with a string
test: do the first two words of the organism name appear in VFDB's species list. A pathogen UniProt has
renamed fails it and is scored benign, which puts genuine pathogen proteins in the benign cells. Entry
57 recorded the direction -- dilution, therefore conservative -- and recorded that it was not
quantified. This quantifies it.

⚠️ No new inference: same rows, same embeddings, study G's clean fold, and only the definition of one
binary factor changes.

🔒 The resolver is two-stage and exact-verified, because the obvious routes are wrong: `scientific:`
alone misses every renamed pathogen, and taking a free-text search's top hit maps Escherichia coli to
Escherichia phage 1. An unverified hit is never accepted.

Usage:
    python src/88_taxid_provenance.py --resolve      # cache VFDB name -> canonical species (once)
    python src/88_taxid_provenance.py --arm esm2_650M
    python src/88_taxid_provenance.py --arm esm2_35M
    python src/88_taxid_provenance.py --selftest
"""

import argparse
import hashlib
import importlib.util
import json
import re
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
CACHE = ROOT / "data" / "external" / "uniprot_taxonomy"
BUILD = ROOT / "results" / "external_class_axis_build.json"
SCREEN = ROOT / "results" / "external_class_axis_screen.json"
OUT_STEM = ROOT / "results" / "taxid_provenance"
SEEDS, N_TRAIN, N_CAL, SPEC = 30, 1000, 500, 0.95
VFDB_RATE = 0.7348585427532796
FLOOR, RR_LO, RR_HI, UNRESOLVED_CEIL = 150, 0.67, 1.5, 0.10
LINEAGE_SPECIES_RE = re.compile(r"([^,]+?)\s*\(species\)")
# 🔒 study G's clean-condition results, used as a reproduction gate on the string factor
G_CLEAN_RR = {"esm2_650M": 0.999, "esm2_35M": 1.358}


def _load(stem):
    spec = importlib.util.spec_from_file_location(f"_{stem}", ROOT / "src" / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M84 = _load("84_localization_control")
M83 = _load("83_provenance_control")


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))


def fold_sizes(n):
    return (N_TRAIN, N_CAL) if n >= N_TRAIN + N_CAL else (2 * n // 3, n - 2 * n // 3)


def _tax_search(query, size):
    q = urllib.parse.urlencode({"query": query, "format": "json", "size": size})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(
                    f"https://rest.uniprot.org/taxonomy/search?{q}", timeout=90) as r:
                return json.load(r).get("results", [])
        except Exception:                                              # noqa: BLE001
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    return []


def resolve_species(name):
    """Two-stage, exact-verified. Returns (taxid, canonical_name, how)."""
    low = name.lower()
    for d in _tax_search(f'scientific:"{name}" AND rank:species', 10):
        if str(d.get("scientificName", "")).lower() == low:
            return d["taxonId"], d.get("scientificName"), "scientific"
    for d in _tax_search(f'"{name}" AND rank:species', 50):
        names = {str(d.get("scientificName", "")).lower()}
        names |= {str(x).lower() for x in (d.get("synonyms") or [])}
        names |= {str(x).lower() for x in (d.get("otherNames") or [])}
        if low in names:
            return d["taxonId"], d.get("scientificName"), "synonym"
    # 🔴 never accept an unverified hit: this is what stops E. coli becoming a phage
    return None, None, "UNRESOLVED"


def fetch_lineage(accs, batch=500, pause=0.34):
    """Rank-annotated lineage per accession, content-addressed cache (entry 55's lesson)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    got = {}
    for i in range(0, len(accs), batch):
        chunk = accs[i:i + batch]
        tag = hashlib.sha256("\n".join(chunk).encode()).hexdigest()[:16]
        shard = CACHE / f"lin_{i // batch:04d}_{tag}.tsv"
        if not shard.exists():
            q = urllib.parse.urlencode({"accessions": ",".join(chunk), "format": "tsv",
                                        "fields": "accession,organism_id,lineage,organism_name"})
            url = f"https://rest.uniprot.org/uniprotkb/accessions?{q}"
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(url, timeout=180) as r:
                        shard.write_text(r.read().decode())
                    break
                except Exception as e:                                 # noqa: BLE001
                    if attempt == 4:
                        raise SystemExit(f"lineage fetch failed on shard {i // batch}: {e}")
                    time.sleep(2 ** attempt)
            time.sleep(pause)
            print(f"  fetched {i + len(chunk):>6}/{len(accs)}", flush=True)
        rows = {}
        for line in shard.read_text().splitlines()[1:]:
            f = line.split("\t")
            if f and f[0]:
                rows[f[0]] = {"taxid": f[1] if len(f) > 1 else "",
                              "lineage": f[2] if len(f) > 2 else "",
                              "organism": f[3] if len(f) > 3 else ""}
        stray = set(rows) - set(chunk)
        if stray:
            raise SystemExit(f"cache shard {shard.name} holds unrequested accessions; delete it")
        got.update(rows)
    return got


def canonical_species(rec):
    """Current canonical species name from a rank-annotated lineage, else a marked fallback."""
    m = LINEAGE_SPECIES_RE.search(rec.get("lineage", ""))
    if m:
        return m.group(1).strip(), False
    return " ".join(rec.get("organism", "").replace("(", "").split()[:2]), True


def string_species(organism):
    """Exactly what src/85 line 125 does, reproduced so the comparison is like-for-like."""
    return " ".join(organism.replace("(", "").split()[:2])


def selftest():
    assert string_species("Escherichia coli (strain K12)") == "Escherichia coli"
    lin = ("cellular organisms (no rank), Bacteria (domain), Enterobacterales (order), "
           "Escherichia (genus), Escherichia coli (species)")
    assert canonical_species({"lineage": lin}) == ("Escherichia coli", False)
    assert canonical_species({"lineage": "", "organism": "Foo bar (strain X)"}) == ("Foo bar", True)
    lin2 = "cellular organisms (no rank), Bacteria (domain), Mycoplasmoides pneumoniae (species)"
    assert canonical_species({"lineage": lin2})[0] == "Mycoplasmoides pneumoniae"
    assert fold_sizes(1450) == (966, 484)
    print("selftest PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="esm2_650M", choices=["esm2_650M", "esm2_35M"])
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--resolve", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    vfdb_names = sorted(M83.vfdb_species())
    CACHE.mkdir(parents=True, exist_ok=True)
    resolved_path = CACHE / "vfdb_species_resolved.json"

    if a.resolve or not resolved_path.exists():
        done = json.loads(resolved_path.read_text()) if resolved_path.exists() else {}
        print(f"resolving {len(vfdb_names)} VFDB species names ({len(done)} cached)")
        for k, n in enumerate(vfdb_names):
            if n in done:
                continue
            tid, canon, how = resolve_species(n)
            done[n] = {"taxid": tid, "canonical": canon, "how": how}
            if (k + 1) % 25 == 0:
                resolved_path.write_text(json.dumps(done, indent=1) + "\n")
                print(f"  {k + 1}/{len(vfdb_names)}", flush=True)
            time.sleep(0.2)
        resolved_path.write_text(json.dumps(done, indent=1) + "\n")
        if a.resolve:
            un = [n for n in vfdb_names if done[n]["how"] == "UNRESOLVED"]
            print(f"  resolved {len(vfdb_names) - len(un)}/{len(vfdb_names)}, {len(un)} unresolved")
            return

    res_map = json.loads(resolved_path.read_text())
    unresolved = sorted(n for n in vfdb_names if res_map[n]["how"] == "UNRESOLVED")
    by_how = {h: sum(1 for n in vfdb_names if res_map[n]["how"] == h)
              for h in ("scientific", "synonym", "UNRESOLVED")}
    canon_set = {res_map[n]["canonical"] for n in vfdb_names if res_map[n]["canonical"]}
    frac_un = len(unresolved) / len(vfdb_names)
    print(f"VFDB species names {len(vfdb_names)}: {by_how} -> {len(canon_set)} canonical species")
    print(f"  unresolved {frac_un * 100:.1f}% (ceiling {UNRESOLVED_CEIL * 100:.0f}%)")

    pool = json.loads(POOL_JSON.read_text())["proteins"]
    test = json.loads(BUILD.read_text())["pool_partition"]["test_rows"]
    accs = [pool[i]["uniprot"] for i in test]
    kw = M84.fetch_keywords(accs)
    lin = fetch_lineage(accs)
    missing = [x for x in accs if x not in lin]
    if missing:
        raise SystemExit(f"{len(missing)} accessions have no lineage; rerun")
    vf_seqs = M83.vfdb_sequences()

    # ---- the two factors, on identical rows ------------------------------------------------------
    n_fallback = 0
    factors = {"string": {}, "taxid": {}}
    for jj, i in enumerate(test):
        canon, fell_back = canonical_species(lin[accs[jj]])
        n_fallback += int(fell_back)
        factors["string"][jj] = string_species(pool[i]["organism"]) in set(vfdb_names)
        factors["taxid"][jj] = canon in canon_set
    moved = [jj for jj in factors["string"] if factors["string"][jj] != factors["taxid"][jj]]

    def cells_for(which):
        c = {(p, s): [] for p in ("pathogen", "benign_species")
             for s in ("extracellular", "intracellular")}
        elig = []
        for jj, i in enumerate(test):
            if pool[i]["kingdom"] not in M84.KINGDOMS or pool[i]["sequence"] in vf_seqs:
                continue
            elig.append(jj)
            s = M84.stratum_of(kw[accs[jj]])
            if s in ("extracellular", "intracellular"):
                c[("pathogen" if factors[which][jj] else "benign_species", s)].append(jj)
        return c, elig

    cs, elig = cells_for("string")
    ct, _ = cells_for("taxid")
    moved_elig = [jj for jj in moved if jj in set(elig)]
    # 🔒 A 31% fallback rate needs explaining, not reporting. UniProt's lineage lists ANCESTORS, so
    # an organism that is itself at species rank has no "(species)" entry -- and for those the current
    # organism name IS the canonical species name, which is exactly what the fallback returns. The
    # check is whether a fallback name is two words with no strain suffix.
    fb = [accs[jj] for jj, i in enumerate(test) if canonical_species(lin[accs[jj]])[1]]
    fb_clean = sum(1 for x in fb if len(lin[x]["organism"].split()) == 2 and "(" not in lin[x]["organism"])
    print(f"  lineage fallback used for {n_fallback} of {len(test)} proteins; "
          f"{fb_clean} of them ({fb_clean / max(len(fb), 1) * 100:.1f}%) have a bare two-word organism "
          f"name, i.e. are already at species rank")
    if fb:
        print(f"    examples: {[lin[x]['organism'] for x in fb[:3]]}")
    print(f"\n{'cell':<34}{'string':>9}{'taxid':>9}{'delta':>8}")
    for k in cs:
        print(f"  {k[0] + ' x ' + k[1]:<32}{len(cs[k]):>9}{len(ct[k]):>9}{len(ct[k]) - len(cs[k]):>+8}")
    print(f"  {'changed side (eligible)':<32}{'':>9}{len(moved_elig):>9}"
          f"   = {len(moved_elig) / len(elig) * 100:.2f}%")
    if not moved_elig:
        raise SystemExit("0 proteins changed side — per § 3 that is a bug, not a finding")

    # ---- study G's clean fold, unchanged ---------------------------------------------------------
    sfx = "" if a.arm == "esm2_650M" else f"_{a.arm}"
    P = np.load(RES / f"embeddings_class_axis_positives{sfx}.npy")
    POOL = np.load(RES / f"embeddings_pool_large_{a.arm}.npy")
    union = [i for i in json.loads(SCREEN.read_text())["admitted_rows"]
             if pool[i]["sequence"] not in vf_seqs]
    n_tr, n_ca = fold_sizes(len(union))
    y = np.r_[np.ones(P.shape[0]), np.zeros(n_tr)]
    rows = np.array(test)

    out = {}
    t0 = time.time()
    for which, cells in (("string", cs), ("taxid", ct)):
        per = {k: [] for k in cells}
        per["all"] = []
        for seed in range(a.seeds):
            r = np.random.default_rng(seed)
            perm = r.permutation(len(union))
            tr = [union[i] for i in perm[:n_tr]]
            ca = [union[i] for i in perm[n_tr:n_tr + n_ca]]
            model = clf().fit(np.vstack([P, POOL[tr]]), y)
            t = float(np.quantile(model.predict_proba(POOL[ca])[:, 1], SPEC))
            flag = model.predict_proba(POOL[rows])[:, 1] >= t
            for k, v in cells.items():
                per[k].append(float(flag[v].mean()))
            per["all"].append(float(flag[elig].mean()))
        rr = ((np.array(per[("pathogen", "extracellular")]) / np.array(per[("pathogen", "intracellular")]))
              / (np.array(per[("benign_species", "extracellular")])
                 / np.array(per[("benign_species", "intracellular")])))
        pool_rate = float(np.mean(per["all"]))
        out[which] = {
            "cell_n": {f"{p}|{s}": len(v) for (p, s), v in cells.items()},
            "rates": {("|".join(k) if isinstance(k, tuple) else k): float(np.mean(v))
                      for k, v in per.items()},
            "RR": float(np.mean(rr)),
            "RR_ci": [float(np.mean(rr) - 1.96 * np.std(rr, ddof=1) / a.seeds ** 0.5),
                      float(np.mean(rr) + 1.96 * np.std(rr, ddof=1) / a.seeds ** 0.5)],
            "joint_share": float((np.mean(per[("pathogen", "extracellular")]) - pool_rate)
                                 / (VFDB_RATE - pool_rate)),
            "F5": float(np.mean(per[("pathogen", "intracellular")])
                        / np.mean(per[("benign_species", "intracellular")])),
        }
    print(f"  {time.time() - t0:.0f}s")

    # 🔒 reproduction gate: the string factor must reproduce study G's clean RR
    gate = abs(out["string"]["RR"] - G_CLEAN_RR[a.arm])
    if gate > 0.02:
        raise SystemExit(f"string factor gives RR {out['string']['RR']:.3f}, study G's clean "
                         f"condition gave {G_CLEAN_RR[a.arm]:.3f}; the two are not the same analysis")
    print(f"  🔒 reproduction gate: string RR {out['string']['RR']:.3f} vs study G {G_CLEAN_RR[a.arm]}"
          f" (|Δ| = {gate:.4f})")

    s, t_ = out["string"], out["taxid"]
    band = "I-1 < 1%, footnote" if len(moved_elig) / len(elig) < 0.01 else \
        "I-1 1-5%, material" if len(moved_elig) / len(elig) <= 0.05 else "I-1 > 5%, substantially wrong"
    res = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "arm": a.arm, "seeds": a.seeds,
           "n_vfdb_names": len(vfdb_names), "resolved_by": by_how,
           "n_canonical_species": len(canon_set),
           "unresolved_frac": frac_un, "unresolved_ok": bool(frac_un <= UNRESOLVED_CEIL),
           "unresolved_names": unresolved,
           "n_lineage_fallback": n_fallback, "n_fallback_bare_species_name": fb_clean,
           "n_eligible": len(elig), "n_moved": len(moved_elig),
           "moved_frac": len(moved_elig) / len(elig), "I1_band": band,
           "reproduction_gate_delta": gate,
           "string": s, "taxid": t_,
           "I2": {"RR_in_band": bool(RR_LO <= t_["RR"] <= RR_HI),
                  "joint_within_10pp": bool(abs(t_["joint_share"] - s["joint_share"]) <= 0.10),
                  "F5_above_1": bool(t_["F5"] > 1.0),
                  "floors_met": bool(min(t_["cell_n"].values()) >= FLOOR)}}
    dest = Path(f"{OUT_STEM}{sfx}.json")
    dest.write_text(json.dumps(res, indent=2) + "\n")

    print(f"\n{'quantity':<26}{'string':>11}{'taxid':>11}{'delta':>10}")
    print("-" * 58)
    for k in ("RR", "joint_share", "F5"):
        print(f"  {k:<24}{s[k]:>11.3f}{t_[k]:>11.3f}{t_[k] - s[k]:>+10.3f}")
    print(f"\nI-1  {len(moved_elig)}/{len(elig)} changed side = "
          f"{len(moved_elig) / len(elig) * 100:.2f}%  ->  {band}")
    v = res["I2"]
    print(f"I-2  RR {t_['RR']:.3f} [{t_['RR_ci'][0]:.3f}, {t_['RR_ci'][1]:.3f}] "
          f"{'in band' if v['RR_in_band'] else 'OUT OF BAND'}; joint "
          f"{'within' if v['joint_within_10pp'] else 'OUTSIDE'} 10pp; F-5 "
          f"{'> 1' if v['F5_above_1'] else 'NOT > 1'}; floors "
          f"{'met' if v['floors_met'] else 'NOT met'}")
    print(f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
