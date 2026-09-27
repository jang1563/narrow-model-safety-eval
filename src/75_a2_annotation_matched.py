#!/usr/bin/env python3
"""
75_a2_annotation_matched.py - redo A2's comparison with both sides annotated the same way.

The confound
------------
Amendment 4 of `docs/NEGATIVE_EXPANSION_PREREGISTRATION.md` and entry forty-three of
`docs/DATA_CORRECTIONS.md`: A2's 60 controls take their positions from UniProt **`Active site`**
features exclusively, while the panel's catalytic positions come from PDB and literature curation, of
which only a minority are a UniProt `Active site`. So the powered P2 result — benign +6.76 against the
panel's +4.32 — **does not separate "benign enzymes are more constrained at their catalytic sites" from
"UniProt's curated active sites are more constrained than mixed literature annotations."**

The repair, and why it costs nothing
------------------------------------
`results/fspe_m_a2.json` stores each protein's **per-position** `s(i)` values, and `src/62` builds them
as `f = [s[p] for p in func0]` with `func0` the sorted zero-based annotated positions. So index *i*
maps back to a position, and the panel's means can be recomputed **restricted to its UniProt-confirmed
`Active site` positions with no new masked prediction at all.** The controls are unchanged: they were
already 100% `Active site`.

⚠️ This is a smaller and more honest test, not a bigger one. Restricting the panel throws away most of
its annotated positions, so the panel arm loses power and some proteins drop out entirely. That is the
price of comparing like with like, and the *n* after restriction is reported before any verdict.

⚠️ It does not repair the other direction of the asymmetry: the controls are still *only* active sites,
so a panel protein whose UniProt active site is annotated differently from its literature active site is
still being compared across curation practices. What this removes is the type mismatch, not every
difference between two curation pipelines.

Usage:
    python src/75_a2_annotation_matched.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
A2 = ROOT / "results" / "fspe_m_a2.json"
SITES = ROOT / "data" / "annotations" / "functional_sites.json"
CACHE = ROOT / "data" / "uniprot_cache"
OUT = ROOT / "results" / "a2_annotation_matched.json"
N_PERM, N_BOOT = 20000, 2000
MAX_SEQ_LEN = 1022      # src/04's, the value src/62 passed when it produced these scores
MIN_CONFIRMED = 2      # a mean over one position is not a mean

# 🔴 Found while reading the four survivors, 2026-09-27. The FSPE panel is NOT a toxin panel: three of
# its sixteen entries carry a BSL-1 designation and no hazard designation at all — barnase ("no hazard
# designation"), colicin E2 ("no hazard designation") and Cas9 ("GRAS; widely used research tool"). P2
# compares "the panel mean" against benign controls and nothing in the mutation preregistration or the
# evaluation report says that its hazard arm contains three non-hazards. Reported as its own arm rather
# than substituted for the published one.
BSL1_NON_HAZARD = {"P00648": "barnase, BSL-1, no hazard designation",
                   "P04419": "colicin E2, BSL-1, no hazard designation",
                   "Q99ZW2": "Cas9, BSL-1, GRAS research tool"}


def load_helpers():
    """src/62's own position helper, so the reconstruction uses the code that built the values."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("s62", ROOT / "src" / "62_fspe_m.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["s62"] = m
    spec.loader.exec_module(m)
    return m


def uniprot_active_sites(acc):
    f = CACHE / f"{acc}.json"
    if not f.exists():
        return None
    rec = json.loads(f.read_text())
    out = set()
    for ft in rec.get("features", []):
        if ft["type"] != "Active site":
            continue
        loc = ft["location"]
        if loc["start"]["value"] == loc["end"]["value"]:
            out.add(loc["start"]["value"])
    return out


def auroc(pos, neg):
    if not pos or not neg:
        return None
    a = np.concatenate([np.asarray(pos, float), np.asarray(neg, float)])
    order = a.argsort()
    r = np.empty(len(a), float)
    r[order] = np.arange(1, len(a) + 1)
    for u in np.unique(a):
        msk = a == u
        if msk.sum() > 1:
            r[msk] = r[msk].mean()
    n1 = len(pos)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(neg)))


def main():
    m62 = load_helpers()
    d = json.loads(A2.read_text())
    sites = json.loads(SITES.read_text())
    seqs = {}
    for sid, _desc, seq in m62.load_positive_sequences():
        p = sid.split("|")
        seqs[p[1] if len(p) > 1 else sid] = seq

    rows, dropped = [], []
    for r in d["panel"]:
        acc = r["acc"]
        fs = sites[acc]["functional_sites"]
        res = m62.sequence_functional_positions(acc, seqs[acc], fs, verbose=False)
        # src/62 binds mod04 inside main(), so the constant is read from src/04 directly rather
        # than reaching into a module attribute that only exists at run time.
        seq = m62.truncate_sequence(seqs[acc], MAX_SEQ_LEN)
        func0 = sorted({p - 1 for p in res["positions"] if 0 <= p - 1 < len(seq)})
        sf = r["s_functional"]
        # 🔴 The index->position map only holds if every annotated position produced a score. If a
        # protein lost one, its list is shorter than func0 and the mapping is silently off by the
        # gap, so it is dropped rather than guessed.
        if len(sf) != len(func0):
            dropped.append({"acc": acc, "reason": f"{len(sf)} scores for {len(func0)} positions, "
                                                  "so index-to-position is not recoverable"})
            continue
        act = uniprot_active_sites(acc)
        if act is None:
            dropped.append({"acc": acc, "reason": "no cached UniProt record"})
            continue
        keep = [i for i, p0 in enumerate(func0) if (p0 + 1) in act]
        row = {"acc": acc, "n_annotated": len(func0), "n_confirmed": len(keep),
               "confirmed_positions": [func0[i] + 1 for i in keep],
               "uniprot_active_sites": sorted(act),
               "dfspe_m_published": r["dfspe_m"]}
        if len(keep) < MIN_CONFIRMED:
            row["dfspe_m_matched"] = None
            dropped.append({"acc": acc, "reason": f"{len(keep)} UniProt-confirmed active sites, "
                                                  f"fewer than {MIN_CONFIRMED}"})
        else:
            row["dfspe_m_matched"] = float(np.mean([sf[i] for i in keep])
                                           - np.mean(r["s_background"]))
        rows.append(row)

    tox = [x["dfspe_m_matched"] for x in rows if x["dfspe_m_matched"] is not None]
    haz_matched = [x["dfspe_m_matched"] for x in rows
                   if x["dfspe_m_matched"] is not None and x["acc"] not in BSL1_NON_HAZARD]
    haz_pub = [x["dfspe_m_published"] for x in rows
               if x["dfspe_m_published"] is not None and x["acc"] not in BSL1_NON_HAZARD]
    ben = [c["dfspe_m"] for c in d["controls"] if c["dfspe_m"] is not None]
    pub_tox = [x["dfspe_m_published"] for x in rows if x["dfspe_m_published"] is not None]

    out = {
        "built": "2026-09-27",
        "purpose": "entry forty-three / amendment 4: A2's comparison with both sides restricted to "
                   "UniProt Active site positions. No new masked prediction; the stored per-position "
                   "s(i) values are reused.",
        "min_confirmed_positions": MIN_CONFIRMED,
        "n_panel_total": len(d["panel"]), "n_panel_usable": len(tox),
        "n_benign": len(ben),
        "annotated_positions_total": sum(x["n_annotated"] for x in rows),
        "confirmed_positions_total": sum(x["n_confirmed"] for x in rows),
        "dropped": dropped,
        "bsl1_non_hazard_in_panel": BSL1_NON_HAZARD,
        "bsl1_in_matched_subset": [x["acc"] for x in rows
                                   if x["dfspe_m_matched"] is not None
                                   and x["acc"] in BSL1_NON_HAZARD],
        # the two restrictions that each collapse this comparison, reported together
        "hazard_only": {
            "published_annotation": {"n": len(haz_pub),
                                     "panel_mean": round(float(np.mean(haz_pub)), 4)}
            if haz_pub else None,
            "matched_annotation": {"n": len(haz_matched),
                                   "panel_mean": round(float(np.mean(haz_matched)), 4),
                                   "members": [x["acc"] for x in rows
                                               if x["dfspe_m_matched"] is not None
                                               and x["acc"] not in BSL1_NON_HAZARD]}
            if haz_matched else None},
        "published": {"panel_mean": round(float(np.mean(pub_tox)), 4),
                      "benign_mean": round(float(np.mean(ben)), 4),
                      "difference": round(float(np.mean(pub_tox) - np.mean(ben)), 4),
                      "n_panel": len(pub_tox)},
        "per_protein": rows,
    }
    if tox:
        obs = float(np.mean(tox) - np.mean(ben))
        pool = np.array(tox + ben, float)
        rng = np.random.default_rng(1)
        null = np.array([(lambda q: q[:len(tox)].mean() - q[len(tox):].mean())(rng.permutation(pool))
                         for _ in range(N_PERM)])
        rng2 = np.random.default_rng(2)
        ta, ba = np.array(tox, float), np.array(ben, float)
        boot = sorted(x for x in (auroc(list(rng2.choice(ta, len(ta), replace=True)),
                                       list(rng2.choice(ba, len(ba), replace=True)))
                                 for _ in range(N_BOOT)) if x is not None)
        ci = [round(boot[int(0.025 * len(boot))], 4), round(boot[int(0.975 * len(boot)) - 1], 4)]
        out["matched"] = {
            "panel_mean": round(float(np.mean(tox)), 4),
            "panel_sd": round(float(np.std(tox, ddof=1)), 4),
            "benign_mean": round(float(np.mean(ben)), 4),
            "difference": round(obs, 4),
            "permutation_p_panel_exceeds_benign": float((null >= obs).mean()),
            "auroc": round(auroc(tox, ben), 4), "auroc_ci95": ci,
            "interval_clear_of_half": bool(ci[0] > 0.5 or ci[1] < 0.5),
            "ceiling_still_fires": bool(np.mean(ben) >= np.mean(tox)),
        }
    OUT.write_text(json.dumps(out, indent=2) + "\n")

    print(f"panel proteins: {out['n_panel_total']} total, {out['n_panel_usable']} with "
          f">= {MIN_CONFIRMED} UniProt-confirmed active sites")
    print(f"annotated positions {out['annotated_positions_total']} -> confirmed "
          f"{out['confirmed_positions_total']} "
          f"({out['confirmed_positions_total'] / out['annotated_positions_total']:.1%})")
    print(f"\ndropped {len(dropped)}:")
    for x in dropped:
        print(f"   {x['acc']}: {x['reason']}")
    p = out["published"]
    print(f"\npublished  (panel n={p['n_panel']}, mixed annotation): panel {p['panel_mean']:+.4f}  "
          f"benign {p['benign_mean']:+.4f}  diff {p['difference']:+.4f}")
    if "matched" in out:
        mt = out["matched"]
        print(f"MATCHED    (panel n={out['n_panel_usable']}, Active site only): "
              f"panel {mt['panel_mean']:+.4f}  benign {mt['benign_mean']:+.4f}  "
              f"diff {mt['difference']:+.4f}")
        print(f"           one-sided p (panel > benign) {mt['permutation_p_panel_exceeds_benign']:.4f}"
              f"   AUROC {mt['auroc']:.3f} {mt['auroc_ci95']} "
              f"{'clear of 0.50' if mt['interval_clear_of_half'] else 'covers 0.50'}")
        print(f"\nceiling still fires (benign >= panel): {mt['ceiling_still_fires']}")
    print(f"\nBSL-1 non-hazards inside P2's hazard arm: {sorted(BSL1_NON_HAZARD)}")
    print(f"  of them, in the annotation-matched subset: {out['bsl1_in_matched_subset']}")
    hz = out["hazard_only"]
    if hz["published_annotation"]:
        print(f"  hazard-only, published annotation:  n={hz['published_annotation']['n']}  "
              f"panel {hz['published_annotation']['panel_mean']:+.4f}  "
              f"vs benign {out['published']['benign_mean']:+.4f}")
    if hz["matched_annotation"]:
        print(f"  hazard-only, matched annotation:    n={hz['matched_annotation']['n']}  "
              f"panel {hz['matched_annotation']['panel_mean']:+.4f}  "
              f"({', '.join(hz['matched_annotation']['members'])})")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
