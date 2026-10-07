#!/usr/bin/env python3
"""Study AT: three modalities, two encoder families each.

`docs/GENERALITY_GRID_PREREGISTRATION.md`, frozen before this file existed.

    margin(C) = mean over members of C of [ max cos to a positive outside C
                                            - max cos to a UNION negative ]

🔒 The union negatives, matching src/125 and src/81 -- the arm the +0.60 floor was set on.
src/138 used all negatives; on 20NG the two rank classes at rho +0.9945 and AS clears the
floor under both, and the preregistration records that before this runs.

🔒 The fold is src/133's constants and src/30's margin, imported. Nothing about the method is
restated here -- only the data. Each cell is independent and checkpointed; the VERDICT is
computed only when all six have landed, because the grid is the unit of the claim.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results/v3"
OUT = ROOT / "results/generality_grid.json"
SEEDS, FLOOR, PERMS = 30, 7, 20000
BAND_OK, BAND_NOT = 0.60, 0.20
ALPHA = 0.05 / 3
AT0_RECOVERY, AT0_FPR = 0.20, (0.045, 0.056)

# modality -> encoder -> how to get (P, N, y)
CELLS = {
    "protein/esm2_650M": ("protein", ""),
    "protein/prott5_xl": ("protein", "_prott5_xl"),
    "text/minilm_l6": ("text", "sentence-transformers/all-MiniLM-L6-v2"),
    "text/mpnet_base": ("text", "sentence-transformers/all-mpnet-base-v2"),
    "image/clip_vit_b32": ("image", "openai/clip-vit-base-patch32"),
    "image/dinov2_vits14": ("image", "facebook/dinov2-small"),
}
# 🔒 ESM-2 35M is kept as a third protein cell and reported, but AT-1 is scored on the
# two-FAMILY row: 650M and ProtT5. The preregistration says so.
EXTRA = {"protein/esm2_35M": ("protein", "_esm2_35M")}

# CIFAR-100's own superclasses, split animal / not-animal. `people` is on the ANIMAL side
# because people are animals; the alternative was considered and rejected before any
# embedding was computed, and the preregistration records that.
ANIMAL_SUPER = (
    "aquatic_mammals", "fish", "insects", "large_carnivores",
    "large_omnivores_and_herbivores", "medium_mammals",
    "non-insect_invertebrates", "people", "reptiles", "small_mammals",
)
POS_TOP = ("comp", "rec", "sci")
NEG_TOP = ("alt", "misc", "soc", "talk")


def load(mod):
    spec = importlib.util.spec_from_file_location(mod[:4], ROOT / "src" / mod)
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod[:4]] = m
    spec.loader.exec_module(m)
    return m


def cache_for(cell):
    return RES / f"at_{cell.replace('/', '_')}.npz"


# ----------------------------------------------------------------- data, per modality
def data_protein(sfx):
    E = np.load(RES / f"embeddings_deepvic{sfx}.npy")
    man = json.loads((RES / f"embedding_manifest_deepvic{sfx}.json").read_text())
    lab, cls = np.array(man["label_of_row"]), np.array(man["class_of_row"])
    if E.shape[0] != len(lab):
        raise SystemExit(f"embedding {E.shape[0]} rows against manifest {len(lab)}")
    pos, neg = np.where(lab == 1)[0], np.where(lab == 0)[0]
    return E[pos].astype(np.float64), E[neg].astype(np.float64), cls[pos]


def data_text(encoder, cell):
    c = cache_for(cell)
    if not c.exists():
        from sentence_transformers import SentenceTransformer
        from sklearn.datasets import fetch_20newsgroups
        rm = ("headers", "footers", "quotes")
        docs = []
        for sub in ("train", "test"):
            d = fetch_20newsgroups(subset=sub, remove=rm, random_state=0)
            docs += [(t, d.target_names[y]) for t, y in zip(d.data, d.target)
                     if len(t.strip()) >= 40]
        pos = [(t, g) for t, g in docs if g.split(".")[0] in POS_TOP]
        neg = [t for t, g in docs if g.split(".")[0] in NEG_TOP]
        enc = SentenceTransformer(encoder)
        kw = dict(batch_size=128, show_progress_bar=False, convert_to_numpy=True,
                  normalize_embeddings=False)
        P = enc.encode([t for t, _ in pos], **kw)
        N = enc.encode(neg, **kw)
        np.savez_compressed(c, P=P.astype(np.float32), N=N.astype(np.float32),
                            y=np.array([g for _, g in pos]))
        print(f"  cached {c.name}: P {P.shape} N {N.shape}", flush=True)
    z = np.load(c, allow_pickle=True)
    return z["P"].astype(np.float64), z["N"].astype(np.float64), z["y"]


def _cifar100():
    """CIFAR-100's own pickles, fetched and parsed directly.

    🔒 No torchvision: the only thing it was used for was downloading the archive and
    exposing `.data`, and both are in the pickles the archive contains. One less dependency
    between the published dataset and the numbers.

    Returns (images uint8 (N,32,32,3), fine names per row, superclass name per row).
    """
    import tarfile
    import urllib.request
    d = ROOT / "data/external/cifar100"
    d.mkdir(parents=True, exist_ok=True)
    inner = d / "cifar-100-python"
    if not (inner / "train").exists():
        tgz = d / "cifar-100-python.tar.gz"
        if not tgz.exists():
            url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
            print(f"  fetching {url}", flush=True)
            urllib.request.urlretrieve(url, tgz)
        with tarfile.open(tgz) as t:
            t.extractall(d, filter="data")
    import pickle

    def rd(name):
        raw = pickle.loads((inner / name).read_bytes(), encoding="latin1")
        # the pickles were written under Python 2, so keys arrive as bytes or str
        return {(k.decode() if isinstance(k, bytes) else k): v for k, v in raw.items()}

    meta, tr = rd("meta"), rd("train")
    fine_names = meta["fine_label_names"]
    coarse_names = meta["coarse_label_names"]
    fine = np.asarray(tr["fine_labels"])
    coarse = np.asarray(tr["coarse_labels"])
    imgs = np.asarray(tr["data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    if len(imgs) != 50000 or len(fine) != 50000:
        raise SystemExit(f"CIFAR-100 train is {len(imgs)} images / {len(fine)} labels, "
                         "expected 50,000 of each")
    missing = set(ANIMAL_SUPER) - set(coarse_names)
    if missing:
        raise SystemExit(f"superclass names not in CIFAR-100 meta: {sorted(missing)}")
    return (imgs,
            np.array([fine_names[int(f)] for f in fine]),
            np.array([coarse_names[int(c)] for c in coarse]))


def data_image(encoder, cell):
    c = cache_for(cell)
    if not c.exists():
        import torch
        from transformers import AutoImageProcessor, AutoModel, CLIPVisionModelWithProjection
        imgs, fine_name, sup_name = _cifar100()
        is_pos = np.isin(sup_name, ANIMAL_SUPER)
        print(f"  CIFAR-100: {is_pos.sum()} animal rows in "
              f"{len(set(fine_name[is_pos]))} fine classes, {(~is_pos).sum()} non-animal in "
              f"{len(set(fine_name[~is_pos]))}", flush=True)
        proc = AutoImageProcessor.from_pretrained(encoder)
        mdl = (CLIPVisionModelWithProjection if "clip" in encoder
               else AutoModel).from_pretrained(encoder).eval()
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        mdl = mdl.to(dev)
        vecs, B = [], 256
        with torch.no_grad():
            for i in range(0, len(imgs), B):
                px = proc(images=list(imgs[i:i + B]),
                          return_tensors="pt")["pixel_values"].to(dev)
                o = mdl(pixel_values=px)
                v = (o.image_embeds if hasattr(o, "image_embeds")
                     else o.last_hidden_state[:, 0])
                vecs.append(v.float().cpu().numpy())
                if i % 10240 == 0:
                    print(f"    {i}/{len(imgs)}", flush=True)
        E = np.vstack(vecs)
        np.savez_compressed(c, P=E[is_pos].astype(np.float32),
                            N=E[~is_pos].astype(np.float32), y=fine_name[is_pos])
        print(f"  cached {c.name}: P {int(is_pos.sum())} N {int((~is_pos).sum())} "
              f"dim {E.shape[1]}", flush=True)
    z = np.load(c, allow_pickle=True)
    return z["P"].astype(np.float64), z["N"].astype(np.float64), z["y"]


def get_data(cell):
    modality, spec = {**CELLS, **EXTRA}[cell]
    if modality == "protein":
        return data_protein(spec)
    if modality == "text":
        return data_text(spec, cell)
    return data_image(spec, cell)


# ----------------------------------------------------------------- the fold, shared
def margin_parts(P, NEGU, y, m30):
    """Returns the margin AND each of its two halves, per class.

    🔒 The negative side is the UNION pool, matching src/125. AT-2 needs the halves, because
    a quantity that correlates with recovery is uninteresting if one of its own components
    does it alone.
    """
    cats = sorted(set(y.tolist()))
    mo, mn = np.empty(len(P)), np.empty(len(P))
    CH = 1024
    for i in range(0, len(P), CH):
        sl = slice(i, min(i + CH, len(P)))
        spp = m30.cos(P[sl], P)
        for r in range(spp.shape[0]):
            spp[r, i + r] = -np.inf
        spp[y[sl][:, None] == y[None, :]] = -np.inf
        mo[sl] = spp.max(1)
        mn[sl] = m30.cos(P[sl], NEGU).max(1)
    return ({c: float((mo[y == c] - mn[y == c]).mean()) for c in cats},
            {c: float(mo[y == c].mean()) for c in cats},
            {c: float(mn[y == c].mean()) for c in cats})


# ---- per-class parallelism -------------------------------------------------------------
# 🔒 Each class's fold is EXACTLY independent: the same 30 seeds, the same union/test
# partitions, no shared state and no cross-class ordering. So a per-class parallel map is
# numerically identical to the serial loop, and `--verify-parallel` proves it on a scored
# cell rather than asserting it.
#
# ⚠️ The embedding is passed as a memory-mapped file, not pickled. At 1280 x 46,445 float64
# that is 475 MB per task, and pickling it per class would cost more than the fits.
_W = {}


def _winit(path, shape, dtype, union, test, seeds):
    _W["X"] = np.memmap(path, dtype=dtype, mode="r", shape=tuple(shape))
    _W["union"], _W["test"], _W["seeds"] = union, test, seeds
    _W["m133"] = load("133_estimation_repair.py")


def _winit_serial(X, union, test, seeds, m133):
    """The serial path uses the SAME worker body, so there is one implementation of the fold
    and not two that have to be kept in step."""
    _W.update({"X": X, "union": union, "test": test, "seeds": seeds, "m133": m133})


def _one_class(task):
    """Returns (cat, recovery dict, mean test FPR, chosen C) for one held-out class."""
    cat, ci, h, tri, setting = task
    X, union, test = _W["X"], _W["union"], _W["test"]
    m133 = _W["m133"]
    cv = 1.0
    if setting == "swept":
        cv, _ = m133.select_C(X, tri, union, ci)
    rr, ff = [], []
    for seed in range(_W["seeds"]):
        r = np.random.default_rng(seed)
        p = r.permutation(len(union))
        tr = union[p[:m133.N_TRAIN]]
        ca = union[p[m133.N_TRAIN:m133.N_TRAIN + m133.N_CAL]]
        mdl = m133.clf(cv).fit(np.vstack([X[tri], X[tr]]),
                               np.r_[np.ones(len(tri)), np.zeros(len(tr))])
        t = float(np.quantile(mdl.predict_proba(X[ca])[:, 1], m133.SPEC))
        rr.append(float((mdl.predict_proba(X[h])[:, 1] >= t).mean()))
        ff.append(float((mdl.predict_proba(X[test])[:, 1] >= t).mean()))
    return (cat, {"n": int(len(h)), "mean": float(np.mean(rr)),
                  "sd": float(np.std(rr, ddof=1))}, float(np.mean(ff)), cv)


def score_cell(cell, seeds, m30, m133, jobs=1):
    P, N, y = get_data(cell)
    X = np.vstack([P, N])
    pos = np.arange(len(P))
    neg = np.arange(len(P), len(P) + len(N))
    perm = np.random.default_rng(12345).permutation(len(neg))
    cut = 2 * len(neg) // 3
    union, test = neg[perm[:cut]], neg[perm[cut:]]
    cats = sorted(set(y.tolist()))
    print(f"  {cell}: P {P.shape} N {N.shape}  {len(cats)} classes  "
          f"{len(union)} union / {len(test)} test", flush=True)

    mar, hi_pos, hi_neg = margin_parts(P, X[union], y, m30)
    res = {"cell": cell, "dim": int(P.shape[1]), "n_classes": len(cats),
           "n_positives": int(len(P)), "n_negatives": int(len(N)),
           "margin": mar, "nearest_positive": hi_pos, "nearest_negative": hi_neg,
           "class_size": {c: int((y == c).sum()) for c in cats},
           "settings": {}}

    mm = None
    if jobs > 1:
        mm = ROOT / f"results/v3/.at_mm_{cell.replace('/', '_')}.dat"
        mm.parent.mkdir(parents=True, exist_ok=True)
        np.memmap(mm, dtype=np.float64, mode="w+", shape=X.shape)[:] = X
    for setting in ("fixed", "swept"):
        rec, fpr, chosen = {}, [], {}
        tasks = []
        for ci, cat in enumerate(cats):
            h = pos[y == cat]
            if len(h) < FLOOR:
                continue
            tasks.append((cat, ci, h, pos[y != cat], setting))
        if jobs > 1:
            import concurrent.futures as cf
            with cf.ProcessPoolExecutor(
                    max_workers=jobs, initializer=_winit,
                    initargs=(str(mm), X.shape, np.float64, union, test, seeds)) as ex:
                got = list(ex.map(_one_class, tasks))
        else:
            _winit_serial(X, union, test, seeds, m133)
            got = [_one_class(t) for t in tasks]
        for cat, r, f, cv in got:
            rec[cat], chosen[cat] = r, cv
            fpr.append(f)
            print(f"    {setting:<6} {cat:<30}{r['mean']*100:6.2f}%  n={r['n']:<5} "
                  f"C={cv:g}  margin {mar[cat]:+.5f}", flush=True)
        ks = sorted(rec)
        rv = [rec[c]["mean"] for c in ks]
        rho = m30._spearman([mar[c] for c in ks], rv)
        rng = np.random.default_rng(0)
        rva = np.array(rv)
        null = np.array([abs(m30._spearman([mar[c] for c in ks], rng.permutation(rva)))
                         for _ in range(PERMS)])
        pval = float((1 + (null >= abs(rho) - 1e-12).sum()) / (PERMS + 1))
        mean_rec = float(np.mean(rv))
        fmin, fmax = float(min(fpr)), float(max(fpr))
        # 🔒 AT-2: margin must beat class size AND both halves of its own definition
        rivals = {
            "class_size": m30._spearman([res["class_size"][c] for c in ks], rv),
            "nearest_positive": m30._spearman([hi_pos[c] for c in ks], rv),
            "nearest_negative": m30._spearman([hi_neg[c] for c in ks], rv),
        }
        res["settings"][setting] = {
            "recovery": rec, "chosen_C": chosen, "mean_recovery": mean_rec,
            "AT1_rho": rho, "perm_p": pval, "significant": bool(pval < ALPHA),
            "n_classes_scored": len(ks),
            "test_fpr": {"min": fmin, "max": fmax},
            # 🔒 preconditions FIRST: a cell failing either is uninterpretable, and is removed
            # from AT-1's numerator AND denominator rather than counted as a failure.
            "AT0a_mean_recovery_over_20pct": bool(mean_rec > AT0_RECOVERY),
            "AT0b_fpr_in_band": bool(AT0_FPR[0] <= fmin and fmax <= AT0_FPR[1]),
            "rivals": rivals,
            "AT2_beats_all_rivals": bool(all(abs(rho) > abs(x) for x in rivals.values())),
            "verdict": ("clears" if rho >= BAND_OK
                        else "fails" if rho <= BAND_NOT else "between bands"),
        }
        v = res["settings"][setting]
        print(f"    {setting:<6} rho {rho:+.4f}  p {pval:.5f}  mean {mean_rec*100:5.2f}%  "
              f"FPR {fmin*100:.2f}-{fmax*100:.2f}%  AT-0 "
              f"{'ok' if v['AT0a_mean_recovery_over_20pct'] and v['AT0b_fpr_in_band'] else 'FAIL'}"
              f"  rivals {'beaten' if v['AT2_beats_all_rivals'] else 'NOT beaten'}"
              f"  -> {v['verdict']}", flush=True)

    if mm is not None and mm.exists():
        mm.unlink()
    res["jobs"] = jobs
    F, S = res["settings"]["fixed"], res["settings"]["swept"]
    # 🔒 Amendment 2: TWO readings, both stored. `interpretable_strict` is § 3 as frozen
    # (AT-0a and AT-0b). `interpretable` is AT-1's own condition, AT-0a alone, because AT-1 is
    # a rank correlation WITHIN a cell and AT-0b is a condition on comparing levels ACROSS
    # cells. AT-0b continues to gate P4, the only cross-cell level claim here.
    # ⚠️ The amendment was written after the first cell failed AT-0b, which is disclosed; the
    # strict reading is the one with no hindsight in it and is reported alongside.
    res.update(preconditions(F, S))
    res.update(derived_flags(F, S))
    res["tuning_gain"] = S["mean_recovery"] - F["mean_recovery"]
    return res


def preconditions(F, S):
    """The two readings of section 3's preconditions, as a pure function so it is testable.

    🔒 `interpretable_strict` is section 3 as frozen: AT-0a and AT-0b, both settings.
    🔒 `interpretable` is AT-1's own condition, AT-0a alone, per amendment 2 -- AT-1 is a rank
    correlation WITHIN a cell and AT-0b is a condition on comparing levels ACROSS cells.
    🔒 `fpr_in_band` is kept separately because it gates P4, the one cross-cell level claim.
    """
    a = bool(F["AT0a_mean_recovery_over_20pct"] and S["AT0a_mean_recovery_over_20pct"])
    b = bool(F["AT0b_fpr_in_band"] and S["AT0b_fpr_in_band"])
    return {"interpretable_strict": a and b, "interpretable": a, "fpr_in_band": b}


def derived_flags(F, S):
    """Does the cell clear, fail, and beat its rivals -- on BOTH settings.

    🔴 These were gated on `interpretable` when the cell was scored, so the interpretability
    filter lived in two places: here AND in `verdict()`, which already filters by it. That
    made the flags unrefreshable -- amendment 2 changed how `interpretable` is read, and a
    cell scored under the old reading had `clears_both_settings` frozen at False even though
    both of its settings said "clears". The gate belongs in `verdict()` alone.
    """
    return {
        "clears_both_settings": F["verdict"] == S["verdict"] == "clears",
        "fails_both_settings": F["verdict"] == S["verdict"] == "fails",
        "beats_rivals_both": bool(F["AT2_beats_all_rivals"] and S["AT2_beats_all_rivals"]),
    }


def verdict(cells):
    """AT-1 and AT-1b, read as a conjunction. Uninterpretable cells leave the denominator."""
    scored = {k: v for k, v in cells.items() if k in CELLS}
    interp = {k: v for k, v in scored.items() if v["interpretable"]}
    n = len(interp)
    clears = sum(1 for v in interp.values() if v["clears_both_settings"])
    beats = sum(1 for v in interp.values() if v["beats_rivals_both"])
    # 🔒 § 3 as frozen, reported next to the amended reading so a reader who rejects
    # amendment 2 gets a number rather than an argument
    strict = {k: v for k, v in scored.items() if v.get("interpretable_strict")}
    by_mod = {}
    for k, v in interp.items():
        by_mod.setdefault(k.split("/")[0], []).append(v)
    dead = sorted(m for m, vs in by_mod.items()
                  if len(vs) == 2 and all(x["fails_both_settings"] for x in vs))
    out = {"cells_total": len(CELLS), "cells_run": len(scored),
           "cells_interpretable": n,
           "uninterpretable": sorted(k for k, v in scored.items()
                                     if not v["interpretable"]),
           "AT1_clears": clears, "AT1_denominator": n,
           "AT1_clears_strict": sum(1 for v in strict.values()
                                    if v["clears_both_settings"]),
           "AT1_denominator_strict": len(strict),
           "cells_failing_AT0b": sorted(k for k, v in scored.items()
                                        if not v.get("fpr_in_band", True)),
           "readings": ("AT1_clears applies AT-0a alone, per amendment 2, because AT-1 is a "
                        "within-cell ordering; AT1_clears_strict applies section 3 as frozen "
                        "(AT-0a and AT-0b). AT-0b gates P4, the cross-cell level claim."),
           "AT1b_modalities_dead_on_both_encoders": dead,
           "AT2_beats_rivals": beats,
           "complete": len(scored) == len(CELLS)}
    if not out["complete"]:
        out["verdict"] = (f"INCOMPLETE -- {len(scored)} of {len(CELLS)} cells; the grid is "
                          "the unit of the claim and no verdict is read until all six land")
        return out
    # 🔴 Amendment 1: AT-1b is IMPLIED by AT-1 at six cells. A modality dead on both its
    # encoders is two non-clearing cells, leaving at most four clears -- already inside AT-1's
    # modality-dependent band. It stays computed and REPORTED, because which modality is
    # weakest is worth knowing, and it is out of the verdict because it cannot change one.
    out["AT1b_implied_by_AT1"] = True
    out["verdict"] = ("general" if clears >= 5 else "does not generalize" if clears <= 2
                      else "modality-dependent")
    out["AT2_verdict"] = ("separable" if beats >= 5 else "not separable" if beats <= 2
                          else "mixed")
    return out


def save(out, mine):
    """Write the artifact by MERGING this process's cells into whatever is on disk.

    🔴 Each process read the artifact at startup and wrote its own in-memory copy back, so two
    processes scoring different cells silently dropped each other's work -- the second writer
    won and the first cell vanished. A cell is expensive enough that losing one to a race is
    not acceptable, and the race is invisible: the file is valid JSON either way.

    🔒 Cells this process scored overwrite; everything else on disk is kept.
    """
    disk = json.loads(OUT.read_text()) if OUT.exists() else {}
    cells = dict(disk.get("cells", {}))
    for k in mine:
        cells[k] = out["cells"][k]
    merged = {**disk, **out, "cells": cells}
    merged["verdict_block"] = verdict(cells)
    OUT.write_text(json.dumps(merged, indent=2) + "\n")
    return merged


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
    if (BAND_OK, BAND_NOT) != (0.60, 0.20):
        print(f"  XX bands are {(BAND_OK, BAND_NOT)}, frozen at (0.60, 0.20)")
        ok = False
    L = ROOT / "results/deepvic_loco.json"
    if L.exists():
        b = json.loads(L.read_text())["bands"]["rho_replicate_floor"]
        if b != BAND_OK:
            print(f"  XX AI-2's floor is {b}, this uses {BAND_OK}")
            ok = False
    if len(CELLS) != 6 or len({k.split("/")[0] for k in CELLS}) != 3:
        print(f"  XX the grid is {len(CELLS)} cells over "
              f"{len({k.split('/')[0] for k in CELLS})} modalities, frozen at 6 over 3")
        ok = False
    if len(ANIMAL_SUPER) != 10:
        print(f"  XX {len(ANIMAL_SUPER)} animal superclasses, CIFAR-100's split is 10")
        ok = False
    # margin must order a planted geometry, and its halves must be returned separately
    rng = np.random.default_rng(0)
    A = rng.normal(size=(40, 16)) + 5.0
    B = rng.normal(size=(40, 16)) + 5.0
    Cc = rng.normal(size=(40, 16)) - 5.0
    NEG = rng.normal(size=(200, 16)) - 5.0
    Pm = np.vstack([A, B, Cc])
    yc = np.array(["a"] * 40 + ["b"] * 40 + ["c"] * 40)
    mar, hp, hn = margin_parts(Pm, NEG, yc, m30)
    if not (mar["a"] > mar["c"] and mar["b"] > mar["c"]):
        print(f"  XX margin does not order a planted geometry: {mar}")
        ok = False
    if not all(abs(mar[c] - (hp[c] - hn[c])) < 1e-9 for c in mar):
        print("  XX margin is not its two halves subtracted")
        ok = False
    # ⚠️ an UNINTERPRETABLE cell must leave the denominator, not count as a failure. A fake
    # grid checks it, because getting this backwards is the error
    # EXTERNAL_VALIDATION_PREREGISTRATION made in the other direction.
    def cellv(interp=True, strict=True, clears=True, fails=False, rivals=True, fpr=True):
        return {"interpretable": interp, "interpretable_strict": strict,
                "clears_both_settings": clears, "fails_both_settings": fails,
                "beats_rivals_both": rivals, "fpr_in_band": fpr}

    fake = {k: cellv() for k in CELLS}
    if verdict(fake)["verdict"] != "general":
        print(f"  XX a 6/6 clearing grid is not general: {verdict(fake)}")
        ok = False
    one_bad = dict(fake)
    one_bad["image/clip_vit_b32"] = cellv(interp=False, strict=False, clears=False,
                                          rivals=False)
    v = verdict(one_bad)
    if v["AT1_denominator"] != 5 or v["AT1_clears"] != 5 or v["verdict"] != "general":
        print(f"  XX an uninterpretable cell did not leave the denominator: {v}")
        ok = False
    # a modality dead on both encoders must override a 5-of-6 count
    dead = {k: cellv() for k in CELLS}
    for k in ("image/clip_vit_b32", "image/dinov2_vits14"):
        dead[k] = cellv(clears=False, fails=True, rivals=False)
    v = verdict(dead)
    # 🔒 Amendment 1: assert the IMPLICATION, not an override that cannot be reached. A dead
    # modality must force the clearing count below AT-1's floor of 5 -- that is the whole
    # content of AT-1b, and if the band structure ever changes so that it does not hold, this
    # fails. ⚠️ The previous assertion passed with the override DELETED, which is how the
    # redundancy was found.
    if v["AT1b_modalities_dead_on_both_encoders"] != ["image"]:
        print(f"  XX a dead modality was not detected: {v}")
        ok = False
    if v["AT1_clears"] >= 5:
        print(f"  XX a dead modality left the count at AT-1's floor: {v['AT1_clears']}")
        ok = False
    if v["verdict"] != "modality-dependent":
        print(f"  XX a dead modality did not read as modality-dependent: {v['verdict']}")
        ok = False
    if not v.get("AT1b_implied_by_AT1"):
        print("  XX the artifact does not record that AT-1b is implied")
        ok = False
    # 🔒 save() must MERGE, not overwrite -- two processes scoring different cells dropped
    # each other's work before this. Checked against a real file, since the bug was that the
    # in-memory copy looked right.
    import tempfile
    global OUT
    keep = OUT
    try:
        with tempfile.TemporaryDirectory() as td:
            OUT = Path(td) / "grid.json"
            save({"cells": {"protein/esm2_650M": cellv()}, "tag": "A"},
                 ["protein/esm2_650M"])
            m = save({"cells": {"text/minilm_l6": cellv()}, "tag": "B"},
                     ["text/minilm_l6"])
            if sorted(m["cells"]) != ["protein/esm2_650M", "text/minilm_l6"]:
                print(f"  XX save() did not merge: {sorted(m['cells'])}")
                ok = False
            if m.get("tag") != "B":
                print(f"  XX save() did not take the writer's own top-level fields: {m}")
                ok = False
            # and a re-score of an existing cell must REPLACE it
            m = save({"cells": {"text/minilm_l6": cellv(clears=False)}, "tag": "C"},
                     ["text/minilm_l6"])
            if m["cells"]["text/minilm_l6"]["clears_both_settings"]:
                print("  XX save() did not replace a re-scored cell")
                ok = False
            if len(m["cells"]) != 2:
                print(f"  XX save() lost a cell on replace: {sorted(m['cells'])}")
                ok = False
    finally:
        OUT = keep

    # and an incomplete grid must refuse to return a verdict at all
    v = verdict({k: fake[k] for k in list(CELLS)[:4]})
    if not v["verdict"].startswith("INCOMPLETE") or v["complete"]:
        print(f"  XX an incomplete grid returned a verdict: {v}")
        ok = False
    # 🔒 Amendment 2: the STRICT reading must be reported and must be able to disagree with
    # the amended one. A cell that passes AT-0a and fails AT-0b -- the text cells' situation
    # -- counts for AT-1 and NOT for the strict count, and both numbers must appear.
    amended = {k: cellv() for k in CELLS}
    for k in ("text/minilm_l6", "text/mpnet_base"):
        amended[k] = cellv(interp=True, strict=False, fpr=False)
    v = verdict(amended)
    if v["AT1_clears"] != 6 or v["AT1_denominator"] != 6:
        print(f"  XX the amended reading did not keep an AT-0b failure: {v['AT1_clears']}"
              f"/{v['AT1_denominator']}")
        ok = False
    if v["AT1_clears_strict"] != 4 or v["AT1_denominator_strict"] != 4:
        print(f"  XX the strict reading did not drop the AT-0b failures: "
              f"{v['AT1_clears_strict']}/{v['AT1_denominator_strict']}")
        ok = False
    if v["cells_failing_AT0b"] != ["text/minilm_l6", "text/mpnet_base"]:
        print(f"  XX AT-0b failures are not named: {v['cells_failing_AT0b']}")
        ok = False
    if "within-cell ordering" not in v["readings"]:
        print("  XX the artifact does not state which reading is which")
        ok = False
    # 🔒 and the per-cell flags themselves, which the verdict fixtures above bypass. Without
    # this, reverting amendment 2 inside score_cell would leave every check above passing.
    def st(a, b):
        return {"AT0a_mean_recovery_over_20pct": a, "AT0b_fpr_in_band": b}

    want = {
        ((True, True), (True, True)): (True, True, True),
        # AT-0a passes, AT-0b fails: counts for AT-1, not for the strict reading
        ((True, False), (True, False)): (False, True, False),
        # AT-0a fails: out of both
        ((False, True), (False, True)): (False, False, True),
        # one setting fails AT-0b and the other does not -- still out of strict
        ((True, True), (True, False)): (False, True, False),
    }
    for (fa, sa), exp in want.items():
        got = preconditions(st(*fa), st(*sa))
        g = (got["interpretable_strict"], got["interpretable"], got["fpr_in_band"])
        if g != exp:
            print(f"  XX preconditions({fa}, {sa}) = {g}, expected {exp}")
            ok = False
    # 🔴 and the derived flags must NOT consult interpretability -- that gate is verdict()'s,
    # and having it here too froze `clears_both_settings` at False for a cell scored before
    # amendment 2 changed how `interpretable` is read.
    def sv(verd, riv):
        return {"verdict": verd, "AT2_beats_all_rivals": riv,
                "AT0a_mean_recovery_over_20pct": False, "AT0b_fpr_in_band": False}

    df = derived_flags(sv("clears", True), sv("clears", True))
    if not (df["clears_both_settings"] and df["beats_rivals_both"]):
        print(f"  XX derived_flags consults interpretability: {df}")
        ok = False
    if derived_flags(sv("clears", True), sv("fails", True))["clears_both_settings"]:
        print("  XX a cell clearing on one setting only was marked as clearing both")
        ok = False
    if not derived_flags(sv("fails", False), sv("fails", False))["fails_both_settings"]:
        print("  XX a cell failing both settings was not marked as failing")
        ok = False
    print("selftest PASS" if ok else "selftest FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cell", action="append", default=[],
                    help="a cell to run, repeatable; default every cell not yet scored")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel workers over held-out classes; the folds are exactly "
                         "independent, and --verify-parallel checks that on a scored cell")
    ap.add_argument("--refresh-flags", action="store_true",
                    help="recompute every scored cell's precondition flags and the verdict "
                         "block from the STORED settings; no refitting, for when the reading "
                         "of a precondition changes by amendment")
    ap.add_argument("--verify-parallel", metavar="CELL",
                    help="re-score CELL with --jobs and compare to the stored serial result")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    t0 = time.time()
    m30, m133 = load("30_margin_across_arms.py"), load("133_estimation_repair.py")

    if a.refresh_flags:
        # 🔒 Pure bookkeeping on numbers already in the artifact: the per-cell flags are a
        # function of the stored settings, so an amendment to how a precondition is READ does
        # not require a single fit to be repeated. The settings themselves are untouched.
        d = json.loads(OUT.read_text())
        before = json.dumps(
            {k: v["settings"] for k, v in d["cells"].items()}, sort_keys=True)
        for cell, cv in d["cells"].items():
            F, S = cv["settings"]["fixed"], cv["settings"]["swept"]
            cv.update(preconditions(F, S))
            cv.update(derived_flags(F, S))
            print(f"  {cell:<24} interpretable={cv['interpretable']} "
                  f"strict={cv['interpretable_strict']} fpr_in_band={cv['fpr_in_band']} "
                  f"clears={cv['clears_both_settings']} "
                  f"rivals={cv['beats_rivals_both']}")
        after = json.dumps(
            {k: v["settings"] for k, v in d["cells"].items()}, sort_keys=True)
        if before != after:
            raise SystemExit("XX refreshing flags changed a stored setting; it must not")
        d["verdict_block"] = verdict(d["cells"])
        OUT.write_text(json.dumps(d, indent=2) + "\n")
        print(f"  AT-1 {d['verdict_block']['AT1_clears']}/"
              f"{d['verdict_block']['AT1_denominator']} amended, "
              f"{d['verdict_block']['AT1_clears_strict']}/"
              f"{d['verdict_block']['AT1_denominator_strict']} strict")
        return 0

    if a.verify_parallel:
        # 🔒 The claim being checked: a per-class parallel map is BIT-IDENTICAL to the serial
        # loop. Checked against a cell already scored serially, not asserted from the fact
        # that the folds look independent.
        cell = a.verify_parallel
        stored = json.loads(OUT.read_text())["cells"].get(cell)
        if stored is None:
            raise SystemExit(f"{cell} is not in {OUT.name}; score it serially first")
        if stored.get("jobs", 1) != 1:
            raise SystemExit(f"{cell} was scored with jobs={stored['jobs']}, "
                             "so it is not a serial reference")
        jobs = max(2, a.jobs)
        print(f"re-scoring {cell} with jobs={jobs} against the stored serial result",
              flush=True)
        fresh = score_cell(cell, a.seeds, m30, m133, jobs)
        bad = []
        for setting in ("fixed", "swept"):
            sv, fv = stored["settings"][setting], fresh["settings"][setting]
            for cat, r in sv["recovery"].items():
                g = fv["recovery"].get(cat)
                if g is None or g["mean"] != r["mean"] or g["sd"] != r["sd"]:
                    bad.append(f"{setting}/{cat}: {r} vs {g}")
            for k in ("AT1_rho", "mean_recovery", "perm_p"):
                if sv[k] != fv[k]:
                    bad.append(f"{setting}/{k}: {sv[k]!r} vs {fv[k]!r}")
            if sv["chosen_C"] != fv["chosen_C"]:
                bad.append(f"{setting}/chosen_C differs")
            if sv["test_fpr"] != fv["test_fpr"]:
                bad.append(f"{setting}/test_fpr: {sv['test_fpr']} vs {fv['test_fpr']}")
        if bad:
            print(f"  XX parallel differs from serial in {len(bad)} place(s):")
            for b in bad[:8]:
                print(f"     {b}")
            return 1
        print(f"  OK parallel == serial, bit-for-bit, on every recovery, C, rho and FPR "
              f"in {cell} ({round(time.time() - t0, 1)}s)")
        return 0

    out = json.loads(OUT.read_text()) if OUT.exists() else {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "study": "AT -- three modalities, two encoder families each",
        "preregistration": "docs/GENERALITY_GRID_PREREGISTRATION.md",
        "bands": {"ok": BAND_OK, "not": BAND_NOT, "alpha": ALPHA,
                  "AT1_general_floor": 5, "AT1_fail_ceiling": 2,
                  "AT0a_recovery": AT0_RECOVERY, "AT0b_fpr": list(AT0_FPR)},
        "negative_side": ("the UNION negatives, matching src/125 and src/81 -- the arm the "
                          "+0.60 floor was set on; src/138 used all negatives and the "
                          "preregistration records the sensitivity"),
        "seeds": a.seeds, "cells": {}}

    todo = a.cell or [k for k in {**CELLS, **EXTRA} if k not in out["cells"]]
    done = []
    for cell in todo:
        if cell not in {**CELLS, **EXTRA}:
            raise SystemExit(f"unknown cell {cell!r}; known: {sorted({**CELLS, **EXTRA})}")
        try:
            out["cells"][cell] = score_cell(cell, a.seeds, m30, m133, a.jobs)
        except FileNotFoundError as e:
            print(f"  -- {cell} skipped: {e}", flush=True)
            continue
        done.append(cell)
        out = save(out, done)
        print(f"  wrote {OUT.name} after {cell} "
              f"({len(out['cells'])} cell(s) on disk)", flush=True)

    out["seconds"] = round(time.time() - t0, 1)
    out = save(out, done)
    v = out["verdict_block"]
    print(f"\nAT-1 {v['AT1_clears']}/{v['AT1_denominator']} clear  "
          f"AT-2 {v['AT2_beats_rivals']}/{v['AT1_denominator']} beat rivals")
    if v["uninterpretable"]:
        print(f"uninterpretable: {v['uninterpretable']}")
    print(f"verdict: {v['verdict']}")
    print(f"wrote {OUT.relative_to(ROOT)} in {out['seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
