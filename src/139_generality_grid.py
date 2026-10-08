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
# 🔒 `spawn` passes the parent's sys.path to each child, so putting src/ here is what lets a
# worker child import `at_fold_worker` and unpickle the fold body.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
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

# 🔒 Study AU (docs/IMAGE_AXIS_REPAIR_PREREGISTRATION.md): the SAME embeddings and the same
# fold, with the held-out unit changed from fine class to SUPERCLASS. Kept out of CELLS so
# study AT's frozen grid is untouched -- these are a different study, not extra AT cells.
REPAIR = {
    "image_sc/clip_vit_b32": ("image_sc", "openai/clip-vit-base-patch32"),
    "image_sc/dinov2_vits14": ("image_sc", "facebook/dinov2-small"),
}

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


def data_image_superclass(encoder, cell):
    """Study AU: the identical embedding cache, relabelled by SUPERCLASS.

    🔒 Nothing is re-embedded. The fine-class cache built for study AT is reused and only `y`
    changes, so the vectors underlying both studies are bit-identical and any difference in
    the result is the axis and nothing else.
    """
    base = cell.replace("image_sc/", "image/")
    z = np.load(cache_for(base), allow_pickle=True)
    _, fine_name, sup_name = _cifar100()
    is_pos = np.isin(sup_name, ANIMAL_SUPER)
    fine_pos, sup_pos = fine_name[is_pos], sup_name[is_pos]
    # 🔒 the cache stores y as the fine-class name in the SAME row order, so the mapping from
    # fine class to superclass is enough -- and it is checked rather than assumed
    cached = z["y"]
    if len(cached) != len(fine_pos) or not np.array_equal(cached, fine_pos):
        raise SystemExit(f"{cache_for(base).name}'s row order does not match CIFAR-100's; "
                         "the superclass relabelling would be misaligned")
    return z["P"].astype(np.float64), z["N"].astype(np.float64), sup_pos


def get_data(cell):
    modality, spec = {**CELLS, **EXTRA, **REPAIR}[cell]
    if modality == "protein":
        return data_protein(spec)
    if modality == "text":
        return data_text(spec, cell)
    if modality == "image_sc":
        return data_image_superclass(spec, cell)
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
# 🔒 The worker body lives in `src/at_fold_worker.py`, which has an IMPORTABLE name. It was
# in this file until 2026-10-07 and every spawned child died with
# `ModuleNotFoundError: No module named '139_'` as soon as another file imported this one and
# used the pool -- see that module's docstring. `--verify-parallel` re-checks bit-identity.
from at_fold_worker import one_class as _one_class  # noqa: E402
from at_fold_worker import winit as _winit  # noqa: E402
from at_fold_worker import winit_serial as _winit_serial  # noqa: E402


def score_cell(cell, seeds, m30, m133, jobs=1, draw=12345):
    P, N, y = get_data(cell)
    X = np.vstack([P, N])
    pos = np.arange(len(P))
    neg = np.arange(len(P), len(P) + len(N))
    # 🔒 `draw` is the negative-PARTITION seed, 12345 everywhere in this project. A second
    # value re-draws the union/test split, which is study AR's stability test -- the one that
    # showed this rho moves 0.105 between draws while the levels move 0.023.
    perm = np.random.default_rng(draw).permutation(len(neg))
    cut = 2 * len(neg) // 3
    union, test = neg[perm[:cut]], neg[perm[cut:]]
    cats = sorted(set(y.tolist()))
    print(f"  {cell}: P {P.shape} N {N.shape}  {len(cats)} classes  "
          f"{len(union)} union / {len(test)} test", flush=True)

    mar, hi_pos, hi_neg = margin_parts(P, X[union], y, m30)
    # 🔒 Library versions, recorded because a rho reproduced to 6e-08 on the margin still
    # moved 0.0044 on the recoveries between 2026-09-30 and 2026-10-07, and the older
    # artifact records no versions so the cause cannot be confirmed.
    import sklearn
    import scipy
    res = {"cell": cell, "draw": draw, "dim": int(P.shape[1]), "n_classes": len(cats),
           "versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                        "sklearn": sklearn.__version__, "python": sys.version.split()[0]},
           # 🔑 Spearman over n classes has a QUANTUM: one adjacent swap moves it by
           # 12/(n(n^2-1)). Differences below it are not resolvable, and this project quotes
           # rho to four decimals.
           "rho_quantum_one_adjacent_swap": 12.0 / (len(cats) * (len(cats) ** 2 - 1)),
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
            import os
            # 🔴 Each worker ran multi-threaded BLAS, so 8 processes x 8 threads thrashed 8
            # cores -- load average 214. Process parallelism wants ONE BLAS thread per
            # worker. Set before the pool spawns: with the `spawn` start method children
            # inherit os.environ at spawn time, while the parent's already-loaded BLAS keeps
            # its threads for the one-off margin matmuls.
            keep = {k: os.environ.get(k) for k in
                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}
            os.environ.update({k: "1" for k in keep})
            try:
                with cf.ProcessPoolExecutor(
                        max_workers=jobs, initializer=_winit,
                        initargs=(str(mm), X.shape, np.float64, union, test,
                                  seeds)) as ex:
                    got = list(ex.map(_one_class, tasks))
            finally:
                for k, was in keep.items():
                    if was is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = was
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
            **at2(rho, rivals),
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


def at2(rho, rivals):
    """AT-2: does margin's |rho| exceed class size and both halves of its own definition?

    🔴 `all(abs(rho) > abs(x) for x in rivals.values())` read False when a rival's rho was
    NaN, so a rival with NO VARIANCE counted as unbeatable. Every CIFAR-100 fine class holds
    exactly 500 images, so `class_size` is constant there and src/30's Spearman correctly
    returns NaN -- and that made AT-2 fail on a technicality.

    🔒 A constant rival carries no information and cannot be a rival. It is DROPPED and named,
    rather than silently skipped or silently fatal.
    ⚠️ On the image cell this does not change the verdict: `nearest_positive` beats margin
    there on merit, +0.609 against +0.523 and +0.637 against +0.542.
    """
    live = {k: x for k, x in rivals.items() if x is not None and np.isfinite(x)}
    dropped = sorted(set(rivals) - set(live))
    return {
        "AT2_beats_all_rivals": bool(live) and all(abs(rho) > abs(x)
                                                   for x in live.values()),
        "AT2_rivals_scored": sorted(live),
        "AT2_rivals_dropped_no_variance": dropped,
        "AT2_beaten_by": sorted(k for k, x in live.items() if abs(x) >= abs(rho)),
    }


def draw_artifact(seed):
    """Where a second-draw run writes, DERIVED from the artifact it is measuring.

    🔴 This was inline as `OUT.with_name(f"generality_grid_draw{seed}.json")` -- a hardcoded
    stem. Study AU ran with `--out results/image_axis_repair.json` and its second draw still
    wrote into study AT's stability file. It is a function now so a test can call it.
    """
    return OUT.with_name(f"{OUT.stem}_draw{seed}.json")


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


def score_predictions(cells, m30):
    """Section 5's four predictions, scored from the grid. Returns None until it is complete.

    🔒 The grid is the unit of the claim, so no prediction is scored on a partial grid --
    the same rule the verdict follows.
    🔒 P4 is gated on AT-0b per amendment 2: it is the one CROSS-CELL LEVEL claim here, which
    is what AT-0b was always for. Both the gated and ungated numbers are reported.
    """
    scored = {k: v for k, v in cells.items() if k in CELLS}
    if len(scored) != len(CELLS):
        return None
    v = verdict(cells)

    # P2: the weakest modality, by mean rho over its cells and both settings
    by_mod = {}
    for k, cv in scored.items():
        by_mod.setdefault(k.split("/")[0], []).extend(
            cv["settings"][s]["AT1_rho"] for s in ("fixed", "swept"))
    mod_mean = {m: float(np.mean(r)) for m, r in by_mod.items()}
    weakest = min(mod_mean, key=mod_mean.get)

    # P3: in every interpretable cell, the lowest-margin class has margin <= 0 and sits in
    # the bottom third of that cell's recovery ranking
    p3 = {}
    for k, cv in scored.items():
        if not cv["interpretable"]:
            continue
        low = min(cv["margin"], key=cv["margin"].get)
        per = {}
        for setting in ("fixed", "swept"):
            rec = cv["settings"][setting]["recovery"]
            if low not in rec:
                per[setting] = None
                continue
            order = sorted(rec, key=lambda c: rec[c]["mean"])
            per[setting] = bool(cv["margin"][low] <= 0.0
                                and order.index(low) < max(1, len(order) // 3))
        p3[k] = {"lowest_margin_class": low, "margin": cv["margin"][low], "by_setting": per}

    # P4: tuning gain against embedding dimension, across cells
    def gain_vs_dim(keys):
        if len(keys) < 3:
            return None
        return m30._spearman([scored[k]["dim"] for k in keys],
                             [scored[k]["tuning_gain"] for k in keys])
    gated = sorted(k for k in scored if scored[k].get("fpr_in_band"))
    rho_gated, rho_all = gain_vs_dim(gated), gain_vs_dim(sorted(scored))

    return {
        "P1_AT1_at_least_5_of_6": {
            "kind": "derived",
            "held": bool(v["AT1_clears"] >= 5),
            "values": {"amended": [v["AT1_clears"], v["AT1_denominator"]],
                       "strict": [v["AT1_clears_strict"], v["AT1_denominator_strict"]]}},
        "P2_image_cells_are_the_weakest_modality": {
            "kind": "reasoned",
            "held": bool(weakest == "image"),
            "values": {"mean_rho_by_modality": mod_mean, "weakest": weakest}},
        "P3_floor_class_has_nonpositive_margin_and_bottom_third_recovery": {
            "kind": "derived",
            "held": bool(p3) and all(x["by_setting"][s] is True
                                     for x in p3.values() for s in ("fixed", "swept")),
            "values": p3},
        "P4_tuning_gain_tracks_embedding_dimension": {
            "kind": "derived",
            "held": bool(rho_gated is not None and rho_gated >= 0.5),
            "values": {"rho_AT0b_gated": rho_gated, "cells_gated": gated,
                       "rho_all_cells": rho_all,
                       "note": ("AT-0b gates this per amendment 2: it is a cross-cell LEVEL "
                                "claim. Both numbers are reported; the gated one is primary. "
                                "None means fewer than three cells qualified.")}},
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
    pr = score_predictions(cells, load("30_margin_across_arms.py"))
    if pr is not None:
        merged["predictions"] = pr
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
    # 🔒 The second-draw artifact's name must DERIVE from OUT, or one study's stability run
    # lands in another's file -- which it did on 2026-10-07.
    # ⚠️ The first version of this check compared `Path.with_name` against itself and so could
    # not fail: restoring the hardcoded stem left the selftest green. It calls the real
    # function now, against a rebound OUT.
    global OUT
    _keep = OUT
    try:
        for stem in ("generality_grid", "image_axis_repair", "zzz_other"):
            OUT = ROOT / f"results/{stem}.json"
            got = draw_artifact(67890).name
            if got != f"{stem}_draw67890.json":
                print(f"  XX second-draw artifact for {stem}.json is {got}")
                ok = False
    finally:
        OUT = _keep

    # 🔒 save() must MERGE, not overwrite -- two processes scoring different cells dropped
    # each other's work before this. Checked against a real file, since the bug was that the
    # in-memory copy looked right.
    import tempfile
    # ⚠️ `global OUT` is declared once, above, in the second-draw-name check; Python allows
    # only one declaration and it must precede every use in the function.
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

    # 🔒 AT-2's NaN handling. A rival with no variance cannot be a rival, and the old
    # `abs(rho) > abs(nan)` made it unbeatable.
    r = at2(0.80, {"class_size": float("nan"), "nearest_positive": 0.5,
                   "nearest_negative": -0.3})
    if not r["AT2_beats_all_rivals"]:
        print(f"  XX a constant rival still blocks AT-2: {r}")
        ok = False
    if r["AT2_rivals_dropped_no_variance"] != ["class_size"]:
        print(f"  XX the dropped rival is not named: {r}")
        ok = False
    # and a rival that genuinely beats margin must still fail it, and be named
    r = at2(0.52, {"class_size": float("nan"), "nearest_positive": 0.61,
                   "nearest_negative": -0.14})
    if r["AT2_beats_all_rivals"] or r["AT2_beaten_by"] != ["nearest_positive"]:
        print(f"  XX a real rival did not fail AT-2 or was not named: {r}")
        ok = False
    # a tie must not count as beating
    if at2(0.5, {"a": 0.5})["AT2_beats_all_rivals"]:
        print("  XX a tie counted as beating the rival")
        ok = False
    # and if EVERY rival is constant there is nothing to beat, which is not a pass
    if at2(0.9, {"a": float("nan")})["AT2_beats_all_rivals"]:
        print("  XX AT-2 passed with no scorable rival at all")
        ok = False

    # 🔒 section 5's predictions, each driven BOTH ways. Untested prediction scoring is how a
    # study reports a verdict nobody checked.
    def gridcell(dim, rho, gain, low_margin, low_is_worst, fpr=True):
        """A synthetic cell with just the fields score_predictions reads."""
        cats = ["z", "b", "c", "d", "e", "f"]
        mar = {c: 0.5 for c in cats}
        mar["z"] = low_margin
        rec = {c: {"mean": 0.9 - 0.01 * i} for i, c in enumerate(cats)}
        rec["z"] = {"mean": 0.01 if low_is_worst else 0.99}
        st = {"AT1_rho": rho, "recovery": rec,
              "AT0a_mean_recovery_over_20pct": True, "AT0b_fpr_in_band": fpr,
              "verdict": "clears" if rho >= BAND_OK else "fails",
              "AT2_beats_all_rivals": True}
        cv = {"dim": dim, "margin": mar, "tuning_gain": gain,
              "settings": {"fixed": dict(st), "swept": dict(st)}}
        cv.update(preconditions(cv["settings"]["fixed"], cv["settings"]["swept"]))
        cv.update(derived_flags(cv["settings"]["fixed"], cv["settings"]["swept"]))
        return cv

    # all six clear, image weakest, floor class negative-margin and worst, gain tracks dim
    dims = {"protein/esm2_650M": 1280, "protein/prott5_xl": 1024,
            "text/minilm_l6": 384, "text/mpnet_base": 768,
            "image/clip_vit_b32": 512, "image/dinov2_vits14": 384}
    g = {k: gridcell(dims[k], 0.7 if k.startswith("image") else 0.9,
                     dims[k] / 10000.0, -0.1, True) for k in CELLS}
    pr = score_predictions(g, m30)
    if pr is None:
        print("  XX score_predictions returned None on a complete grid")
        ok = False
    else:
        for name, want in (("P1_AT1_at_least_5_of_6", True),
                           ("P2_image_cells_are_the_weakest_modality", True),
                           ("P3_floor_class_has_nonpositive_margin_and_bottom_third_recovery",
                            True),
                           ("P4_tuning_gain_tracks_embedding_dimension", True)):
            if pr[name]["held"] is not want:
                print(f"  XX {name} = {pr[name]['held']}, expected {want}: "
                      f"{pr[name]['values']}")
                ok = False
    # now drive each one the other way
    g2 = {k: gridcell(dims[k], 0.1 if k.startswith("image") else 0.9,
                      dims[k] / 10000.0, -0.1, True) for k in CELLS}
    if score_predictions(g2, m30)["P1_AT1_at_least_5_of_6"]["held"]:
        print("  XX P1 held with only 4 of 6 clearing")
        ok = False
    g3 = {k: gridcell(dims[k], 0.9 if k.startswith("image") else 0.7,
                      dims[k] / 10000.0, -0.1, True) for k in CELLS}
    if score_predictions(g3, m30)["P2_image_cells_are_the_weakest_modality"]["held"]:
        print("  XX P2 held with image the STRONGEST modality")
        ok = False
    # floor class with a POSITIVE margin must break P3
    g4 = dict(g)
    g4["text/minilm_l6"] = gridcell(384, 0.9, 0.0384, +0.2, True)
    if score_predictions(g4, m30)["P3_floor_class_has_nonpositive_margin_and_bottom_third_recovery"]["held"]:
        print("  XX P3 held with a positive-margin floor class")
        ok = False
    # floor class that is the BEST recovered must break P3 too
    g5 = dict(g)
    g5["text/mpnet_base"] = gridcell(768, 0.9, 0.0768, -0.1, False)
    if score_predictions(g5, m30)["P3_floor_class_has_nonpositive_margin_and_bottom_third_recovery"]["held"]:
        print("  XX P3 held with the floor class the best recovered")
        ok = False
    # gain running AGAINST dimension must break P4
    g6 = {k: gridcell(dims[k], 0.9, -dims[k] / 10000.0, -0.1, True) for k in CELLS}
    p4 = score_predictions(g6, m30)["P4_tuning_gain_tracks_embedding_dimension"]
    if p4["held"] or p4["values"]["rho_AT0b_gated"] > -0.5:
        print(f"  XX P4 held with gain running against dimension: {p4['values']}")
        ok = False
    # 🔒 and AT-0b must GATE P4: with fewer than three cells in band it is unscorable, not
    # silently computed on the rest
    g7 = {k: gridcell(dims[k], 0.9, dims[k] / 10000.0, -0.1, True,
                      fpr=k.startswith("protein")) for k in CELLS}
    p4 = score_predictions(g7, m30)["P4_tuning_gain_tracks_embedding_dimension"]
    if p4["values"]["rho_AT0b_gated"] is not None or p4["held"]:
        print(f"  XX P4 was scored on 2 in-band cells: {p4['values']}")
        ok = False
    if p4["values"]["rho_all_cells"] is None:
        print("  XX P4 does not report the ungated number alongside")
        ok = False

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
    ap.add_argument("--out", metavar="PATH",
                    help="write to PATH instead of results/generality_grid.json -- study AU "
                         "uses this so the repaired axis cannot land in AT's frozen grid")
    ap.add_argument("--second-draw", type=int, metavar="SEED",
                    help="re-score the named cells with a different negative PARTITION seed "
                         "and write to results/generality_grid_draw<SEED>.json -- study AR's "
                         "stability test. The primary artifact is never touched.")
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
    if a.out:
        # 🔒 Rebinding the module-level OUT is what `save()`, `verdict()` and the second-draw
        # path all read, so one assignment keeps every write on the same artifact.
        global OUT
        OUT = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
        print(f"writing to {OUT.relative_to(ROOT)}", flush=True)

    if a.second_draw:
        # 🔒 A SEPARATE artifact, so a stability measurement cannot contaminate the verdict
        # it is measuring. The frozen draw stays 12345 in results/generality_grid.json.
        if a.second_draw == 12345:
            raise SystemExit("--second-draw 12345 is the frozen draw, not a second one")
        # 🔴 Was `OUT.with_name(f"generality_grid_draw{seed}.json")` -- a HARDCODED stem.
        # With `--out results/image_axis_repair.json` study AU's second draw still wrote to
        # study AT's stability artifact, merging two studies' cells under AT's label. No data
        # was lost only because the merge is key-wise and the cell names differ.
        alt = draw_artifact(a.second_draw)
        base = json.loads(OUT.read_text()) if OUT.exists() else {"cells": {}}
        got = json.loads(alt.read_text()) if alt.exists() else {
            "built": time.strftime("%Y-%m-%d %H:%M:%S"),
            "study": "AT stability: study AR's second-draw test on the grid",
            "exploratory": ("not a band. The preregistration froze one negative partition; "
                            "this re-draws it to measure how far rho moves, because the "
                            "verdict turns on cells sitting within 0.05 of the floor and "
                            "study AR measured this rho moving 0.105 between draws"),
            "draw": a.second_draw, "frozen_draw": 12345, "cells": {}}
        for cell in (a.cell or sorted(base.get("cells", {}))):
            fresh = score_cell(cell, a.seeds, m30, m133, a.jobs, a.second_draw)
            ref = base.get("cells", {}).get(cell, {})
            row = {"draw": a.second_draw, "settings": {}}
            for setting in ("fixed", "swept"):
                f = fresh["settings"][setting]
                r = ref.get("settings", {}).get(setting, {})
                row["settings"][setting] = {
                    "rho_draw2": f["AT1_rho"], "rho_draw1": r.get("AT1_rho"),
                    "mean_recovery_draw2": f["mean_recovery"],
                    "mean_recovery_draw1": r.get("mean_recovery"),
                    "clears_draw2": f["verdict"] == "clears",
                    "clears_draw1": r.get("verdict") == "clears",
                    "verdict_draw2": f["verdict"],
                    "rivals_draw2": f["rivals"],
                }
                d = row["settings"][setting]
                if d["rho_draw1"] is not None:
                    d["rho_shift"] = d["rho_draw2"] - d["rho_draw1"]
                    d["verdict_flipped"] = d["clears_draw2"] != d["clears_draw1"]
                    print(f"    {cell} {setting}: rho {d['rho_draw1']:+.4f} -> "
                          f"{d['rho_draw2']:+.4f}  ({d['rho_shift']:+.4f})  "
                          f"{'FLIPPED' if d['verdict_flipped'] else 'same verdict'}",
                          flush=True)
            got["cells"][cell] = row
            alt.write_text(json.dumps(got, indent=2) + "\n")
        flips = sorted(f"{c}/{s}" for c, v in got["cells"].items()
                       for s, d in v["settings"].items() if d.get("verdict_flipped"))
        shifts = [abs(d["rho_shift"]) for v in got["cells"].values()
                  for d in v["settings"].values() if "rho_shift" in d]
        got["summary"] = {
            "verdicts_flipped": flips,
            "mean_abs_rho_shift": float(np.mean(shifts)) if shifts else None,
            "max_abs_rho_shift": float(max(shifts)) if shifts else None,
            "reading": ("a cell whose verdict flips between two negative draws is not "
                        "evidence either way, and the grid count should be read with it "
                        "named rather than silently included"),
        }
        alt.write_text(json.dumps(got, indent=2) + "\n")
        print(f"\n  flipped: {flips or 'none'}   mean |shift| "
              f"{got['summary']['mean_abs_rho_shift']}")
        print(f"  wrote {alt.relative_to(ROOT)}")
        return 0

    if a.refresh_flags:
        # 🔒 Pure bookkeeping on numbers already in the artifact: the per-cell flags are a
        # function of the stored settings, so an amendment to how a precondition is READ does
        # not require a single fit to be repeated. The settings themselves are untouched.
        d = json.loads(OUT.read_text())
        # ⚠️ AT-2's flag is recomputed from the stored rivals, so the guard below compares
        # the MEASUREMENTS -- recoveries, rho, FPR, chosen C -- and not the derived verdicts.
        def measured(cells):
            keep = ("recovery", "chosen_C", "AT1_rho", "perm_p", "mean_recovery",
                    "test_fpr", "rivals")
            return json.dumps({k: {s: {kk: sv[kk] for kk in keep if kk in sv}
                                   for s, sv in v["settings"].items()}
                               for k, v in cells.items()}, sort_keys=True)

        before = measured(d["cells"])
        for cell, cv in d["cells"].items():
            F, S = cv["settings"]["fixed"], cv["settings"]["swept"]
            for sv in (F, S):
                sv.update(at2(sv["AT1_rho"], sv["rivals"]))
            cv.update(preconditions(F, S))
            cv.update(derived_flags(F, S))
            print(f"  {cell:<24} interpretable={cv['interpretable']} "
                  f"strict={cv['interpretable_strict']} fpr_in_band={cv['fpr_in_band']} "
                  f"clears={cv['clears_both_settings']} "
                  f"rivals={cv['beats_rivals_both']}")
        if measured(d["cells"]) != before:
            raise SystemExit("XX refreshing flags changed a stored MEASUREMENT; it must not")
        d["verdict_block"] = verdict(d["cells"])
        pr = score_predictions(d["cells"], m30)
        if pr is not None:
            d["predictions"] = pr
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
        if cell not in {**CELLS, **EXTRA, **REPAIR}:
            raise SystemExit(f"unknown cell {cell!r}; "
                             f"known: {sorted({**CELLS, **EXTRA, **REPAIR})}")
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
