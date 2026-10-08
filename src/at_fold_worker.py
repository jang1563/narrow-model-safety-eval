#!/usr/bin/env python3
"""The per-class fold body, in an IMPORTABLE module.

🔴 This lived in `src/139_generality_grid.py`. With the `spawn` start method a worker child
unpickles the function by module name, and src/139 is loaded under the name `139_` by a
helper that takes the first four characters of the filename -- which is not an importable
module name. It worked only because running src/139 DIRECTLY put the function in
`__main__`, which spawn can reconstruct by re-running the script. The moment another file
imported src/139 and used its pool, every child died with
`ModuleNotFoundError: No module named '139_'`.

🔒 Filenames beginning with a digit cannot be imported at all, so the fix is a module whose
name is a legal identifier. The arithmetic is unchanged and `--verify-parallel` re-checks
that against an already-scored cell.

⚠️ Parallelism here is numerically exact: each class's fold uses the same 30 seeds, the same
union/test partitions and no shared state.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
_W = {}


def _load(mod):
    """Import a numerically-named sibling module.

    🔴 The first version of this file took a `loader` CALLABLE in `initargs`. That callable
    was src/139's own `load`, whose `__module__` is the un-importable `139_`, so pickling it
    reproduced the exact failure this module was created to fix -- one layer further in.
    🔒 Nothing module-bound crosses the process boundary now: the worker does its own import
    from a path it computes itself.
    """
    spec = importlib.util.spec_from_file_location(mod[:4], SRC / mod)
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod[:4]] = m
    spec.loader.exec_module(m)
    return m


def winit(path, shape, dtype, union, test, seeds):
    """Worker setup. The embedding arrives as a memory-mapped file, not pickled -- at
    1280 x 46,445 float64 that is 475 MB per task and pickling it per class would cost more
    than the fits."""
    _W["X"] = np.memmap(path, dtype=dtype, mode="r", shape=tuple(shape))
    _W["union"], _W["test"], _W["seeds"] = union, test, seeds
    _W["m133"] = _load("133_estimation_repair.py")


def winit_serial(X, union, test, seeds, m133):
    """The serial path uses the SAME body, so there is one implementation of the fold and not
    two that have to be kept in step."""
    _W.update({"X": X, "union": union, "test": test, "seeds": seeds, "m133": m133})


def one_class(task):
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
