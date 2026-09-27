#!/usr/bin/env python3
"""
70_coherent_pooling.py - reductions that keep residues coherent, written as artifacts src/03b reads.

The question
------------
§ 9 of `docs/MECHANISM_GENERALIZATION.md` says "Pooling is not the fix" on the strength of three
reductions: mean 15.7%, CLS 16.4%, per-dimension max 1.9% on beta-lactamase at 30 seeds. § 9.1.1
records why that covers less than it claims — per-dimension max takes an independent maximum in each
of 1,280 dimensions, so dimension 5's value can come from residue 12 and dimension 6's from residue
300. It does not preserve a local signal, it destroys **residue coherence**. Mean and CLS are
whole-protein summaries. Nothing has tested a reduction that keeps a residue, or a window of
residues, intact.

The reductions here all do, and all are **label-free**, which is the point: they can be precomputed
into a fixed feature matrix, so `src/03b` evaluates them **unchanged** and the leave-one-mechanism-out
protocol is not reimplemented. A supervised direction would have to be fitted inside each fold, which
means either a rewritten protocol or one artifact per held-out class; `--supervised` does the latter
and is a separate, later run.

    mean_res      residues only, no special tokens. The control: must reproduce the published mean.
    dev_topk      the k residues furthest from the protein's own mean vector, averaged. Keeps whole
                  residues. k in {1, 5, 10, 20, 50}.
    dev_attn      softmax(beta * ||h_i - mean||) weights over residues. The soft version of dev_topk;
                  beta in {1, 4, 16}.
    win_best      window of width w, mean within the window, the window whose mean deviates most from
                  the protein mean. A motif detector with no learned filter. w in {5, 9, 15, 25}.
    win_max       window means, then per-dimension max ACROSS windows. This is src/02f's max made
                  coherent at the window level: the value in every dimension comes from a real
                  window rather than from an unrelated residue. w in {9, 15}.

🔴 Every grid point is reported. None is nominated as primary after the fact, and the decision rule
is fixed here before the run: **a reduction counts as movement only if beta-lactamase's 5-seed
recovery exceeds mean's, and then only as a screen** — it must survive 30 seeds (`src/03x`) and the
provenance, localization and target-host controls before it means anything, per § 9.1.1's two
standing constraints. There are 14 grid points, so an uncorrected best-of-14 is expected to beat the
control by chance; that is why nothing here is called a result.

Usage:
    python src/70_coherent_pooling.py --panel v2
    # then, for each tag it prints:
    python src/03b_leave_one_mechanism_out.py --panel v2 --tag <tag>
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

TOPK = (1, 5, 10, 20, 50)
BETAS = (1.0, 4.0, 16.0)
WINS_BEST = (5, 9, 15, 25)
WINS_MAX = (9, 15)


def dev_scores(X):
    """||h_i - mean(h)|| per residue. Label-free, and the only ranking any reduction here uses."""
    mu = X.mean(0, keepdims=True)
    return np.linalg.norm(X - mu, axis=1)


def window_means(X, w):
    """Mean vector of every contiguous window of width w, via a cumulative sum."""
    if X.shape[0] <= w:
        return X.mean(0, keepdims=True)
    c = np.concatenate([np.zeros((1, X.shape[1]), X.dtype), np.cumsum(X, 0)], 0)
    return (c[w:] - c[:-w]) / w


def reduce_all(X):
    """{name: vector} for one protein's residue stack. X is float32, [n_res, d]."""
    out = {"mean_res": X.mean(0)}
    d = dev_scores(X)
    order = np.argsort(-d)
    for k in TOPK:
        out[f"dev_topk{k}"] = X[order[:min(k, X.shape[0])]].mean(0)
    for b in BETAS:
        # subtract the max before exponentiating, and scale by the spread so beta means the same
        # thing on a short protein as on a long one
        z = d / (d.std() + 1e-8)
        wts = np.exp(b * (z - z.max()))
        out[f"dev_attn{b:g}"] = (X * (wts / wts.sum())[:, None]).sum(0)
    mu = X.mean(0, keepdims=True)
    for w in WINS_BEST:
        wm = window_means(X, w)
        out[f"win_best{w}"] = wm[int(np.argmax(np.linalg.norm(wm - mu, axis=1)))]
    for w in WINS_MAX:
        out[f"win_max{w}"] = window_means(X, w).max(0)
    return out


def selftest():
    """A positive control: if these reductions cannot find a planted motif, they cannot find a real one.

    Nine consecutive residues are shifted off the distribution and the reductions must localise them:
    dev_topk1 must return a residue from inside the planted run, and win_best9 must return exactly
    the planted window. Mean pooling, by construction, cannot do either — which is the hypothesis
    these reductions exist to test.
    """
    rng = np.random.default_rng(0)
    for n in (3, 8, 40, 400):
        X = rng.normal(size=(n, 1280)).astype(np.float32)
        if n >= 40:
            X[20:29] += 6.0
        r = reduce_all(X)
        assert all(v.shape == (1280,) and np.isfinite(v).all() for v in r.values()), n
        if n >= 40:
            d = dev_scores(X)
            top = int(np.argmax(d))
            assert 20 <= top < 29, (n, top)
            assert np.allclose(r["dev_topk1"], X[top]), n
            wm = window_means(X, 9)
            assert wm.shape == (n - 9 + 1, 1280), (n, wm.shape)
            mu = X.mean(0, keepdims=True)
            assert int(np.argmax(np.linalg.norm(wm - mu, axis=1))) == 20, n
            # and the control must NOT localise it, or the comparison is meaningless
            assert np.linalg.norm(r["mean_res"] - X[top]) > np.linalg.norm(r["dev_topk1"] - X[top])
    print(f"SELFTEST PASS: {len(r)} reductions, planted 9-residue motif localised by dev_topk1 "
          f"and win_best9, and not by mean_res")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="v2", choices=["v2", "v3"])
    ap.add_argument("--selftest", action="store_true",
                    help="run the planted-motif positive control and exit")
    ap.add_argument("--supervised", action="store_true",
                    help="also write one artifact set per held-out mechanism class, ranking "
                         "residues by a direction fitted WITHOUT that class. Not run by default: it "
                         "multiplies the artifacts by the class count and needs its own gate.")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    out = ROOT / "results" / a.panel
    idx_path = out / f"residue_stack_index_{a.panel}.json"
    if not idx_path.exists():
        raise SystemExit(f"{idx_path} missing; run python src/69_residue_stack_embed.py "
                         f"--panel {a.panel}")
    if a.supervised:
        raise SystemExit("--supervised is declared in the docstring and not implemented; it needs a "
                         "per-class gate and is a separate run, not a flag flipped in passing")

    meta = json.loads(idx_path.read_text())
    stack = np.load(out / f"residue_stack_{a.panel}.npy", mmap_mode="r")
    print(f"stack {stack.shape} {stack.dtype}, {meta['total_residues']} residues")

    pooled = {}
    for role in ("positive", "negative"):
        rows = meta["rows"][role]
        print(f"{role}: {len(rows)} proteins")
        t0 = time.time()
        for n, r in enumerate(rows):
            X = np.asarray(stack[r["offset"]:r["offset"] + r["n_res"]], dtype=np.float32)
            for name, v in reduce_all(X).items():
                pooled.setdefault((role, name), []).append(v)
            if (n + 1) % 50 == 0 or n + 1 == len(rows):
                print(f"  {n + 1}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)
        pooled = {k: v for k, v in pooled.items()}

    names = sorted({k[1] for k in pooled})
    # src/03b reads a manifest beside each embedding pair for the row order, so one is written per
    # tag from the residue stack's own index. The ids come from the same FASTA read that produced
    # the stack, which is the FASTA order src/02b and src/03b use.
    written = []
    for name in names:
        tag = f"esm2_650M_{name}"
        for role in ("positive", "negative"):
            arr = np.vstack(pooled[(role, name)]).astype(np.float32)
            np.save(out / f"embeddings_{role}_{a.panel}_{tag}.npy", arr)
        (out / f"embedding_manifest_{a.panel}_{tag}.json").write_text(json.dumps({
            "model": meta["model"], "device": meta["device"], "dry_run_tag": tag,
            "pooling": name, "built": time.strftime("%Y-%m-%d %H:%M:%S"),
            "embedding_dim": int(stack.shape[1]), "max_len": meta["max_len"],
            "note": f"reduction '{name}' of residue_stack_{a.panel}.npy; every tag comes from the "
                    "one forward pass src/69 made, so the reductions differ and nothing else does",
            "positive_rows": [{"row": i, "acc": r["id"], "name": r["id"], "len": r["seq_len"]}
                              for i, r in enumerate(meta["rows"]["positive"])],
            "negative_rows": [{"row": i, "acc": r["id"], "name": r["id"], "len": r["seq_len"]}
                              for i, r in enumerate(meta["rows"]["negative"])],
        }, indent=2) + "\n")
        written.append(tag)

    # the control has to reproduce the published mean, or every comparison below is against a
    # different feature matrix than the one the published number used
    gate = {}
    for role in ("positive", "negative"):
        pub = out / (f"embeddings_{role}_v2_esm2_650M_mean.npy" if a.panel == "v2"
                     else f"embeddings_{role}_v3.npy")
        if pub.exists():
            p = np.load(pub)
            mine = np.vstack(pooled[(role, "mean_res")]).astype(np.float32)
            gate[role] = {"max_abs_delta": float(np.abs(p - mine).max()),
                          "against": pub.name, "shape_ok": p.shape == mine.shape}
    (out / f"coherent_pooling_manifest_{a.panel}.json").write_text(json.dumps({
        "built": time.strftime("%Y-%m-%d %H:%M:%S"), "panel": a.panel,
        "source": f"residue_stack_{a.panel}.npy",
        "reductions": names, "n_grid_points": len(names) - 1,
        "mean_res_gate": gate,
        "decision_rule": ("fixed before the run: a reduction counts as movement only if "
                          "beta-lactamase's 5-seed recovery exceeds mean_res's, and then only as a "
                          "screen requiring 30 seeds plus the provenance, localization and "
                          "target-host controls. 14 grid points, so an uncorrected best-of-14 is "
                          "expected to beat the control by chance."),
        "tags": written,
    }, indent=2) + "\n")

    print(f"\nmean_res gate: {gate}")
    print(f"wrote {len(written)} tagged artifact pairs. Evaluate each with:")
    print(f"  for t in {' '.join(written)}; do "
          f"python src/03b_leave_one_mechanism_out.py --panel {a.panel} --tag $t; done")


if __name__ == "__main__":
    raise SystemExit(main() or 0)
