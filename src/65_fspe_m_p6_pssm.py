#!/usr/bin/env python3
"""
65_fspe_m_p6_pssm.py - P6: does dFSPE-M beat a position-specific scoring matrix from a homolog
                       alignment, reduced the same way?

P6 as frozen in section 4
-------------------------
    The strictest baseline: a position-specific scoring matrix built from a homolog alignment for
    each panel protein, reduced the same way as FSPE-M. Supported if FSPE-M exceeds the PSSM version
    by at least 0.05 AUROC at p < 0.0083. **If the PSSM matches FSPE-M, the honest report is that a
    multiple sequence alignment does this as well as a protein language model**, and that is a
    publishable and more useful result than a marginal win for the model.

That last sentence is why this test matters more than its threshold. The sixth amendment concluded
from P2 that the axis measures evolutionary constraint rather than hazard. P6 can turn that from an
inference into a demonstration: if an alignment reproduces the metric, the metric is measuring what
alignments measure.

The reduction, mirrored exactly
------------------------------
For each query position i with wild-type residue w, from the recruited alignment column:

    f(a, i) = (count of a in the column + 0.5) / (depth + 10)      add-0.5 over 20 residues
    s(i)    = log f(w, i) - log mean_{a != w} f(a, i)
    dPSSM   = mean s over catalytic positions - mean s over background positions

**The same positions dFSPE-M used**, read from `results/fspe_m.json`, so the only thing that differs
between dFSPE-M and dPSSM is where the per-position score comes from: a masked language model or a
column of homologues.

⚠️ The AUROC comparison P6 specifies inherits the n_benign = 4 power problem the fifth and eighth
amendments measured: the null standard deviation of such an AUROC is 0.167, so a 0.05 margin is
inside the noise and is reported without a pass or fail. The **per-protein agreement** between the two
scores over 19 proteins is far better powered, and is the comparison section 4's own sentence about
alignments is really asking for. It is reported as a descriptive addition, clearly labelled as such,
not as a redefinition of P6.

Usage:
    python src/65_fspe_m_p6_pssm.py --db .external/db/uniprot_sprot.fasta
    python src/65_fspe_m_p6_pssm.py --db ... --limit 3     # smoke test
"""

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from utils import (  # noqa: E402
    load_functional_sites,
    load_positive_sequences,
    sequence_functional_positions,
    truncate_sequence,
)

FSPE_M = ROOT / "results" / "fspe_m.json"
CONTROL_FASTA = ROOT / "data" / "sequences" / "benign_controls.fasta"
OUT = ROOT / "results" / "fspe_m_p6_pssm.json"
AA = "ACDEFGHIKLMNPQRSTVWY"
PSEUDO = 0.5
N_BG, SEED = 20, 42       # src/04's background, which src/62 used and this must reuse exactly
MARGIN = 0.05
N_PERM = 20000


def read_fasta(path):
    out, acc, buf = {}, None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            if acc:
                out[acc] = "".join(buf)
            p = line[1:].split()[0].split("|")
            acc, buf = (p[1] if len(p) > 1 else line[1:].split()[0]), []
        elif acc:
            buf.append(line.strip())
    if acc:
        out[acc] = "".join(buf)
    return out


def read_stockholm(path):
    """name -> aligned sequence, concatenating the interleaved blocks Stockholm may use."""
    rows = {}
    for line in Path(path).read_text().splitlines():
        line = line.rstrip()
        if not line or line.startswith("#") or line == "//":
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        rows[parts[0]] = rows.get(parts[0], "") + parts[1].replace(".", "-").strip()
    return rows


def recruit(acc, seq, db, iters, cpu, workdir):
    """jackhmmer the query against the database; return the recruited alignment rows."""
    q = workdir / f"{acc}.fa"
    q.write_text(f">{acc}\n" + "\n".join(seq[i:i + 60] for i in range(0, len(seq), 60)) + "\n")
    sto = workdir / f"{acc}.sto"
    r = subprocess.run(
        ["jackhmmer", "-N", str(iters), "--cpu", str(cpu), "-A", str(sto),
         "--noali", "-o", str(workdir / f"{acc}.out"), str(q), str(db)],
        capture_output=True, text=True)
    if r.returncode != 0 or not sto.exists():
        return None, (r.stderr or "jackhmmer produced no alignment")[:200]
    return read_stockholm(sto), None


def pssm_scores(query_seq, rows, query_name):
    """s(i) per query position, from the alignment columns. Returns {0-indexed pos: s}."""
    qrow = None
    for name, aligned in rows.items():
        if name.split("/")[0] == query_name or name.startswith(query_name):
            qrow = aligned
            break
    if qrow is None:
        return None, "the query is not present in the recruited alignment"
    others = [v for k, v in rows.items() if v is not qrow]
    if not others:
        return None, "no homologues recruited besides the query"

    # map alignment columns to query positions, skipping columns where the query has a gap
    col_of, qpos = {}, 0
    for col, ch in enumerate(qrow):
        if ch not in "-.":
            col_of[qpos] = col
            qpos += 1
    if qpos != len(query_seq):
        return None, f"query row is {qpos} residues, sequence is {len(query_seq)}"

    out, depth = {}, []
    for pos, col in col_of.items():
        column = [s[col].upper() for s in others if col < len(s)]
        counts = Counter(c for c in column if c in AA)
        n = sum(counts.values())
        depth.append(n)
        if n == 0:
            continue
        denom = n + PSEUDO * len(AA)
        f = {a: (counts.get(a, 0) + PSEUDO) / denom for a in AA}
        w = query_seq[pos].upper()
        if w not in f:
            continue
        others_mean = float(np.mean([p for a, p in f.items() if a != w]))
        out[pos] = math.log(f[w]) - math.log(max(others_mean, 1e-12))
    return out, (float(np.mean(depth)) if depth else 0.0)


def position_sets(acc, seq, sites_all, controls):
    """The SAME functional and background positions src/62 scored, regenerated deterministically.

    src/62 stores the per-position s values but not the positions, so they are rebuilt here from the
    same two deterministic sources it used: the offset-resolved annotations, and RandomState(42) over
    20 draws from the non-functional positions. The counts are asserted against src/62's artifact, so
    a mismatch is an error rather than a silently different comparison.
    """
    if acc in sites_all:
        fs = sites_all[acc]["functional_sites"]
        res = sequence_functional_positions(acc, seq, fs, verbose=False)
        func = sorted({p - 1 for p in res["positions"] if 0 <= p - 1 < len(seq)})
    else:
        v = controls.get(acc)
        if v is None:
            return None, None
        func = sorted({p - 1 for p in v["positions"] if 0 <= p - 1 < len(seq)})
    rng = np.random.RandomState(SEED)
    cand = sorted(set(range(len(seq))) - set(func))
    bg = sorted(int(x) for x in rng.choice(cand, min(N_BG, len(cand)), replace=False))
    return func, bg


def auroc(pos, neg):
    if not len(pos) or not len(neg):
        return None
    return float(np.mean([(1.0 if a > b else 0.5 if a == b else 0.0) for a in pos for b in neg]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="sequence database FASTA for jackhmmer")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--cpu", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    d = json.load(open(FSPE_M))
    sites_all = load_functional_sites()
    controls = json.load(open(ROOT / "data/annotations/benign_control_sites.json"))["verified"]
    controls = {v["uniprot"]: v for v in controls.values()}
    seqs = {}
    for sid, _desc, s in load_positive_sequences():
        p = sid.split("|")
        seqs[p[1] if len(p) > 1 else sid] = s
    seqs.update(read_fasta(CONTROL_FASTA))

    # the SAME positions dFSPE-M scored, so the reduction is the only difference
    targets = []
    for r in d["panel"]:
        targets.append((r["acc"], 1, r))
    for c in d["controls"]:
        targets.append((c["acc"], 0, c))
    if a.limit:
        targets = targets[:a.limit]

    print(f"db={a.db}  iters={a.iters}  cpu={a.cpu}  targets={len(targets)}")
    rows, work = [], Path(tempfile.mkdtemp(prefix="p6_"))
    for acc, is_panel, rec in targets:
        if acc not in seqs:
            rows.append({"acc": acc, "panel": is_panel, "error": "no sequence"})
            continue
        seq = truncate_sequence(seqs[acc], 1022)
        t0 = time.time()
        aln, err = recruit(acc, seq, a.db, a.iters, a.cpu, work)
        if aln is None:
            print(f"  {acc}: jackhmmer failed: {err}")
            rows.append({"acc": acc, "panel": is_panel, "error": err})
            continue
        s, depth = pssm_scores(seq, aln, acc)
        if s is None:
            print(f"  {acc}: {depth}")
            rows.append({"acc": acc, "panel": is_panel, "error": depth})
            continue

        func, bg = position_sets(acc, seqs[acc], sites_all, controls)
        if func is None:
            rows.append({"acc": acc, "panel": is_panel, "error": "no position set"})
            continue
        # the counts must match src/62's, or the two reductions are not on the same positions
        if len(func) != rec["n_functional"] or len(bg) != rec["n_background"]:
            print(f"XX {acc}: regenerated {len(func)}/{len(bg)} positions against src/62's "
                  f"{rec['n_functional']}/{rec['n_background']}")
            return 2

        f = [s[p] for p in func if p in s]
        b = [s[p] for p in bg if p in s]
        dp = (float(np.mean(f)) - float(np.mean(b))) if (f and b) else None
        rows.append({"acc": acc, "panel": is_panel, "n_aligned": len(aln),
                     "mean_column_depth": depth, "seconds": round(time.time() - t0, 1),
                     "n_functional": len(f), "n_background": len(b),
                     "dpssm": dp, "dfspe_m": rec["dfspe_m"],
                     "s_functional": f, "s_background": b})
        print(f"  {acc}: {len(aln)} rows, depth {depth:.0f}, dPSSM "
              f"{'None' if dp is None else f'{dp:+.4f}'} vs dFSPE-M "
              f"{rec['dfspe_m']:+.4f}  [{time.time() - t0:.0f}s]", flush=True)

    ok = [r for r in rows if r.get("dpssm") is not None and r.get("dfspe_m") is not None]
    pan = [r for r in ok if r["panel"] == 1]
    ctl = [r for r in ok if r["panel"] == 0]
    a_pssm = auroc([r["dpssm"] for r in pan], [r["dpssm"] for r in ctl])
    a_dfm = auroc([r["dfspe_m"] for r in pan], [r["dfspe_m"] for r in ctl])
    diff = (a_dfm - a_pssm) if (a_dfm is not None and a_pssm is not None) else None

    # A smoke run with --limit can select no controls at all, which makes every permuted AUROC
    # None. Degrade to the descriptive half rather than crashing on a sum of Nones.
    if ctl and pan:
        rng = np.random.default_rng(0)
        vals = np.array([r["dpssm"] for r in ok], float)
        n1 = len(pan)
        null = np.array([auroc(v[:n1], v[n1:]) for v in
                         (rng.permutation(vals) for _ in range(N_PERM))])
        null_sd = float(null.std())
    else:
        print("!! no control rows scored, so the AUROC half is skipped")
        null_sd = None

    x = np.array([r["dfspe_m"] for r in ok], float)
    y = np.array([r["dpssm"] for r in ok], float)
    pear = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else None
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    spear = float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 else None
    # does the alignment reproduce P1's direction on the panel?
    p1_pssm_k = sum(1 for r in pan if r["dpssm"] > 0)
    p1_pssm_p = sum(math.comb(len(pan), i) for i in range(p1_pssm_k, len(pan) + 1)) / 2 ** len(pan)

    res = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "db": str(a.db),
           "iters": a.iters, "pseudocount": PSEUDO, "n_scored": len(ok),
           "auroc_dfspe_m": a_dfm, "auroc_pssm": a_pssm, "difference": diff,
           "required_margin": MARGIN, "null_auroc_sd": null_sd,
           "margin_inside_noise": bool(diff is not None and null_sd is not None
                                      and abs(diff) < null_sd),
           "descriptive_agreement": {
               "note": ("NOT part of P6 as frozen. Section 4's own sentence asks whether an "
                        "alignment does this as well as a language model, and over 19 proteins "
                        "that is far better powered than the AUROC comparison the threshold names."),
               "pearson": pear, "spearman": spear,
               "pssm_p1_k_above_0": p1_pssm_k, "pssm_p1_n": len(pan),
               "pssm_p1_sign_p": p1_pssm_p},
           "rows": rows}
    res["verdict"] = (
        "INDETERMINATE: the 0.05 margin is inside the sampling noise of an AUROC at "
        f"n_control = {len(ctl)}, null sd {null_sd:.3f}. Reported without a pass or fail, on the "
        "same grounds as step 4's AUROC half and P5's composition half."
        if res["margin_inside_noise"] else
        ("SUPPORTED" if diff is not None and diff >= MARGIN else "NOT SUPPORTED"))
    json.dump(res, open(OUT, "w"), indent=2)

    print()
    print(f"n scored {len(ok)}  (panel {len(pan)}, control {len(ctl)})")
    if a_dfm is not None and a_pssm is not None:
        print(f"AUROC dFSPE-M {a_dfm:.4f}   AUROC dPSSM {a_pssm:.4f}   difference {diff:+.4f}")
        print(f"null AUROC sd {null_sd:.4f}  -> margin inside noise: "
              f"{res['margin_inside_noise']}")
    print(f"\n{res['verdict']}")
    print(f"\nDescriptive, not P6: Pearson {pear:.4f}, Spearman {spear:.4f} between the two "
          f"per-protein scores")
    print(f"  the alignment's own P1: {p1_pssm_k}/{len(pan)} above 0, sign p {p1_pssm_p:.4f}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
