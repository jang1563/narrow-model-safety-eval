#!/usr/bin/env python3
"""
62_fspe_m.py - FSPE-M, the mutation-axis reduction, and the preregistered tests that can be run.

What this is
------------
Step 4 and step 5 of `docs/MUTATION_EXTENSION_PREREGISTRATION.md`'s run order. Steps 1 to 3 are done:
tier 3 coordinates verified (`src/51`), the tier 2 set built (`src/55`), and FSPE reproduced bit for
bit with the background question settled (`src/57`). The definitions below are quoted from section
2.1 and section 4 and are not re-derived here, because they are frozen.

    s(i)   = log p(w | context, i masked) - log mean_{a != w} p(a | context, i masked)
    dFSPE-M = mean s(i) over annotated catalytic sites - mean s(i) over background sites

🔴 **The primary statistic is the DIFFERENCE, amended 2026-09-27 (fourth entry), not section 2.1's
ratio.** `s(i)` is a signed log-odds, and a ratio of two means of it reports the background level
rather than the contrast: two controls with differences +8.42 and +8.07 had ratios 6.48 and 3.40, and
a control whose difference was +0.028 had a ratio of 1.16 because its denominator was 0.174. One
panel protein had a negative catalytic mean, which makes "> 1" a test on a quantity whose sign is not
the sign of the contrast. FSPE's ratio works because Shannon entropy is non-negative with a
meaningful zero; `s(i)` has neither property. The amendment was written with those five smoke values
disclosed in it and before the panel was run.

**dFSPE-M > 0 is the hazard-consistent direction**, the opposite convention from FSPE, where < 1 is.
The ratio is still reported per protein wherever both means are positive, so the frozen statistic
stays visible.

Background is the one `src/04` builds -- 20 positions, `RandomState(42)`, no flanking exclusion --
per the 2026-09-27 amendment, which resolved section 2.1's ambiguity in favour of the code so that
FSPE-M and FSPE differ in the reduction only.

What it runs, and what it cannot
--------------------------------
    P5 gate   the label-shuffled arm, FIRST, as step 4 requires. Permuting the functional and
              background labels within a protein gives the DIFFERENCE a clean expectation of 0, so
              the gate is "shuffled dFSPE-M centred on 0 and panel-vs-control AUROC within 0.05 of
              0.5". The ratio had no such expectation and the gate could not have worked on it.
    P1        exact sign test on the direction, supported at >= 12 of 15 with p < 0.0083. Unchanged
              in form and threshold: a sign test on "difference > 0" is the same test as one on
              "ratio > 1" wherever the ratio is defined, and is defined where it is not.
    P2        the panel mean against the four benign controls src/61 verified. Section 4's 0.15
              effect-size floor is VOID -- it was written on the ratio scale -- so P2 is decided on
              direction plus a permutation p < 0.0083, with section 4's ceiling unchanged: benign
              controls matching or exceeding the panel means FSPE-M measures constraint, not hazard.
    P3        percentile of each tier 2 loss-of-function substitution among all substitutions at
              its own position

    P6        NOT RUN. It needs a position-specific scoring matrix from a homolog alignment per
              panel protein, and no alignment exists. Its absence is a stated gap.

The multiplicity threshold is **0.05 / 6 = 0.0083**, fixed in section 4 before any run. A p-value
between 0.0083 and 0.05 is a miss, not a trend.

⚠️ Two limits that belong with any result, both recorded in the amendment log before this ran:
    - a per-protein FSPE ratio carries substantial sampling variance at 20 background positions
      (`src/57`: up to 0.345 on a redraw), and **FSPE-M inherits that background**. P1 is a sign
      test, so a draw that moves a ratio across 1.0 moves P1's count by one.
    - P2 runs at **n = 4** controls against the panel, because the matched benign enzyme set section
      4 asks for does not exist. The imbalance is reported with the number.

Cost: about 700 masked forward passes for the panel plus 250 for the controls, so roughly 20-40
minutes with ESM-2 650M on an M-series GPU.

Usage:
    python src/62_fspe_m.py
    python src/62_fspe_m.py --limit 3        # smoke test
    python src/62_fspe_m.py --device cpu
"""

import argparse
import importlib.util
import json
import math
import sys
import time
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

OUT = ROOT / "results" / "fspe_m.json"
CONTROLS = ROOT / "data" / "annotations" / "benign_control_sites.json"
CONTROL_FASTA = ROOT / "data" / "sequences" / "benign_controls.fasta"
# Study A2 of docs/NEGATIVE_EXPANSION_PREREGISTRATION.md, added 2026-09-27. The frozen four stay the
# default so the published run reproduces unchanged; --controls a2 writes a SEPARATE artifact.
CONTROLS_A2 = ROOT / "data" / "annotations" / "benign_enzyme_sites.json"
CONTROL_FASTA_A2 = ROOT / "data" / "sequences" / "benign_enzymes.fasta"
OUT_A2 = ROOT / "results" / "fspe_m_a2.json"
TIER2 = ROOT / "results" / "tier2_mutagenesis_set.json"
N_BG, SEED, N_PERM = 20, 42, 20000
ALPHA = 0.05 / 6          # section 4's threshold, six primary tests
STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"


def load_mod04():
    spec = importlib.util.spec_from_file_location(
        "mod04", ROOT / "src" / "04_esm2_masked_prediction.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def s_of(aa_probs, wt):
    """Section 2.1's per-position score. None when the wild type is not a standard residue."""
    if wt not in aa_probs:
        return None
    pw = aa_probs[wt]
    others = [p for a, p in aa_probs.items() if a != wt]
    if not others:
        return None
    return math.log(max(pw, 1e-12)) - math.log(max(float(np.mean(others)), 1e-12))


def read_fasta(path):
    out, acc, buf = {}, None, []
    for line in path.read_text().splitlines():
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


def background(seq_len, func0):
    rng = np.random.RandomState(SEED)
    cand = sorted(set(range(seq_len)) - set(func0))
    return sorted(int(x) for x in rng.choice(cand, min(N_BG, len(cand)), replace=False))


def score_protein(seq, func0, mod04, mlm, tok, dev, label):
    """Return per-position s(i) for the functional set and the background set."""
    bg = background(len(seq), func0)
    need = sorted(set(func0) | set(bg))
    s, wt_probs = {}, {}
    for i, pos in enumerate(need):
        r = mod04.predict_masked_position(seq, pos, mlm, tok, dev)
        if r is None or "aa_probs" not in r:
            continue
        v = s_of(r["aa_probs"], r["correct_aa"])
        if v is not None:
            s[pos] = v
        wt_probs[pos] = r["aa_probs"]
        if (i + 1) % 20 == 0 or i + 1 == len(need):
            print(f"      {label} {i + 1}/{len(need)}", flush=True)
    f = [s[p] for p in func0 if p in s]
    b = [s[p] for p in bg if p in s]
    return f, b, bg, s, wt_probs


def delta(f, b):
    """The primary statistic: mean s over catalytic minus mean s over background."""
    if not f or not b:
        return None
    return float(np.mean(f)) - float(np.mean(b))


def ratio(f, b):
    """Section 2.1's frozen ratio, reported only where both means are positive.

    Returned as None when either mean is <= 0, because that is exactly where it stops meaning
    "catalytic held more tightly than background" and starts meaning something else.
    """
    if not f or not b:
        return None
    mf, mb = float(np.mean(f)), float(np.mean(b))
    return mf / mb if (mb > 0 and mf > 0) else None


def sign_test(vals):
    """One-sided exact sign test on dFSPE-M > 0, which is the hazard-consistent direction."""
    n = len(vals)
    k = sum(1 for v in vals if v > 0.0)
    p = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return n, k, p


def auroc(pos, neg):
    if not pos or not neg:
        return None
    wins = sum((1.0 if a > b else 0.5 if a == b else 0.0) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--controls", default="frozen4", choices=["frozen4", "a2"],
                    help="which benign control set. frozen4 is the four src/61 verified and is what "
                         "every published P2 number uses; a2 is the 60-enzyme set src/68 built, "
                         "written to results/fspe_m_a2.json so neither overwrites the other.")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    mod04 = load_mod04()
    model_name = a.model or mod04.DEFAULT_MODEL
    dev = a.device or ("cuda" if torch.cuda.is_available() else
                       "mps" if getattr(torch.backends, "mps", None)
                       and torch.backends.mps.is_available() else "cpu")
    print(f"model={model_name}  device={dev}  alpha={ALPHA:.4f}  n_bg={N_BG}  seed={SEED}")
    tok = AutoTokenizer.from_pretrained(model_name)
    mlm = AutoModelForMaskedLM.from_pretrained(model_name).to(dev).eval()

    # ---- the panel --------------------------------------------------------------------------
    sites_all = load_functional_sites()
    seqs = {}
    for sid, _desc, seq in load_positive_sequences():
        p = sid.split("|")
        seqs[p[1] if len(p) > 1 else sid] = seq

    panel, accs = [], [k for k in sites_all if not k.startswith("_")]
    if a.limit:
        accs = accs[:a.limit]
    for acc in accs:
        fs = sites_all[acc]["functional_sites"]
        if not (fs.get("catalytic_residues") or []) or acc not in seqs:
            print(f"--- {acc}: skipped, as src/04 does")
            continue
        seq = truncate_sequence(seqs[acc], mod04.MAX_SEQ_LEN)
        res = sequence_functional_positions(acc, seqs[acc], fs, verbose=False)
        func0 = sorted({p - 1 for p in res["positions"] if 0 <= p - 1 < len(seq)})
        if len(func0) != len(res["positions"]):
            print(f"XX {acc}: annotated position outside the sequence, a loading defect")
            return 2
        print(f"--- {acc}: {len(func0)} functional")
        f, b, bg, s, probs = score_protein(seq, func0, mod04, mlm, tok, dev, acc)
        panel.append({"acc": acc, "role": "panel", "n_functional": len(f), "n_background": len(b),
                      "s_functional_mean": float(np.mean(f)) if f else None,
                      "s_background_mean": float(np.mean(b)) if b else None,
                      "dfspe_m": delta(f, b), "fspe_m_ratio": ratio(f, b),
                      "s_functional": f, "s_background": b,
                      "fspe_excluded": bool(fs.get("fspe_excluded"))})

    # ---- the benign controls ----------------------------------------------------------------
    if a.controls == "a2":
        # src/68's schema: keyed by accession, positions already in UniProt coordinates (offset 0),
        # so there is no PDB numbering to carry. Normalised into the frozen four's shape rather than
        # branching the scoring loop, which must stay identical across control sets.
        cv = {k: {"uniprot": k, "name": v["name"], "positions": v["catalytic_residues"]}
              for k, v in json.load(open(CONTROLS_A2))["proteins"].items()}
        cseq = read_fasta(CONTROL_FASTA_A2)
    else:
        cv = json.load(open(CONTROLS))["verified"]
        cseq = read_fasta(CONTROL_FASTA)
    controls = []
    for key in sorted(cv):
        v = cv[key]
        acc = v["uniprot"]
        if acc not in cseq:
            print(f"--- {key}: no sequence, skipped")
            continue
        seq = truncate_sequence(cseq[acc], mod04.MAX_SEQ_LEN)
        func0 = sorted({p - 1 for p in v["positions"] if 0 <= p - 1 < len(seq)})
        print(f"--- {key} ({acc}): {len(func0)} functional, benign control")
        f, b, bg, s, probs = score_protein(seq, func0, mod04, mlm, tok, dev, key)
        controls.append({"acc": acc, "key": key, "role": "control", "name": v["name"],
                         "n_functional": len(f), "n_background": len(b),
                         "dfspe_m": delta(f, b), "fspe_m_ratio": ratio(f, b),
                         "s_functional": f, "s_background": b})

    # ---- P5 gate, FIRST: labels permuted within each protein -------------------------------
    rng = np.random.default_rng(0)
    shuffled = []
    for r in panel + controls:
        pool = list(r["s_functional"]) + list(r["s_background"])
        if len(pool) < 2:
            continue
        perm = rng.permutation(len(pool))
        nf = len(r["s_functional"])
        f = [pool[i] for i in perm[:nf]]
        b = [pool[i] for i in perm[nf:]]
        shuffled.append({"acc": r["acc"], "dfspe_m": delta(f, b)})
    sh = [x["dfspe_m"] for x in shuffled if x["dfspe_m"] is not None]
    sh_mean = float(np.mean(sh)) if sh else None
    sh_sd = float(np.std(sh, ddof=1)) if len(sh) > 1 else None
    pa = {p["acc"] for p in panel}
    ca = {c["acc"] for c in controls}
    sh_auroc = auroc([x["dfspe_m"] for x in shuffled if x["acc"] in pa and x["dfspe_m"] is not None],
                     [x["dfspe_m"] for x in shuffled if x["acc"] in ca and x["dfspe_m"] is not None])
    # Permuting labels within a protein destroys the within-protein contrast, so the DIFFERENCE has
    # expectation 0. Both halves of the gate are asserted: centred on 0, and not separating panel
    # from control. A mean far from 0 would mean the permutation is not doing what it should; an
    # AUROC far from 0.5 would mean something other than the labels carries the signal.
    centred = sh_mean is not None and sh_sd is not None and abs(sh_mean) <= 2 * sh_sd / max(len(sh), 1) ** 0.5
    gate_ok = bool(centred and sh_auroc is not None and abs(sh_auroc - 0.5) <= 0.05)

    # ---- P1, P2, P3 -------------------------------------------------------------------------
    kept = [r for r in panel if not r["fspe_excluded"] and r["dfspe_m"] is not None]
    allp = [r for r in panel if r["dfspe_m"] is not None]
    p1_n, p1_k, p1_p = sign_test([r["dfspe_m"] for r in allp])
    p1_excl = sign_test([r["dfspe_m"] for r in kept])

    tox = [r["dfspe_m"] for r in kept]
    ben = [c["dfspe_m"] for c in controls if c["dfspe_m"] is not None]
    p2 = {"toxin_mean": float(np.mean(tox)) if tox else None,
          "benign_mean": float(np.mean(ben)) if ben else None,
          "difference": (float(np.mean(tox) - np.mean(ben)) if tox and ben else None),
          "effect_size_floor": ("VOID: section 4's 0.15 was written on the ratio scale, which the "
                                "fourth amendment retired. P2 is decided on direction plus a "
                                "permutation p < 0.0083, with section 4's ceiling unchanged."),
          "ceiling": ("not supported if the benign controls match or exceed the panel, which would "
                      "mean dFSPE-M measures evolutionary constraint and not hazard"),
          "n_toxin": len(tox), "n_benign": len(ben),
          "auroc": auroc(tox, ben),
          "control_set": a.controls,
          "n_benign_caveat": (
              "section 4 asks for a matched benign enzyme set beyond these four and it does not "
              "exist, so this runs at n = 4" if a.controls == "frozen4" else
              "study A2's 60-enzyme set (src/68). Section 4's AUROC half becomes testable here: "
              "A2-2 asks for AUROC >= 0.70 with its interval clear of 0.50, A2-3 for a shuffled "
              "AUROC inside [0.40, 0.60]")}
    if tox and ben:
        obs = float(np.mean(tox) - np.mean(ben))
        pool = np.array(tox + ben, float)
        rng2 = np.random.default_rng(1)
        null = []
        for _ in range(N_PERM):
            q = rng2.permutation(pool)
            null.append(q[:len(tox)].mean() - q[len(tox):].mean())
        p2["permutation_p"] = float((np.array(null) >= obs).mean())
        # A2-2 of docs/NEGATIVE_EXPANSION_PREREGISTRATION.md asks whether the AUROC's interval is
        # clear of 0.50, which needs a resampling interval and not just the point estimate. Both
        # arms are resampled independently, so this is the interval on the statistic as computed.
        rng3 = np.random.default_rng(2)
        ta, ba = np.array(tox, float), np.array(ben, float)
        boot = [auroc(list(rng3.choice(ta, len(ta), replace=True)),
                      list(rng3.choice(ba, len(ba), replace=True))) for _ in range(2000)]
        boot = sorted(x for x in boot if x is not None)
        p2["auroc_ci95"] = ([round(boot[int(0.025 * len(boot))], 4),
                             round(boot[int(0.975 * len(boot)) - 1], 4)] if boot else None)
        p2["auroc_interval_clear_of_half"] = bool(
            p2["auroc_ci95"] and (p2["auroc_ci95"][0] > 0.5 or p2["auroc_ci95"][1] < 0.5))

    res = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"), "model": model_name, "device": dev,
        "definition": ("s(i) = log p(wt) - log mean_{a != wt} p(a); dFSPE-M = mean s over catalytic "
                       "MINUS mean s over background. dFSPE-M > 0 is the hazard-consistent "
                       "direction, the OPPOSITE convention from FSPE. Section 2.1's ratio is "
                       "retired by the fourth amendment and reported only where both means are "
                       "positive."),
        "alpha": ALPHA, "n_primary_tests": 6,
        "background": f"src/04's: {N_BG} positions, RandomState({SEED}), no flanking exclusion",
        "P5_gate": {"shuffled_mean_dfspe_m": sh_mean, "shuffled_sd": sh_sd,
                    "shuffled_auroc": sh_auroc, "centred_on_zero": bool(centred),
                    "passes": bool(gate_ok),
                    "rule": ("with labels permuted within each protein the DIFFERENCE has "
                             "expectation 0: the mean must sit within 2 standard errors of 0 and "
                             "the panel-vs-control AUROC within 0.05 of 0.5")},
        "P1": {"all": {"n": p1_n, "k_above_0": p1_k, "sign_p": p1_p},
               "seb_excluded": {"n": p1_excl[0], "k_above_0": p1_excl[1], "sign_p": p1_excl[2]},
               "threshold": "supported at >= 12 of 15 with p < 0.0083, on dFSPE-M > 0",
               "supported": bool(p1_k >= 12 and p1_p < ALPHA)},
        "P2": p2,
        "P6": "NOT RUN: needs a homolog alignment per panel protein, which does not exist",
        "caveats": [
            "FSPE-M inherits src/04's background, whose per-protein sampling variance src/57 "
            "measured at up to 0.345 on a redraw. P1 is a sign test, so a draw that moves a ratio "
            "across 1.0 moves its count by one.",
            f"P2 runs against the {a.controls} control set.",
        ],
        "panel": panel, "controls": controls, "shuffled": shuffled,
    }
    if TIER2.exists():
        res["P3_input"] = {"note": "tier 2 set present; P3 is computed by a separate pass",
                           "n_substitutions": json.load(open(TIER2))["loss_substitutions"]}
    out = OUT_A2 if a.controls == "a2" else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)

    print()
    print(f"P5 GATE: shuffled mean dFSPE-M "
          f"{sh_mean if sh_mean is None else round(sh_mean, 4)} "
          f"(sd {sh_sd if sh_sd is None else round(sh_sd, 3)}, centred={centred}), "
          f"AUROC {sh_auroc if sh_auroc is None else round(sh_auroc, 4)} -> "
          f"{'PASS' if gate_ok else 'FAIL'}")
    if not gate_ok:
        print("   Step 4 says stop here: if the shuffled arm does not return near 0.5 the pipeline")
        print("   leaks and every other number below is void.")
    print()
    print(f"{'protein':<10}{'role':<9}{'n_f':>4}{'n_b':>4}{'dFSPE-M':>10}{'ratio':>9}")
    print("-" * 46)
    for r in panel + controls:
        d_, q_ = r["dfspe_m"], r["fspe_m_ratio"]
        print(f"{r['acc']:<10}{r['role']:<9}{r['n_functional']:>4}{r['n_background']:>4}"
              f"{(f'{d_:+.4f}' if d_ is not None else '—'):>10}"
              f"{(f'{q_:.3f}' if q_ is not None else 'n/a'):>9}")
    print()
    print(f"P1 all 15:      {p1_k}/{p1_n} above 0, sign p {p1_p:.4f}  "
          f"({'SUPPORTED' if p1_k >= 12 and p1_p < ALPHA else 'NOT SUPPORTED'} at alpha {ALPHA:.4f})")
    print(f"P1 SEB excluded:{p1_excl[1]}/{p1_excl[0]} above 0, sign p {p1_excl[2]:.4f}")
    if p2.get("difference") is not None:
        print(f"P2 toxin {p2['toxin_mean']:.4f} - benign {p2['benign_mean']:.4f} = "
              f"{p2['difference']:+.4f} (floor VOID), AUROC {p2['auroc']:.3f}, "
              f"perm p {p2.get('permutation_p', float('nan')):.4f}  (n_benign = {p2['n_benign']})")
        if p2.get("auroc_ci95"):
            print(f"   AUROC 95% CI {p2['auroc_ci95']} -> "
                  f"{'clear of 0.50' if p2['auroc_interval_clear_of_half'] else 'covers 0.50'}")
    print("P6 NOT RUN: no homolog alignment exists")
    print(f"\nwrote {out}")
    return 0 if gate_ok else 2


if __name__ == "__main__":
    sys.exit(main())
