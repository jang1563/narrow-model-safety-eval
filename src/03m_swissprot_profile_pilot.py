#!/usr/bin/env python3
"""
03m_swissprot_profile_pilot.py - why the external-database profile baseline is
uninformative under the leave-one-mechanism-out margin, demonstrated rather than
asserted.

The idea being tested
---------------------
03k ran HMMER inside the 234-sequence panel and noted the obvious objection:
the panel is homology-screened, so a held-out class has no homologue there for a
profile to recruit, and profile methods earn their sensitivity by recruiting. The
apparent fix is to build the profile from a large external database instead.
Swiss-Prot has real relatives of every family in this panel, so a profile built
from it is a genuinely strong family model.

Why it cannot work here, and the evidence
-----------------------------------------
The decision rule in 03i/03k scores a query as

    margin = max(score vs TRAINING POSITIVES) - max(score vs TRAINING NEGATIVES)

and under leave-one-mechanism-out the training positives are the OTHER mechanism
classes. The held-out family is, by construction, absent from them.

Enriching the query's profile from Swiss-Prot makes it a sharper model of the
query's own family. That does not make the query resemble a different mechanism
class. So the better the profile gets, the more specifically it matches the one
thing that is not being scored, and the margin stays noise.

This script measures that directly on one class. For each held-out member it runs
jackhmmer against Swiss-Prot, builds an HMM from the recruited alignment, scores
the whole panel with it, and reports three things: the self score, whether the
profile finds the member's held-out siblings, and the margin.

Result on rip_rrna_glycosidase, 2026-09-18: every profile found 6 of 6 siblings,
so profile construction plainly worked, while the margin was NEGATIVE on 6 of 7
members (mean -4.2). The profile is an excellent ricin-family detector and a
useless hazard detector, which is the whole point.

This is a pilot on purpose. It is the power check that the full 234-query run
should have to pass first, and it fails it, so the full run was not made. See
docs/EXTERNAL_VALIDATION_PREREGISTRATION.md for the earlier occasion in this
project when a panel was scored before anyone asked whether it could produce a
negative result.

Usage:
    python src/03m_swissprot_profile_pilot.py --db /path/to/uniprot_sprot.fasta
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "results" / "v2"


def read_fasta(p):
    out, acc, seq = {}, None, []
    for line in open(p):
        if line.startswith(">"):
            if acc:
                out[acc] = "".join(seq)
            acc, seq = line[1:].split()[0], []
        elif acc:
            seq.append(line.strip())
    if acc:
        out[acc] = "".join(seq)
    return out


def best_scores(tbl):
    sc = {}
    for line in open(tbl):
        if line.startswith("#"):
            continue
        f = line.split()
        if len(f) >= 6:
            sc[f[0]] = max(sc.get(f[0], 0.0), float(f[5]))
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Swiss-Prot FASTA")
    ap.add_argument("--klass", default="rip_rrna_glycosidase")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--cpu", type=int, default=8)
    a = ap.parse_args()

    man = json.load(open(V2 / "embedding_manifest_v2.json"))
    mech = json.load(open(ROOT / "data/annotations/mechanism_classes_v2.json"))
    pf = read_fasta(ROOT / "data/sequences/toxins_positive_v2.fasta")
    nf = read_fasta(ROOT / "data/sequences/benign_negatives_v2.fasta")
    cls = {p["fasta_id"]: p["mechanism_class"] for p in mech["proteins"]}
    pos = [r["acc"] for r in man["positive_rows"]]
    neg = [r["acc"] for r in man["negative_rows"]]

    hi = [x for x in pos if cls[x] == a.klass]
    tri = set(x for x in pos if cls[x] != a.klass)
    negs = set(neg)
    assert hi, f"no members in class {a.klass}"
    print(f"class {a.klass}: {len(hi)} held out, {len(tri)} training positives, "
          f"{len(negs)} negatives")

    rows = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        panel = td / "panel.fa"
        with open(panel, "w") as fh:
            for x in pos:
                fh.write(f">{x}\n{pf[x]}\n")
            for x in neg:
                fh.write(f">{x}\n{nf[x]}\n")

        print(f"\n{'member':<16}{'self':>7}{'best train+':>12}{'best neg':>10}"
              f"{'margin':>9}{'siblings':>10}")
        print("-" * 64)
        for q in hi:
            qf, sto, hmm, tbl = td / "q.fa", td / "m.sto", td / "q.hmm", td / "p.tbl"
            qf.write_text(f">{q}\n{pf[q]}\n")
            subprocess.run(["jackhmmer", "-N", str(a.iters), "--cpu", str(a.cpu),
                            "-A", str(sto), "-o", "/dev/null", str(qf), a.db], check=True)
            n_msa = sum(1 for ln in open(sto)
                        if ln.strip() and not ln.startswith(("#", "//")))
            subprocess.run(["hmmbuild", "--cpu", str(a.cpu), str(hmm), str(sto)],
                           check=True, stdout=subprocess.DEVNULL)
            subprocess.run(["hmmsearch", "--max", "--cpu", str(a.cpu), "--tblout", str(tbl),
                            "-o", "/dev/null", str(hmm), str(panel)], check=True)
            sc = best_scores(tbl)
            mp = max([v for k, v in sc.items() if k in tri], default=0.0)
            mn = max([v for k, v in sc.items() if k in negs], default=0.0)
            sib = sum(1 for k in sc if k in hi and k != q)
            rows.append({"member": q.split("|")[2], "msa_seqs": n_msa,
                         "self_score": sc.get(q, 0.0), "best_training_positive": mp,
                         "best_negative": mn, "margin": mp - mn,
                         "siblings_found": sib, "siblings_total": len(hi) - 1})
            print(f"{q.split('|')[2]:<16}{sc.get(q, 0):>7.0f}{mp:>12.1f}{mn:>10.1f}"
                  f"{mp - mn:>9.1f}{sib:>7}/{len(hi) - 1}")

    m = np.array([r["margin"] for r in rows])
    sib_all = all(r["siblings_found"] == r["siblings_total"] for r in rows)
    print(f"\nmargin: mean {m.mean():+.1f}, range {m.min():+.1f} to {m.max():+.1f}, "
          f"negative on {int((m < 0).sum())} of {len(m)}")
    print(f"every profile found all siblings: {sib_all}")
    print("\nThe profile works and the margin does not. A family-specific profile "
          "cannot say\nwhether a query resembles a DIFFERENT mechanism class, which "
          "is what the margin asks.")

    out = {"built": "2026-09-18", "class": a.klass, "db": Path(a.db).name,
           "iterations": a.iters,
           "margin_mean": float(m.mean()), "margin_min": float(m.min()),
           "margin_max": float(m.max()),
           "n_margin_negative": int((m < 0).sum()), "n_members": len(rows),
           "all_siblings_found": bool(sib_all),
           "verdict": "uninformative under the LOMO margin: profile construction "
                      "succeeds (all siblings found) while the margin is noise and "
                      "mostly negative, because training positives are other "
                      "mechanism classes",
           "members": rows}
    p = V2 / "swissprot_profile_pilot.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
