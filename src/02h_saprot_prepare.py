#!/usr/bin/env python3
"""
02h_saprot_prepare.py - fetch structures and derive the 3Di tokens SaProt needs.

Why
---
Every model tested so far reads sequence only. Beta-lactamase is the one class
where plain alignment beats every embedding method (§6i) and no classifier head
rescues (§6j), and it is a family defined by a conserved fold and active site. A
structure-aware representation is the natural test of whether that information is
recoverable at all.

SaProt consumes a structure-aware alphabet: each position is the amino acid
followed by a Foldseek 3Di letter, so "M" with 3Di "d" becomes the token "Md".
That requires a structure per protein, which AlphaFold DB provides keyed by
UniProt accession.

Three steps, each skipped if its output already exists:
  1. foldseek static binary, since it is not on PyPI and no module provides it
  2. AlphaFold DB structures. ⚠️ the current path is `-model_v6.cif`; the older
     `-model_v4.pdb` returns 404 and made an earlier coverage check report zero
  3. `foldseek structureto3didescriptor` over the downloaded structures

Coverage is 218 of 220 panel proteins. The two without a structure are kept and
given the SaProt mask token "#" at every position, which makes them
sequence-only rather than dropping them and changing the panel.

⚠️ A 3Di string must be the same length as the sequence it describes. AFDB models
the full UniProt entry, so a length mismatch means the panel sequence and the
structure are different records; those are reported and masked rather than
silently truncated.

Output: data/annotations/structure_3di_v2.json

Usage:
    python src/02h_saprot_prepare.py [--workdir /path/for/structures]
"""

import argparse
import concurrent.futures as cf
import json
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "annotations" / "structure_3di_v2.json"
FOLDSEEK_URL = "https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz"
AFDB = "https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v6.cif"


def read_fasta(path):
    recs, acc, seq = [], None, []
    for line in open(path):
        if line.startswith(">"):
            if acc:
                recs.append((acc, "".join(seq)))
            acc, seq = line[1:].split()[0], []
        elif acc:
            seq.append(line.strip())
    if acc:
        recs.append((acc, "".join(seq)))
    return recs


def get_foldseek(work):
    exe = work / "foldseek" / "bin" / "foldseek"
    if exe.exists():
        print(f"foldseek already present at {exe}")
        return exe
    tgz = work / "foldseek.tar.gz"
    print(f"downloading foldseek from {FOLDSEEK_URL}")
    urllib.request.urlretrieve(FOLDSEEK_URL, tgz)
    with tarfile.open(tgz) as t:
        t.extractall(work)
    exe.chmod(0o755)
    print(f"foldseek ready at {exe}")
    return exe


def fetch_structures(accs, sdir):
    sdir.mkdir(parents=True, exist_ok=True)

    def one(acc):
        p = sdir / f"AF-{acc}-F1.cif"
        if p.exists() and p.stat().st_size > 1000:
            return acc, True
        try:
            urllib.request.urlretrieve(AFDB.format(acc=acc), p)
            return acc, p.stat().st_size > 1000
        except Exception:
            p.unlink(missing_ok=True)
            return acc, False

    t0 = time.time()
    with cf.ThreadPoolExecutor(12) as ex:
        res = list(ex.map(one, accs))
    ok = [a for a, v in res if v]
    print(f"structures: {len(ok)}/{len(accs)} in {time.time() - t0:.0f}s")
    return set(ok)


def run_3di(exe, sdir, work):
    tsv = work / "3di.tsv"
    if tsv.exists() and tsv.stat().st_size > 0:
        print(f"reusing {tsv}")
    else:
        print("running foldseek structureto3didescriptor")
        r = subprocess.run([str(exe), "structureto3didescriptor", str(sdir), str(tsv)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"foldseek failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    out = {}
    for line in open(tsv):
        f = line.rstrip("\n").split("\t")
        if len(f) < 3:
            continue
        name = f[0].split()[0]
        acc = name.replace("AF-", "").split("-F1")[0]
        out[acc] = {"seq": f[1], "threedi": f[2]}
    print(f"parsed 3Di for {len(out)} structures")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=str(ROOT / ".saprot_work"))
    a = ap.parse_args()
    work = Path(a.workdir)
    work.mkdir(parents=True, exist_ok=True)

    pos = read_fasta(ROOT / "data/sequences/toxins_positive_v2.fasta")
    neg = read_fasta(ROOT / "data/sequences/benign_negatives_v2.fasta")
    recs = [(f, s, "positive") for f, s in pos] + [(f, s, "negative") for f, s in neg]
    accs = [f.split("|")[1] for f, _, _ in recs]

    exe = get_foldseek(work)
    have = fetch_structures(accs, work / "structures")
    tri = run_3di(exe, work / "structures", work)
    print(f"downloaded {len(have)} structures, foldseek described {len(tri)}")

    ann, stats = {}, {"matched": 0, "length_mismatch": 0, "no_structure": 0}
    for fid, seq, side in recs:
        acc = fid.split("|")[1]
        rec = tri.get(acc)
        if rec is None:
            status, td = "no_structure", "#" * len(seq)
            stats["no_structure"] += 1
        elif len(rec["threedi"]) != len(seq):
            # AFDB models the full UniProt entry; a mismatch means different records
            status, td = "length_mismatch", "#" * len(seq)
            stats["length_mismatch"] += 1
        else:
            status, td = "ok", rec["threedi"]
            stats["matched"] += 1
        ann[fid] = {"acc": acc, "side": side, "len": len(seq),
                    "threedi": td, "status": status}

    payload = {"built": time.strftime("%Y-%m-%d %H:%M:%S"),
               "source": "AlphaFold DB v6 cif + foldseek structureto3didescriptor",
               "afdb_url_template": AFDB,
               "note": "positions without usable structure carry the SaProt mask '#', "
                       "which makes those proteins sequence-only rather than dropped",
               "stats": stats, "proteins": ann}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(OUT, "w"))
    print(f"\n{stats}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
