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
import sys
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "data" / "annotations"   # the panel picks the output file at runtime
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
    ap.add_argument(
        "--panel",
        default="v2",
        choices=["v2", "v3"],
        help="Panel version. Switches the input FASTA and the output annotation together. v3's "
        "members were masked as no_structure by src/27 because foldseek is a Linux binary and the "
        "existing 3Di strings were produced on the HPC; running this with --panel v3 there is what "
        "replaces the mask with real structure.",
    )
    a = ap.parse_args()
    work = Path(a.workdir)
    work.mkdir(parents=True, exist_ok=True)
    pv = a.panel
    OUT = ANN / f"structure_3di_{pv}.json"

    pos = read_fasta(ROOT / f"data/sequences/toxins_positive_{pv}.fasta")
    neg = read_fasta(ROOT / f"data/sequences/benign_negatives_{pv}.fasta")
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

    # 🔴 Reproduction check, 2026-09-27. v3's file was built by src/27 inheriting v2's 231 real
    # strings and masking the rest, so a v3 rebuild recomputes those 231 from AFDB and foldseek. If
    # any of them comes back different, the source or the tool has moved and every 3Di number in the
    # repository is in question -- so it is compared rather than overwritten silently.
    prior, changed = {}, []
    if OUT.exists():
        prior = json.load(open(OUT)).get("proteins", {})
    for fid, rec in ann.items():
        old_rec = prior.get(fid)
        if old_rec and old_rec.get("status") == "ok" and rec["status"] == "ok" \
                and old_rec["threedi"] != rec["threedi"]:
            changed.append(fid.split("|")[1])

    payload = {"built": time.strftime("%Y-%m-%d %H:%M:%S"),
               "panel": pv,
               "source": "AlphaFold DB v6 cif + foldseek structureto3didescriptor",
               "afdb_url_template": AFDB,
               "reproduction": {
                   "previously_ok_entries": sum(1 for r in prior.values()
                                                if r.get("status") == "ok"),
                   "changed_threedi": sorted(changed),
                   "note": ("entries that were already status ok and whose 3Di string changed on "
                            "this rebuild; a non-empty list means AFDB or foldseek moved under the "
                            "published strings")},
               "note": "positions without usable structure carry the SaProt mask '#', "
                       "which makes those proteins sequence-only rather than dropped",
               "stats": stats, "proteins": ann}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(OUT, "w"), indent=2)
    print(f"\n{stats}")
    if changed:
        print(f"XX {len(changed)} previously-ok entries changed their 3Di string: {changed[:8]}")
        print("   AFDB or foldseek has moved under the published strings. Do not use this file")
        print("   until that is explained.")
    else:
        print(f"OK every previously-ok entry reproduced its 3Di string "
              f"({payload['reproduction']['previously_ok_entries']} checked)")
    print(f"wrote {OUT}")
    return 2 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
