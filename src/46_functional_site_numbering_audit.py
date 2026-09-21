#!/usr/bin/env python3
"""
46_functional_site_numbering_audit.py - 5 of 16 functional_sites entries fail a direct
                                        residue-identity check, and the FSPE pipeline masked them.

The defect
----------
`data/annotations/functional_sites.json` carries catalytic residues by number. `src/04` masks each
one on the sequence from `toxins_positive.fasta` (`load_positive_sequences`, indexed `r - 1`). But that
FASTA holds the **precursor** for several proteins, while the annotations are numbered on the **mature
chain or PDB**. Checking each annotated residue against the amino acid its own text names (e.g.
"Tyr80"):

    P02879 Ricin A-chain      576 aa precursor, annotations need +35 (signal peptide)   0/5 direct
    P00648 Barnase            157 aa,           need +47                                 0/5 direct
    P00588 Diphtheria toxin   567 aa,           need +32                                 1/3 direct
    P01552 enterotoxin B      266 aa,           NO offset matches                        0/9 direct
    Q51451 ExoS               453 aa,           offset 0 gives 4/5, residue 234 wrong    4/5 direct

The `use_pdb_numbering: true` entries are all correct (8/8 checkable match directly). The defect is
confined to entries where that flag is absent and the panel sequence is a precursor.

🔑 What the correction did, now that it has been applied. The mis-numbered entries sat in the weak
tail of the published n=15, so the old headline "12/15 below 1.0, sign test p=0.018" was counting
mis-masked proteins as failures. Applying the three clean offsets moved the headline to **13/15
below 1.0, sign test p=0.0037, permutation p=0.0002**, and it moved nothing else: the other twelve
proteins differ from the pre-fix snapshot by at most 2.1e-6, which is forward-pass noise.

⚠️ Per protein the result was MIXED, not uniformly stronger, and an earlier version of this
docstring predicted otherwise. Only two of the three ratios fell:

    P00648 1.283 -> 0.051   fell, and is the one protein that crossed below 1.0
    P00588 0.955 -> 0.562   fell, was already below 1.0
    P02879 1.226 -> 1.230   ROSE, and stays above 1.0

So the entire headline gain came from barnase. Ricin got very slightly worse under its own
correction, which is the useful detail: the offsets were fixed by residue identity and by the
UniProt chain boundary, not by which direction they pushed the metric, and one of them pushed the
wrong way. See `offset_uniqueness()` below for why that is not available for fitting.

⚠️ The concentration in the weak tail is real but it is NOT one-to-one, and an earlier version of
this docstring overstated it as "exactly the five weakest." Verified PRE-FIX ranking (the ordering
the published n=15 had, which is what the concentration argument is about), weakest signal first:

    rank 1  P00648 1.283  mis-numbered (+47)
    rank 2  P02879 1.226  mis-numbered (+35)
    rank 3  P11140 1.073  CLEAN, 5/5 direct, no offset needed
    rank 4  P01552 0.956  mis-numbered, no offset fixes it
    rank 5  P00588 0.955  mis-numbered (+32)
    rank 6  Q51451 0.662  4/5 direct, one mislabelled residue, no offset needed

So the mis-numbered set is ranks 1, 2, 4, 5, 6: it skips rank 3 and reaches past the weakest five.
That gap is the load-bearing part. Abrin sits third-weakest and is verified clean at offset 0
(it has no signal peptide, UniProt Chain 1-251), so its ratio above 1.0 cannot be an artifact of
this defect. Abrin and Ricin are both type-2 RIPs with the same catalytic tetrad, which is why the
post-correction panel still has exactly two sign-flips and both are RIPs. Numbering explains part
of the weak tail; what it does not explain is a mechanism-class signal.

⚠️ P01552 and Q51451 are NOT recomputed here: SEB matches at no offset, so its annotation is wrong in
a way an offset cannot fix, and ExoS is the mislabelled entry from the twelfth corrections entry with
one bad residue (234). Both need re-annotation, not re-indexing.

Usage:
    python src/46_functional_site_numbering_audit.py --model facebook/esm2_t33_650M_UR50D
"""
import argparse
import json
import re
import sys
from pathlib import Path

from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import importlib  # noqa: E402  (must follow the sys.path.insert above)
mp = importlib.import_module("04_esm2_masked_prediction")
from utils import _AA3  # noqa: E402  (same sys.path requirement)

OFFSETS = {"P02879": 35, "P00648": 47, "P00588": 32}  # clean single offsets, verified 5/5, 5/5, 3/3


def read_fasta(path):
    out, cur, buf = {}, None, []
    for line in open(path):
        if line.startswith(">"):
            if cur:
                out[cur] = "".join(buf)
            cur = line[1:].split("|")[1] if "|" in line else line[1:].split()[0]
            buf = []
        else:
            buf.append(line.strip())
    if cur:
        out[cur] = "".join(buf)
    return out


def identity_matches(sites, seq, offset):
    """(n_matched, n_checkable) annotated identities at a given offset."""
    expected = {}
    for k, text in (sites.get("residue_annotations") or {}).items():
        if not k.isdigit():
            continue
        m = re.match(r"([A-Z][a-z]{2})(\d+)", text.strip())
        if m and m.group(1) in _AA3 and int(m.group(2)) == int(k):
            expected[int(k)] = _AA3[m.group(1)]
    cat = set(sites.get("catalytic_residues") or [])
    scored = {p: v for p, v in expected.items() if p in cat}
    n = sum(1 for p, v in scored.items()
            if 0 <= p + offset - 1 < len(seq) and seq[p + offset - 1] == v)
    return n, len(scored)


def offset_uniqueness(sites, seq):
    """Every offset that makes ALL annotated identities land correctly.

    This is the anti-p-hacking check. Each correction in OFFSETS moves a ratio in the direction the
    project's own claim wants, so the offset must be fixed by something other than the outcome. Two
    independent criteria do that, and neither one touches FSPE: the offset is the UNIQUE integer that
    aligns every annotated residue identity, and it independently equals the UniProt signal (or
    signal + propeptide) length preceding the mature chain. A value pinned by two criteria that do
    not mention the metric is not available for fitting.
    """
    _, total = identity_matches(sites, seq, 0)
    if total == 0:
        return {"full_match_offsets": [], "n_checkable": 0, "best": None}
    span = len(seq) - max(sites.get("catalytic_residues") or [0])
    full, best = [], (-1, None)
    for off in range(0, max(span, 0) + 1):
        n, _ = identity_matches(sites, seq, off)
        if n > best[0]:
            best = (n, off)
        if n == total:
            full.append(off)
    return {"full_match_offsets": full, "n_checkable": total,
            "unique": len(full) == 1, "best_n": best[0], "best_offset": best[1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    import torch
    dev = a.device or ("cuda" if torch.cuda.is_available()
                       else "mps" if getattr(torch.backends, "mps", None)
                       and torch.backends.mps.is_available() else "cpu")
    print(f"model {a.model} on {dev}\n")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForMaskedLM.from_pretrained(a.model).to(dev).eval()

    fs = json.load(open(ROOT / "data/annotations/functional_sites.json"))
    seqs = read_fasta(ROOT / "data/sequences/toxins_positive.fasta")
    # Deliberately the PRE-fix snapshot: `fspe_results.json` now holds the CORRECTED values, so
    # reading it here would compare the fix against itself and print a zero difference.
    _pub_src = ROOT / "results/fspe_results_PRE_NUMBERING_FIX_2026_05_22.json"
    if not _pub_src.exists():
        _pub_src = ROOT / "results/fspe_results.json"
    published = {e.get("uniprot_id") or e.get("accession"): e
                 for e in json.load(open(_pub_src))["per_protein"]}

    out = {}
    for acc, off in OFFSETS.items():
        seq = seqs[acc]
        cat = fs[acc]["functional_sites"]["catalytic_residues"]
        wrong = cat
        fixed = [c + off for c in cat]
        rw = mp.evaluate_protein_fspe(acc, seq, wrong, model, tok, dev)
        rf = mp.evaluate_protein_fspe(acc, seq, fixed, model, tok, dev)
        pub = published.get(acc, {}).get("fspe_ratio")
        uniq = offset_uniqueness(fs[acc]["functional_sites"], seq)
        out[acc] = {"name": fs[acc]["name"], "offset": off,
                    "offset_uniqueness": uniq,
                    "offset_is_uniquely_determined": uniq.get("unique") and uniq["full_match_offsets"] == [off],
                    "residues_wrong": wrong, "residues_fixed": fixed,
                    "ratio_wrong": rw["fspe_ratio"], "ratio_fixed": rf["fspe_ratio"],
                    "published_ratio": pub,
                    "func_entropy_wrong": rw["fspe_functional"],
                    "func_entropy_fixed": rf["fspe_functional"],
                    "improved": rf["fspe_ratio"] < rw["fspe_ratio"]}
        print(f"{acc} {fs[acc]['name'][:30]:<30} published {pub}  "
              f"wrong-num {rw['fspe_ratio']:.3f}  corrected {rf['fspe_ratio']:.3f}  "
              f"{'DOWN (stronger)' if rf['fspe_ratio'] < rw['fspe_ratio'] else 'up'}", flush=True)

    print("\noffset uniqueness (the anti-p-hacking check):")
    for acc, v in out.items():
        u = v["offset_uniqueness"]
        print(f"  {acc}: offsets achieving a full {u['n_checkable']}/{u['n_checkable']} identity "
              f"match = {u['full_match_offsets']}  -> uniquely determined: "
              f"{v['offset_is_uniquely_determined']}")

    n_down = sum(v["improved"] for v in out.values())
    all_below_1 = all(v["ratio_fixed"] < 1.0 for v in out.values())
    verdict = (f"correcting the numbering lowered the FSPE ratio for {n_down} of {len(out)} proteins"
               + (", and all three now sit below 1.0, so the mis-numbering made the published headline "
                  "CONSERVATIVE" if all_below_1 else
                  ", a mixed result that needs reading per protein"))
    print(f"\nverdict: {verdict}")
    dest = ROOT / "results/v3/functional_site_numbering_audit.json"
    json.dump({"model": a.model, "recomputed": out,
               "not_recomputed": {
                   "P01552": "matches at no offset (0/9 direct; best over 0-60 is 3/9, tied at +6 "
                             "and +60, so not even a unique candidate). Annotation is wrong beyond "
                             "numbering: these are MHC-II/TCR interface residues on a superantigen "
                             "with no catalytic site. Left at offset 0 so this run isolates the "
                             "three repaired entries.",
                   "Q51451": "mislabelled entry (12th correction). 4/5 direct at offset 0, and "
                             "offset 0 is the best over the whole sweep, so it needs NO offset; "
                             "residue 234 is annotated Trp but the precursor carries Asp and 1HE1 "
                             "resolves only the GAP domain, so it cannot be adjudicated.",
                   "P55981": "FALSE POSITIVE of an earlier pass of this audit. `catalytic_residues` "
                             "is deliberately EMPTY (pore-forming channel, no classical active "
                             "site), so src/04 skips it and no number depends on it. The two "
                             "mature-numbered keys in `residue_annotations` are explicitly "
                             "labelled 'not catalytic' and are never read. An audit that scans "
                             "`residue_annotations` instead of `catalytic_residues` will flag this "
                             "entry as broken; it is not.",
               },
               "verdict": verdict}, open(dest, "w"), indent=2)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
