#!/usr/bin/env python3
"""
79_external_class_axis_screen.py - screen study B's train and calibrate negatives against its positives.

`docs/EXTERNAL_CLASS_AXIS_PREREGISTRATION.md` § 2: the 1,500 pool proteins that will be fitted on and
calibrated with are screened against all 746 VFDB representatives, and any at or above **0.282** — the
panel negatives' own maximum similarity to a hazard — is excluded and logged.

⚠️ **The 6,758 test negatives are deliberately not screened.** A deployed screen does not get to remove
the proteins it will meet, and screening the test set would make the out-of-sample rate optimistic in
exactly the direction § 2.6.1 of `docs/MECHANISM_GENERALIZATION.md` warns about. This is the one place
where doing less work is the methodologically correct choice, so it is stated rather than assumed.

Aligner is the repository's, unchanged: `Bio.Align.PairwiseAligner` local, BLOSUM62, gap −11/−1,
normalized `score / sqrt(self_i · self_j)`, the same screen as `02d`, `27`, `42` and `74`.

Usage:
    python src/79_external_class_axis_screen.py
"""

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "results" / "external_class_axis_build.json"
POS = ROOT / "data" / "sequences" / "vfdb_class_axis_positives.fasta"
POOL = ROOT / "data" / "sequences" / "benign_pool_large.fasta"
SCREEN_REF = ROOT / "results" / "v3" / "pool_homology_against_panel.json"
OUT = ROOT / "results" / "external_class_axis_screen.json"


# 🔴 BLOSUM62's alphabet is ARNDCQEGHILKMFPSTWYVBZX*: it has X, B and Z but NOT U
# (selenocysteine) or O (pyrrolysine), and 10 of the 1,500 train+calibrate negatives contain them.
# Substituted for the ALIGNMENT ONLY, conservatively — U is a cysteine analogue and O a lysine
# analogue — and logged. The stored sequences are untouched, so nothing the probe sees changes.
# The alternative, dropping those ten, would shrink a partition that was frozen before this was
# known, for a reason that has nothing to do with what they are.
ALIGN_SUB = {"U": "C", "O": "K"}


def for_alignment(s):
    return "".join(ALIGN_SUB.get(c, c) for c in s)


def read_fasta(path):
    out, hid, buf = [], None, []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith(">"):
            if hid is not None:
                out.append((hid, "".join(buf)))
            hid, buf = line[1:].split()[0], []
        elif hid is not None:
            buf.append(line.strip())
    if hid is not None:
        out.append((hid, "".join(buf)))
    return out


def main():
    build = json.loads(BUILD.read_text())
    bound = json.loads(SCREEN_REF.read_text())["panel_negative_reference"]["max"]
    pos = read_fasta(POS)
    pool = read_fasta(POOL)
    pp = build["pool_partition"]
    idx = pp["train_rows"] + pp["calibrate_rows"]
    cand = [pool[i] for i in idx]
    print(f"{len(pos)} representatives, {len(cand)} train+calibrate negatives to screen, "
          f"bound {bound:.6f}")
    print(f"⚠️ {len(pp['test_rows'])} test negatives are NOT screened, by design")

    from Bio.Align import PairwiseAligner, substitution_matrices
    al = PairwiseAligner()
    al.mode = "local"
    al.substitution_matrix = substitution_matrices.load("BLOSUM62")
    al.open_gap_score, al.extend_gap_score = -11, -1
    n_subbed = 0
    pos = [(h, for_alignment(s)) for h, s in pos]
    selfpos = {h: al.score(s, s) for h, s in pos}

    admitted, rejected, t0 = [], [], time.time()
    for n, (h, s0) in enumerate(cand):
        s = for_alignment(s0)
        if s != s0:
            n_subbed += 1
        ss = al.score(s, s)
        best, who = 0.0, None
        for ph, ps in pos:
            sim = al.score(s, ps) / (selfpos[ph] * ss) ** 0.5
            if sim > best:
                best, who = sim, ph
        rec = {"row": idx[n], "id": h, "max_similarity": round(best, 6), "nearest": who}
        (rejected if best >= bound else admitted).append(rec)
        if (n + 1) % 100 == 0 or n + 1 == len(cand):
            el = time.time() - t0
            print(f"  {n + 1}/{len(cand)}  {el / 60:.1f} min  "
                  f"(eta {el / (n + 1) * (len(cand) - n - 1) / 60:.0f} min)  "
                  f"admitted {len(admitted)}", flush=True)

    OUT.write_text(json.dumps({
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "bound": bound, "aligner": "Bio.Align.PairwiseAligner local BLOSUM62 -11/-1, "
                                   "score/sqrt(self_i*self_j), as 02d/27/42/74",
        "n_positives": len(pos), "n_candidates": len(cand),
        "alignment_substitutions": {"map": ALIGN_SUB, "n_candidates_affected": n_subbed,
                                    "note": "U and O are absent from BLOSUM62; substituted for the "
                                            "alignment only, stored sequences untouched"},
        "n_admitted": len(admitted), "n_rejected": len(rejected),
        "test_negatives_unscreened": len(pp["test_rows"]),
        "max_similarity_admitted": max(a["max_similarity"] for a in admitted) if admitted else None,
        "admitted_rows": [a["row"] for a in admitted],
        "rejected": rejected,
    }, indent=2) + "\n")
    print(f"\n{n_subbed} candidate(s) carried U or O, substituted for the alignment only")
    print(f"admitted {len(admitted)} of {len(cand)}; {len(rejected)} at or above {bound:.4f}")
    if admitted:
        print(f"highest similarity among the admitted: "
              f"{max(a['max_similarity'] for a in admitted):.4f}")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
