"""Ingest VFDB as an external class axis, and measure what it does and does not resolve.

Why this exists. Every class-level number in this project rests on eleven to twelve mechanism
classes that the author defined, which is the binding limitation named in § 1.1b-ter of
`research/05_v2_related_work_survey.md`. VFDB supplies a published alternative: its own category
assignment, maintained by someone else, at larger class count. DeepVIC (Bioinformatics Advances
2026) uses exactly that axis — 14 categories over 12,989 annotated VFs — and runs no
leave-one-category-out evaluation, so the axis is available and the question is unoccupied.

Two things this script establishes before any probe is trained, because both change the design:

1. **"Virulence factor" is not "toxin".** Exotoxin is one VFDB category out of fourteen and a few
   per cent of the records. A panel built on this axis inverts the composition of the v2 panel,
   which is toxin-centric with a ten-member non-toxin virulence control.
2. **Target host needs species resolution, and only the full set has the contrast.** VFDB's
   organism field is the *producing pathogen*, not the target, so host is an inference from the
   pathogen — and § 2.4 of `docs/MECHANISM_GENERALIZATION.md` already records that producer and
   target come apart (six of seven RIPs are plant-produced and act on animal ribosomes). Genus is
   too coarse to infer it from: *Pseudomonas* holds *aeruginosa* (human) beside *syringae* (plant)
   and *entomophila* (insect), and *Bacillus* holds *anthracis* beside *thuringiensis*.

The raw downloads are not committed. The download page states no license, the full protein set is
19 MB, and both are reproducible from here; what is tracked is this script, the aggregate summary,
and nothing that reproduces VFDB's table.

Usage:
    python src/67_vfdb_ingest.py --download      # fetch into data/external/vfdb/
    python src/67_vfdb_ingest.py                 # parse what is already there
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/external/vfdb"
OUT = ROOT / "results/vfdb_ingest.json"

BASE = "https://www.mgc.ac.cn/VFs/Down"
FILES = {"setA": "VFDB_setA_pro.fas.gz", "setB": "VFDB_setB_pro.fas.gz", "table": "VFs.xls.gz"}

# >VFG037176(gb|WP_001081735) (plc1) phospholipase C [Phospholipase C (VF0470) - Exotoxin
#  (VFC0235)] [Acinetobacter baumannii ACICU]
# One setB record, VFG042213, carries no "(gb|...)" accession block at all, so that group is
# optional. It was found by the unparsed-header failure below rather than by inspection.
HEADER = re.compile(
    r"^>(?P<vfg>VFG\d+)(?:\((?P<db>[^)]*)\))?\s*"
    r"(?:\((?P<gene>[^)]*)\)\s*)?(?P<name>.*?)\s*"
    r"\[(?P<vf>[^\[\]]*?)\((?P<vfid>VF\d+)\)\s*-\s*(?P<cat>[^\[\]]*?)\((?P<vfc>VFC\d+)\)\]\s*"
    r"\[(?P<org>[^\[\]]+)\]\s*$"
)

# Species -> the host the producing pathogen is primarily described against. Species level
# deliberately: genus is wrong here, see the module docstring. Anything not listed stays
# "unassigned" and is counted as such rather than defaulted to human, because defaulting to the
# majority class is the exact failure § 2.4.1 found in the probe (every held-out class came back
# "animal", scoring 0.500 = always-animal).
HOST = {
    "Pseudomonas syringae": "plant", "Pseudomonas entomophila": "insect",
    "Xanthomonas campestris": "plant", "Xanthomonas axonopodis": "plant",
    "Xanthomonas oryzae": "plant", "Ralstonia solanacearum": "plant",
    "Erwinia amylovora": "plant", "Erwinia pyrifoliae": "plant",
    "Pectobacterium atrosepticum": "plant", "Pectobacterium carotovorum": "plant",
    "Dickeya dadantii": "plant", "Dickeya zeae": "plant", "Dickeya chrysanthemi": "plant",
    "Xylella fastidiosa": "plant", "Pantoea stewartii": "plant", "Pantoea ananatis": "plant",
    "Agrobacterium tumefaciens": "plant", "Agrobacterium fabrum": "plant",
    "Bacillus thuringiensis": "insect", "Paenibacillus larvae": "insect",
    "Photorhabdus luminescens": "insect", "Photorhabdus asymbiotica": "insect",
    "Xenorhabdus nematophila": "insect", "Xenorhabdus bovienii": "insect",
    "Serratia proteamaculans": "insect",
}
# Genera whose members in VFDB are human or other mammalian pathogens throughout. Listed rather
# than inferred, and any genus absent from both tables is left unassigned.
HOST_GENUS_HUMAN = {
    "Acinetobacter", "Actinobacillus", "Actinomyces", "Aggregatibacter", "Anaplasma",
    "Arcanobacterium", "Bacteroides", "Bartonella", "Bordetella", "Borrelia", "Brucella",
    "Campylobacter", "Chlamydia", "Chlamydophila", "Citrobacter", "Clostridium",
    "Corynebacterium", "Coxiella", "Dichelobacter", "Ehrlichia", "Enterococcus",
    "Erysipelothrix", "Escherichia", "Francisella", "Fusobacterium", "Haemophilus",
    "Helicobacter", "Klebsiella", "Legionella", "Leptospira", "Listeria", "Mannheimia",
    "Moraxella", "Mycobacterium", "Mycoplasma", "Neisseria", "Neorickettsia", "Pasteurella",
    "Porphyromonas", "Rickettsia", "Salmonella", "Shigella", "Staphylococcus", "Streptococcus",
    "Tannerella", "Treponema", "Yersinia",
}


def fetch() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name in FILES.values():
        gz = RAW / name
        # curl -L: the site 301s http -> https, and a redirect-less fetch writes an empty file
        # rather than failing, which is how the first attempt produced three 0-byte downloads.
        subprocess.run(["curl", "-sSL", "--fail", "-o", str(gz), f"{BASE}/{name}"], check=True)
        if gz.stat().st_size == 0:
            raise SystemExit(f"{name} downloaded as 0 bytes")
        subprocess.run(["gunzip", "-kf", str(gz)], check=True)
        print(f"  {name}  {gz.stat().st_size / 1e6:.2f} MB gz  "
              f"sha256 {hashlib.sha256(gz.read_bytes()).hexdigest()[:16]}")


def parse(path: Path) -> list[dict]:
    recs, bad = [], []
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith(">"):
            continue
        m = HEADER.match(line)
        if not m:
            bad.append(line)
            continue
        # The non-greedy name groups keep their trailing space, and an unstripped "Exotoxin "
        # silently compares unequal to "Exotoxin" — which it did, reporting 0 exotoxins out of
        # 248 while still counting fourteen categories.
        d = {k: (v.strip() if isinstance(v, str) else v) for k, v in m.groupdict().items()}
        d["species"] = " ".join(d["org"].split()[:2])
        d["genus"] = d["org"].split()[0]
        d["host"] = HOST.get(d["species"]) or (
            "human_or_mammal" if d["genus"] in HOST_GENUS_HUMAN else "unassigned")
        recs.append(d)
    if bad:
        # A header this script cannot read is a hard failure: a silently dropped record would
        # shrink a category and nothing downstream would notice.
        raise SystemExit(f"{len(bad)} unparsed headers in {path.name}, first:\n{bad[0]}")
    return recs


def summarize(recs: list[dict]) -> dict:
    cats = collections.Counter(r["cat"] for r in recs)
    hosts = collections.Counter(r["host"] for r in recs)
    tox = [r for r in recs if r["cat"] == "Exotoxin"]
    return {
        "n_records": len(recs),
        "n_categories": len(cats),
        "n_vf_ids": len(set(r["vfid"] for r in recs)),
        "n_genera": len(set(r["genus"] for r in recs)),
        "n_species": len(set(r["species"] for r in recs)),
        "categories": dict(cats.most_common()),
        # the distinction the axis makes that this project's panel assumes away
        "exotoxin": len(tox),
        "exotoxin_frac": round(len(tox) / len(recs), 4),
        "non_toxin_virulence": len(recs) - len(tox),
        "hosts": dict(hosts.most_common()),
        "host_non_human": hosts["plant"] + hosts["insect"],
        "host_unassigned_frac": round(hosts["unassigned"] / len(recs), 4),
        # a category is only usable for leave-one-category-out if it has members to hold out
        "categories_ge_20": sorted(c for c, n in cats.items() if n >= 20),
        "exotoxin_by_host": dict(collections.Counter(r["host"] for r in tox).most_common()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="fetch the VFDB files first")
    args = ap.parse_args()

    if args.download:
        print("Downloading VFDB (no license stated on the download page; not redistributed):")
        fetch()

    out = {"source": "VFDB, https://www.mgc.ac.cn/VFs/download.htm",
           "license": "none stated on the download page",
           "note": "raw files are not committed; rerun with --download to reproduce"}
    for key in ("setA", "setB"):
        fas = RAW / FILES[key].removesuffix(".gz")
        if not fas.exists():
            raise SystemExit(f"{fas} missing; run with --download")
        recs = parse(fas)
        out[key] = summarize(recs)
        out[key]["sha256"] = hashlib.sha256(fas.read_bytes()).hexdigest()

    # setA is the experimentally verified core, setB the full set including predicted VFs. The
    # host contrast lives almost entirely in setB, which is the cost side of the choice.
    out["host_contrast_is_setB_only"] = {
        "setA_non_human": out["setA"]["host_non_human"],
        "setB_non_human": out["setB"]["host_non_human"],
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")

    for key in ("setA", "setB"):
        s = out[key]
        print(f"\n{key}: {s['n_records']} records, {s['n_categories']} categories, "
              f"{s['n_vf_ids']} VF ids, {s['n_species']} species")
        print(f"  Exotoxin {s['exotoxin']} ({s['exotoxin_frac']:.1%}), "
              f"non-toxin virulence {s['non_toxin_virulence']}")
        print("  host: " + ", ".join(f"{k} {v}" for k, v in s["hosts"].items()))
        print(f"  categories with n >= 20: {len(s['categories_ge_20'])}")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
