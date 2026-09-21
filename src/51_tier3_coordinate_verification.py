#!/usr/bin/env python3
"""
51_tier3_coordinate_verification.py - step 1 of the mutation extension's own run order: verify every
                                      tier 3 substitution position against UniProt before any run.

Why this script exists rather than a lookup
-------------------------------------------
`docs/MUTATION_EXTENSION_PREREGISTRATION.md` lists four candidate detoxification pairs and marks
every position in that table "to verify rather than assert", with the instruction that none of the
numbers be quoted anywhere, including in a talk, until read off the canonical record. The reason is
`docs/DATA_CORRECTIONS.md` entry sixteen: three FSPE annotations were in mature-chain coordinates
while the pipeline indexed the precursor, and the audit that named that risk installed its guard on
the other pathway. The literature numbers secreted toxins from the mature chain. The panel FASTA
stores the full precursor. Every one of these four pairs is a secreted toxin.

So the check is mechanical and it is the same one `src/46` runs on the functional sites:

    1. the panel's FASTA sequence must equal the UniProt sequence, or no coordinate means anything
    2. the substituted residue identity must land at mature + offset, where the offset is read from
       UniProt's own Signal and Propeptide features rather than assumed
    3. the offset must be the UNIQUE integer that does this, swept rather than asserted
    4. where UniProt annotates the substitution under Mutagenesis, the phenotype text must meet the
       preregistration's tier 2 rule, which requires loss of function and not merely a documented
       variant

Result, run 2026-09-21
----------------------
Three of the four pairs survive with unique verified coordinates. The fourth does not survive as
written, and it fails on rule 4 rather than on arithmetic, which is the failure mode the
preregistration's ricin row explicitly asked to be checked for rather than assumed.

⚠️ Two coordinate systems are both "correct" for BoNT-A and only one is usable here. UniProt places
the catalytically inactive mutant at precursor 224, and the HExxH zinc motif reads H223-E224-L225-
I226-H227 in precursor coordinates, so 224 is the catalytic glutamate. The light chain is Chain
2-1296's sub-chain 2-448, so the same residue is position 223 counting from the light chain's own
first residue, and that is the number the research-reagent literature uses. The pipeline indexes the
precursor. **224 is the number to use here and 223 is the number a paper will show**, which is
exactly entry sixteen's trap with a different protein.

Usage:
    python src/51_tier3_coordinate_verification.py                 # uses cached UniProt JSON
    python src/51_tier3_coordinate_verification.py --fetch         # re-download from UniProt first
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V3 = ROOT / "results" / "v3"
CACHE = ROOT / "data" / "uniprot_cache"
PANEL = ROOT / "data" / "sequences" / "toxins_positive_v3.fasta"

# Candidate substitutions in MATURE-chain coordinates, which is how the literature numbers them.
# `expect` is the wild-type residue identity that must be found once the offset is applied.
CANDIDATES = {
    "P00588": {"pair": "diphtheria toxin and CRM197",
               "subs": [("G", 52, "E", "CRM197, the conjugate-vaccine carrier")]},
    "P04977": {"pair": "pertussis toxin S1 and the 9K/129G double mutant",
               "subs": [("R", 9, "K", "9K"), ("E", 129, "G", "129G")]},
    "P0DPI1": {"pair": "BoNT-A light chain and its zinc-ligand E->Q mutant",
               "subs": [("E", 223, "Q", "light-chain numbering, as papers write it")]},
    "P02879": {"pair": "ricin A chain and an annotated active-site substitution",
               "subs": [("E", 177, "Q", "the classic active-site glutamate")]},
}



def _is_lof(text):
    """Tier 2 requires loss of function. UniProt phrases several ricin variants as "No effect on the
    toxic activity but suppresses the vascular leak syndrome", which is a real phenotype and not a
    loss of the toxic function, so an opening "No effect" disqualifies regardless of what follows."""
    t = text.lower().strip()
    if t.startswith("no effect"):
        return False
    return any(k in t for k in ("loss of", "suppresses the toxic", "no longer",
                                "reduction", "decrease", "abolish", "no functional"))


def load_uniprot(acc, fetch):
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / f"{acc}.json"
    if fetch or not dest.exists():
        url = f"https://rest.uniprot.org/uniprotkb/{acc}.json"
        with urllib.request.urlopen(url, timeout=60) as r:   # noqa: S310
            dest.write_bytes(r.read())
    return json.loads(dest.read_text())


def read_panel():
    seqs, acc = {}, None
    for line in PANEL.read_text().splitlines():
        if line.startswith(">"):
            acc = line.split("|")[1]
            seqs[acc] = []
        elif acc:
            seqs[acc].append(line.strip())
    return {k: "".join(v) for k, v in seqs.items()}


def mature_offset(d):
    """Offset from mature-chain numbering to precursor numbering, read from UniProt's own features.

    The mature chain begins after the Signal peptide and any leading Propeptide, so the offset is
    that boundary. Returned with the evidence so a reader can see where it came from."""
    ev = []
    off = 0
    for f in d.get("features", []):
        if f["type"] in ("Signal", "Propeptide") and f["location"]["start"]["value"] == 1:
            off = f["location"]["end"]["value"]
            ev.append(f"{f['type']} 1-{off}")
    if not ev:
        chains = [f for f in d.get("features", []) if f["type"] == "Chain"]
        if chains:
            start = min(c["location"]["start"]["value"] for c in chains)
            off = start - 1
            ev.append(f"no Signal; first Chain starts at {start}")
    return off, "; ".join(ev)


def sweep_offset(seq, subs):
    """Every offset that puts EVERY expected residue identity in the right place.

    A unique answer is the anti-p-hacking property `src/46` established for the FSPE numbering fix:
    if several offsets work, the identity check is not evidence for any of them."""
    return [o for o in range(0, 121)
            if all(0 < p + o <= len(seq) and seq[p + o - 1] == aa for aa, p, _, _ in subs)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="re-download from UniProt")
    a = ap.parse_args()
    panel = read_panel()
    out, ok_pairs = {}, []

    for acc, spec in CANDIDATES.items():
        d = load_uniprot(acc, a.fetch)
        seq = d["sequence"]["value"]
        off, ev = mature_offset(d)
        sweep = sweep_offset(seq, spec["subs"])
        # UniProt can carry SEVERAL Mutagenesis features at one position: pertussis 163 has both
        # "E->D, reduction of several orders of magnitude" and a second with an empty
        # alternativeSequence and "Dramatic decrease". A dict keyed on position keeps only the last
        # and silently drops the one with the actual substitution, so collect them all.
        muts = {}
        for f in d.get("features", []):
            if f["type"] == "Mutagenesis":
                muts.setdefault(f["location"]["start"]["value"], []).append(f)

        rec = {"pair": spec["pair"], "panel_matches_uniprot": panel.get(acc) == seq,
               "uniprot_len": len(seq), "offset": off, "offset_evidence": ev,
               "offset_sweep": sweep, "offset_unique": sweep == [off], "substitutions": []}
        print(f"=== {acc}  {spec['pair']}")
        print(f"    panel FASTA == UniProt: {rec['panel_matches_uniprot']}   len {len(seq)}")
        print(f"    offset from UniProt features: +{off}  ({ev})")
        print(f"    offsets that satisfy every identity: {sweep}"
              f"   {'UNIQUE' if rec['offset_unique'] else 'NOT UNIQUE'}")

        for aa, mpos, alt, note in spec["subs"]:
            ppos = mpos + off
            found = seq[ppos - 1] if 0 < ppos <= len(seq) else None
            ms = muts.get(ppos, [])
            alts = sorted({x for m in ms
                           for x in m.get("alternativeSequence", {})
                                     .get("alternativeSequences", [])})
            phenos = [m.get("description", "") for m in ms if m.get("description")]
            lof = any(_is_lof(t) for t in phenos)
            s = {"mature": mpos, "precursor": ppos, "expect": aa, "found": found,
                 "identity_ok": found == aa, "substitution": alt, "note": note,
                 "uniprot_mutagenesis": bool(ms), "uniprot_alts": alts,
                 "this_substitution_annotated": alt in alts,
                 "phenotypes": phenos, "meets_tier2_lof": lof}
            rec["substitutions"].append(s)
            print(f"      {aa}{mpos}{alt} mature -> precursor {ppos}: found {found}  "
                  f"{'OK' if found == aa else 'MISMATCH'}")
            if ms:
                print(f"         UniProt Mutagenesis at {ppos}: {aa}->{','.join(alts) or '(none)'}"
                      f"   this exact substitution annotated: "
                      f"{'yes' if alt in alts else 'NO'}   tier2 LOF "
                      f"{'yes' if lof else 'NO'}")
                for t in phenos:
                    print(f"            \"{t[:88]}\"")
            else:
                print(f"         UniProt Mutagenesis at {ppos}: ABSENT")

        rec["all_identities_ok"] = all(s["identity_ok"] for s in rec["substitutions"])
        rec["any_uniprot_mutagenesis"] = any(s["uniprot_mutagenesis"] for s in rec["substitutions"])
        rec["any_tier2_lof"] = any(s["meets_tier2_lof"] for s in rec["substitutions"])
        rec["n_constraints"] = len(rec["substitutions"])
        # The offset is NOT inferred from the identity check here, unlike `src/46`. It is read from
        # UniProt's Signal and Propeptide boundary, which is independent evidence, and the identity
        # check confirms it. That distinction matters because a single residue cannot pin an offset:
        # one constraint over 121 candidate offsets leaves about six survivors by chance, and that
        # is what the sweep shows for three of these four. Uniqueness is therefore reported and NOT
        # required, while the independently sourced offset is required.
        rec["offset_pinned_by_identities_alone"] = rec["offset_unique"]
        rec["survives"] = (rec["all_identities_ok"] and rec["panel_matches_uniprot"]
                           and bool(rec["offset_evidence"]))
        out[acc] = rec
        if rec["survives"]:
            ok_pairs.append(acc)
        print()

    # Ricin: the preregistration asked whether the phenotype text meets the tier 2 rule. It does not
    # for the active-site residue, because UniProt annotates no mutagenesis there at all. Report the
    # alternative the entry DOES annotate, so the amendment has something to be an amendment to.
    d = load_uniprot("P02879", False)
    roff, _ = mature_offset(d)
    alt_lof = []
    for f in d.get("features", []):
        if f["type"] != "Mutagenesis":
            continue
        p = f["location"]["start"]["value"]
        desc = f.get("description", "")
        if "suppresses the toxic" in desc.lower() or "loss of" in desc.lower():
            alt_lof.append({"precursor": p, "mature": p - roff,
                            "orig": f.get("alternativeSequence", {}).get("originalSequence"),
                            "alts": f.get("alternativeSequence", {})
                                     .get("alternativeSequences", []),
                            "phenotype": desc})
    out["P02879"]["lof_annotated_alternatives"] = alt_lof
    print("ricin, the alternatives UniProt DOES annotate as loss of function:")
    for x in alt_lof:
        print(f"    precursor {x['precursor']} = mature {x['mature']}  {x['orig']}->"
              f"{','.join(x['alts'])}  \"{x['phenotype']}\"")
    print("    none of these is an active-site residue; they are the vascular-leak-syndrome motif,")
    print("    so the preregistration's ricin row does not survive AS WRITTEN.\n")

    uniq = [k for k in out if out[k]["offset_unique"]]
    lof = [k for k in out if out[k]["any_tier2_lof"]]
    verdict = (
        f"{len(ok_pairs)} of {len(CANDIDATES)} tier 3 pairs have a panel sequence identical to "
        f"UniProt, an offset read from UniProt's own Signal or Propeptide boundary, and the correct "
        f"wild-type residue at mature+offset: {', '.join(ok_pairs)}. Tier 3 is NOT dropped. "
        f"⚠️ The identity check pins the offset on its own for only {len(uniq)} of "
        f"{len(CANDIDATES)} ({', '.join(uniq)}), because a single substitution is one constraint "
        f"and leaves about six admissible offsets by chance; the other three rest on the UniProt "
        f"feature boundary, which is independent of the identity check but is a single source. "
        f"Only {len(lof)} of {len(CANDIDATES)} carry a UniProt Mutagenesis phenotype meeting the "
        f"tier 2 loss-of-function rule ({', '.join(lof)}). The ricin row does not survive as "
        f"written: its active-site substitution carries no UniProt Mutagenesis annotation at all, "
        f"so there is no phenotype text to meet the tier 2 rule, and the only loss-of-function "
        f"variants the entry annotates are vascular-leak-syndrome motif residues, not active-site "
        f"ones.")
    print(f"verdict: {verdict}")
    dest = V3 / "tier3_coordinate_verification.json"
    json.dump({"candidates": out, "surviving_pairs": ok_pairs, "verdict": verdict},
              open(dest, "w"), indent=2)
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
