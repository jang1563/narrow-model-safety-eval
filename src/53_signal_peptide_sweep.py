#!/usr/bin/env python3
"""
53_signal_peptide_sweep.py - no annotated functional position may sit inside a cleaved signal
                             peptide or propeptide. Run that check over the whole panel.

Where this rule came from
-------------------------
`docs/DATA_CORRECTIONS.md` entry nineteen. SEB's nine annotated positions all failed the
residue-identity check at the published offset of 0 and the entry ran anyway for months, because the
identity check that `src/46` installed can only catch a wrong coordinate frame when the identities
are RIGHT. SEB's identities are also wrong, so the check returned zero matches and no signal.

A cheaper rule would have caught it on day one and needs no identities at all: positions 23 and 25
were annotated "MHC-II binding interface" while sitting inside the cleaved signal peptide
`MYKRLFISHVILIFALILVISTPNVLA`. A secreted protein cannot present a functional site on a peptide that
is removed before secretion. The rule is:

    for every annotated position p of every entry, p must lie OUTSIDE every Signal and Propeptide
    region that UniProt places on that accession

That is a falsifiable, identity-free necessary condition on the coordinate frame. It cannot prove a
frame right. It can prove one wrong, which is what is wanted from a cheap gate.

⚠️ What a hit does and does not mean. A hit means the entry's positions are not in precursor
coordinates, because no precursor-numbered functional site lands in a signal peptide. It does NOT
tell you what the right frame is, and it does not mean the entry is unusable. It means the entry
must not be scored at its current offset without a stated reason.

⚠️ Propeptides are included and are the subtler case. A propeptide can be functional before
cleavage, so a hit inside a propeptide is weaker evidence than a hit inside a signal peptide.
Reported separately for that reason rather than pooled.

Usage:
    python src/53_signal_peptide_sweep.py
    python src/53_signal_peptide_sweep.py --fetch     # refresh the UniProt cache first
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "uniprot_cache"
SITES = ROOT / "data" / "annotations" / "functional_sites.json"


def load(acc, fetch):
    dest = CACHE / f"{acc}.json"
    if fetch or not dest.exists():
        url = f"https://rest.uniprot.org/uniprotkb/{acc}.json"
        with urllib.request.urlopen(url, timeout=60) as r:   # noqa: S310
            dest.write_bytes(r.read())
    return json.loads(dest.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    fs = json.loads(SITES.read_text())
    accs = [k for k in fs if not k.startswith("_")]

    rows, hits = {}, []
    print("positions are shown AFTER applying each entry's own precursor_offset, "
          "as the pipeline does\n")
    print(f"{'acc':<9}{'len':>5}{'off':>5}  {'cleaved regions':<28}{'positions':<24}{'inside'}")
    print("-" * 92)
    for acc in sorted(accs):
        d = load(acc, a.fetch)
        seq = d["sequence"]["value"]
        cleaved = []
        for f in d.get("features", []):
            if f["type"] in ("Signal", "Propeptide"):
                L = f["location"]
                cleaved.append((f["type"], L["start"]["value"], L["end"]["value"]))
        # 🔴 The three entries `src/46` repaired store their positions in MATURE coordinates and
        # carry a `precursor_offset` that the pipeline applies before masking. Reading
        # `catalytic_residues` raw scores them in the wrong frame and flags all three, which is this
        # check firing on its own repairs. The first run of this script did exactly that: it reported
        # P00588 and P00648 as hits alongside P01552, and those two are clean once the offset they
        # already carry is applied. Apply it here the same way the pipeline does.
        site = fs[acc]["functional_sites"]
        off = site.get("precursor_offset") or 0
        pos = [q + off for q in site["catalytic_residues"]]
        in_sig = sorted({p for p in pos for t, s_, e_ in cleaved
                         if t == "Signal" and s_ <= p <= e_})
        in_pro = sorted({p for p in pos for t, s_, e_ in cleaved
                         if t == "Propeptide" and s_ <= p <= e_})
        over = [p for p in pos if p > len(seq)]
        reg = ", ".join(f"{t[:4]} {s_}-{e_}" for t, s_, e_ in cleaved) or "none"
        flag = ""
        if in_sig:
            flag = f"🔴 SIGNAL {in_sig}"
        elif in_pro:
            flag = f"⚠️ propeptide {in_pro}"
        elif over:
            flag = f"🔴 PAST END {over}"
        print(f"{acc:<9}{len(seq):>5}{off:>5}  {reg:<28}{str(pos)[:22]:<24}{flag}")
        rows[acc] = {"len": len(seq), "cleaved": cleaved, "positions": pos,
                     "precursor_offset": off,
                     "stored_positions": list(site["catalytic_residues"]),
                     "in_signal": in_sig, "in_propeptide": in_pro, "past_end": over,
                     "clean": not (in_sig or in_pro or over)}
        if not rows[acc]["clean"]:
            hits.append(acc)

    n_sig = [a_ for a_ in rows if rows[a_]["in_signal"]]
    n_pro = [a_ for a_ in rows if rows[a_]["in_propeptide"]]
    n_end = [a_ for a_ in rows if rows[a_]["past_end"]]
    verdict = (
        f"{len(rows) - len(hits)} of {len(rows)} entries place every annotated position outside "
        f"every cleaved region. {len(n_sig)} put a position inside a SIGNAL peptide "
        f"({', '.join(n_sig) or 'none'}), which proves those positions are not in precursor "
        f"coordinates. {len(n_pro)} put one inside a propeptide ({', '.join(n_pro) or 'none'}), "
        f"which is weaker because a propeptide can be functional before cleavage. {len(n_end)} "
        f"index past the end of the sequence ({', '.join(n_end) or 'none'}).")
    print(f"\nverdict: {verdict}")
    dest = ROOT / "results/v3/signal_peptide_sweep.json"
    json.dump({"n_entries": len(rows), "entries": rows, "hits": hits,
               "in_signal": n_sig, "in_propeptide": n_pro, "past_end": n_end,
               "verdict": verdict}, open(dest, "w"), indent=2)
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
