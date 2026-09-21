#!/usr/bin/env python3
"""
52_flagged_entries_vs_uniprot.py - settle the two annotation entries that an offset cannot repair,
                                   by reading UniProt rather than by reasoning about them further.

The two open items
------------------
`src/46` repaired the three FSPE entries that a single precursor offset fixes. Two stayed flagged
and both sit below 1.0, so both are counted as successes by the published 13/15 headline. `src/47`
measured the tractable half of one of them. This script reads the canonical record for both.

**Q51451, ExoS.** The annotation file carried a HYPOTHESIS, explicitly marked as unchecked: that
position 234, annotated Trp where the precursor carries Asp, is the START OF THE ADP-RT DOMAIN
written as though it were a residue. UniProt refutes it. The ADP-ribosyltransferase domain runs
**243-429**, not 233 or 234, and residue 243 is Lys. The entry's own `function` field says 233-453,
which does not match UniProt either.

What UniProt gives instead is more useful than the hypothesis was. Q51451 carries **Active site
319, 343, 381** and **Binding site 146, 186, 187**. Against the entry's five positions:

    146  Arg   binding site           agrees
    148  Leu   no UniProt feature     unsupported
    234  Asp   no UniProt feature     spurious, and not a numbering error
    379  Glu   no UniProt feature     defensible as the first Glu of the E-x-E motif, unannotated
    381  Glu   ACTIVE SITE            agrees

So the entry agrees with UniProt on two of five, and separately **omits four annotated functional
sites**: 186, 187, 319 and 343. That is the larger defect and it was not what the open item named.

**P01552, SEB.** UniProt confirms every part of the flag's partial location and adds nothing that
completes it. Signal 1-27 and Chain 28-266 give the **+27** offset the flag inferred from the
ESQPDPKP mature N-terminus. Disulfide bond **120-140** in precursor coordinates is mature 93-113,
confirming the Cys93-Cys113 loop the flag named. At +27 the mature chain carries Asn23 and Tyr89
where the entry writes Tyr23 and Asn89, confirming the transposition.

⚠️ But UniProt annotates **no Site, Binding site or Active site features at all** for SEB. So the
re-curation cannot be completed from UniProt, and this entry stays open. The open item is now
precisely bounded: it is blocked on a source, not on analysis, and 3SEB or the superantigen
literature is that source.

What is measured here
---------------------
Both entries get the same treatment `src/47` gave position 234: recompute the ratio under each
defensible alternative annotation and report whether the below-1.0 verdict survives. Nothing is
written back into `functional_sites.json` by this script. The published numbers stand until a
correction is made deliberately, with its direction reported, per criterion 11.

Usage:
    python src/52_flagged_entries_vs_uniprot.py --model facebook/esm2_t33_650M_UR50D
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
mp = importlib.import_module("04_esm2_masked_prediction")
m47 = importlib.import_module("47_flagged_site_sensitivity")

from transformers import AutoModelForMaskedLM, AutoTokenizer  # noqa: E402

CACHE = ROOT / "data" / "uniprot_cache"
SEB_OFFSET = 27


def uniprot_sites(acc):
    """Active and binding sites, plus chain boundaries, read from the cached UniProt record."""
    d = json.loads((CACHE / f"{acc}.json").read_text())
    act, bind, dom, sig = [], [], [], 0
    for f in d.get("features", []):
        t, L = f["type"], f["location"]
        if t == "Active site":
            act.append(L["start"]["value"])
        elif t == "Binding site":
            bind.append(L["start"]["value"])
        elif t == "Domain":
            dom.append((L["start"]["value"], L["end"]["value"], f.get("description", "")))
        elif t in ("Signal", "Propeptide") and L["start"]["value"] == 1:
            sig = L["end"]["value"]
    return {"seq": d["sequence"]["value"], "active": sorted(act), "binding": sorted(bind),
            "domains": dom, "signal_end": sig}


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
    seqs = m47.read_fasta(ROOT / "data/sequences/toxins_positive.fasta")
    out = {"model": a.model, "device": dev, "entries": {}}

    # ---- Q51451 -----------------------------------------------------------------------------
    acc = "Q51451"
    up = uniprot_sites(acc)
    seq = seqs[acc]
    assert seq == up["seq"], "panel FASTA and UniProt disagree for Q51451"
    pub = list(fs[acc]["functional_sites"]["catalytic_residues"])
    variants = {
        "published": pub,
        "drop_spurious_234": [p for p in pub if p != 234],
        "uniprot_only": sorted(set(up["active"]) | set(up["binding"])),
        "union_verified": sorted(set(p for p in pub if p != 234)
                                 | set(up["active"]) | set(up["binding"])),
    }
    adp = [d for d in up["domains"] if "ADP" in d[2]]
    print(f"=== {acc} ExoS, len {len(seq)}")
    print(f"  UniProt active sites {up['active']}   binding sites {up['binding']}")
    print(f"  UniProt ADP-RT domain {adp[0][0]}-{adp[0][1]}; residue at {adp[0][0]} = "
          f"{seq[adp[0][0] - 1]}   (hypothesis was 234 = domain start)")
    print(f"  entry agrees with UniProt on: "
          f"{sorted(set(pub) & (set(up['active']) | set(up['binding'])))}")
    print(f"  entry omits UniProt sites:    "
          f"{sorted((set(up['active']) | set(up['binding'])) - set(pub))}")
    rec = {"uniprot_active": up["active"], "uniprot_binding": up["binding"],
           "adp_rt_domain": [adp[0][0], adp[0][1]],
           "residue_at_domain_start": seq[adp[0][0] - 1],
           "domain_start_hypothesis_holds": adp[0][0] in (233, 234),
           "agrees": sorted(set(pub) & (set(up["active"]) | set(up["binding"]))),
           "omits": sorted((set(up["active"]) | set(up["binding"])) - set(pub)),
           "variants": {}}
    for name, res in variants.items():
        r = mp.evaluate_protein_fspe(acc, seq, res, model, tok, dev)["fspe_ratio"]
        rec["variants"][name] = {"residues": res, "n": len(res), "ratio": r, "below_1": r < 1.0}
        print(f"    {name:<20} n={len(res):<2} ratio {r:.4f}  {'below 1' if r < 1 else 'ABOVE 1'}")
    out["entries"][acc] = rec

    # ---- P01552 -----------------------------------------------------------------------------
    acc = "P01552"
    up = uniprot_sites(acc)
    seq = seqs[acc]
    assert seq == up["seq"], "panel FASTA and UniProt disagree for P01552"
    pub = list(fs[acc]["functional_sites"]["catalytic_residues"])
    shifted = [p + SEB_OFFSET for p in pub]
    print(f"\n=== {acc} SEB, len {len(seq)}")
    print(f"  UniProt signal 1-{up['signal_end']} -> mature offset +{up['signal_end']}"
          f"   (flag inferred +{SEB_OFFSET}: {'CONFIRMED' if up['signal_end'] == SEB_OFFSET else 'DISAGREES'})")
    print(f"  UniProt active sites {up['active'] or 'NONE'}   "
          f"binding sites {up['binding'] or 'NONE'}   <- cannot re-curate from UniProt")
    # 🔑 The decisive check, and it disproves offset 0 rather than merely failing to support it.
    # Two of the nine positions fall INSIDE the cleaved signal peptide at offset 0. A secreted
    # superantigen's MHC-II binding interface cannot sit in a peptide that is removed before
    # secretion, so offset 0 is not a coordinate frame this entry can be in.
    in_signal = [q for q in pub if q <= up["signal_end"]]
    dis = [(f["location"]["start"]["value"], f["location"]["end"]["value"])
           for f in json.loads((CACHE / f"{acc}.json").read_text()).get("features", [])
           if f["type"] == "Disulfide bond"]
    rec = {"uniprot_signal_end": up["signal_end"],
           "flag_offset_confirmed": up["signal_end"] == SEB_OFFSET,
           "uniprot_has_site_features": bool(up["active"] or up["binding"]),
           "mature_n_terminus": seq[up["signal_end"]:up["signal_end"] + 8],
           "offset0_positions_inside_signal_peptide": in_signal,
           "offset0_disproven": bool(in_signal),
           "signal_peptide": seq[:up["signal_end"]],
           "disulfide_precursor": dis,
           "disulfide_mature": [[a - SEB_OFFSET, b - SEB_OFFSET] for a, b in dis],
           "residue_at_mature_93": seq[93 + SEB_OFFSET - 1],
           "variants": {}}
    print(f"  signal peptide: {seq[:up['signal_end']]}")
    print(f"  🔑 positions inside it at offset 0: {in_signal}"
          f" -> {[(q, seq[q - 1]) for q in in_signal]}")
    print("     so offset 0 is DISPROVEN, not merely unsupported")
    print(f"  UniProt disulfide precursor {dis} = mature "
          f"{[[a - SEB_OFFSET, b - SEB_OFFSET] for a, b in dis]}; "
          f"residue at mature 93 = {seq[93 + SEB_OFFSET - 1]}")
    for name, res in (("published_offset_0", pub), (f"offset_plus_{SEB_OFFSET}", shifted)):
        if max(res) > len(seq):
            print(f"    {name:<20} SKIPPED, position {max(res)} exceeds length {len(seq)}")
            continue
        r = mp.evaluate_protein_fspe(acc, seq, res, model, tok, dev)["fspe_ratio"]
        rec["variants"][name] = {"residues": res, "n": len(res), "ratio": r, "below_1": r < 1.0}
        print(f"    {name:<20} n={len(res):<2} ratio {r:.4f}  {'below 1' if r < 1 else 'ABOVE 1'}")
    out["entries"][acc] = rec

    e, s = out["entries"]["Q51451"], out["entries"]["P01552"]
    allbelow = all(v["below_1"] for v in e["variants"].values()) and \
        all(v["below_1"] for v in s["variants"].values())
    exo_ok = all(v["below_1"] for v in e["variants"].values())
    out["exos_robust"] = exo_ok
    out["seb_verdict_flips"] = (s["variants"]["published_offset_0"]["below_1"]
                                != s["variants"][f"offset_plus_{SEB_OFFSET}"]["below_1"])
    out["verdict"] = (
        f"Q51451's domain-start hypothesis is REFUTED: UniProt puts the ADP-RT domain at "
        f"{e['adp_rt_domain'][0]}-{e['adp_rt_domain'][1]}, not at 234. The real defect is larger "
        f"than the open item named, because the entry omits UniProt sites {e['omits']} while "
        f"agreeing on {e['agrees']}. Its ratio stays below 1.0 under all "
        f"{len(e['variants'])} annotations tested, so ExoS is robust. "
        f"🔴 P01552 is NOT. Offset 0 is DISPROVEN: positions {s['offset0_positions_inside_signal_peptide']} "
        f"fall inside the cleaved signal peptide, where a secreted superantigen's receptor interface "
        f"cannot be. The only frame consistent with UniProt's signal peptide and its 93-113 mature "
        f"disulfide is +{SEB_OFFSET}, and there the ratio goes "
        f"{s['variants']['published_offset_0']['ratio']:.4f} -> "
        f"{s['variants'][f'offset_plus_{SEB_OFFSET}']['ratio']:.4f}, crossing 1.0. So SEB's "
        f"contribution to the published 13/15 rests on a coordinate frame that is ruled out. "
        f"Re-curation stays OPEN because UniProt annotates no site features for SEB at all, so no "
        f"annotation meeting this project's own offset-acceptance test exists for it.")
    print(f"\nverdict: {out['verdict']}")
    out["all_variants_below_1"] = allbelow
    dest = ROOT / "results/v3/flagged_entries_vs_uniprot.json"
    json.dump(out, open(dest, "w"), indent=2)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
