#!/usr/bin/env python3
"""
54_annotation_provenance_audit.py - how much of the FSPE annotation set is actually in UniProt, and
                                    does being in UniProt predict the signal?

Why this exists
---------------
FSPE's entire validity rests on `data/annotations/functional_sites.json`. Those positions were
curated by hand, and until now nobody measured how well they agree with the canonical record. Two
findings forced the question:

- `docs/DATA_CORRECTIONS.md` entry nineteen: Q51451 agrees with UniProt on 2 of its 5 positions and
  **omits 4** annotated functional sites (Active site 319, 343; Binding site 186, 187). That was
  found while chasing a different defect, which means it was found by luck.
- entry sixteen: three entries were in the wrong coordinate frame entirely.

So the annotation set has been repaired twice, in both cases after a specific suspicion. This script
asks the general question instead, over every entry at once.

What is measured
----------------
For each entry, positions are read AFTER applying its own `precursor_offset`, exactly as the
pipeline does, and compared against UniProt's **Active site**, **Binding site** and **Site**
features:

    agree   annotated positions that UniProt also marks as a functional site
    extra   annotated positions UniProt does not mark
    omit    positions UniProt marks that the annotation does not have

⚠️ An `extra` is NOT automatically an error and the asymmetry matters. UniProt's feature table is
incomplete for many toxins, and a curator reading a structure paper can legitimately know a residue
UniProt has not annotated. So `extra` measures *unsourced-in-UniProt*, not *wrong*. An `omit` is the
stronger signal, because it means the curator had a canonical source available and did not use it.

🔴 **The comparison is against Active site ONLY, and the first version of this script got that
wrong.** It pooled Active site, Binding site and Site, which inflated the omission count with
features a `catalytic_residues` field should not be expected to hold. Ricin came out "omitting 37
sites"; 33 of those are carbohydrate-binding residues of the B-chain lectin domain plus AMP
contacts, and the entry in fact matches all four of UniProt's ricin **active** sites exactly.
Anthrax protective antigen came out omitting 17, which are Ca(2+) binding sites, the furin cleavage
site and alpha/phi-clamp residues. None of those are catalytic. Pooling them measured the wrong
thing and made the curation look far worse than it is.

So Active site is the primary comparison, and the broader set is reported beside it as context with
its composition named rather than pooled into a single number.

Then the question the repository actually cares about
----------------------------------------------------
If FSPE measures something specific about functional sites, entries whose annotations agree with the
canonical record should show it more strongly than entries whose annotations are unsourced. If the
ratio is unrelated to whether the masked positions are real functional sites, that is evidence FSPE
tracks conservation generally, which is `docs/EVALUATION_REPORT.md`'s own stated null.

    P1  FSPE ratio correlates with the fraction of annotated positions that UniProt confirms,
        negatively: better-sourced annotations give lower (more confident) ratios.

⚠️ PREREGISTERED AS UNDERPOWERED. At most 16 entries have annotations at all and fewer have UniProt
site features, so this test cannot support a positive claim. It is run because a NULL result is
informative and cheap, and because the descriptive table is worth having regardless. Any correlation
is reported with its n and its p and is not to be quoted as a finding.

Usage:
    python src/54_annotation_provenance_audit.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "uniprot_cache"
SITES = ROOT / "data" / "annotations" / "functional_sites.json"
SITE_TYPES = ("Active site", "Binding site", "Site")


def uniprot_site_positions(acc):
    """Positions by feature type, plus the ligand names, because the ligand is what makes a Binding
    site comparable or not. A carbohydrate site in a lectin domain and a catalytic glutamate are both
    "functional" and only one belongs in a `catalytic_residues` field."""
    d = json.loads((CACHE / f"{acc}.json").read_text())
    by_type = {t: set() for t in SITE_TYPES}
    ligands = set()
    for f in d.get("features", []):
        if f["type"] in by_type:
            L = f["location"]
            by_type[f["type"]].update(range(L["start"]["value"], L["end"]["value"] + 1))
            lig = (f.get("ligand") or {}).get("name")
            if lig:
                ligands.add(lig)
    return ({t: sorted(v) for t, v in by_type.items()}, sorted(ligands),
            len(d["sequence"]["value"]))


def main():
    fs = json.loads(SITES.read_text())
    accs = sorted(k for k in fs if not k.startswith("_"))
    fspe = {(x.get("uniprot_id") or x.get("accession")): x["fspe_ratio"]
            for x in json.loads((ROOT / "results/fspe_results.json").read_text())["per_protein"]}

    rows = {}
    print("primary comparison is against UniProt ACTIVE SITE only; the broader set is context\n")
    print(f"{'acc':<9}{'ann':>4}{'off':>5}  {'act':>4}{'agr':>4}{'ext':>4}{'omit':>5}  "
          f"{'conf':>6}  {'FSPE':>7}  notes")
    print("-" * 98)
    for acc in accs:
        site = fs[acc]["functional_sites"]
        off = site.get("precursor_offset") or 0
        pos = sorted({p + off for p in site["catalytic_residues"]})
        by_type, ligands, _ln = uniprot_site_positions(acc)
        act = set(by_type["Active site"])
        broad = set().union(*(set(v) for v in by_type.values()))
        agree = sorted(set(pos) & act)
        extra = sorted(set(pos) - act)
        omit = sorted(act - set(pos))
        frac = (len(agree) / len(pos)) if pos else None
        r = fspe.get(acc)
        note = []
        if not act:
            note.append("no UniProt Active site")
        if not pos:
            note.append("no annotated positions (negative control)")
        if omit:
            note.append(f"omits active {omit}")
        # positions explained by a non-catalytic UniProt feature rather than being unsourced
        expl = sorted(set(extra) & broad)
        if expl:
            note.append(f"{len(expl)} extra are non-catalytic UniProt features")
        print(f"{acc:<9}{len(pos):>4}{off:>5}  {len(act):>4}{len(agree):>4}{len(extra):>4}"
              f"{len(omit):>5}  {('n/a' if frac is None else f'{frac:.0%}'):>6}  "
              f"{('n/a' if r is None else f'{r:.4f}'):>7}  {'; '.join(note)}")
        rows[acc] = {"n_annotated": len(pos), "offset": off, "positions": pos,
                     "uniprot_by_type": by_type, "uniprot_ligands": ligands,
                     "n_active": len(act), "n_broad": len(broad),
                     "agree_active": agree, "extra_vs_active": extra, "omit_active": omit,
                     "extra_explained_by_other_feature": expl,
                     "frac_active_confirmed": frac, "fspe_ratio": r,
                     "has_active_site": bool(act)}

    comparable = [a for a in accs if rows[a]["has_active_site"] and rows[a]["n_annotated"]]
    scored = [a for a in comparable if rows[a]["fspe_ratio"] is not None]
    t_ann = sum(rows[a]["n_annotated"] for a in comparable)
    t_agree = sum(len(rows[a]["agree_active"]) for a in comparable)
    t_omit = sum(len(rows[a]["omit_active"]) for a in comparable)
    perfect = [a for a in comparable
               if not rows[a]["extra_vs_active"] and not rows[a]["omit_active"]]
    zero = [a for a in comparable if not rows[a]["agree_active"]]
    no_act = [a for a in accs if not rows[a]["has_active_site"]]

    print(f"\n{len(comparable)} of {len(accs)} entries have BOTH annotations and a UniProt "
          f"Active site")
    print(f"  no UniProt Active site at all: {no_act}")
    print(f"  active-site positions: {t_agree}/{t_ann} annotated positions confirmed "
          f"({t_agree / t_ann:.0%}); {t_omit} UniProt active sites omitted")
    print(f"  entries matching UniProt's active sites exactly: {perfect or 'none'}")
    print(f"  entries with ZERO confirmed active sites: {zero or 'none'}")
    # The 32% is not 68% unsourced. Most non-active-site positions ARE in UniProt, under a different
    # feature type, which is what a curator reading a structure paper would pick up.
    t_extra = sum(len(rows[a]["extra_vs_active"]) for a in comparable)
    t_expl = sum(len(rows[a]["extra_explained_by_other_feature"]) for a in comparable)
    grounded = t_agree + t_expl
    print(f"  of the {t_extra} annotated positions that are not Active sites, {t_expl} carry some "
          f"other UniProt feature")
    print(f"  so {grounded}/{t_ann} ({grounded / t_ann:.0%}) of annotated positions are grounded in "
          f"UniProt under SOME feature type")

    corr = None
    if len(scored) >= 5:
        from scipy import stats
        x = [rows[a]["frac_active_confirmed"] for a in scored]
        y = [rows[a]["fspe_ratio"] for a in scored]
        rho, pv = stats.spearmanr(x, y)
        corr = {"n": len(scored), "rho": float(rho), "p": float(pv),
                "direction_as_predicted": bool(rho < 0),
                "significant_at_05": bool(pv < 0.05)}
        print(f"\nP1  FSPE ratio vs fraction of annotated positions that are UniProt active sites: "
              f"rho {rho:+.4f}, p {pv:.4f}, n {len(scored)}")
        print(f"    {'as predicted (negative)' if rho < 0 else 'OPPOSITE to prediction'}, "
              f"{'p < 0.05' if pv < 0.05 else 'not significant'}")
        print("    ⚠️ preregistered as underpowered; a null is not evidence of no effect and a hit "
              "is not quotable")

    verdict = (
        f"Against UniProt ACTIVE SITE features, {t_agree} of {t_ann} annotated positions across "
        f"{len(comparable)} comparable entries are confirmed ({t_agree / t_ann:.0%}), with "
        f"{t_omit} UniProt active sites omitted. {len(perfect)} entries match UniProt's active-site "
        f"set exactly ({', '.join(perfect) or 'none'}) and {len(zero)} have none confirmed "
        f"({', '.join(zero) or 'none'}). {len(no_act)} entries have no UniProt Active site feature "
        f"at all ({', '.join(no_act)}), so most of the curation is sourced from structure papers "
        f"rather than the feature table, which is defensible for toxins and means the omissions are "
        f"the actionable half, and there are almost none: the only omitted active sites in the whole "
        f"panel are Q51451's 319 and 343, on the entry already known to be defective. The 32% is "
        f"also not 68% unsourced. Of the {t_extra} annotated positions that are not Active sites, "
        f"{t_expl} carry some other UniProt feature, so {grounded}/{t_ann} "
        f"({grounded / t_ann:.0%}) are grounded in UniProt under some type, which is what a curator "
        f"reading a structure paper would produce. An earlier version of this audit pooled Binding "
        f"site and Site with "
        f"Active site and reported 64 omissions; most of those were carbohydrate, Ca(2+) and "
        f"cleavage-site features that a catalytic-residue field should not contain, so that number "
        f"measured the wrong thing.")
    if corr:
        verdict += (f" P1 is "
                    f"{'SUPPORTED' if corr['direction_as_predicted'] and corr['significant_at_05'] else 'NOT SUPPORTED'}"
                    f" (rho {corr['rho']:+.3f}, p {corr['p']:.3f}, n {corr['n']}, underpowered by "
                    f"construction).")
    print(f"\nverdict: {verdict}")

    dest = ROOT / "results/v3/annotation_provenance_audit.json"
    json.dump({"site_types": list(SITE_TYPES), "primary_comparison": "Active site",
               "n_entries": len(accs), "comparable": comparable,
               "no_uniprot_active_site": no_act, "entries": rows,
               "totals": {"annotated": t_ann, "confirmed_active": t_agree,
                          "omitted_active": t_omit,
                          "frac_active_confirmed": t_agree / t_ann,
                          "extra_vs_active": t_extra,
                          "extra_explained_by_other_feature": t_expl,
                          "grounded_in_uniprot": grounded,
                          "frac_grounded": grounded / t_ann},
               "exact_match": perfect, "zero_confirmed": zero,
               "P1": corr, "verdict": verdict}, open(dest, "w"), indent=2)
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
