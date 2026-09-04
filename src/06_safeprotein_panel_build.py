#!/usr/bin/env python3
"""
06_safeprotein_panel_build.py - recover the SafeProtein-Bench panel for attempt 2.

Registered in `docs/EXTERNAL_VALIDATION_PREREGISTRATION.md` Amendment 2 before this
was run. Attempt 1 failed on a self-drawn panel; that failure stands. This builds
the externally *curated* panel attempt 1 could not obtain.

Provenance chain, because it is unusual and needs to be checkable:
  SafeProtein-Bench (Fan et al. 2025) curated the hazard set. Its own repository
  `github.com/jigang-fan/SafeProtein` is 404 and the author account has no public
  repositories, so the benchmark is not directly downloadable. VFUSE and SAEBER
  published derived artifacts on Hugging Face, and the per-protein filenames in
  `michaelwaves/saeber-rfd3-safeprotein-activations` encode UniProt accessions with
  hazard or benign labels. Those filenames are the only thing taken from there;
  sequences are fetched from UniProt by accession.

Rules, quoted from Amendment 2 rather than re-derived:
  1. labels come from the filenames and are not reviewed or re-assigned
  2. accessions already in panel_v2_manifest.json are excluded, count reported
  3. NO homology screen against the internal panel, deliberately, because external
     scoring trains and tests entirely within the external panel
  4. classes assigned from UniProt protein names with 05's CLASS_TERMS; no match
     means dropped
  5. thresholds and the three confirmatory numbers are unchanged

Outputs: data/sequences/safeprotein_{positives,negatives}.fasta
         data/sequences/safeprotein_panel_manifest.json
         data/annotations/safeprotein_mechanism_classes.json
"""

import collections
import hashlib
import importlib.util
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEQ = ROOT / "data" / "sequences"
ANN = ROOT / "data" / "annotations"
HF = ("https://huggingface.co/api/datasets/"
      "michaelwaves/saeber-rfd3-safeprotein-activations")
UNIPROT = "https://rest.uniprot.org/uniprotkb/accessions"
BATCH = 50
NAME_PAT = re.compile(r"train_inputs_(\w+?)_([A-Z0-9]{6,10})_\d+_metadata\.pkl$")


def load_05():
    p = ROOT / "src" / "05_external_panel_build.py"
    spec = importlib.util.spec_from_file_location("mod05", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def H(s):
    return hashlib.sha256(s.encode()).hexdigest()


def recover_accessions():
    d = json.load(urllib.request.urlopen(HF, timeout=60))
    out = {}
    for s in d.get("siblings", []):
        m = NAME_PAT.search(s["rfilename"])
        if m:
            out[m.group(2)] = m.group(1)
    return out


def fetch_sequences(accs):
    """UniProt by accession. Returns {acc: {seq, name, id, organism}}."""
    got = {}
    for i in range(0, len(accs), BATCH):
        chunk = accs[i : i + BATCH]
        url = UNIPROT + "?" + urllib.parse.urlencode({
            "accessions": ",".join(chunk),
            "fields": "accession,id,protein_name,organism_name,sequence,length",
            "format": "json"})
        with urllib.request.urlopen(url, timeout=90) as r:
            data = json.load(r)
        for rec in data.get("results", []):
            got[rec["primaryAccession"]] = rec
        print(f"    fetched {min(i + BATCH, len(accs))}/{len(accs)}", flush=True)
        time.sleep(0.3)
    return got


def write_fasta(path, entries):
    with open(path, "w") as f:
        for e in entries:
            f.write(f">{e['acc']} {e['protein_name']} OS={e['organism']}\n")
            for i in range(0, len(e["sequence"]), 60):
                f.write(e["sequence"][i : i + 60] + "\n")


def main():
    m05 = load_05()
    labels = recover_accessions()
    print(f"recovered {len(labels)} accessions from the SAEBER artifact: "
          f"{dict(collections.Counter(labels.values()))}")

    internal = json.load(open(SEQ / "panel_v2_manifest.json"))
    int_acc = {e["acc"].split("|")[1] for e in internal["positives"] + internal["negatives"]}
    overlap = sorted(set(labels) & int_acc)
    for a in overlap:
        labels.pop(a, None)
    print(f"rule 2: {len(overlap)} accessions overlapped the internal panel and were removed")

    print(f"fetching {len(labels)} sequences from UniProt")
    recs = fetch_sequences(sorted(labels))
    missing = sorted(set(labels) - set(recs))
    print(f"UniProt returned {len(recs)}; {len(missing)} not retrievable")

    pos, neg, dropped = [], [], collections.Counter()
    for acc, lab in sorted(labels.items()):
        r = recs.get(acc)
        if r is None:
            dropped["not_in_uniprot"] += 1
            continue
        seq = r.get("sequence", {}).get("value", "")
        if not seq:
            dropped["empty_sequence"] += 1
            continue
        name = m05.protein_name(r)
        entry = {"acc": f"sp|{acc}|{r.get('uniProtkbId', acc)}", "uniprot": acc,
                 "name": r.get("uniProtkbId", acc), "protein_name": name,
                 "organism": r.get("organism", {}).get("scientificName", ""),
                 "len": len(seq), "sha256": H(seq), "sequence": seq,
                 "saeber_label": lab}
        if lab == "hazard":
            cls = m05.assign_class(name)
            if cls is None:
                dropped["hazard_no_class_match"] += 1
                continue
            entry["mechanism_class"] = cls
            pos.append(entry)
        else:
            neg.append(entry)

    print(f"\ndropped: {dict(dropped)}")
    cc = collections.Counter(p["mechanism_class"] for p in pos)
    print(f"panel: {len(pos)} positives / {len(neg)} negatives")
    print(f"classes: {dict(cc)}")

    SEQ.mkdir(parents=True, exist_ok=True)
    ANN.mkdir(parents=True, exist_ok=True)
    write_fasta(SEQ / "safeprotein_positives.fasta", pos)
    write_fasta(SEQ / "safeprotein_negatives.fasta", neg)
    json.dump({"built": time.strftime("%Y-%m-%d %H:%M:%S"),
               "provenance": "SafeProtein-Bench hazard curation (Fan et al. 2025) "
                             "recovered from VFUSE/SAEBER HF artifact filenames; "
                             "sequences fetched from UniProt by accession",
               "hf_source": HF,
               "n_recovered": len(labels) + len(overlap),
               "internal_overlap_removed": overlap,
               "homology_screen": "none, per Amendment 2 rule 3",
               "dropped": dict(dropped), "classes": dict(cc),
               "counts": {"positives": len(pos), "negatives": len(neg)},
               "positives": [{k: v for k, v in e.items() if k != "sequence"} for e in pos],
               "negatives": [{k: v for k, v in e.items() if k != "sequence"} for e in neg]},
              open(SEQ / "safeprotein_panel_manifest.json", "w"), indent=2)
    json.dump({"built": time.strftime("%Y-%m-%d %H:%M:%S"),
               "note": "classes assigned from UniProt protein names using 05's "
                       "CLASS_TERMS, per Amendment 2 rule 4; SafeProtein's own class "
                       "labels were not recovered",
               "holdout_eligible_classes": [c for c, n in cc.items() if n >= 3],
               "proteins": [{"fasta_id": e["acc"], "mechanism_class": e["mechanism_class"]}
                            for e in pos]},
              open(ANN / "safeprotein_mechanism_classes.json", "w"), indent=2)
    print(f"\nwrote the panel and manifests under {SEQ}")


if __name__ == "__main__":
    main()
