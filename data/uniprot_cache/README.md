# Cached UniProt records

Four UniProtKB entries, fetched 2026-09-21 from `https://rest.uniprot.org/uniprotkb/{acc}.json`,
cached so that `src/51_tier3_coordinate_verification.py` reproduces without a network and so that a
later UniProt revision cannot silently move a coordinate that a frozen preregistration depends on.

| accession | protein | why it is here |
|---|---|---|
| P00588 | Diphtheria toxin | tier 3 pair, CRM197 |
| P04977 | Pertussis toxin subunit 1 | tier 3 pair, 9K/129G |
| P0DPI1 | Botulinum neurotoxin type A | tier 3 pair, zinc-ligand E->Q |
| P02879 | Ricin | tier 3 candidate, demoted; see the preregistration's amendment log |

Re-fetch with `python src/51_tier3_coordinate_verification.py --fetch`, which overwrites these
files. Do that deliberately: if a coordinate moves, that is a finding to record in
`docs/DATA_CORRECTIONS.md`, not a cache to refresh quietly.

UniProt is released under CC BY 4.0.
