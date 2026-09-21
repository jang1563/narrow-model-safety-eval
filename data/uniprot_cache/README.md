# Cached UniProt records

UniProtKB entries fetched 2026-09-21 from `https://rest.uniprot.org/uniprotkb/{acc}.json`,
cached so that `src/51_tier3_coordinate_verification.py`, `src/52_flagged_entries_vs_uniprot.py`
and `src/53_signal_peptide_sweep.py` reproduce without a network, and so that a later UniProt
revision cannot silently move a coordinate that a frozen preregistration or a published headline
depends on.

| accession | protein | len | entry | why it is here |
|---|---|---|---|---|
| O34208 | ExoU | 687 | TrEMBL | panel FSPE annotation, signal-peptide sweep |
| P00588 | Diphtheria toxin | 567 | Swiss-Prot | tier 3 mutation pair |
| P00648 | Ribonuclease | 157 | Swiss-Prot | panel FSPE annotation, signal-peptide sweep |
| P01552 | Enterotoxin type B | 266 | Swiss-Prot | 🔴 the one signal-peptide hit; its published offset is ruled out |
| P01555 | Cholera enterotoxin subunit A | 258 | Swiss-Prot | panel FSPE annotation, signal-peptide sweep |
| P02879 | Ricin | 576 | Swiss-Prot | tier 3 mutation pair |
| P04419 | Colicin-E2 | 581 | Swiss-Prot | panel FSPE annotation, signal-peptide sweep |
| P04958 | Tetanus toxin | 1315 | Swiss-Prot | panel FSPE annotation, signal-peptide sweep |
| P04977 | Pertussis toxin subunit 1 | 269 | Swiss-Prot | tier 3 mutation pair |
| P0DF97 | Streptolysin O | 571 | Swiss-Prot | panel FSPE annotation, signal-peptide sweep |
| P0DPI1 | Botulinum neurotoxin type A | 1296 | Swiss-Prot | tier 3 mutation pair |
| P11140 | Abrin-a | 528 | Swiss-Prot | panel FSPE annotation, signal-peptide sweep |
| P13423 | Protective antigen | 764 | Swiss-Prot | panel FSPE annotation, signal-peptide sweep |
| P15273 | Tyrosine-protein phosphatase YopH | 468 | Swiss-Prot | panel FSPE annotation, signal-peptide sweep |
| P55981 | Vacuolating cytotoxin autotransporter | 1290 | Swiss-Prot | panel FSPE annotation, signal-peptide sweep |
| Q51451 | Exoenzyme S | 453 | TrEMBL | annotation-flagged; domain-start hypothesis refuted here |
| Q99ZW2 | CRISPR-associated endonuclease Cas9/Csn1 | 1368 | Swiss-Prot | panel FSPE annotation, signal-peptide sweep |

Re-fetch with any of those scripts' `--fetch` flag, which overwrites these files. Do that
deliberately: if a coordinate moves, that is a finding to record in
`docs/DATA_CORRECTIONS.md`, not a cache to refresh quietly.

UniProt is released under CC BY 4.0.
