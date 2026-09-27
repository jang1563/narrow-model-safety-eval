# VFDB as an external class axis — what the database actually resolves

**Status: design note, not a preregistration.** It records what `src/67_vfdb_ingest.py` measured on
the downloaded database and the one design decision those measurements force. Thresholds get frozen
in a preregistration after that decision, not here.

Written 2026-09-27, prompted by two questions: *virulence and toxicity are not the same thing, and
acting on a human is not the same as acting on another species — are those distinguished, and does
this project cover them?* Both turn out to be measurable against VFDB rather than arguable, and the
answer to the second half is **partly, at an n small enough to be the limitation**.

---

## 1. What was downloaded

[VFDB](https://www.mgc.ac.cn/VFs/download.htm), last updated 2026-09-25, **no license stated on the
download page**. `VFDB_setA_pro.fas.gz` (1.3 MB), `VFDB_setB_pro.fas.gz` (5.6 MB), `VFs.xls.gz`
(0.25 MB). setA is the core set of experimentally verified VFs; setB is the full set including
predicted ones. The raw files are **not committed** — unstated license, 19 MB uncompressed, and
reproducible with `python src/67_vfdb_ingest.py --download`. What is tracked is the script and
`results/vfdb_ingest.json`.

The category assignment is in the FASTA headers, so no separate join is needed:

```
>VFG037176(gb|WP_001081735) (plc1) phospholipase C
 [Phospholipase C (VF0470) - Exotoxin (VFC0235)] [Acinetobacter baumannii ACICU]
```

| | records | categories | VF ids | species | with n ≥ 20 |
|---|---:|---:|---:|---:|---:|
| **setA** (verified) | 4,755 | 14 | 746 | 73 | 12 |
| **setB** (full) | 30,215 | 14 | 1,322 | 283 | 14 |

Fourteen categories at n ≥ 20 in setB against this project's **eleven to twelve** hand-curated
mechanism classes, from a published assignment maintained by someone else. That is the § 1.1b-ter
limitation answered, and it is why this route was chosen.

---

## 2. 🔑 Virulence is not toxicity, and VFDB's own ontology says so

**Exotoxin is one category out of fourteen, and a few per cent of the records.**

| | Exotoxin | non-toxin virulence |
|---|---:|---:|
| setA | **248 (5.2%)** | 4,507 |
| setB | **1,218 (4.0%)** | 28,997 |

The other thirteen, setB counts: Effector delivery system 8,804 · Immune modulation 4,677 ·
Adherence 4,677 · Nutritional/Metabolic factor 3,964 · Motility 3,358 · Biofilm 748 · Enzyme 712 ·
Others 600 · Regulation 585 · Invasion 332 · Stress survival 260 · Post-translational modification
157 · Antimicrobial activity/Competitive advantage 123.

⚠️ **This inverts the v2 panel.** v2 is toxin-centric — clostridial neurotoxins, RIPs,
superantigens, ADP-ribosylating AB toxins, pore-forming cytolysins — with a **ten-member**
`virulence_associated_non_toxin` class carried as a labelled *control*. On the VFDB axis the control
becomes 95% of the data. Adherence proteins, flagellar motility components and iron-uptake systems
are virulence factors and are not hazardous in the sense FSPE and the physical-realizability tiers
were built around.

**What the project already knows about that boundary**, and it is a real result rather than a gap:
the labelled non-toxin virulence control is the **worst-recovering class that is not one of the two
documented failures** — 50% at 95% specificity, 32% at 99%, AUROC 0.844, against 100% for four of
the toxin classes. A probe trained on toxins already does not transfer to non-toxin virulence
factors. The VFDB axis would test that at n = 28,997 instead of n = 10.

---

## 3. 🔑 Target host: the contrast exists, in setB only, and at species resolution

VFDB's organism field names the **producing pathogen, not the target**. § 2.4 of
`docs/MECHANISM_GENERALIZATION.md` already records that these come apart — six of seven
ribosome-inactivating proteins are plant-produced and act on animal ribosomes — so a host label
derived from the pathogen is weaker than v2's hand-annotated target and must be labelled as such.

⚠️ **Genus is the wrong resolution.** *Pseudomonas* holds *aeruginosa* (935 records, human) beside
*syringae* (681, plant) and *entomophila* (139, insect); *Bacillus* holds *anthracis* and *cereus*
beside *thuringiensis*. `src/67` assigns host at species level from an explicit table and leaves
everything else `unassigned` rather than defaulting to the majority — defaulting to "animal" is the
exact failure § 2.4.1 found in the probe.

| host of the producing pathogen | setA | setB |
|---|---:|---:|
| human or mammal | 3,640 | 22,976 |
| **plant** | **0** | **1,093** |
| **insect** | **0** | **511** |
| unassigned | 1,115 | 5,635 |

🔴 **The host contrast is a setB-only property.** setA, the experimentally verified core, is host-
homogeneous: not one plant or insect pathogen in 4,755 records. Asking the species question means
accepting predicted VFs, and that trade is the second decision this note surfaces.

🔑 **And the two axes cross, which is the part worth having.** Exotoxin by host in setB:

| | human or mammal | insect | plant | unassigned |
|---|---:|---:|---:|---:|
| **Exotoxin** | 759 | **181** | **61** | 217 |

242 exotoxins whose producer is an insect or plant pathogen — *Bacillus thuringiensis* Cry toxins,
*Photorhabdus* and *Xenorhabdus* toxin complexes, phytotoxins — against 759 mammalian ones. That is
a matched toxin-versus-toxin, host-versus-host contrast, which is the comparison neither the v2
panel nor any paper in `research/05_v2_related_work_survey.md` runs.

---

## 4. What this project already covers on these two axes

Both questions have been asked here, and the second one has a **better** answer than a first pass
through this document said. The paragraph this replaces was written from v2's annotation and put the
species axis at n = 7; v3's is `data/annotations/target_host_v3.json` and the holdout has already been
run on it. Corrections entry thirty-four.

### 4.1 Toxin versus non-toxin virulence — distinguished, and the probe does not transfer

`virulence_associated_non_toxin` is a **labelled control class**, not an assumption:

| | flagged@95 | flagged@99 | AUROC |
|---|---:|---:|---:|
| v2 (n = 10) | 50% | 32% | 0.844 |
| v3 (n = 10) | **34%** | **12%** | 0.820 |

Against 100% at 95% for four of the toxin classes. 🔑 **A probe trained on toxins already fails to
transfer to virulence factors that are not toxins**, and it is the worst-recovering class that is not
one of the two documented failures. The VFDB axis would test that at n = 28,997 rather than n = 10.

### 4.2 Target host — annotated at species level, and it survives class holdout on v3

`data/annotations/target_host_v3.json`, 149 positives, target kept separate from producer:

| target | n | | target | n |
|---|---:|---|---|---:|
| **bacteria** | 52 | | small molecule | 15 |
| **animal** | 51 | | regulatory / self | 4 |
| **insect** | 22 | | **plant** | **1** |

🟢 **The host axis reaches a held-out mechanism class on v3, and did not on v2.** § 2.4.1 returned
balanced class accuracy **0.500** — exactly always-say-animal — on v2's nine classes split seven to
two, and diagnosed class diversity as the cause. On v3's eleven classes with four non-animal:
**0.804** (animal 0.86, non-animal 0.75), permutation *p* = **0.0150**, no null draw reaching a
perfect score. By the preregistered rule: **SUPPORTED**.

**And recovery does not follow the host label**, which is the part that matters for the VFDB design:

| target | class | flagged@95 |
|---|---|---:|
| insect | cry_insecticidal (n=22) | **87.3%** |
| bacteria | bacteriocin (n=15) | 84.0% |
| bacteria | contact_dependent_inhibition (n=4) | 75.0% |
| bacteria | **phage_peptidoglycan_hydrolase (n=32)** | **10.0%** |
| small molecule | **beta_lactamase (n=14)** | **18.6%** |
| animal | seven classes | 80% to 100% |

🔑 **Both documented failures act on something other than an animal, and acting on something other
than an animal does not predict failure** — an insect-target toxin class recovers at 87.3%, as well as
the mammalian ones. So target host is not the axis the failures lie on; margin is (§ 10.4). An
insect-targeting Cry toxin is as detectable as a mammalian neurotoxin, and a phage lysin is not.

⚠️ **The one host with no support at all is plant, at n = 1** — the *Agrobacterium* T-pilus subunit,
and it sits inside the mixed virulence control rather than in a class of its own. That is the specific
gap VFDB closes, not "other species" in general.

## 5. The decision this forces

Two studies share one download and one embedding run, and they are not the same study.

**Option A — hazard stays "toxin", and VFDB supplies the negatives.** Positives: the 248 verified
(or 1,218 full) exotoxins. Negatives: the 4,507 verified non-toxin virulence factors, which are
ontology-labelled, pathogen-produced, and a far harder negative class than a random benign pool.
This fixes the problem named earlier in this project as the binding one — **n_benign = 4**, which
makes three of the mutation preregistration's four AUROC thresholds unusable. It keeps the hazard
construct that FSPE, FSI and the realizability tiers were built for.

**Option B — hazard becomes "virulence factor", and VFDB supplies the class axis.** Leave-one-
category-out over 14 categories, DeepVIC's own axis, asking the question DeepVIC does not ask. It
tests margin as a pre-hoc predictor of which held-out category fails, out of sample, on someone
else's labels and someone else's class definitions — the strongest available test of § 10.4. It
changes what "hazard" means for the duration of that experiment, and that has to be said in the
title rather than in a footnote.

**Recommendation: A first, then B.** A repairs a stated blocker and leaves the construct intact; B
is the stronger novelty claim and needs A's infrastructure anyway.

The host question rides on either one, and § 4.2 says what to ask of it. The axis already survives
class holdout at 0.804 and recovery already fails to follow it, so the open question is **not** whether
the representation encodes target host. It is (i) **plant**, which has n = 1 here and 1,093 in setB,
and (ii) whether a toxin's detectability tracks its target species once mechanism is held constant —
which setB's Exotoxin × host cross-tab (759 mammal / 181 insect / 61 plant) can answer and the v2/v3
panel cannot, because Cry is the only insect-target class it has. Both require restricting to setB and
stratifying on the species-level host label, with the **producer-not-target** caveat stated wherever
the number appears.
