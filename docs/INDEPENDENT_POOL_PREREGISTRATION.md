# Preregistration — do the headline findings survive a benign population they have never seen?

**Frozen 2026-09-29, before the second pool has been fetched.** Two primaries, one composition check.
Append-only.

---

## 0. The structural weakness this addresses

The step-by-step review earlier today found that **eleven analysis scripts draw from the same
`admitted_rows` union and the same `test_rows`**. Studies E through Q are not independent replications;
they are re-analyses of **one fit on one population of 8,259 Swiss-Prot proteins**. 🔴 **And the "second
arm" this project relies on is a different *embedding of the same proteins*, not different data** — a
weaker guarantee than the phrase suggests, and one I used repeatedly without qualifying it.

## 🔒 0.1 Why a disjoint sample is cheap here

The original pool's build log records the binding constraint: a **per-organism cap of 30** rejected
**28,728** proteins that the identical query had already returned. 🔑 **A second, disjoint sample from
the same query and the same filters is therefore available without changing anything about how the
population was defined** — it is the tranche the cap discarded.

⚠️ **Declared limitation, because it is not a random resample.** Tranche 2 is the proteins at positions
31 and beyond in UniProt's return order for each organism. That order is not random, and nothing
guarantees positions 31–60 resemble 1–30. 🔒 **So a composition check runs first and is reported
whatever it shows**: length distribution, localization strata proportions, kingdom mix and organism
overlap against pool 1. **If the two differ materially, the replication is confounded by composition
and says so.**

---

## 1. Design

Pool 2 is built by `src/100` with the **query string copied verbatim** from
`data/sequences/_scaled_negative_pool.json`, the same per-organism cap of 30, and every accession
already in pool 1 excluded. Target **6,000**, which scales the original's cells to roughly
extracellular 475 / intracellular 1,975 / short 200 / mid 610 — all above the floors those studies
froze.

⚠️ **Embedding is the cost and it is staged.** ESM-2 **35M first** (~15 minutes); **650M is a separate
decision** (~3 hours) and is not assumed here. 🔒 **A 35M-only result is single-arm and therefore
indicative**, as this project's standing rule requires — but it is indicative on a *new axis*, an
unseen population, which is the axis this study exists to test.

---

## 2. Primary tests, frozen bands — unchanged from the studies being replicated

**S-1 — the adverse finding.** The localization ratio `R` = extracellular / intracellular, study L's
strata, study G's fold procedure refitted on pool 2.

| band | reading |
|---|---|
| **R ≥ 3.0** | 🔴 study E's verdict replicates on an unseen population |
| 1.5 – 3.0 | ⚠️ partial |
| **R ≤ 1.5** | 🟢 it does not replicate — and studies E, L, M, N, O and P all rest on it |

**S-2 — the positive finding.** `C/A` = VFDB-registered (study K's 778, unchanged) over pool 2's benign
extracellular cell.

| band | reading |
|---|---|
| **≥ 2.0** | 🟢 the registration contrast replicates |
| 1.2 – 2.0 | ⚠️ partial |
| **≤ 1.2** | 🔴 it does not replicate |

🔴 **What a failure obliges, written before the data exists.** If `R ≤ 1.5`, the project's main adverse
finding is a property of one draw of Swiss-Prot and **every document stating it must say so in the
same commit**. If `C/A ≤ 1.2`, the same applies to the registration finding. **Both obligations are
written now because the whole point of this study is that I do not know which way it goes.**

## 3. Frozen predictions

- **S-1: `R` between 4 and 7**, replicating. The localization effect has survived coverage expansion,
  length adjustment, residue-only pooling and four model sizes; a fresh draw of the same query is a
  weaker challenge than those were.
- **S-2: `C/A` between 2.0 and 3.0**, replicating.
- ⚠️ **The composition check is where I expect trouble, not the ratios** — the per-organism cap means
  pool 2 is drawn from organisms that had **more than 30** qualifying proteins, so it should be
  **biased toward well-studied organisms** relative to pool 1.
- 🔒 **Checkable against a wrong answer**: if pool 2's accessions overlap pool 1 at all, the exclusion
  failed and nothing here is independent.

## 4. Floors

🔒 Extracellular and intracellular cells need ≥ 300 each, as study E froze. Below that, S-1 is
indicative.
🔒 If fewer than 4,000 proteins are kept, the cells fall below those floors and the study is reported
as underpowered rather than run to a verdict.

---

## 🔒 Amendment 1 — 2026-09-29, after the composition check and before any probe touches pool 2

§ 1 set the target at **6,000** on the assumption that pool 2's kingdom mix would resemble pool 1's.
**The composition check says it does not**, and § 0.1 required that check to be reported whatever it
showed:

| | pool 1 | pool 2 (target 6,000) |
|---|---:|---:|
| median length | 433 | 423 |
| mean length | 460 | 464 |
| **Bacteria** | **87.5%** | **60.2%** |
| **Eukaryota** | **3.3%** | **33.4%** |
| distinct species | 1,356 | 1,116 |
| species shared with pool 1 | — | 398 (36% of pool 2) |

🔑 **Length matches closely; the kingdom mix does not.** The cause is in pool 1's own build note: its
per-organism cap "bit hard on eukaryotic Swiss-Prot", so pool 1 **under-filled its Eukaryota quota**,
while pool 2 — drawing the tranche the cap had rejected — filled every quota exactly.

⚠️ **This matters only through the eligible count**, because every analysis being replicated restricts
to **Bacteria and Archaea**. Pool 2 at 6,000 holds **3,825** of them, which projects to an
extracellular stratum of **~301 against a frozen floor of 300**. 🔴 **A projection that lands one above
a floor will land below it about as often as above**, and the floor exists to stop exactly that.

🔒 **So the target is raised to 11,300**, which matches pool 1's eligible count of 6,247. **The
population rule is unchanged** — same query, same filters, same per-organism cap of 30, same
proportional quotas, same exclusion of pool 1. **Only the size changes, and it changes before any
probe has been applied.**

⚠️ **The kingdom-mix difference does not go away by drawing more**; it is a property of which tranche
each pool took. It is carried with every number this study produces, and the shared-species figure
(**36%**) is the more informative one for a replication: **most of pool 2's organisms are not pool 1's
organisms.**
