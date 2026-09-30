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

---

## 🔴 Amendment 2 — 2026-09-29, on running it: § 2 did not say which benign cell S-2 meant

§ 2 defined S-2 as *"`C/A` = VFDB-registered over pool 2's benign extracellular cell"*. 🔴 **There are
two such cells and they differ by more than a factor of two.**

| denominator | rule | pool 1 | pool 2 |
|---|---|---:|---:|
| **study K's** | keyword strata, Bacteria, **restricted to VFDB species**, contaminants out | **2.890** | **1.629** |
| study L's | GO-augmented strata, all Bacteria + Archaea, no species restriction | 5.400 | 2.296 |

🔒 **The band is judged on study K's**, because that is the only one comparable to the 2.617 this study
is trying to replicate — and the script reproduces study K closely on pool 1 (**2.890** against 2.617,
the residual being the similarity screen this study skips). **The other is reported beside it rather
than dropped**, because the ambiguity was mine and hiding it would let the more favourable number stand
unchallenged.

⚠️ **A second gap in § 4**: floors were frozen for the localization strata (300) and for the pool size
(4,000) but **not for S-2's denominator cell**, which holds **115** in pool 2 against 171 in pool 1.
**S-2 is therefore reported with that number attached.**

---

## Results, 2026-09-29 — ESM-2 35M, single arm, indicative

`src/101_independent_pool_replication.py`. Pool 2: 11,301 proteins, **zero overlap** with pool 1, the
partition rule reproducing pool 1's published split at **100%**.

| | pool 1 (same script) | **pool 2** |
|---|---:|---:|
| union / fold | 1,475 → (983, 492) | 1,483 → (988, 495) |
| extracellular / intracellular | 619 / 2,708 | **320 / 3,441** |
| ⭐ **S-1, localization `R`** | 6.059 | **10.235 [9.585, 10.884]** |
| ⭐ **S-2, `C/A`** (study K's denominator) | 2.890 *(n=171)* | **1.629 [1.594, 1.664]** *(n=115)* |

### 🟢 S-1: the adverse finding replicates, and strengthens

**R = 10.235** against a frozen band of ≥ 3.0. 🔒 **On a population that shares only 32% of its species
with pool 1, the localization gradient is not merely present but larger than on the pool every earlier
study used.** Floors met (320 and 3,441 against 300).

### 🔴 S-2: the positive finding weakens from supported to partial

**C/A = 1.629**, in the **partial** band, where study K read **2.617** — *supported*. The same script on
pool 1 returns **2.890**, so this is not a script difference: **the registration contrast is about
44% smaller on a benign population the probe has not seen.**

🔒 **The frozen obligation does not fire** — it was written for `C/A ≤ 1.2` — so nothing is retracted.
⚠️ **But "supported on both arms" was a statement about one pool**, and on a second pool of the same
construction it is partial. **That qualification belongs on the finding and is added to it.**

### 🔒 What the skipped similarity screen was worth, measured

Pool 1 refitted **without** its screen gives `R` = **6.059** against study L's screened, decontaminated
**5.353** — so the screen is worth about **0.7** on this quantity, in the direction of *lowering* it.
🔑 **Pool 2's 10.235 is therefore, if anything, understated relative to a screened comparison**, and
the procedural shortcut cannot explain the replication.

### 🔒 Scoring the frozen predictions — both primaries missed, in opposite directions

| | frozen | actual | |
|---|---|---|---|
| S-1 `R` between 4 and 7 | | **10.235** | ❌ replicates, but far above the band |
| S-2 `C/A` between 2.0 and 3.0 | | **1.629** | ❌ below it |
| the composition check is where I expect trouble | | kingdom mix 87.5% vs 60.2% bacterial | ✅ |
| any overlap ⇒ the exclusion failed | | 0 | ✅ guard silent |

⚠️ **I predicted both headline quantities would land in narrow bands and neither did.** The direction
that matters: **the finding I expected to be fragile held and grew; the finding I was most confident in
shrank.**

---

## Results, second arm — ESM-2 650M, 2026-09-30

`src/102` embedded pool 2 at 650M with shard checkpointing, its gate passing at **9.060e-06**; the run
was stopped once at the user's request and **resumed from shard 2**, which is the first time this
project's checkpointing has been exercised for real.

| | pool 1 (same script) | **pool 2** | |
|---|---:|---:|---|
| **S-1, `R`** — 35M | 6.059 | **10.235 [9.585, 10.884]** | replicates |
| **S-1, `R`** — 650M | 7.682 | **11.774 [10.423, 13.125]** | replicates |
| **S-2, `C/A`** — 35M | 2.890 | **1.629 [1.594, 1.664]** | partial |
| **S-2, `C/A`** — 650M | 2.813 | **1.594 [1.567, 1.620]** | partial |

### 🔴 Both primaries now have two-arm verdicts, and they point opposite ways

🟢 **S-1 replicates on both arms and is larger on the unseen pool on both** — 10.235 against 6.059, and
11.774 against 7.682, against a frozen band of ≥ 3.0. 🔒 **The localization gradient is not a property
of one draw of Swiss-Prot.** This is no longer indicative; it is a verdict.

🔴 **S-2 lands in the *partial* band on both arms** — 1.629 and 1.594, where study K read **2.617** and
**2.418** as *supported*. The same script on pool 1 returns **2.890** and **2.813**, reproducing study K
closely, so **the drop is the population and not the code.** 🔒 The frozen retraction threshold of
`C/A ≤ 1.2` is **not crossed on either arm**, so nothing is withdrawn — but **"supported on both arms"
described one pool**, and on a second pool of identical construction the same quantity is partial on
both arms.

### ⚠️ The skipped screen is worth more on 650M than I said on 35M

Pool 1 refitted without its similarity screen gives `R` = **6.059** on 35M against study L's screened
5.353, and **7.682** on 650M against study L's screened 4.965 — so the screen is worth about **0.7** on
one arm and **2.7** on the other. 🔴 **My 35M write-up called it "about 0.7" without saying that was
arm-specific, and it is not.**

🔒 **What holds on both arms is the direction**: the screen *lowers* `R`, so pool 2's 10.235 and 11.774
are conservative relative to a screened comparison and the procedural shortcut cannot manufacture the
replication. ⚠️ **The magnitude is arm-dependent and is not to be quoted as a single number.**
