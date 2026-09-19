# Claim Receipts

Notes on the load-bearing claims in this repo's launch copy and track summaries — what is sourced, what was narrowed, and what still needs a citation.

The rule applied throughout: **narrow the population until the claim becomes checkable.**

---

## The Claim Under Test

Original phrasing, used in the launch copy and in the Track B takeaways:

> "Most production RL looks like B — a bandit in front of a frozen model. It just never gets a tutorial."

This is three separate claims of different strength:

| # | Claim | Type | Verdict |
|:-:|-------|------|---------|
| 1 | "Most production RL looks like B" | Quantitative | **Not supported — narrowed** |
| 2 | "...a bandit in front of a frozen model" | Architectural | **Supported, but narrower than written** |
| 3 | "It just never gets a tutorial" | Content supply | **Supported, with a caveat** |

---

## 1. "Most production RL looks like B" — why it was narrowed

Two problems with the quantifier:

- **No denominator exists.** There is no census of production RL deployments. "Most production RL" is unfalsifiable as written — there is no dataset you could check it against. A claim with no possible falsifying observation is rhetorical, not empirical.
- **Rigorous sources say RL is *rare* in production, period.** A 2025 real-world systems survey describes RL as appearing *"rarely"* in production, usually confined to tightly scoped optimization where experimentation is cheap. A 2024 industry review puts manufacturing RL adoption *"in its early stages."* If RL overall is rare in production, "most production RL is X" is a claim about a small, poorly-measured population.

**Resolution:** drop the unsupported quantifier and scope the claim to a population that *is* measurable — online learning in ads, ranking, and recommendation.

---

## 2. "A bandit in front of a frozen model" — supported, but narrower

The *architecture* is real and documented in the literature, but it is one named pattern, not the definition of production RL:

| Source | What it shows |
|--------|---------------|
| **OrcaRouter** | Production LLM routing via a LinUCB contextual bandit over prompt features and embeddings, with an offline→online deployment protocol and a frozen exploitation mode for stability |
| **"A Control System, a Dataset, and a Recipe for Making Production LLM Agents Learn from Their Own Experience"** | Frames production agents explicitly as *"a frozen model wrapped in a harness"*; learns a policy over the harness with a small, enumerable action space (prompt style, retrieval policy, memory policy, planning depth, step budget) |
| **"Learning to Route: A Shadow-First Contextual Bandit for LLM Routing"** | Shadow-mode bandit, then cautious ε-greedy serving rollout with deterministic constraints |
| **BaRP** | LLM routing as a preference-conditioned contextual bandit with a frozen encoder |

**Caveat:** most documented bandit deployments are ads/recs/ranking — *not* wrapped around an LLM. The bandit-on-frozen-LLM pattern is the newest slice. So "bandits dominate online decision-making" is well supported; "in front of a frozen model" narrows past what the evidence covers.

---

## What *is* strongly supported

Contextual bandits as the **industrial default** for online decision-making in ads, ranking, and recommendation:

| Source | Evidence |
|--------|----------|
| **Microsoft Decision Service** | Production contextual-bandit system with a full explore/log/learn/deploy loop; reported 25–30% CTR improvement and 18% revenue lift on a landing page |
| **MSN / Microsoft News** | Contextual-bandit systems reported ~26% click increase (January 2016) over a static baseline |
| **Microsoft Responsive Search Ads** | Sequential contextual-bandit framework with per-position Thompson Sampling |

Ads, ranking, recommendation, pricing, and experiment allocation are consistently described as bandit-dominated domains. This is the strongest version of the claim and the one the repo now uses.

---

## 3. "It never gets a tutorial" — defensible, with a caveat

The *classic* bandit tutorial literature is enormous — Vowpal Wabbit, Microsoft's Decision Service writeups, Auer et al. on UCB, the Thompson Sampling line.

What is genuinely underserved is the **LLM-specific** version: *a bandit selecting prompt / route / style in front of a frozen model, with production guardrails.* That gap is real, which is why the repo's phrasing is now narrower than "never gets a tutorial."

---

## What Changed in the Repo

The claim was narrowed in two places to a scoped, checkable population:

| File | Before | After |
|------|--------|-------|
| `README.md` (Track B takeaway) | "Most production RL in the world looks like this, not like RLHF." | "In ads, ranking, and recommendation, the online-learning default has been a contextual bandit for a decade — and increasingly, a bandit wrapped around a frozen model. It almost never gets a tutorial." |
| `docs/tracks/b-decision-optimization/index.md` (Key Takeaway) | "Most production RL in the world looks like this, not like RLHF." | Same scoped framing as above. |
| `docs/launch/launch-copy.md` (LinkedIn short version) | "Production RL looks like B — a bandit in front of a frozen model." | "Production RL is B — a contextual bandit in front of a frozen model." (drops the "most" quantifier) |

Launch copy (`launch-copy.md`) uses the same narrowed line throughout.

---

## Still Needs a Citation

These are asserted in the repo and are checkable, but the receipts are not yet collected here:

- The 2025 real-world systems survey and the 2024 industry review referenced above (exact titles, authors, venues).
- The specific CTR/revenue figures for Microsoft Decision Service and MSN (primary source, not secondary reporting).
- Whether any of the LLM-routing papers listed under §2 have moved from shadow/canary deployment to full production traffic.

Add URLs and full citations as they are confirmed. Until then, treat the figures above as *reported*, not *verified*.
