# Track B — Decision Optimization in Systems

> **The question this track answers:**
> *"Which action should my system take, given the current context, right now?"*

This is the track for people shipping software. It runs online, per-request, in milliseconds, on a CPU. It needs no GPUs, no reward model, and no preference data — only a measurable outcome and a place to store one number.

If you are building a product and wondering how RL applies to *your* code, this is your track.

---

## Who This Track Is For

| You are… | What to do |
|----------|-----------|
| An app/product engineer | Start at **B0**, go in order. This is your entire curriculum. |
| Building personalization, ranking, routing, or A/B tests | **B1 → B2** is the highest-value two hours you will spend. |
| Modeling a multi-step user journey | **B3 → B4**. Sequences are a different problem shape than picks. |
| Working with raw text/images as state | **B5**. Last resort — try B4 first. |
| About to ship any of the above to real users | **B6**. Read it before, not after. |

**The honest default:** B0–B4 solve the overwhelming majority of real production personalization problems. B5 is for genuinely high-dimensional state. If you are reading B5 first, you are probably over-engineering.

---

## Prerequisites

- Basic Python (NumPy helps, but the early guides are nearly pure Python)
- No machine learning background required — that is the point of starting at B0
- No GPU, no framework, no API key

---

## The Guides

| # | Guide | Problem Shape | Complexity | Online? |
|:-:|-------|---------------|:----------:|:-------:|
| **B0** | [Random Baseline](b0-random-baseline.md) | Establish the floor | ⭐ | N/A |
| **B1** | [Epsilon-Greedy Bandit](b1-epsilon-greedy-bandit.md) | Pick best option; **no context** | ⭐ | ✅ |
| **B2** | [Contextual Bandit (LinUCB)](b2-contextual-bandit-linucb.md) | Pick best option **given context** | ⭐⭐ | ✅ |
| **B3** | [Q-Learning (Tabular)](b3-q-learning-tabular.md) | **Sequence** of decisions, small state space | ⭐⭐ | ✅ |
| **B4** | [Feature-Based Q-Learning](b4-feature-based-q-learning.md) | Sequence, large/continuous state space | ⭐⭐⭐ | ✅ |
| **B5** | [DQN](b5-dqn.md) | Sequence, raw high-dimensional state (text/images) | ⭐⭐⭐⭐ | ⚠ |
| **B6** | [Production Online Systems](b6-production-online-systems.md) | Shipping any of the above to real users | ⭐⭐⭐ | ✅ |

---

## The Core Loop

Every guide in this track is a variation on one loop:

```
──────────────────────────────────────────────────────────────┐
│                                                              │
│   CONTEXT ──► POLICY ──► serve action to user                │
│                  ▲              │                            │
│                  │              ▼                            │
│            UPDATE ONE VALUE ◄── OBSERVE OUTCOME              │
│            (milliseconds)       (reply, click, return visit) │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

Compare this to [Track A's loop](../a-llm-post-training/index.md#the-core-loop). Same shape, opposite economics: Track A updates *weights* in GPU-hours; Track B updates *a few numbers* in microseconds.

---

## The Two Paradigms

The whole track divides in two, and knowing which side you are on determines which half of these guides matter:

```
                        Do the ORDER of your decisions matter?
                                        │
                        ┌───────────────┴───────────────┐
                        │                               │
                       NO                              YES
                        │                               │
                        ▼                               ▼
                  ┌───────────┐                   ┌───────────
                  │  BANDITS  │                   │    MDP    │
                  │           │                   │           │
                  │  B0, B1,  │                   │  B3, B4,  │
                  │     B2    │                   │    B5     │
                  └───────────                   └───────────┘

  "Pick the best option        "Pick the action that leads to the
   right now. Each decision     best FUTURE. What I do now changes
   is independent."             what happens next."
```

**The most common mistake in this track is using a bandit for a sequential problem, or an MDP for an independent one.** If yesterday's decision does not change today's options, it is a bandit problem and B3 is overkill. If your user journey is greeting → question → answer → confirmation, the order matters and B1 will not find it.

---

## The Complexity Ladder

```
Cheapest to run / easiest to debug
        │
        │   B0   Random                     zero storage, zero code
        │   B1   Epsilon-Greedy             ~20 numbers
        │   B2   LinUCB                      ~500 numbers, uses features
        │   B3   Q-Table                     states × actions
        │   B4   Feature Q-Learn             ~50 weights
        │   B5   DQN                         neural network (MBs), needs GPU
        │
Most expensive / hardest to debug

   ▲ Climb only when the level below provably breaks.
   ▲ Every level costs more to operate, monitor, and explain.
```

**The single most valuable rule in this track: always run the level below your target as a baseline.** B0 is not a warm-up exercise — it is the control condition that tells you whether your careful algorithm is doing anything at all. Ship B0, ship B1, and compare. This is the discipline that separates working personalization from cargo cult.

---

## Storage: Where the State Lives

This is the part of the repo that is specific to Track B, because only Track B needs per-request O(1) state.

Track A's storage is a GPU filesystem and an object store. Track B's storage is a key-value read that must complete inside a request's latency budget.

| Data | Store | Key Pattern | Why |
|------|-------|-------------|-----|
| Q-table per user | key-value store | `rl-qtable-{user_id}` | O(1) lookup per decision |
| LinUCB weights per user | key-value store | `linucb-user-{user_id}` | O(1) lookup per decision |
| Global weights (shared) | key-value store | `linucb-global` | Cold-start baseline for new users |
| Reward events | append-only log | tags: `[user_id, "reward"]` | Searchable analytics and offline eval |
| Interaction history | append-only log | tags: `[user_id, "interaction"]` | Debugging and replay |
| User preferences / profile | key-value store | `rl-prefs-{user_id}` | Fast profile lookup |

The layout rule that makes this work: **the key-value store holds the mutable learning state the policy reads on every request; the append-only log holds the immutable events you analyze later.** Keep them separate. Mixing them means either slow reads or no history.

See [B6 — Production Online Systems](b6-production-online-systems.md) for the full serving architecture, drift handling, and safety rails.

---

## Two Problems That Will Bite You

| Problem | Why it hurts | Where it's covered |
|---------|--------------|--------------------|
| **Cold start** | A new user has an empty table. Pure per-user learning treats them like a stranger for their first 20 interactions. | B2 (hybrid global + personal weights), B6 |
| **Non-stationarity** | User tastes drift. Data from six months ago actively misleads you. A policy that never forgets is a policy that decays. | B6 |

Both are handled in B6, but note *why* they are handled late: neither is visible in a notebook. They only appear when real users, real time, and real traffic hit your policy. That is the gap B6 exists to close.

---

## Recommended Order

```
Just want the 80/20?            B0 → B1 → B2       (covers most personalization)
Personalization is working,
now model the journey?          B3 → B4            (sequential decisions)
Text/images are the state?      B5                 (only if B4 provably fails)
About to ship to real users?    B6                 (mandatory)
```

---

## Key Takeaway

```
Track A optimizes THE MODEL that generates actions.
Track B optimizes WHICH ACTION your system takes.

This track needs no GPUs, no reward model, and no preference data —
just a measurable outcome, a context vector, and one key-value store.

In ads, ranking, and recommendation, the online-learning default has been
a contextual bandit for a decade. Increasingly, it's a bandit wrapped
around a frozen model — and it almost never gets a tutorial.
```

---

**See Also**: [Track A — Post-Training & Agentic RL](../a-llm-post-training/index.md) · [Master RL Methods Guide](../../RL_METHODS_GUIDE.md)

---

**Last updated:** 2026-09-17