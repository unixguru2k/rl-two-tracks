# Track A — Post-Training & Agentic RL

> **The question this track answers:**
> *"How do I make the model itself better at reasoning and at acting in an environment?"*

This is the track that dominates research, headlines, and job postings. It is also the track most app developers should **consume rather than build**. This index says plainly which is which.

---

## Who This Track Is For

| You are… | What to do |
|----------|-----------|
| An app/product engineer | **Read A1–A4 for the vocabulary.** Then stop. You consume these models via API — you don't train them. |
| Someone with a verifiable task and GPUs | A4 is your entry point. RLVR is the cheapest real RL win available. |
| Someone with preference pairs and one GPU | A3 is your entry point. DPO/RFT, not RLHF. |
| Building a tool-using or computer-use agent | A5 and A6. This is the actual frontier. |
| **Training a foundation model** | A2. This is the only reason to read A2. It costs millions. |

**The honest default:** unless you have (a) GPUs, (b) an automatic way to grade correctness, and (c) a model you own — you are a consumer of Track A, not a contributor to it. That is not a limitation, it is the intended design of the ecosystem. Track A produces *judges and reasoners*; Track B *uses* them.

---

## Prerequisites

- Comfortable reading Python
- Basic understanding of gradient descent and neural networks
- Ideally: read [Track B](../b-decision-optimization/index.md) first — the RL vocabulary (state, action, reward, policy, value) is far easier to absorb on a Q-table than on a 7-billion-parameter transformer

---

## The Guides

| # | Guide | What It Covers | Reward Source | Compute |
|:-:|-------|----------------|---------------|---------|
| **A0** | [Hands-On Post-Training](a0-hands-on-post-training.md) | **Runnable** SFT → DPO → GRPO on your own hardware (M4 + Colab) | Cross-entropy / preferences / verifier | GPU |
| **A1** | [Policy Gradient / PPO](a1-policy-gradient-ppo.md) | Optimizing the policy directly; the clipped objective | Environment / reward model | High |
| **A2** | [RLHF](a2-rlhf.md) | Preference data → reward model → PPO | Human preferences | Extreme |
| **A3** | [DPO & RFT](a3-dpo-and-rft.md) | Alignment without a reward model or RL loop; generate-filter-finetune | Preference pairs / filtered samples | Medium |
| **A4** | [GRPO & RLVR](a4-grpo-and-rlvr.md) | Critic-free group-relative optimization; verifiable rewards | **Programmatic verifier** | High |
| **A5** | [Agentic RL & Environments](a5-agentic-rl-environments.md) | Multi-turn rollouts, tool use, sandboxes, the environment as a packaged unit | Verifier + environment state | High |
| **A6** | [Rewards & Test-Time Compute](a6-rewards-and-test-time-compute.md) | Outcome vs process rewards, generative verifiers, search and reranking at inference | Verifier / reward model / judge | Medium–High |

---

## The Core Loop

Every guide in this track is a variation on one loop:

```
──────────────────────────────────────────────────────────────┐
│                                                              │
│   prompt / task ──► POLICY ──► output or action sequence      │
│                       ▲                  │                   │
│                       │                  ▼                   │
│                  UPDATE WEIGHTS ◄── SCORE                    │
│                  (gradient step)     (who scores it?)        │
│                                                              │
└──────────────────────────────────────────────────────────────
```

The only thing that meaningfully differs between A1 → A6 is **who does the scoring**, and **how many turns the action sequence spans**.

---

## The Reward Source Progression

This is the single most useful axis for orienting yourself in this track. It is also the axis along which the field actually moved.

```
TRUSTWORTHY ──────────────────────────────────────────► CHEAP
EXPENSIVE ────────────────────────────────────────────► SCALABLE

  Human preferences        AI judge            Programmatic verifier
  ────────────────        ────────            ─────────────────────
   A2 (RLHF)               RLAIF               A4 (RLVR)
                           A3 (pairs)          A5 (environments)

  Slow, costly,            Cheap, biased       Instant, exact,
  high quality             like the judge      impossible to argue with

  "Which answer would       "Which answer        "Is the answer 42?
   a human prefer?"          does GPT-4 prefer?"  Do the tests pass?"
```

**The 2026 shift in one line:** the interesting work moved from the left of that axis to the right. Verifiers replaced annotators wherever correctness could be checked by machine.

That move has a hard boundary, and it is the boundary that matters most:

```
Verifiable tasks                    Non-verifiable tasks
─────────────────                   ────────────────────
math, code, logic,                  empathy, creativity,
tool-call success,                  tone, open-ended
schema/format,                      writing, taste
constraint satisfaction
        │                                    │
        ▼                                    ▼
   RLVR works great                  Fall back to A2/A3
   (A4, A5)                          (preferences / judges)
```

---

## The 2026 Post-Training Stack

If you were building this from scratch today, the canonical order is:

```
1. SFT                    supervised fine-tune on demonstrations
                          (teach the format and the task)
        │
        ▼
2. LIGHT PREFERENCE       DPO / SimPO / ORPO / KTO
   OPTIMIZATION           cheap, stable, no reward model
        │                 (A3)
        ▼
3. RLVR — IF NEEDED       GRPO / DAPO / GSPO
                          only when you have a verifier AND need
                          reasoning or agency gains (A4)
        │
        ▼
4. TEST-TIME COMPUTE      search / reranking / self-consistency
                          often cheaper than more training (A6)
```

Three things worth internalizing:

1. **You rarely need step 3.** Most teams stop at step 2 and get most of the benefit.
2. **Steps 3 and 4 are complementary, not substitutes.** RL training (step 3) raises the model's reasoning *ceiling*; test-time compute (step 4) extracts that ceiling at inference. Verifier-based RL typically scales *better* with a larger test-time budget than training-free methods — so this is not an either/or choice.
3. **Steps are not exclusive.** RLHF-era thinking assumed one pipeline; 2026 practice mixes them per capability.

---

## Failure Modes Specific To This Track

| Failure | What it looks like | Where it appears |
|---------|-------------------|------------------|
| **Reward hacking** | Model games the scorer: writes plausible-looking but wrong math, discovers test-file shortcuts | A4, A5 |
| **Verifier gaming** | Exploits the *checker*, not the task — asserts pass, edits tests, prints expected output | A5 |
| **Length bias** | Reward accidentally correlates with verbosity → model rambles | A4 (Dr.GRPO addresses) |
| **KL collapse** | Policy drifts too far from reference → incoherent, repetitive | A1, A2, A4 |
| **Preference overfitting** | Model matches your pairs, generalizes to nothing | A3 |
| **Vanishing advantage** | All outputs in a group score identically → zero gradient → no learning | A4 |
| **Sim-to-real gap** | Great in the sandbox, useless in production | A5 |

Reward hacking deserves emphasis: **it is not a bug you fix once, it is a permanent adversarial dynamic.** Any time a model is optimized against a fixed scorer, it will find the scorer's weaknesses. Design for it from the start with held-out verifiers and human spot-checks.

---

## The Cross-Track Bridge

Track A and Track B are coupled at exactly one point, and it is worth stating explicitly because it is the most useful idea in this whole repo:

```
   TRACK A                          TRACK B
   ──────                          ───────
   Produces JUDGES       ──────►    Consumes judges as reward signals
   (RLHF'd / DPO'd                  (score each served action)
    reward models,
    preferences, verifiers)
```

**Track A outputs are Track B inputs.**

If your product has an action whose outcome you cannot measure directly — "was this empathetic?", "was this explanation clear?" — you cannot bandit-test it, because there is no reward signal. Track A supplies one: run an aligned model as a scorer, and that score becomes your Track B reward.

```
Without the bridge:
   serve response ──► ??? no measurable outcome ──► can't learn

With the bridge:
   serve response ──► Track A judge scores it 0..1 ──► Track B bandit updates
```

This is the answer to "how do I do RL on things I can't measure with a click." It is not RLHF on your own model — it is using someone else's RLHF'd model as a measuring instrument.

---

## Recommended Order

```
Want to actually RUN the pipeline?           A0             (hands-on: SFT → DPO → GRPO)
Curious about how ChatGPT was made?          A2 → A4 → A5   (read for understanding)
Want to align your own small model?          A3 → A4        (build)
Building a reasoning or coding assistant?    A4 → A6        (build)
Building a tool-using or computer-use agent? A5 → A6        (build — the frontier)
Building a product with personalization?     Go to Track B, come back only for the bridge
```

---

## Key Takeaway

```
Track B optimizes WHICH ACTION your system takes.
Track A optimizes THE MODEL that generates the action.

Both are "RL." They share the vocabulary and almost nothing else:
different reward sources, different runtimes, different costs,
different people doing the work.
```

---

**See Also**: [Track B — Decision Optimization in Systems](../b-decision-optimization/index.md) · [Master RL Methods Guide](../../RL_METHODS_GUIDE.md)

---

**Last updated:** 2026-09-20