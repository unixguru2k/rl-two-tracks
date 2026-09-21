# A6 — Rewards & Test-Time Compute

> **Track A · Guide 7 of 7** — [Track A Index](index.md) · [RL Methods Guide](../../RL_METHODS_GUIDE.md)

## What You'll Learn

A4 and A5 taught a model to reason and to act. A6 is about the two levers that decide *how good* that reasoning gets: **where the reward comes from** (outcome vs process) and **how much compute you spend at inference** (search, reranking, self-consistency). Training raises the model's reasoning *ceiling*; test-time compute is how you *extract* it.

> **TL;DR:** A4/A5 reward the *answer*. A6 asks two harder questions — *can you reward the reasoning steps, not just the result?* and *can you get more out of a frozen model by thinking longer at inference?* Both are cheaper than more training, and both are where the frontier moved.

---

## The Concept

```
┌─────────────────────────────────────────────────────────────────┐
│              TWO LEVERS, ONE GOAL                               │
│                                                                 │
│   LEVER 1 — REWARD GRANULARITY                                  │
│     Outcome reward (ORM):  score the FINAL answer only          │
│     Process reward (PRM):  score EACH STEP of the reasoning     │
│                                                                 │
│   LEVER 2 — TEST-TIME COMPUTE                                   │
│     Spend more compute at INFERENCE, not training:              │
│       sample N → verify → rerank → vote → search                │
│                                                                 │
│   Training raises the CEILING.                                  │
│   Test-time compute EXTRACTS it.                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why A6 Exists

A4's RLVR gives one 0/1 signal for a whole reasoning trace. That works — but it leaves two things on the table.

```
PROBLEM 1 — SPARSE CREDIT
  A 20-step chain gets ONE reward at the end.
  Which step was wrong? The model can't tell.
  → A PROCESS reward model scores each step → denser signal.

PROBLEM 2 — FROZEN MODELS CAN STILL DO BETTER
  You already have a strong model. You can't retrain it.
  But you CAN let it think longer, sample more, and pick the best.
  → TEST-TIME COMPUTE turns inference budget into accuracy.
```

Both are the same insight from two directions: **the reward signal and the inference budget are design choices, not fixed constraints.**

---

## Lever 1 — Outcome vs Process Rewards

```
OUTCOME REWARD MODEL (ORM)          PROCESS REWARD MODEL (PRM)
──────────────────────────          ──────────────────────────
score(prompt, final_answer)         score(prompt, step_1..step_k)

  "Is the answer 42?"                 "Is step 3 valid?"

  ✓ cheap to collect (verifier)       ✓ dense, per-step signal
  ✓ exact when verifiable             ✓ catches errors early
  ✗ sparse — one bit per trace        ✗ expensive to label
  ✗ can't say WHERE it went wrong     ✗ needs step-level data
```

```
WHY PROCESS REWARDS HELP:

  trace:  step1 ✓ → step2 ✓ → step3 ✗ → step4 → answer ✗
  ORM:    reward = 0        (no idea which step failed)
  PRM:    [1.0, 1.0, 0.1, 0.2]   (step 3 is the culprit)

  → PRM-guided search can PRUNE at step 3 instead of finishing a doomed trace.
  → PRM-guided training gives credit to the steps that actually mattered.
```

### Where the Process Labels Come From

```
1. HUMAN step annotation      accurate, slow, expensive
2. AUTOMATIC step checks      math: does each line follow? code: does it run?
3. MONTE-CARLO rollouts       label a step by how often it leads to success
4. AI judge (RLAIF)           a strong model scores each step
```

---

## Lever 2 — Test-Time Compute

```
SAME FROZEN MODEL, MORE COMPUTE AT INFERENCE:

  greedy           1 sample, take it                    (cheapest)
  best-of-N        sample N, keep the verified best     (needs a verifier)
  self-consistency sample N, majority-vote the answer   (no verifier needed)
  beam / tree search  explore branches, score partials  (needs a PRM)
  reranking        generate N, a judge picks the best   (needs a judge)
```

```
THE SCALING PICTURE:

  accuracy
     │                    ┌──────────  search + PRM
     │                 ┌──┘
     │              ┌──┘  ┌──────────  best-of-N + verifier
     │           ┌──┘  ┌──┘
     │        ┌──┘  ┌──┘  ┌────────────  self-consistency
     │     ┌──┘  ┌──┘  ┌──┘
     │  ┌──┘  ┌──┘  ┌──┘
     │──┘  ┌──┘  ┌──┘
     └────────────────────────────────────► inference compute
        greedy   N=4    N=16   N=64

  More compute → more accuracy, with DIMINISHING returns.
  A verifier or PRM makes the curve steeper (you can tell good from bad).
```

**The key result:** verifier-based RL (A4) scales *better* with a larger test-time budget than training-free methods. Training and test-time compute are **complementary, not substitutes** — training raises the ceiling, test-time compute reaches for it.

---

## Python Example

A toy reasoning task, four inference strategies, one comparison. No framework — just NumPy.

```python
import numpy as np

# ============================================================
# SIMPLIFIED illustration of A6.
# A "model" produces reasoning traces. We compare four ways to
# spend inference compute on the SAME frozen model:
#   1. greedy            — one sample
#   2. best-of-N         — sample N, keep the VERIFIED best
#   3. self-consistency  — sample N, majority-vote the answer
#   4. process-guided    — sample N, keep the best PROCESS score
# ============================================================


class ToyReasoner:
    """Stand-in for an LLM: samples reasoning traces for a problem.

    A trace = (steps, final_answer, step_scores).
    The model is correct with probability p_correct per sample.
    """

    def __init__(self, n_answers: int = 5, p_correct: float = 0.35, seed: int = 0):
        self.n_answers = n_answers
        self.p_correct = p_correct
        self.rng = np.random.default_rng(seed)

    def sample(self, correct: int) -> tuple:
        n_steps = int(self.rng.integers(2, 5))
        steps = self.rng.integers(0, 10, size=n_steps)

        if self.rng.random() < self.p_correct:
            final = correct
            # a CORRECT trace has consistently good steps
            step_scores = self.rng.uniform(0.7, 1.0, size=n_steps)
        else:
            final = int(self.rng.integers(0, self.n_answers))
            # an INCORRECT trace has one bad step — the PRM's whole point
            step_scores = self.rng.uniform(0.0, 1.0, size=n_steps)
            err = int(self.rng.integers(0, n_steps))
            step_scores[err] *= 0.2

        return steps, final, step_scores


def outcome_verifier(final: int, correct: int) -> float:
    """RLVR-style check: is the final answer right? 0 or 1."""
    return 1.0 if final == correct else 0.0


def process_reward(step_scores: np.ndarray) -> float:
    """PRM-style score: average step quality (denser than outcome)."""
    return float(np.mean(step_scores))


# --- Inference strategies (all use the SAME frozen model) ---

def greedy(model: ToyReasoner, correct: int) -> int:
    _, final, _ = model.sample(correct)
    return final


def best_of_n(model: ToyReasoner, correct: int, n: int) -> int:
    best_final, best_r = None, -1.0
    for _ in range(n):
        _, final, _ = model.sample(correct)
        r = outcome_verifier(final, correct)
        if r > best_r:
            best_r, best_final = r, final
    return best_final


def self_consistency(model: ToyReasoner, correct: int, n: int) -> int:
    finals = [model.sample(correct)[1] for _ in range(n)]
    counts = {}
    for f in finals:
        counts[f] = counts.get(f, 0) + 1
    return max(counts, key=counts.get)


def process_guided(model: ToyReasoner, correct: int, n: int) -> int:
    best_final, best_prm = None, -1.0
    for _ in range(n):
        _, final, step_scores = model.sample(correct)
        prm = process_reward(step_scores)
        if prm > best_prm:
            best_prm, best_final = prm, final
    return best_final


# --- Evaluation ---

def evaluate(strategy: str, model: ToyReasoner, n_problems: int = 2000,
             n_samples: int = 1, seed: int = 1) -> float:
    rng = np.random.default_rng(seed)
    correct_count = 0
    for _ in range(n_problems):
        correct = int(rng.integers(0, model.n_answers))
        if strategy == "greedy":
            pred = greedy(model, correct)
        elif strategy == "best_of_n":
            pred = best_of_n(model, correct, n_samples)
        elif strategy == "self_consistency":
            pred = self_consistency(model, correct, n_samples)
        elif strategy == "process_guided":
            pred = process_guided(model, correct, n_samples)
        else:
            raise ValueError(strategy)
        if pred == correct:
            correct_count += 1
    return correct_count / n_problems


# --- Run the comparison ---
model = ToyReasoner(n_answers=5, p_correct=0.35, seed=0)

print("=== ACCURACY vs INFERENCE COMPUTE (same frozen model) ===")
print(f"{'strategy':<20}{'N':>4}{'accuracy':>12}{'samples/problem':>18}")
print("-" * 54)

rows = [
    ("greedy", 1),
    ("self_consistency", 4),
    ("self_consistency", 16),
    ("best_of_n", 4),
    ("best_of_n", 16),
    ("process_guided", 4),
    ("process_guided", 16),
]

for strategy, n in rows:
    acc = evaluate(strategy, model, n_samples=n)
    print(f"{strategy:<20}{n:>4}{acc:>12.3f}{n:>18}")

# Expected shape (approximate — exact numbers vary by seed):
#   greedy            ~0.48   the floor
#   self_consistency  rises with N, no verifier needed
#   best_of_n         rises faster — it can RECOGNIZE a correct answer
#   process_guided    rises fastest — it can also PRUNE bad traces
#
# The lesson: the SAME model gets better purely by spending more
# compute at inference — and a verifier/PRM makes that compute count.
```

### Exercise: The Diminishing-Returns Curve

```python
# Sweep N and watch the curve flatten.
# In evaluate(), call best_of_n with n_samples in [1, 2, 4, 8, 16, 32, 64].
# Plot accuracy vs N (or just print it).
#
# You should see: steep gains early, then diminishing returns.
# That shape is the whole argument for test-time compute —
# and the reason "how much should I sample?" is a budget question,
# not a correctness question.
```

---

## When to Use A6

| Use Case | Why |
|----------|-----|
| You have a verifier but no GPUs | Best-of-N + verifier is pure inference |
| You can't retrain the model | Test-time compute works on a frozen model |
| Reasoning quality matters more than latency | Spend inference budget, not training budget |
| You need to catch *where* reasoning fails | Process rewards localize the error |
| You want to squeeze a strong model further | Search + reranking extract the ceiling |

---

## Key Concepts Visualized

### ORM vs PRM

```
OUTCOME (ORM)                        PROCESS (PRM)
┌──────────────────────────┐         ┌──────────────────────────┐
│ trace ──────────► answer │         │ step1 ─► 1.0            │
│                      │   │         │ step2 ─► 1.0            │
│                      ▼   │         │ step3 ─► 0.1  ← the bug  │
│                   reward │         │ step4 ─► 0.2            │
│                   0 or 1 │         │                          │
└──────────────────────────┘         └──────────────────────────┘
  one bit per trace                    a score per step
```

### The Test-Time Compute Ladder

```
  CHEAPEST ─────────────────────────────────────────► MOST EXPENSIVE

  greedy      best-of-N      self-consistency     search + PRM
  1 sample    N + verifier   N + majority vote    tree/beam + step scores
     │             │                │                    │
  no signal    needs verifier   no verifier         needs a PRM
```

### Verifier Types

```
Task        Verifier                              Signal
──────────────────────────────────────────────────────────
Math        compare final answer to ground truth  0 / 1
Code        run the unit tests                    pass / fail
Logic       check constraints                    0 / 1
Format      regex / JSON-schema                  0 / 1
Open-ended  AI judge (RLAIF)                      score
Reasoning   process reward model (PRM)            per-step score
```

### Where A6 Sits

```
  TRAINING (A1–A5)                 INFERENCE (A6)
  ────────────────                 ─────────────
  raises the CEILING               reaches for the ceiling
  weights, GPU-hours               samples, CPU/GPU-seconds
  one-time cost                    per-request cost
        │                                │
        └──────────► complementary ◄─────┘
```

---

## Key Limitations

```
1. TEST-TIME COMPUTE COSTS MONEY PER REQUEST
   Best-of-64 is 64× the inference cost. It is a budget decision,
   not a free win — and it does not help if the model can't
   produce a correct answer at all.

2. DIMINISHING RETURNS
   Accuracy rises with N, then flattens. Past a point you are
   paying for noise. Tune N to the task, not to a round number.

3. VERIFIER QUALITY IS THE CEILING
   Best-of-N can only pick a correct answer if the verifier can
   RECOGNIZE one. A weak verifier caps the whole strategy.

4. PRMs ARE EXPENSIVE TO BUILD
   Step-level labels are costly (human or rollout-based), and a
   bad PRM will confidently prefer wrong traces.

5. REWARD HACKING MOVES TO INFERENCE
   Optimize against a fixed verifier and the model learns to
   satisfy the checker, not the task. Held-out verifiers still apply.

6. LATENCY
   Search and reranking add wall-clock time. For interactive
   products, the budget may not allow it — measure before adopting.
```

---

## Connecting to Real RL Methods Guide

From the [RL Methods Guide](../../RL_METHODS_GUIDE.md):

> **Progression Path**
> ... YEAR 2: DQN + PPO (optional) ... FINE-TUNE: DPO (Level 7) ... NEVER: RLHF / GRPO + RLVR

A6 is the layer the master guide's progression path points at but does not name: the inference-time counterpart to training. The guide's core split still holds — **Track A optimizes the model, Track B optimizes the action** — and A6 is the cheapest way to improve a Track A model you did not train yourself.

---

## Real-Life Applications

| System | Method | Result |
|--------|--------|--------|
| OpenAI o1 / o3 | RL on reasoning + test-time compute | Accuracy scales with thinking budget |
| DeepSeek-R1 | GRPO + RLVR, then best-of-N at inference | Reasoning extracted at serve time |
| Math solvers | Best-of-N + verifier | Higher pass@1 with no retraining |
| Code assistants | Sample N, run tests, keep the passing one | Fewer broken suggestions |
| Open-ended writing | Reranking with an AI judge | Better drafts from a frozen model |

---

## Where You Fit In

```
YOUR ROLE AS AN APP DEVELOPER:
  ❌ Don't train a reasoning model yourself
  ✅ USE reasoning models via API — and spend inference budget wisely
  ✅ If you have a verifier, best-of-N is the cheapest quality win there is
  ✅ Your personalization still lives in Track B (LinUCB + Q-Learning)

YOUR PRACTICAL STACK:
  Base model:      a reasoning model via API
  Quality lever:   best-of-N + verifier (A6) when latency allows
  Personalization: LinUCB (B2) + Q-Learning (B3)   ← still the sweet spot
```

---

## Summary: The Complete RL Journey

```
Level 0: Random         → No learning, establish baseline
Level 1a: Classic MAB   → Learn best GLOBAL default
Level 1b: LinUCB        → Personalize with context
Level 2: Q-Learning     → Handle sequential decisions
Level 3: Feature Q      → Scale to large state spaces
Level 4: DQN            → Raw inputs, non-linear patterns
Level 5: PPO            → Direct policy optimization (has critic)
Level 6: RLHF           → Reward model + PPO (classic alignment)
Level 7: DPO            → Direct preference optimization (no RM, offline)
Level 8: GRPO + RLVR    → Group-relative, verifiable rewards (reasoning)
         A5: Agentic RL → Multi-turn action inside a packaged environment
         A6: Rewards & Test-Time Compute → process rewards + inference search

For most chatbot applications: Levels 1b–3 are still the sweet spot.
For alignment/fine-tuning: DPO (Level 7) is the pragmatic default.
For reasoning with verifiers: GRPO + RLVR (A4).
For agents that act: A5 — the environment, not the model, is the work.
For squeezing a frozen model: A6 — spend inference compute, not training compute.
```

---

## Further Reading

- **Let's Verify Step by Step** (Lightman et al., 2023) — process reward models
- **Self-Consistency** (Wang et al., 2022) — majority voting over samples
- **Tree of Thoughts** (Yao et al., 2023) — search over reasoning steps
- **Scaling test-time compute** (2024–2025) — inference budget vs accuracy
- **Generative verifiers** (2024–2025) — verifiers that reason about correctness

---

**Previous:** [A5 — Agentic RL & Environments](a5-agentic-rl-environments.md)

**Up:** [Track A Index](index.md) · **Reference:** [RL Methods Guide](../../RL_METHODS_GUIDE.md)