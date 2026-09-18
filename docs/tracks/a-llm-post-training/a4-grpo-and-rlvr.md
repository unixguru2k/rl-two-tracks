# A4 — GRPO & RLVR (Reasoning Models)

> **Track A · Guide 5 of 6** — [Track A Index](index.md) · [RL Methods Guide](../../RL_METHODS_GUIDE.md)

## What You'll Learn

GRPO (Group Relative Policy Optimization) is the algorithm behind **DeepSeek-R1**, and RLVR (Reinforcement Learning with **Verifiable** Rewards) is the paradigm behind **reasoning models** (o1/o3, DeepSeek-R1, QwQ). Together they showed that RL can teach a model to *think* — using **automatic, checkable rewards** instead of human preferences, and **without a value network**.

> **TL;DR:** GRPO = PPO minus the critic (use the *group mean* as the baseline). RLVR = reward from a *verifier* (is the math answer right? do the tests pass?) instead of a human or reward model.

---

## The Concept

```
┌─────────────────────────────────────────────────────────────────┐
│                 GRPO + RLVR                                     │
│                                                                 │
│   For each prompt, sample a GROUP of G outputs from the policy: │
│                                                                 │
│   prompt ──► [ output 1, output 2, ..., output G ]              │
│                      │        │             │                   │
│                 verifier  verifier     verifier                 │
│                      ▼        ▼             ▼                   │
│                   r=1      r=0           r=1                    │
│                                                                 │
│   GRPO advantage: A_i = (r_i - mean(r)) / std(r)   ← GROUP RELATIVE
│                                                                 │
│   No value network! The group's average IS the baseline.        │
│                                                                 │
│   Update policy to make high-advantage outputs more likely.     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why GRPO Exists

```
PPO (Level 5) needs a VALUE NETWORK (critic) to estimate baselines:
  - Doubles memory (actor + critic + reward model + reference)
  - The critic is hard to train and often the source of instability
  - For LLMs, a good critic is nearly as big as the policy itself

GRPO'S INSIGHT (DeepSeekMath, 2024):
  For a given prompt, generate G samples instead of 1.
  Use the GROUP's mean reward as the baseline → no critic needed!

  A_i = (r_i - mean(r_1..r_G)) / std(r_1..r_G)

  Advantages:
    ✓ No value network  → ~half the memory
    ✓ No critic training → simpler, more stable
    ✓ Group normalization → automatically scales advantages
    ✓ Trivially parallelizable (G rollouts at once)
```

---

## Why RLVR Exists (The Reasoning-Model Breakthrough)

```
RLHF / DPO need HUMANS or a REWARD MODEL for every judgment.
That's expensive, slow, and impossible to scale to millions of problems.

RLVR: use a PROGRAMMATIC VERIFIER as the reward.
  Math problem    → check if final answer matches the ground truth
  Code generation → run the unit tests
  Logic puzzle    → validate constraints
  Formatting      → regex / schema check

  Reward is 0 or 1 (or a rule-based score). No humans. No reward model.

WHY IT MATTERS:
  → Infinite, cheap, exact reward signal
  → Lets you RL-train on millions of verifiable problems
  → Teaches models to produce LONGER chains of thought that self-correct
  → This is what made DeepSeek-R1 / o1-style reasoning emerge
```

### The "Aha" Moment

```
A model trained with RLVR on math learns to:
  - write out intermediate steps
  - double-check its work
  - backtrack and try again
  - use more tokens of "thinking" when the problem is hard

This reasoning behavior EMERGED from RL — it was never hand-coded.
```

---

## The GRPO Objective

```
For each prompt, sample group {o_1, ..., o_G} from the OLD policy.
Compute rewards {r_1, ..., r_G} with the verifier.
Normalize to advantages A_i (group-relative).

Then apply a PPO-style CLIPPED objective (like Level 5), but with A_i:

  L_GRPO = E[ (1/G) Σ_i  min( ρ_i · A_i ,  clip(ρ_i, 1-ε, 1+ε) · A_i ) ]
           - β · KL( π_θ ‖ π_ref )

  where  ρ_i = π_θ(o_i | q) / π_θ_old(o_i | q)     (importance ratio)
         ε   = clip range (~0.2)
         β   = KL penalty to the reference model

Same clipping trick as PPO — but the baseline comes from the GROUP,
not from a learned value function.
```

---

## Python Example

```python
import numpy as np

# ============================================================
# SIMPLIFIED illustration of GRPO + RLVR.
# Real GRPO runs on transformers over long reasoning traces.
# Here a tiny policy answers "which option is correct?" and the
# verifier checks the answer — showing the MECHANICS:
#   group sampling → verifiable reward → group-relative advantage.
# ============================================================


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / np.sum(e)


class GRPOPolicy:
    """Policy over K candidate answers, parameterized by logits."""

    def __init__(self, n_answers: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.logits = rng.normal(0.0, 0.01, n_answers)

    def probs(self) -> np.ndarray:
        return softmax(self.logits)


def verifier(sample: int, correct: int) -> float:
    """RLVR reward: a programmatic check, not a human or reward model."""
    return 1.0 if sample == correct else 0.0


def grpo_step(
    policy: GRPOPolicy,
    correct: int,
    group_size: int = 8,
    lr: float = 0.5,
    clip_eps: float = 0.2,
    kl_coef: float = 0.02,
    ref_logits: np.ndarray = None,
) -> tuple:
    """One GRPO update for a single prompt with a verifiable answer."""
    probs_old = policy.probs()

    # 1) Sample a GROUP of outputs from the current policy
    samples = np.random.choice(len(probs_old), size=group_size, p=probs_old)

    # 2) Score each with the VERIFIER
    rewards = np.array([verifier(int(s), correct) for s in samples])

    # 3) Group-relative advantages (this is what replaces the value network)
    if rewards.std() > 1e-8:
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    else:
        advantages = rewards - rewards.mean()   # all same → no signal

    # 4) Policy-gradient step (REINFORCE with a group baseline)
    grad = np.zeros_like(policy.logits)
    for s, a in zip(samples, advantages):
        one_hot = np.zeros_like(probs_old)
        one_hot[s] = 1.0
        grad += a * (one_hot - probs_old)       # ∇ log π(s)
    grad /= group_size

    # 5) KL penalty to the reference model (keeps the policy on the rails)
    if ref_logits is not None:
        ref = softmax(ref_logits)
        grad -= kl_coef * (probs_old - ref)     # ∇ KL(π ‖ ref)

    # NOTE: real GRPO also clips the importance ratio ρ (PPO-style).
    # We omit clipping here for clarity; the group baseline is the key idea.
    policy.logits += lr * grad

    return rewards.mean(), rewards.std()


def train_grpo(n_prompts: int = 2000, n_options: int = 5, group_size: int = 8, seed: int = 7):
    """Train with GRPO + RLVR on verifiable prompts."""
    rng = np.random.default_rng(seed)

    policy = GRPOPolicy(n_options, seed=3)
    ref_logits = policy.logits.copy()          # frozen reference

    window = []
    for step in range(n_prompts):
        # Each prompt has a verifiable correct answer
        correct = int(rng.integers(n_options))
        mean_r, std_r = grpo_step(
            policy, correct, group_size=group_size, ref_logits=ref_logits
        )
        window.append(mean_r)

        if (step + 1) % 400 == 0:
            print(f"Prompt {step + 1}: avg verifier reward (last 400) = "
                  f"{np.mean(window[-400:]):.3f}")

    return policy, window


# Run training
print("Training GRPO policy on verifiable prompts...")
policy, reward_history = train_grpo()

# --- Inspect what GRPO learned ---
print("\n=== LEARNED POLICY ===")
probs = policy.probs()
for i, p in enumerate(probs):
    bar = "█" * int(p * 50)
    print(f"  answer_{i}  {p:.3f} {bar}")

print(f"\n  Max probability: {probs.max():.3f}")
print(f"  Reward trend: {np.mean(reward_history[:200]):.3f} "
      f"→ {np.mean(reward_history[-200:]):.3f}")

# Because the correct answer is RANDOM per prompt, the best a policy can do
# is spread probability roughly uniformly. The LEARNING SIGNAL is what matters:
# verifier reward climbs and stays stable with NO value network.
#
# To see sharp convergence, make the correct answer FIXED (see the exercise
# at the bottom of this guide).
```

### Exercise: Watch It Converge

```python
# Change the correct answer to be the SAME for every prompt.
# The policy should concentrate almost all probability on it.
#
# In train_grpo(), replace:
#     correct = int(rng.integers(n_options))
# with:
#     correct = 0
#
# Re-run: answer_0 should climb toward ~1.0, the others toward ~0.0.
# This is the same "learn from verifiable reward" loop that trains
# reasoning models — just without the long chain-of-thought.
```

---

## When to Use GRPO / RLVR

| Use Case | Why |
|----------|-----|
| Reasoning / math / code | Verifiable rewards, no human labels needed |
| Long chain-of-thought | Group sampling rewards intermediate reasoning |
| Limited memory | No value network (unlike PPO) |
| Massive scale | Cheap, exact, programmatic reward |
| RLHF-style alignment | GRPO also works with a reward model (GRPO-RM) |

---

## Key Concepts Visualized

### PPO vs GRPO

```
PPO (Level 5):                       GRPO (Level 8):
┌──────────────┐                     ┌──────────────┐
│ 1 rollout    │                     │ G rollouts   │
└──────┬───────┘                     └──────┬───────┘
       ▼                                    ▼
┌──────────────┐                     ┌──────────────┐
│ value net    │  (baseline)         │ group mean   │  (baseline!)
│ (critic)     │                     │ = free       │
└──────┬───────┘                     └──────┬───────┘
       ▼                                    ▼
┌──────────────┐                     ┌──────────────┐
│ clipped PG   │                     │ clipped PG   │
└──────────────┘                     └──────────────┘

  Memory:  actor + critic              Memory:  actor only
  Cost:    train critic                Cost:    G forward passes
```

### Group-Relative Advantage

```
Group rewards:  [1, 0, 1, 0, 1, 0, 1, 1]
                       │
                       ▼
mean = 0.625, std = 0.5

A_i = (r_i - 0.625) / 0.5

  r=1 → A = +0.75   (better than the group → reinforce)
  r=0 → A = -1.25   (worse than the group  → discourage)

  The group's own average is the baseline. No value network required.
```

### RLVR: The Verifier Is the Reward

```
Task        Verifier                          Reward
──────────────────────────────────────────────────────
Math        compare final answer to GT        0 or 1
Code        run unit tests                    pass / fail
Logic       check constraints satisfied       0 or 1
Format      regex / JSON-schema match         0 or 1
Tool use    did the API call succeed?         shaped
```

---

## The Modern Alignment Landscape

```
2022–2023                     2024–2026
────────────────────────────  ────────────────────────────
RLHF: RM + PPO  ──────────────►  DPO (offline, no RM)          [Level 7]
                                GRPO + RLVR (reasoning)        [Level 8]
                                RLAIF / Constitutional AI
                                Agentic RL (tool use)

RULE OF THUMB:
  Whole-response preference alignment, offline? ──► DPO (Level 7)
  Verifiable task, want reasoning, no critic?  ──► GRPO + RLVR (Level 8)
  Classic, online, have a good RM?             ──► PPO/RLHF (Level 6)
```

### RLAIF (RL from AI Feedback)

```
Replace human annotators with a strong AI judge:

  Human label:  "A is better than B"   (slow, expensive)
  AI label:     "A is better than B"   (LLM judge, cheap, scalable)

Used in Constitutional AI (Anthropic). Feeds DPO or GRPO just like
human preferences would — this is how preference data scales.
```

---

## GRPO vs PPO vs DPO

| Aspect | PPO / RLHF | DPO | GRPO |
|--------|-----------|-----|------|
| Value network | ✅ Yes (critic) | ❌ No | ❌ No |
| Reward model | ✅ Yes | ❌ No (implicit) | Optional (or verifier) |
| Sampling | ✅ Rollouts | ❌ Offline pairs | ✅ Group rollouts |
| Explores new outputs | ✅ | ❌ | ✅ |
| Process-level credit | Weak | ❌ | ✅ (per-step rewards) |
| Memory | Highest | Lowest | Medium |
| Best for | Classic alignment | Offline preference data | Reasoning / RLVR |

---

## Key Limitations

```
1. NEEDS A VERIFIER (for RLVR)
   Works only where correctness is checkable.
   Open-ended creativity / empathy → no automatic verifier → back to
   human preferences or a reward model.

2. REWARD HACKING
   Policy may satisfy the verifier without "truly" solving the task
   (e.g., matching the answer format). Mitigations: held-out verifiers,
   KL penalty, multiple reward checks.

3. GROUP SIZE COST
   G rollouts per prompt → G× the generation cost.
   Small G → noisy advantages; large G → expensive.

4. LONGER OUTPUTS
   Reasoning RL tends to lengthen outputs (more "thinking tokens"),
   raising inference cost. Needs length control / budget tuning.

5. STILL EXPENSIVE TO TRAIN
   Far cheaper than full RLHF, but still GPU-heavy at real model scale.
   → Most teams still USE reasoning models (R1, o-series) via API
     rather than training them.
```

---

## Connecting to Real RL Methods Guide

From the [RL Methods Guide](../../RL_METHODS_GUIDE.md#level-8-grpo--rlvr-reasoning-models):

> **What:** PPO without a value network (group-relative advantages), trained on verifiable rewards.
> **When:** Reasoning/math/code tasks with automatic correctness checks.
> **Key advantage:** No critic, cheap exact rewards, enables emergent reasoning.
> **Key limitation:** Needs a verifier; expensive at scale; can reward-hack.

---

## Real-Life Applications

| System | Method | Result |
|--------|--------|--------|
| DeepSeek-R1 | GRPO + RLVR | Emergent long chain-of-thought reasoning |
| OpenAI o1 / o3 | RL on reasoning (undisclosed) | Test-time-compute scaling |
| Qwen / QwQ | GRPO variants (e.g., GSPO) | Open reasoning models |
| Code assistants | RLVR on unit tests | Better code generation |
| Math tutors | RLVR on answer checking | Step-by-step solutions |

---

## Where You Fit In

```
YOUR ROLE AS AN APP DEVELOPER:
  ❌ Don't train a reasoning model yourself
  ✅ USE reasoning models (DeepSeek-R1, o-series) via API
  ✅ If you fine-tune: DPO (Level 7) is usually enough
  ✅ Reserve GRPO + RLVR for teams with a verifiable task AND GPUs

YOUR PRACTICAL STACK (unchanged for chatbot personalization):
  Base model:  GPT / Claude / R1 via API  (already aligned)
  Personalization: LinUCB (1b) + Q-Learning (2)   ← still the sweet spot
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

For most chatbot applications: Levels 1b–3 are still the sweet spot.
For alignment/fine-tuning: DPO (Level 7) is the pragmatic default.
For reasoning tasks with verifiers: GRPO + RLVR (Level 8).
```

---

## Further Reading

- **DeepSeekMath** (2024) — introduced GRPO
- **DeepSeek-R1** (2025) — GRPO + RLVR at reasoning scale
- **DPO** — Rafailov et al., 2023 (see Level 7)
- **GSPO** — Qwen's sequence-level variant of GRPO
- **RLVR surveys** (2025) — reinforcement learning with verifiable rewards

---

**See Also**: [RL Methods Guide - Level 8](../../RL_METHODS_GUIDE.md#level-8-grpo--rlvr-reasoning-models)
