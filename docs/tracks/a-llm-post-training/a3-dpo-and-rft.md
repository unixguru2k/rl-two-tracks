# A3 — DPO & RFT (Direct Preference Optimization & Rejection Fine-Tuning)

> **Track A · Guide 4 of 7** — [Track A Index](index.md) · [RL Methods Guide](../../RL_METHODS_GUIDE.md)

## What You'll Learn

DPO is the method that **replaced RLHF's reward-model + PPO pipeline** for most open-source alignment. It learns directly from human preference pairs — **no reward model, no RL loop, no sampling during training**. Same objective, dramatically simpler.

> **TL;DR:** If RLHF is "train a reward model, then run PPO against it," DPO is "skip both steps and optimize the policy on preference pairs directly."

---

## The Concept

```
┌─────────────────────────────────────────────────────────────────┐
│              DPO (DIRECT PREFERENCE OPTIMIZATION)               │
│                                                                 │
│   Input: preference pairs from humans (or an AI judge)          │
│     "Response W is better than Response L for this prompt"      │
│                                                                 │
│   RLHF:  preferences → reward model → PPO → policy  (3 stages)  │
│   DPO:   preferences ──────────────────────► policy  (1 stage)  │
│                                                                 │
│   Key math trick: the optimal RLHF policy has a CLOSED FORM     │
│   → we can write the reward in terms of the policy itself       │
│   → the reward model cancels out, leaving a simple loss         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why DPO Exists

```
RLHF (Level 6) PIPELINE:
  1. Collect preferences        (expensive humans)
  2. Train reward model         (separate model, can be gamed)
  3. Run PPO against it         (needs sampling, a value net, careful tuning)

  Problems:
    ✗ Reward model is a lossy proxy for human intent
    ✗ PPO is unstable and needs a value network + reward model in memory
    ✗ Reward hacking: policy exploits reward model blind spots
    ✗ 3 separate moving parts to debug

DPO INSIGHT (Rafailov et al., 2023):
  The KL-constrained RLHF objective has an exact optimal solution:

      π*(y|x) ∝ π_ref(y|x) · exp( r(x,y) / β )

  Rearranged, this gives the "implicit reward":

      r(x,y) = β · log( π*(y|x) / π_ref(y|x) )

  So the reward is IMPLICIT in the policy itself.
  We never need to train a reward model — just optimize the policy
  so that preferred responses get higher implicit reward.
```

---

## The DPO Loss

```
Given a preference pair (prompt x, preferred y_w, rejected y_l):

  L_DPO = - log σ( β · [ log π_θ(y_w|x)/π_ref(y_w|x)
                       - log π_θ(y_l|x)/π_ref(y_l|x) ] )

Where:
  π_θ    = policy being trained
  π_ref  = frozen reference policy (the SFT model)
  β      = temperature (how far you allow the policy to drift; ~0.1)
  σ      = sigmoid

Plain English:
  "Increase the log-prob of the winner, decrease the log-prob of the
   loser — but measure both RELATIVE to the frozen reference model."
```

### Why the Reference Model Matters

```
WITHOUT a reference model:
  The policy could collapse to a degenerate distribution
  ("always output the single winning string, ignore the prompt").

WITH a reference model (the log-ratio):
  A KL-style leash. The policy is rewarded for MOVING AWAY from the
  reference toward preferred outputs, not for being confident in general.

  β small (0.01)  → loose leash, policy drifts far, can collapse
  β large (0.5)   → tight leash, stays close to reference, safe but weak
  β ≈ 0.1         → the common default
```

---

## Python Example

```python
import numpy as np

# ============================================================
# SIMPLIFIED illustration of the DPO objective.
# Real DPO runs on transformer LMs over full sequences.
# Here we use a tiny categorical policy to show the MECHANICS:
# no reward model, no RL loop — just preference pairs.
# ============================================================


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax."""
    z = logits - np.max(logits)
    return z - np.log(np.sum(np.exp(z)))


class CategoricalPolicy:
    """Policy over K discrete responses, parameterized by logits."""

    def __init__(self, n_responses: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.logits = rng.normal(0.0, 0.01, n_responses)

    def log_probs(self) -> np.ndarray:
        return log_softmax(self.logits)

    def probs(self) -> np.ndarray:
        return np.exp(self.log_probs())

    def copy(self) -> "CategoricalPolicy":
        """Freeze a snapshot (used as the reference model)."""
        clone = CategoricalPolicy(len(self.logits))
        clone.logits = self.logits.copy()
        return clone


def dpo_loss_and_grad(
    policy: CategoricalPolicy,
    ref: CategoricalPolicy,
    winner: int,
    loser: int,
    beta: float = 0.1,
) -> tuple:
    """Compute DPO loss and its gradient w.r.t. the policy logits."""
    lp = policy.log_probs()
    lr = ref.log_probs()

    # Log-ratios relative to the frozen reference
    delta_w = lp[winner] - lr[winner]
    delta_l = lp[loser] - lr[loser]

    z = beta * (delta_w - delta_l)

    # L = -log sigmoid(z) = log(1 + e^{-z})   (numerically stable form)
    loss = np.log1p(np.exp(-z))

    # dL/dz = sigmoid(z) - 1
    dloss_dz = (1.0 / (1.0 + np.exp(-z))) - 1.0

    # dL/dlogits = dL/dz * beta * (e_winner - e_loser)
    grad = np.zeros_like(policy.logits)
    grad[winner] += dloss_dz * beta
    grad[loser] -= dloss_dz * beta

    return loss, grad


# --- Synthetic preference data ---
# 5 candidate response styles; humans prefer "empathetic" then "validate".
RESPONSE_NAMES = ["playful", "empathetic", "distract", "validate", "question"]

# True human preference strength (higher = more preferred)
TRUE_PREFERENCE = np.array([0.2, 0.9, 0.3, 0.7, 0.4])


def generate_preference_pair(rng: np.random.Generator) -> tuple:
    """Simulate a human (or AI) judge comparing two responses."""
    a, b = rng.choice(len(RESPONSE_NAMES), size=2, replace=False)

    # Bradley-Terry: P(a preferred over b) ∝ sigmoid(pref_a - pref_b)
    p_a_wins = 1.0 / (1.0 + np.exp(-(TRUE_PREFERENCE[a] - TRUE_PREFERENCE[b])))

    if rng.random() < p_a_wins:
        return int(a), int(b)
    return int(b), int(a)


def train_dpo(n_pairs: int = 2000, beta: float = 0.1, lr: float = 0.5, seed: int = 42):
    """Train a policy directly from preference pairs using DPO."""
    rng = np.random.default_rng(seed)

    policy = CategoricalPolicy(len(RESPONSE_NAMES), seed=1)

    # The reference model is a FROZEN copy of the starting policy (the "SFT" model)
    ref = policy.copy()

    losses = []

    for step in range(n_pairs):
        winner, loser = generate_preference_pair(rng)
        loss, grad = dpo_loss_and_grad(policy, ref, winner, loser, beta)

        policy.logits -= lr * grad          # gradient descent
        losses.append(loss)

        if (step + 1) % 500 == 0:
            avg = np.mean(losses[-500:])
            print(f"Step {step + 1}: avg DPO loss = {avg:.4f}")

    return policy, ref, losses


# Run training
policy, ref, losses = train_dpo()

# --- Inspect what DPO learned ---
print("\n=== REFERENCE POLICY (before DPO) ===")
for name, p in zip(RESPONSE_NAMES, ref.probs()):
    print(f"  {name:<12} {p:.3f}")

print("\n=== DPO POLICY (after training) ===")
final = policy.probs()
for name, p in zip(RESPONSE_NAMES, final):
    bar = "█" * int(p * 50)
    print(f"  {name:<12} {p:.3f} {bar}")

print("\n=== IMPLICIT REWARD (β · log π_θ/π_ref) ===")
implicit = 0.1 * (policy.log_probs() - ref.log_probs())
for name, r in zip(RESPONSE_NAMES, implicit):
    print(f"  {name:<12} {r:+.3f}")

# Expected: probability mass shifts toward "empathetic" (0.9) and "validate" (0.7)
# The implicit reward recovers the human preference ordering WITHOUT any
# reward model being trained.
```

---

## When to Use DPO

| Use Case | Why |
|----------|-----|
| Aligning an SFT model | Simplest path from preferences → aligned model |
| Limited compute | No reward model, no sampling, no RL loop |
| Preference data you already have | Uses pairs directly |
| Iterating fast | One supervised-style training run |
| Open-source fine-tuning | The common default *starting point* (Mistral, Llama, Zephyr, Tulu) — though RLHF still wins on safety-critical, long-horizon, and out-of-distribution tasks |

---

## Key Concepts Visualized

### RLHF vs DPO

```
RLHF (Level 6):                    DPO (Level 7):
┌──────────────┐                   ┌──────────────┐
│ preferences  │                   │ preferences  │
└──────┬───────┘                   └──────┬───────┘
       ▼                                  │
┌──────────────┐                          │
│ reward model │  (extra training)        │
└──────┬───────┘                          │
       ▼                                  ▼
┌──────────────┐                   ┌──────────────┐
│  PPO loop    │  (sampling +      │  DPO loss    │  (one pass,
│  + value net │   value net)      │  on pairs    │   no sampling)
└──────┬───────┘                   └──────┬───────┘
       ▼                                  ▼
┌──────────────┐                   ┌──────────────┐
│   policy     │                   │   policy     │
└──────────────┘                   └──────────────┘
```

### The Implicit Reward

```
DPO never trains a reward model, yet it recovers one:

    r_θ(x, y) = β · log( π_θ(y|x) / π_ref(y|x) )

  → The policy IS the reward model, in disguise.
  → "How much more likely does my policy make this response
     than the reference model did?" = its learned reward.
```

### The β Knob (Leash Length)

```
β = 0.01   → very loose leash
             policy drifts far from reference
             strong signal, but risk of collapse / degenerate outputs

β = 0.10   → common default (good balance)

β = 0.50   → tight leash
             policy stays near reference
             safe, but limited alignment gain
```

---

## The DPO Family (Variants)

DPO opened the door to a whole family of "direct" preference objectives. Pick by your data shape.

| Method | Idea | Data Needed |
|--------|------|-------------|
| **DPO** | Log-ratio on preference pairs | Pairs (W vs L) |
| **IPO** | Identity-preference loss; avoids overfitting on pairs | Pairs |
| **KTO** | Uses *unpaired* thumbs-up / thumbs-down signals | Binary feedback |
| **ORPO** | Merges SFT + preference into one loss (no separate reference) | Pairs + SFT data |
| **SimPO** | Drops the reference model; uses average log-prob + target margin | Pairs |
| **cDPO** | DPO with label-noise tolerance | Noisy pairs |

```
CHOOSING A VARIANT:
  Have clean preference pairs? ───────────► DPO (start here)
  Only thumbs up/down (unpaired)? ────────► KTO
  Want SFT + alignment in one run? ───────► ORPO
  Want no reference model (cheaper)? ─────► SimPO
  Pairs are noisy / annotators disagree? ─► cDPO
```

---

## DPO vs RLHF

| Aspect | RLHF (PPO) | DPO |
|--------|-----------|-----|
| Reward model | ✅ Required (separate training) | ❌ Not needed (implicit) |
| Sampling during training | ✅ Yes (rollouts) | ❌ No |
| Value network | ✅ Yes | ❌ No |
| Training stages | 3 (SFT → RM → PPO) | 1 (on top of SFT) |
| Stability | Fragile, tuning-heavy | Supervised-style, stable |
| Compute | High | Much lower |
| Online exploration | ✅ Can discover new outputs | ❌ Learns only from given pairs |
| Reward hacking | Possible (via RM) | Reduced (no RM to hack) — but the implicit reward can overfit preferences |
| Best for | Large labs, online data | Most teams, offline preferences |

---

## Key Limitations

```
1. OFFLINE — NO EXPLORATION
   DPO only learns from the pairs it's given.
   It cannot discover responses better than anything in the dataset.
   → Ceiling = quality of your preference data.

2. NEEDS A GOOD REFERENCE MODEL
   π_ref must be a competent SFT model. Garbage reference → garbage DPO.

3. DISTRIBUTION SHIFT
   The policy drifts off the data distribution; pairs are only informative
   near where the reference model operates.

4. β AND LR SENSITIVITY
   Too aggressive → degenerate, repetitive outputs.
   Too conservative → barely moves.

5. NO PROCESS SIGNAL
   DPO scores whole responses. For long reasoning chains, it gives no
   credit to intermediate steps — this is exactly the gap GRPO fills.
```

---

## Connecting to Real RL Methods Guide

From the [RL Methods Guide](../../RL_METHODS_GUIDE.md#level-7-dpo-direct-preference-optimization):

> **What:** Optimize the policy directly on preference pairs using a closed-form reparameterization of the RLHF objective.
> **When:** You have preference data and want alignment without a reward model or RL loop.
> **Key advantage:** Simple, stable, cheap — one supervised-style pass.
> **Key limitation:** Offline; cannot explore beyond the data; no process-level credit.

---

## Real-Life Applications

| System | What DPO Did | Note |
|--------|--------------|------|
| Zephyr-7B | Aligned via DPO instead of PPO | Proved DPO at small scale |
| Llama / Mistral fine-tunes | Default alignment recipe | Widely used in open weights |
| Tulu 2/3 | DPO stage after SFT | Mixed preference data |
| Production chatbots | Cheap personalization/alignment pass | Lower ops cost than PPO |

---

## Where DPO Fits In Your Stack

```
YOU (app developer):
  ✅ Use DPO if you fine-tune your OWN small model on YOUR preference data
  ✅ Collect preference pairs (human or AI judge) — see RLAIF in the guide
  ❌ Still don't build a foundation model from scratch

TYPICAL OPEN-SOURCE PIPELINE:
  Base model → SFT → DPO (this level) → optional GRPO (Level 8) → deploy
```

---

## Next Step

DPO is strong for whole-response preferences but gives **no credit for intermediate reasoning steps**. For tasks with verifiable answers (math, code, logic), move to **Level 8: GRPO & RLVR** — the paradigm behind DeepSeek-R1 and reasoning models.

---

**Previous:** [A2 — RLHF](a2-rlhf.md) · **Next:** [A4 — GRPO & RLVR](a4-grpo-and-rlvr.md)

**Up:** [Track A Index](index.md) · **Reference:** [RL Methods Guide — Level 7](../../RL_METHODS_GUIDE.md#level-7-dpo-direct-preference-optimization)
