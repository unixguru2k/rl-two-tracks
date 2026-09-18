# Complete RL Methods Guide: When To Use Each One

## Table of Contents

- [Overview](#overview)
- [Level 0: Random Selection](#level-0-random-selection)
- [Level 1a: Classic Multi-Armed Bandit (Epsilon-Greedy)](#level-1a-classic-multi-armed-bandit-epsilon-greedy)
- [Level 1b: Contextual Bandit (LinUCB)](#level-1b-contextual-bandit-linucb)
- [Level 2: Q-Learning (Tabular)](#level-2-q-learning-tabular)
- [Level 3: Feature-Based Q-Learning (Linear Approximation)](#level-3-feature-based-q-learning-linear-approximation)
- [Level 4: Deep Q-Network (DQN)](#level-4-deep-q-network-dqn)
- [Level 5: Policy Gradient / PPO](#level-5-policy-gradient--ppo)
- [Level 6: RLHF (Reinforcement Learning from Human Feedback)](#level-6-rlhf-reinforcement-learning-from-human-feedback)
- [Level 7: DPO (Direct Preference Optimization)](#level-7-dpo-direct-preference-optimization)
- [Level 8: GRPO & RLVR (Reasoning Models)](#level-8-grpo--rlvr-reasoning-models)
- [Online vs Offline Classification](#online-vs-offline-classification)
- [Complete Comparison Table](#complete-comparison-table)
- [Decision Flowchart](#decision-flowchart)
- [When NOT To Use Each Method](#when-not-to-use-each-method)
- [Q-Tables Explained](#q-tables-explained)
- [LinUCB Explained](#linucb-explained)
- [RL Integration With Your LLM API](#rl-integration-with-your-llm-api)
- [Progression Path](#progression-path)

---

## Overview

### The Full Map

```
                          ┌──────────────────────────────┐
                          │      YOUR PROBLEM             │
                          └──────────────┬───────────────┘
                                         │
                          ┌──────────────┴───────────────┐
                          │  How many actions can you     │
                          │  take?                        │
                          └──────────────┬───────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
              Few actions           Many actions         Continuous
              (2-20)                (100s+)              actions
                    │                    │                  │
              ┌─────┴─────┐        ┌─────┴──────┐    ┌─────┴──────┐
              │ How much  │        │ How much   │    │ Policy     │
              │ context?  │        │ context?   │    │ Gradient   │
              └─────┬─────┘        └─────┬──────┘    │ / PPO      │
                    │                    │           └────────────┘
          ┌────────┼────────┐    ┌───────┼───────┐
          │        │        │    │       │       │
        None    Little   Rich  None   Little   Rich
          │        │        │    │       │       │
       Classic  LinUCB   Q-Learn  Random  LinUCB  DQN
         MAB              /DQN    MAB
```

### The Ladder

```
COMPLEXITY LADDER (lowest to highest):
═══════════════════════════════════════

Level 0: Random Selection
Level 1a: Classic MAB (Epsilon-Greedy)
Level 1b: Contextual Bandit (LinUCB)
Level 2: Q-Learning (Tabular)
Level 3: Linear Function Approximation
Level 4: Deep Q-Network (DQN)
Level 5: Policy Gradient / PPO
Level 6: RLHF
Level 7: DPO (Direct Preference Optimization)
Level 8: GRPO & RLVR (Reasoning Models)
```

---

## Level 0: Random Selection

**What:** Pick an action at random. No learning. No memory.

**When:** Only as a baseline to measure against.

| Aspect | Detail |
|--------|--------|
| **Complexity** | Zero |
| **Online?** | N/A (no learning) |
| **Storage** | None |
| **When to use** | Baseline comparison only |
| **Real example** | A/B testing with no optimization — just randomly show variations |

### App Use Case

Randomly pick response style for each message. Useful ONLY to establish a baseline metric: "How much better does personalization do vs random?"

---

## Level 1a: Classic Multi-Armed Bandit (Epsilon-Greedy)

**What:** Try all options, remember what works, mostly pick the best.

**When:** Few actions, no context matters, simple optimization.

| Aspect | Detail |
|--------|--------|
| **Complexity** | Very low |
| **Online?** | ✅ Fully online |
| **Storage** | One number per action |
| **When to use** | You have 2-20 options and context doesn't matter |
| **Key limitation** | Same answer for everyone — no personalization |

### Real-Life Examples

- **Email subject line optimization**: 5 different subject lines, reward = open rate, no context needed
- **Ad placement on homepage**: 4 banner designs, reward = click-through rate, same for all visitors
- **Game difficulty selection**: Easy/Medium/Hard, reward = session length, find sweet spot for retention

### App Use Case

"Which default response style works best for ALL users?"
- Arms: sweet, playful, flirty, caring
- Reward: average engagement across all users
- Use this to find the GLOBAL default before personalizing

---

## Level 1b: Contextual Bandit (LinUCB)

**What:** Same as MAB, but uses CONTEXT to make different decisions for different situations. Smart automatic exploration.

**When:** 2-20 actions + 5-15 context features matter.

| Aspect | Detail |
|--------|--------|
| **Complexity** | Medium |
| **Online?** | ✅ Fully online |
| **Storage** | Weight vectors per action (small) |
| **When to use** | Decisions depend on context (who, when, where) |
| **Key advantage** | Automatic exploration, no epsilon tuning |
| **Key limitation** | Assumes linear relationship between features and reward |

### Real-Life Examples

- **News article recommendation**: 10 article categories, context = user reading history + time + device, reward = read time / share / comment
- **Personalized treatment selection**: 5 medications, context = patient age / weight / lab results, reward = symptom improvement
- **Music recommendation**: 8 genres/playlists, context = mood / time / activity, reward = listen duration / saves / skips
- **Customer support routing**: 5 support specializations, context = issue type / customer tier / language, reward = resolution time / satisfaction

### App Use Case

"Which response style for THIS user at THIS moment?"
- Arms: sweet, playful, flirty, caring, distract
- Context: mood, time_of_day, engagement, topic, relationship_stage
- Reward: reply speed, message length, sentiment
- **This is the SWEET SPOT for AI girlfriend personalization**

---

## Level 2: Q-Learning (Tabular)

**What:** Score every (state, action) pair. Learn from transitions.

**When:** Discrete states + discrete actions + you care about SEQUENCES.

| Aspect | Detail |
|--------|--------|
| **Complexity** | Low-medium |
| **Online?** | ✅ Fully online |
| **Storage** | State × Action table |
| **When to use** | The ORDER of actions matters (sequential decisions) |
| **Key advantage** | Learns multi-step strategies |
| **Key limitation** | Only works with discrete, small state spaces |

### Key Difference From Bandits

```
Bandit: Each decision is independent.
        "Pick the best action RIGHT NOW."
        Doesn't care what you did last time.

Q-Learning: Decisions are sequential.
            "Pick the action that leads to the BEST FUTURE."
            What you do NOW affects what happens NEXT.
```

### Real-Life Examples

- **Customer service chatbot flow**: States = greeting → identify issue → troubleshoot → resolve → close. Actions = ask question, provide info, escalate, confirm, close. The sequence matters — "ask too many questions" → user abandons.
- **Game AI (chess, Go)**: States = board positions. Actions = legal moves. Each move affects all future moves.
- **Traffic signal control**: States = traffic density per direction. Actions = green/red timing. This cycle's timing affects next cycle's queue.
- **Onboarding flow optimization**: States = step 1 → step 2 → step 3 → complete. Actions = skip, show_tip, require_input. Decisions at step 1 affect step 3.

### App Use Case

"What's the best CONVERSATION FLOW for this user?"
- States: greeting → small_talk → deep_talk → emotional → close
- Actions: ask_question, share_story, comfort, tease, change_topic
- Reward: +10 user opens up, -5 user changes topic, -10 user leaves
- Use this for DIALOG POLICY (conversation flow management)

---

## Level 3: Feature-Based Q-Learning (Linear Approximation)

**What:** Q-learning but uses features instead of a lookup table.

**When:** Sequential decisions + state space too large for a table.

| Aspect | Detail |
|--------|--------|
| **Complexity** | Medium |
| **Online?** | ✅ Fully online |
| **Storage** | Weight vector (small) |
| **When to use** | Sequential decisions + many possible states |
| **Key advantage** | Generalizes to unseen states |
| **Key limitation** | Assumes linear feature-reward relationship |

### Real-Life Examples

- **Robotic assembly line**: Continuous sensor readings, actions = adjust_speed / recalibrate / continue / pause. Too many sensor combinations for a table.
- **Dynamic pricing**: States = demand / supply / time / competitor_price. Actions = raise / lower / hold / discount. Continuous features, sequential dependency.
- **Hospital patient flow**: States = patient_condition / bed_availability / staff_load. Actions = admit / discharge / transfer / monitor. Each decision affects available resources for next patients.

### App Use Case

"Manage conversation flow across hundreds of possible contexts"
- States: (mood × topic × time × engagement × relationship_stage) = too many combinations for a Q-table
- Use when tabular Q-learning can't handle your state space

---

## Level 4: Deep Q-Network (DQN)

**What:** Replace the Q-table/linear function with a neural network.

**When:** States are high-dimensional (images, text, complex data) or have non-linear relationships.

| Aspect | Detail |
|--------|--------|
| **Complexity** | High |
| **Online?** | ⚠ Semi-online (replay buffer) |
| **Storage** | Neural network weights (MB-GB) |
| **Infrastructure** | Needs GPU for training |
| **When to use** | Raw inputs (pixels, text) or very complex states |
| **Key advantage** | Learns non-linear patterns automatically |
| **Key limitation** | Unstable, needs careful tuning, expensive |

### Online DQN (Replay Buffer)

```
Instead of training on one experience at a time (unstable),
store experiences in a buffer and sample mini-batches.

Buffer stores last 10,000 interactions:
┌───────┬───────┬───────┬───────┬───────┐
│ exp 1 │ exp 2 │ exp 3 │  ...  │exp10K │
└───────┴───────┴───────┴───────┴───────┘

Each experience = (state, action, reward, next_state)

On each new message:
1. Add new experience to buffer
2. Sample random batch of 32 from buffer
3. Do ONE gradient step on neural network
4. Use updated network to pick action for THIS message
```

### Real-Life Examples

- **Atari / Video game AI**: States = raw pixels (100,800 values). A table can't store 100K-dimensional states. NN extracts features automatically.
- **Self-driving car**: States = camera feeds + lidar + GPS + speed. Multi-sensor fusion with non-linear interactions.
- **Robot manipulation**: States = camera image + joint angles + force sensors. Vision-based control requires processing raw images.
- **AI girlfriend (advanced)**: States = conversation text embeddings (768-dim). Non-linear interactions between conversation features.

### App Use Case

"Learn from raw conversation text, not just categories"
- Only needed when LinUCB's linear assumption isn't enough
- If conversations have complex non-linear patterns that features like "mood=sad" can't capture

---

## Level 5: Policy Gradient / PPO

**What:** Learn the POLICY directly (probability distribution over actions).

**When:** Continuous action spaces, or you want to optimize LLM output.

| Aspect | Detail |
|--------|--------|
| **Complexity** | Very high |
| **Online?** | ⚠ Semi-online (mini-batch) |
| **Storage** | Neural network weights |
| **Infrastructure** | Needs GPU cluster for serious use |
| **When to use** | Continuous actions, LLM fine-tuning |
| **Key advantage** | Handles continuous/complex action spaces |
| **Key limitation** | Slow, expensive, complex to debug |

### Real-Life Examples

- **Robotic arm control**: Actions = joint torques (continuous values). Can't discretize "0.73 Newton-meters" into table cells.
- **Trading bot**: Actions = "allocate 23.7% to this asset". Action space is continuous (0-100%).
- **Music generation**: Actions = next note frequency, duration, velocity. Continuous output space.
- **ChatGPT/Claude training (RLHF)**: Actions = token probabilities (50K+ vocabulary). This is how OpenAI makes GPT helpful.

### App Use Case

"Fine-tune response generation at the token level"
- Only needed if you're training your OWN language model
- For strategy selection via your LLM API, this is overkill

---

## Level 6: RLHF (Reinforcement Learning from Human Feedback)

**What:** Train a reward model from human preferences, then use PPO to optimize an LLM to maximize that reward.

**When:** You're training/fine-tuning a foundation model.

| Aspect | Detail |
|--------|--------|
| **Complexity** | Extreme |
| **Online?** | ❌ Offline (weeks of GPU training) |
| **Storage** | Multiple large neural networks |
| **Infrastructure** | GPU cluster (thousands of GPU-hours) |
| **When to use** | Fine-tuning foundation models |
| **Key advantage** | Aligns AI with human preferences at scale |
| **Key limitation** | Requires massive compute and data |

### RLHF Pipeline

```
Phase 1 (offline, weeks):
  Collect human preference data
  Train reward model

Phase 2 (offline, days/weeks):
  Use PPO to optimize LLM against reward model
  Train on thousands of GPU-hours

Phase 3 (online, the only online part):
  Deploy optimized LLM
  Collect NEW user feedback
  Feed back into Phase 1 for next training run

The FEEDBACK COLLECTION is online.
The MODEL TRAINING is offline.
```

### Real-Life Examples

- **ChatGPT (OpenAI)**: Base = GPT-4 pretrained on internet text. RLHF = human raters rank responses. Result = helpful, harmless, honest assistant. Cost = millions of dollars.
- **Claude (Anthropic)**: Base = Constitutional AI + RLHF. Preference learning + constitutional rules. Result = safe, helpful assistant.
- **Llama (Meta)**: Base = pretrained model. Community-driven preference data. Result = open-weight helpful assistant.

### App Use Case

You DON'T do RLHF yourself. You USE models that already went through RLHF (Claude, GPT-4). You ADD personalization on top via the lower-level methods.

---

## Level 7: DPO (Direct Preference Optimization)

**What:** Optimize the policy directly on preference pairs — no reward model, no RL loop, no sampling. A closed-form reparameterization of the RLHF objective.

**When:** You have preference data and want alignment without building a reward model or running PPO.

| Aspect | Detail |
|--------|--------|
| **Complexity** | Medium-high |
| **Online?** | ❌ Offline (supervised-style pass) |
| **Storage** | Policy + frozen reference model |
| **Infrastructure** | Single GPU is often enough (vs. cluster for RLHF) |
| **When to use** | Fine-tuning your own SFT model from preference pairs |
| **Key advantage** | Simple, stable, cheap — no reward model, no value net |
| **Key limitation** | Offline: can't explore beyond the data; no process-level credit |

### The Key Idea

```
The KL-constrained RLHF objective has an exact optimal policy:
    π*(y|x) ∝ π_ref(y|x) · exp( r(x,y) / β )

Rearranged, the reward is IMPLICIT in the policy:
    r(x,y) = β · log( π*(y|x) / π_ref(y|x) )

→ The reward model cancels out.
→ Just optimize log π(winner)/π_ref − log π(loser)/π_ref with a sigmoid loss.
```

### DPO Family (Variants)

| Method | Idea | Data Needed |
|--------|------|-------------|
| **DPO** | Log-ratio loss on preference pairs | Pairs (winner vs loser) |
| **IPO** | Identity-preference loss; less overfitting on pairs | Pairs |
| **KTO** | Uses *unpaired* thumbs up/down signals | Binary feedback |
| **ORPO** | Merges SFT + preference into one loss | Pairs + SFT data |
| **SimPO** | Drops the reference model; average log-prob + margin | Pairs |

### Real-Life Examples

- **Zephyr-7B**: aligned via DPO instead of PPO — proved DPO at small scale
- **Llama / Mistral fine-tunes**: DPO is the default open-weight alignment recipe
- **Production chatbots**: cheap personalization/alignment pass on your own model

### App Use Case

If you fine-tune your OWN small model on YOUR preference data, DPO is the pragmatic default — far cheaper than RLHF and usually enough. Collect pairs from humans or an AI judge (RLAIF).

---

## Level 8: GRPO & RLVR (Reasoning Models)

**What:** GRPO = PPO **without a value network** (use the group mean as baseline). RLVR = reward from a programmatic **verifier** (is the math answer right? do the tests pass?) instead of humans or a reward model.

**When:** Reasoning / math / code tasks with automatic correctness checks; limited memory; massive scale.

| Aspect | Detail |
|--------|--------|
| **Complexity** | Extreme |
| **Online?** | ⚠ Semi-online (group rollouts) |
| **Storage** | Policy + reference model (NO critic) |
| **Infrastructure** | GPU cluster |
| **When to use** | Verifiable tasks (math/code/logic), emergent reasoning |
| **Key advantage** | No critic, cheap exact rewards, enables long chain-of-thought |
| **Key limitation** | Needs a verifier; expensive at scale; can reward-hack |

### The Key Ideas

```
GRPO (DeepSeekMath, 2024):
  For each prompt, sample a GROUP of G outputs.
  Use the group's mean reward as the baseline → no value network.

    A_i = (r_i - mean(r)) / std(r)

  → ~half the memory of PPO, simpler, trivially parallelizable.

RLVR (the reasoning breakthrough):
  Reward comes from a VERIFIER, not a human or reward model:
    Math   → compare final answer to ground truth
    Code   → run the unit tests
    Logic  → check constraints / format
  Reward is 0/1. No humans. No reward model. Infinite, cheap, exact.

  → This is what made DeepSeek-R1 / o1-style reasoning EMERGE.
```

### Real-Life Examples

- **DeepSeek-R1**: GRPO + RLVR produced emergent long chain-of-thought reasoning
- **OpenAI o1 / o3**: RL on reasoning (undisclosed) — test-time-compute scaling
- **Qwen / QwQ**: GRPO variants (e.g., GSPO) for open reasoning models
- **Code assistants**: RLVR on unit tests to improve code generation

### App Use Case

Don't train a reasoning model yourself — USE reasoning models (DeepSeek-R1, o-series) via API. DPO (Level 7) is usually the pragmatic default if you must fine-tune. Reserve GRPO + RLVR for teams with a verifiable task AND GPUs.

---

## Online vs Offline Classification

```
                    ONLINE                      OFFLINE
                    (learns per message)        (learns in batches)

Small state space ──► Q-Table ✅                (not needed)
                      Feature-Based ✅
                      Linear Approx ✅

High-dimensional  ──► Online DQN ⚠             Classic DQN ❌
state space           (with replay buffer)      (batch training)

Continuous        ──► Online PPO ⚠             Batch PPO ❌
actions               (mini-batch updates)      (GPU cluster)

LLM fine-tuning  ───► (impractical per msg)     RLHF ❌
                                             (weeks of GPU time)
```

| Method | Online? | Update Cost | Storage | Notes |
|--------|---------|-------------|---------|-------|
| **Q-Table** | ✅ Fully online | One addition | Small table per user | Online by nature |
| **Tile Coding** | ✅ Fully online | Few additions | Sparse table | Online by nature |
| **Linear Approximation** | ✅ Fully online | One vector multiply | Weight vector | Online by nature |
| **Classic DQN (batch)** | ❌ Offline | Hours of training | NN weights | Needs training infra |
| **DQN + Replay Buffer** | ✅ Online-ish | One gradient step | NN weights + buffer | Needs GPU per step |
| **DQN + Periodic Sync** | ⚠ Hybrid | Every N messages | NN weights | Production compromise |
| **PPO batch training** | ❌ Offline | Days/weeks training | NN weights | Needs GPU cluster |
| **Online PPO (mini-batch)** | ⚠ Semi-online | Per batch | NN weights | Needs GPU per batch |
| **RLHF** | ❌ Offline | Weeks of GPU | Multiple NNs | Only collection is online |
| **DPO** | ❌ Offline | One supervised pass | Policy + reference | No RM, no sampling |
| **GRPO + RLVR** | ⚠ Semi-online | Group rollouts | Policy + reference | No critic; needs a verifier |

---

## Complete Comparison Table

```
┌──────────────────┬────────────┬──────────┬───────────┬────────────┬──────────────┐
│ Method           │ Context?   │ Sequence?│ Online?   │ Complexity │ Storage/User │
├──────────────────┼────────────┼──────────┼───────────┼────────────┼──────────────┤
│ Random           │ ❌ None    │ ❌ No    │ N/A       │ Zero       │ Zero         │
│ Classic MAB      │ ❌ None    │ ❌ No    │ ✅ Yes    │ Very Low   │ ~20 numbers  │
│ LinUCB           │ ✅ Yes     │ ❌ No    │ ✅ Yes    │ Medium     │ ~500 numbers │
│ Q-Table          │ ✅ Discrete│ ✅ Yes   │ ✅ Yes    │ Low-Med    │ State×Action │
│ Feature Q-Learn  │ ✅ Yes     │ ✅ Yes   │ ✅ Yes    │ Medium     │ ~50 weights  │
│ DQN              │ ✅ Raw     │ ✅ Yes   │ ⚠ Semi   │ High       │ NN (MBs)     │
│ PPO              │ ✅ Raw     │ ✅ Yes   │ ⚠ Semi   │ Very High  │ NN (MBs)     │
│ RLHF             │ ✅ Raw     │ ✅ Yes   │ ❌ No     │ Extreme    │ NN (GBs)     │
│ DPO              │ ✅ Raw     │ ✅ Yes   │ ❌ No     │ Med-High   │ Policy+Ref   │
│ GRPO + RLVR      │ ✅ Raw     │ ✅ Yes   │ ⚠ Semi   │ Extreme    │ Policy+Ref   │
└──────────────────┴────────────┴──────────┴───────────┴────────────┴──────────────┘
```

---

## Decision Flowchart

```
START: What are you optimizing?
  │
  ├── "Which single response style works best?"
  │    │
  │    ├── Same for all users? ──────────────► Classic MAB
  │    │
  │    └── Depends on context (mood, time)? ──► LinUCB ✅
  │
  ├── "What's the best conversation FLOW?"
  │    │
  │    ├── Small discrete states (< 500)? ────► Q-Table
  │    │
  │    ├── Many states, but have features? ───► Feature-Based Q-Learning
  │    │
  │    └── Raw text as state? ────────────────► DQN
  │
  ├── "Control response at word/token level?"
  │    │
  │    └──────────────────────────────────────► PPO
  │
  └── "Train my own LLM?"
       │
       ├── Have preference pairs, offline? ────► DPO (Level 7) ✅
       │
       ├── Verifiable task (math/code/logic)? ─► GRPO + RLVR (Level 8)
       │
       └── Classic online RL + reward model? ──► RLHF (Level 6)
```

### Quick Decision By State Space Size

```
How many possible states do you have?
    │
    ├── < 1,000 ──────────► Q-Table
    │
    ├── 1,000 - 100,000 ──► Feature-Based Q-Learning / LinUCB
    │
    ├── 100,000 - millions ► DQN
    │
    └── Continuous/infinite ► Policy Gradient / PPO
```

---

## When NOT To Use Each Method

| Method | Don't Use When |
|--------|----------------|
| **Random** | You have ANY data. Even 10 interactions beats random. |
| **Classic MAB** | Context matters. You'll get the same answer for everyone. |
| **LinUCB** | Actions have sequential dependencies. It doesn't plan ahead. |
| **Q-Table** | State space is huge (10K+ combinations) or continuous. |
| **Feature Q-Learn** | Features don't linearly predict reward. Or you have raw pixels/text. |
| **DQN** | You have < 10K training samples. It overfits. |
| **PPO** | You have discrete actions. Q-learning is simpler and works. |
| **RLHF** | You're not training a foundation model. Use a pre-trained one instead. |
| **DPO** | You need online exploration. DPO is offline — it can't beat its data. |
| **GRPO / RLVR** | You have no programmatic verifier (open-ended creativity/empathy). |

---

## Q-Tables Explained

### What Is a Q-Table?

A Q-table is a **cheat sheet** that an RL agent keeps. It's literally a table that says:

> "When I was in situation X and did action Y, how good was the result?"

- **Rows** = situations (states)
- **Columns** = choices (actions)
- **Numbers** = how good that choice was in that situation (Q-values)

### Example: Restaurant Cheat Sheet

```
┌─────────────────────────────────────────────────────────────┐
│                  MY RESTAURANT CHEAT SHEET                   │
│                                                             │
│  Situation          │ Pizza  │ Sushi  │ Tacos  │ Salad      │
│  ───────────────────┼────────┼────────┼────────┼────────────│
│  Date night         │  4.0   │  8.5   │  3.0   │  2.0       │
│  Quick lunch        │  7.0   │  5.0   │  9.0   │  6.0       │
│  Feeling sick       │  3.0   │  2.0   │  6.0   │  8.0       │
│  Celebration        │  5.0   │  9.0   │  4.0   │  1.0       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### How It Learns

```
Step 1: Start with a blank table (all zeros)

Step 2: First interaction — try "Caring" when user is "Sad+Night"
        User opens up → reward: +8
        Q(Sad+Night, Caring) = 0.0 + 0.1 × (8.0 - 0.0) = 0.8

Step 3: Try "Playful" when user is "Sad+Night"
        User goes silent → reward: -2
        Q(Sad+Night, Playful) = 0.0 + 0.1 × (-2.0 - 0.0) = -0.2

Step 4: After 100 interactions, clear patterns emerge:
        Sad+Night → Caring: 7.9, Playful: 1.2, Flirty: 0.8
```

### The Math

```
Q(s, a) ← Q(s, a) + α × [reward - Q(s, a)]

"Move the score TOWARD what actually happened, a little bit at a time."

α (alpha) = learning rate
  0.1 = conservative (trust the trend over many tries)
  0.3 = aggressive (react strongly to each result)
```

### When Q-Tables Break Down

- **State space too large**: 8 moods × 50 topics × 48 time slots × 10 relationship stages = millions of cells
- **States are continuous**: mood_score = 0.347 (not just "sad"/"happy")
- **States need memory**: "sad for 3 days after being happy" matters
- **Need to generalize across users**: New user starts with blank table
- **Actions have complex relationships**: "playful THEN caring" ≠ "caring THEN playful"

---

## LinUCB Explained

### What Is LinUCB?

**Linear Upper Confidence Bound** — a contextual bandit algorithm that uses features instead of a lookup table, with smart automatic exploration.

- **Lin** = Linear (uses features, not a lookup table)
- **U** = Upper (optimistic estimate)
- **C** = Confidence (how uncertain are we?)
- **B** = Bound (the ceiling of our estimate)

### The UCB Part: Optimism in the Face of Uncertainty

```
LINUCB SCORE = predicted_reward + confidence_bonus

  Score(a) = θ_a · x  +  α · √(xᵀ · A_a⁻¹ · x)
             ──────     ─────────────────────
             predicted   uncertainty bonus
             reward      (how unsure we are)
```

Actions we haven't tried much get a big uncertainty bonus, so they get selected automatically. Once we try them enough, the bonus shrinks and actual reward takes over.

```
New action:      uncertainty bonus is LARGE  → gets tried
After 5 tries:   bonus shrinks               → tried if actually good
After 50 tries:  bonus is tiny               → only picked if reward is high

Exploration happens AUTOMATICALLY. No epsilon to tune.
```

### Q-Table vs LinUCB

```
Q-TABLE:   "I've seen this EXACT state before"
            Remembers specific combinations.
            Great with few states. Breaks with many.

LinUCB:    "I haven't seen this exact state, but the
             features are similar to states I HAVE seen"
            Generalizes via features. Works with many.
```

| Aspect | Q-Table + Epsilon-Greedy | LinUCB |
|--------|-------------------------|--------|
| **Exploration** | Fixed ε% random | Automatic via confidence |
| **Context handling** | Discrete buckets only | Continuous features |
| **New states** | Zero knowledge (blank row) | Generalizes from features |
| **Cold start** | Bad (blank table per user) | Decent (shared feature weights) |
| **Online?** | ✅ Yes | ✅ Yes |
| **Update cost** | One addition | One matrix operation |
| **Storage** | O(states × actions) | O(features² × actions) |
| **Works well when** | Few, discrete states | Many states, shared structure |

### LinUCB With Your LLM API

```
EVERY MESSAGE:

Step 1: Extract features
  x = [1.0, mood_score, sin(hour), cos(hour), engagement,
       relationship_stage, topic_embedding..., sentiment_intensity]

Step 2: For each action, compute score
  Score(a) = predicted_reward + uncertainty_bonus

Step 3: Pick action with highest score

Step 4: Build dynamic prompt, call your LLM API, send response

Step 5: On user reaction, calculate reward

Step 6: Update LinUCB weights (online, instant)
  A_action  ← A_action  + x · xᵀ
  b_action  ← b_action  + reward × x
  w_action  ← A_action⁻¹ · b_action
```

### Per-User AND Cross-User Learning

```
Hybrid approach:

  Global weights (all users):         Per-user adaptation:
  ┌──────────────────────────┐       ┌──────────────────────────┐
  │ "Sad people generally    │  +    │ "But THIS user at night  │
  │  respond well to         │       │  specifically prefers    │
  │  empathetic responses"   │       │  humor over empathy"     │
  └──────────────────────────┘       └──────────────────────────┘

  w_effective = w_global + w_user_personal

  New user: w_effective = w_global + 0 (uses crowd wisdom)
  After 20 msgs: w_effective = w_global + small_personal (hybrid)
  After 200 msgs: w_effective = mostly personal (highly adapted)
```

---

## RL Integration With Your LLM API

### Where RL Fits In

RL doesn't sit inside the LLM call. It sits **before** it, deciding **how to instruct the LLM**.

```
┌─────────────────────────────────────────────────────────────┐
│                      YOUR BACKEND                           │
│                                                             │
│  1. CONTEXT EXTRACTOR → state vector                        │
│  2. RL POLICY → "best strategy is: playful + short"         │
│  3. PROMPT BUILDER → dynamic system prompt                  │
│  4. LLM API CALL → same as today, different prompt          │
│  5. REWARD CALCULATOR → score user reaction                 │
│  6. Q-TABLE/LinUCB UPDATE → store learning                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### What Changes vs What Stays the Same

```
STAYS THE SAME (the LLM does this):
  ✓ Understanding user's message
  ✓ Generating human-like text
  ✓ Maintaining conversation context
  ✓ Following persona instructions

CHANGES (RL layer adds this):
  ✓ WHICH persona style to use right now
  ✓ HOW long/short to respond
  ✓ WHAT action to take (comfort vs distract vs tease)
  ✓ WHEN to be creative vs safe (temperature)
  ✓ System prompt is DYNAMIC, not static
```

### Key Insight

```
LLM = WHAT to say (language generation)
RL  = WHICH STRATEGY to use (optimization from outcomes)

LLM = Talented chef who can cook anything
RL  = Learns THIS customer likes it spicy every Tuesday

LLM = Doctor who knows all treatments
RL  = Learns which treatment works for THIS patient
```

### 3 Levels of Integration

**Level 1: Simplest — Multi-Armed Bandit on Response Style**
- 4 pre-written system prompts (sweet, playful, flirty, caring)
- RL learns which prompt works best for which user at what time
- Effort: hours
- Impact: High — immediate personalization

**Level 2: Moderate — Dynamic Prompt Parameters**
- RL controls tone, length, temperature, action type
- These get interpolated into the system prompt dynamically
- Effort: 1-2 days
- Impact: Very high — feels like a different person per user

**Level 3: Advanced — Full RLHF Pipeline**
- Multiple LLM responses generated → user picks best → reward model trained
- Effort: weeks
- Impact: Production-grade personalization

---

## Progression Path

```
MONTH 1:  Classic MAB
          Find the best GLOBAL default style.
          "Across all users, which style gets best engagement?"

MONTH 2:  LinUCB
          Personalize style per user context.
          "For THIS user at 11pm when sad, what works?"

MONTH 4:  Q-Learning + LinUCB
          Add conversation flow optimization.
          "Not just which style, but WHEN to use which style."

MONTH 6:  Feature-Based Q-Learning
          Scale to richer state spaces.
          "Handle 100+ context combinations per user."

YEAR 2:   DQN + PPO (optional)
          Only if you train your own model.
          "Learn directly from conversation text."

FINE-TUNE: DPO (Level 7)
          Only if you fine-tune your OWN small model.
          "Align it to YOUR preference data — no reward model, no RL loop."
          Collect pairs from humans or an AI judge (RLAIF).

NEVER:    RLHF / GRPO + RLVR
          You use pre-trained models. Don't reinvent this.
          For reasoning, USE DeepSeek-R1 / o-series via API.
          Your LLM API gives you access to RLHF'd AND reasoning models.
```

### Recommended Stack for AI Girlfriend App

| Component | Method | Why |
|-----------|--------|-----|
| **Strategy selection** | LinUCB | Context-dependent, few actions, online |
| **Dialog flow** | Q-Learning | Sequential decisions matter |
| **Response generation** | LLM API | Pre-trained, already RLHF'd |
| **Reward calculation** | Heuristic | Computed from user reactions |


