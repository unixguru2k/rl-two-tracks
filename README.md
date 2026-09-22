# RL Two Tracks — Reinforcement Learning for Agents, LLMs & Systems

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

> **Learn Reinforcement Learning from a random baseline to agentic RL** — hands-on, runnable Python examples, organized as **two independent tracks**. They share vocabulary and almost nothing else: different reward sources, runtimes, costs, and audiences.
>
> ✅ **Status:** All 14 guides are live. Every guide is self-contained and NumPy-only **except [A0](docs/tracks/a-llm-post-training/a0-hands-on-post-training.md)**, the hands-on post-training guide, which needs a GPU (free Colab is enough). The interactive web app in `PLAN.md` is a proposal, not built.
>
> 🗓 **Last updated:** 2026-09-20

---

## Choose Your Track

This repo has **two separate tracks**. Pick the one that matches your job — you read **one**, not both.

| If you are… | Track | Read this |
|---|---|---|
| An **agent builder** with your own model, GPUs, and a verifiable task | **A — Post-Training & Agentic RL** | [A1 → A6](#track-a--post-training--agentic-rl) |
| Just curious **how ChatGPT / Claude are trained** | **A** (read-only) | [A2 → A4](#track-a--post-training--agentic-rl) |
| An **app / product / backend engineer** adding personalization, ranking, routing, or A/B tests | **B — Decision Optimization** | [B0 → B6](#track-b--decision-optimization-in-systems) |
| Unsure? | — | [Why This Repo?](#why-this-repo) → then decide |

**The one-sentence difference:**

```
Track A optimizes THE MODEL that generates the action. (weights, offline, GPU-hours, millions of $)
Track B optimizes WHICH ACTION your system takes.      (your code, online, per-request, CPU, milliseconds)
```

> **The most common mistake this repo exists to prevent:** using a bandit for a sequential problem, or an MDP for an independent one. That single decision routes you to one half of the curriculum — [Track B's overview](docs/tracks/b-decision-optimization/index.md) opens with it.

---

## Table of Contents

- [Choose Your Track](#choose-your-track)
- [Why This Repo?](#why-this-repo)
- [What This Repo Is (and Isn't)](#what-this-repo-is-and-isnt)
- [**Track A — Post-Training & Agentic RL**](#track-a--post-training--agentic-rl)
- [**Track B — Decision Optimization in Systems**](#track-b--decision-optimization-in-systems)
- [Cross-Track Bridge](#cross-track-bridge)
- [Quick Start](#quick-start)
- [Presentation](#presentation)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Keywords](#keywords)

---

## Why This Repo?

Most RL tutorials use toy games (CartPole, MountainCar). This guide teaches RL through **real systems** — tool-using agents, verifier-graded reasoning, multi-turn rollouts in sandboxes, response strategy selection, conversation flow optimization, and production personalization. Every concept maps to something you can ship or reason about.

**Who this is for:**
- **App and backend engineers** adding RL-powered personalization, ranking, routing, or A/B testing → **Track B**
- **Agent builders** working on tool use, multi-turn rollouts, sandboxes, and verifiers → **Track A**
- **ML engineers** deciding whether to **train** or **consume** a post-trained model → **Track A** (read-only), then **Track B**
- **Students** learning RL with practical, runnable examples → **Track B first**
- Anyone curious how **ChatGPT, Claude, and reasoning models are actually trained** → **Track A** (read-only)

---

## What This Repo Is (and Isn't)

**What it is:** A documentation-first curriculum organized as two tracks. Each guide is self-contained and NumPy-only — **except [A0](docs/tracks/a-llm-post-training/a0-hands-on-post-training.md)**, the hands-on post-training guide, which needs a GPU (free Colab is enough). No framework, no API key.

**What it isn't:** An application or a library. There is nothing to install and no server to run. The interactive web app in `PLAN.md` is a **proposal** — no `backend/`, `frontend/`, or `start-app.sh` exists yet.

**The core idea this repo is built around:**

```
LLM = WHAT to say           (language generation)
RL  = WHICH STRATEGY to use (optimization from outcomes)
```

You don't reinvent RLHF. You **use** its output (GPT-4, Claude via API) and layer lightweight RL personalization on top.

---

## Track A — Post-Training & Agentic RL

> *"How do I make the model itself better at reasoning and at acting in an environment?"*

**Read this track if:** you have (a) GPUs, (b) an automatic way to grade correctness, and (c) a model you own. Or if you simply want the vocabulary to understand how ChatGPT and reasoning models are trained.

**Skip this track if:** you are building a product with personalization — you are a **consumer** of Track A, not a contributor. Go to [Track B](#track-b--decision-optimization-in-systems).

📖 **Start here:** [Track A Overview](docs/tracks/a-llm-post-training/index.md)

### The A Guides

| # | Guide | What It Covers | Reward Source | Compute |
|:-:|-------|----------------|---------------|:-------:|
| **A0** | [Hands-On Post-Training](docs/tracks/a-llm-post-training/a0-hands-on-post-training.md) | **Runnable** SFT → DPO → GRPO on your own hardware (M4 + Colab) | Cross-entropy / preferences / verifier | GPU |
| **A1** | [Policy Gradient / PPO](docs/tracks/a-llm-post-training/a1-policy-gradient-ppo.md) | Optimizing the policy directly; the clipped objective | Environment / reward model | High |
| **A2** | [RLHF](docs/tracks/a-llm-post-training/a2-rlhf.md) | Preference data → reward model → PPO | Human preferences | Extreme |
| **A3** | [DPO & RFT](docs/tracks/a-llm-post-training/a3-dpo-and-rft.md) | Alignment without a reward model or RL loop | Preference pairs / filtered samples | Medium |
| **A4** | [GRPO & RLVR](docs/tracks/a-llm-post-training/a4-grpo-and-rlvr.md) | Critic-free group-relative optimization; verifiable rewards | **Programmatic verifier** | High |
| **A5** | [Agentic RL & Environments](docs/tracks/a-llm-post-training/a5-agentic-rl-environments.md) | Multi-turn rollouts, tool use, sandboxes | Verifier + environment state | High |
| **A6** | [Rewards & Test-Time Compute](docs/tracks/a-llm-post-training/a6-rewards-and-test-time-compute.md) | Outcome vs process rewards, search, reranking | Verifier / judge | Med–High |

### Track A: The Reward Source Progression

The single most useful axis for orienting yourself in this track — and the axis along which the field actually moved.

```
TRUSTWORTHY / EXPENSIVE  ────────────────────────►  CHEAP / SCALABLE

  Human preferences        AI judge           Programmatic verifier
  ────────────────        ────────           ─────────────────────
   A2 (RLHF)               A3 (pairs)          A4 (RLVR)
                           RLAIF               A5 (environments)

  "Which answer would      "Which answer       "Is the answer 42?
   a human prefer?"         does GPT-4 prefer?" Do the tests pass?"
```

> **The 2026 shift in one line:** the interesting work moved from the left of that axis to the right. Verifiers replaced annotators wherever correctness could be checked by machine.

### Track A: The 2026 Post-Training Stack

```
1. SFT                    supervised fine-tune on demonstrations
        │
        ▼
2. LIGHT PREFERENCE       DPO / SimPO / ORPO / KTO          (A3)
   OPTIMIZATION           cheap, stable, no reward model
        │
        ▼
3. RLVR — IF NEEDED       GRPO / DAPO / GSPO                (A4)
                          only when you have a verifier
        │
        ▼
4. TEST-TIME COMPUTE      search / reranking / self-consistency (A6)
```

> **You rarely need step 3.** Most teams stop at step 2. **Steps 3 and 4 are complementary, not substitutes** — training raises the reasoning ceiling, test-time compute extracts it.

### Track A: Recommended Paths

```
Want to actually RUN the pipeline?     A0             (hands-on: SFT → DPO → GRPO)
Curious how ChatGPT was made?          A2 → A4 → A5   (read for understanding)
Align your own small model?            A3 → A4        (build)
Reasoning or coding assistant?         A4 → A6        (build)
Tool-using / computer-use agent?       A5 → A6        (build — the frontier)
Product with personalization?          Go to Track B  (and come back only for the bridge)
```

**Track A takeaway:** unless you have GPUs, a verifier, and a model you own, you are a **consumer** of this track. That is the intended design — Track A produces *judges and reasoners*; Track B *uses* them.

---

## Track B — Decision Optimization in Systems

> *"Which action should my system take, given the current context, right now?"*

**Read this track if:** you ship software. It runs online, per-request, in milliseconds, on a CPU. No GPUs, no reward model, no preference data — only a measurable outcome and a place to store one number.

**Skip this track if:** you are fine-tuning or training a model (that's Track A).

📖 **Start here:** [Track B Overview](docs/tracks/b-decision-optimization/index.md)

### The B Guides (read in order)

| # | Guide | Problem Shape | Complexity | Online? |
|:-:|-------|---------------|:----------:|:-------:|
| **B0** | [Random Baseline](docs/tracks/b-decision-optimization/b0-random-baseline.md) | Establish the floor (this is a **control**, not a warm-up) | ⭐ | N/A |
| **B1** | [Epsilon-Greedy Bandit](docs/tracks/b-decision-optimization/b1-epsilon-greedy-bandit.md) | Pick best option; **no context** | ⭐ | ✅ |
| **B2** | [Contextual Bandit (LinUCB)](docs/tracks/b-decision-optimization/b2-contextual-bandit-linucb.md) | Pick best option **given context** | ⭐⭐ | ✅ |
| **B3** | [Q-Learning (Tabular)](docs/tracks/b-decision-optimization/b3-q-learning-tabular.md) | **Sequence** of decisions, small state space | ⭐⭐ | ✅ |
| **B4** | [Feature-Based Q-Learning](docs/tracks/b-decision-optimization/b4-feature-based-q-learning.md) | Sequence, large/continuous state space | ⭐⭐⭐ | ✅ |
| **B5** | [DQN](docs/tracks/b-decision-optimization/b5-dqn.md) | Sequence, raw high-dimensional state (text/images) | ⭐⭐⭐⭐ | ⚠ |
| **B6** | [Production Online Systems](docs/tracks/b-decision-optimization/b6-production-online-systems.md) | Shipping any of the above to real users | ⭐⭐⭐ | ✅ |

### Track B: The Two Paradigms

```
                Do the ORDER of your decisions matter?
                                │
                ┌───────────────┴───────────────┐
               NO                              YES
                │                               │
                ▼                               ▼
          ┌───────────┐                   ┌───────────
          │  BANDITS  │                   │    MDP    │
          │  B0, B1,  │                   │  B3, B4,  │
          │     B2    │                   │    B5     │
          └───────────                   └───────────┘
```

### Track B: Complexity Ladder

```
   B0   Random            zero storage, zero code
   B1   Epsilon-Greedy    ~20 numbers
   B2   LinUCB            ~500 numbers, uses features
   B3   Q-Table           states × actions
   B4   Feature Q-Learn   ~50 weights
   B5   DQN               neural network (MBs), needs GPU
        ▲ Climb only when the level below provably breaks.
```

> **The single most valuable rule in Track B:** always run the level below your target as a **baseline**. B0 is not a warm-up — it is the control that tells you whether your careful algorithm is doing anything at all.

### Track B: Recommended Paths

```
Just want the 80/20?               B0 → B1 → B2
Personalization works, now the journey?  B3 → B4
Text/images are the state?         B5  (only if B4 provably fails)
About to ship to real users?       B6  (mandatory)
```

**Track B takeaway:** no GPUs, no reward model, no preference data — just a measurable outcome, a context vector, and one key-value store. In ads, ranking, and recommendation, the online-learning default has been a contextual bandit for a decade — and increasingly, a bandit wrapped around a frozen model. It almost never gets a tutorial.

---

## Cross-Track Bridge

Track A and Track B are coupled at exactly one point — and it is the most useful idea in this repo:

```
   TRACK A                          TRACK B
   ─────                          ───────
   Produces JUDGES       ──────►    Consumes judges as reward signals
   (reward models,                  (score each served action)
    preferences, verifiers)
```

**Track A outputs are Track B inputs.** If your product has an action whose outcome you can't measure directly — *"was this empathetic?"*, *"was this explanation clear?"* — you can't bandit-test it, because there is no reward signal. Track A supplies one: run an aligned model as a scorer, and that score becomes your Track B reward.

This is the answer to *"how do I do RL on things I can't measure with a click."* It is not RLHF on your own model — it is using someone else's RLHF'd model as a measuring instrument.

📚 **Master reference:** [Complete RL Methods Guide (all 9 levels)](docs/RL_METHODS_GUIDE.md)

---

## Quick Start

There is nothing to install. Every guide is copy-paste runnable.

```bash
# 1. Clone
git clone https://github.com/unixguru2k/rl-two-tracks.git
cd rl-two-tracks

# 2. Install the single dependency used by all examples
pip install numpy

# 3. Read your track, then run any Python block from any guide
#    Track A → docs/tracks/a-llm-post-training/
#    Track B → docs/tracks/b-decision-optimization/
```

**No GPU. No API key. No framework. Only NumPy.**

---

## Presentation

A visual overview of the two-track structure, key concepts, and the production RL pattern — designed for talks, onboarding, and quick reference.

📖 **[Open the slides →](https://unixguru2k.github.io/rl-two-tracks/slides/index.html)**

Use **→** / **←** arrow keys, spacebar, or the on-screen buttons to navigate. Works in any browser; no build step required.

---

## Tech Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Examples | **Python 3.9+** | Readable, universally available |
| Numerics | **NumPy** (only dependency) | Every guide runs on CPU with one install — **except A0**, which needs a GPU |
| Docs | **Markdown** | Renders on GitHub and in any editor |
| Diagrams | ASCII | No image assets, no renderer, grep-able |

---

## Project Structure

```
rl-two-tracks/
├── README.md                          ← you are here
├── PLAN.md                            ← proposal for the interactive web app (not built)
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
├── .gitignore
└── docs/
    ├── RL_METHODS_GUIDE.md            ← master reference: all 9 levels
    ├── launch/                        ← launch copy + claim receipts (internal)
    │   ├── launch-copy.md
    │   └── claims.md
    └── tracks/
        ├── a-llm-post-training/       ← Track A (model training)
        │   ├── index.md
        │   ├── a0-hands-on-post-training.md
        │   ├── a1-policy-gradient-ppo.md
        │   ├── a2-rlhf.md
        │   ├── a3-dpo-and-rft.md
        │   ├── a4-grpo-and-rlvr.md
        │   ├── a5-agentic-rl-environments.md
        │   └── a6-rewards-and-test-time-compute.md
        └── b-decision-optimization/   ← Track B (app & product engineers)
            ├── index.md
            ├── b0-random-baseline.md
            ├── b1-epsilon-greedy-bandit.md
            ├── b2-contextual-bandit-linucb.md
            ├── b3-q-learning-tabular.md
            ├── b4-feature-based-q-learning.md
            ├── b5-dqn.md
            └── b6-production-online-systems.md
```

---

## Contributing

Contributions are welcome — new guides, fixes, and clearer examples. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. All 14 guides are live; good first contributions are fixes, clearer examples, and new cross-track material.

---

## License

[MIT](LICENSE)

---

## Keywords

Reinforcement Learning, RLHF, DPO, GRPO, RLVR, PPO, DQN, Q-Learning, Contextual Bandits, LinUCB, Multi-Armed Bandits, Personalization, LLM Post-Training, Agentic RL, Test-Time Compute, Tool Use, Verifiable Rewards, Python, NumPy