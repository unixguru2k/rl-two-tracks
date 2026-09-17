# RL Learning Lab — Reinforcement Learning for Agents, LLMs & Systems

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

> **Learn Reinforcement Learning from a random baseline to agentic RL** with hands-on Python examples. The curriculum is split into **two tracks that share vocabulary and almost nothing else** — different reward sources, runtimes, costs, and audiences.
>
> **Track A — Post-Training & Agentic RL:** make the *model* better at reasoning and acting. PPO, RLHF, DPO & RFT, GRPO & RLVR, agentic RL & environments, rewards & test-time compute.
> **Track B — Decision Optimization in Systems:** choose the *action* your system takes, online, per-request. Random baseline, ε-greedy, LinUCB, tabular Q, feature-Q, DQN, production online systems.

> 📌 **New here?** Most product and app engineers belong in **Track B** — start at [B0: Random Baseline](docs/tracks/b-decision-optimization/b0-random-baseline.md) and climb one level at a time. If you have GPUs and an automatic way to grade correctness, start with the [Track A overview](docs/tracks/a-llm-post-training/index.md).

> 🚧 **Status:** Documentation-first. Ten self-contained example guides are live — **Track A: A1–A4**, **Track B: B0–B5** — plus both track overviews. Three guides are planned but **not yet written**: [A5 — Agentic RL & Environments], [A6 — Rewards & Test-Time Compute], [B6 — Production Online Systems]. The interactive web app described in `PLAN.md` is a proposal, not yet built.
>
> 🗓 **Last updated:** 2026-09-17
>
> ✅ **All example guides are self-contained and runnable** — copy any Python block and run it with just NumPy installed.

---

## Table of Contents

- [Why This Repo?](#why-this-repo)
- [What This Repo Is (and Isn't)](#what-this-repo-is-and-isnt)
- [Algorithms Covered](#reinforcement-learning-algorithms-covered)
- [Algorithm Summaries & Use Cases](#algorithm-summaries--use-cases)
- [Quick Start](#quick-start)
- [How to Navigate This Repo](#how-to-navigate-this-repo)
- [Learning Path](#learning-path)
- [Documentation](#documentation)
- [Which RL Algorithm Should You Use?](#which-rl-algorithm-should-you-use)
- [Real-World Applications](#real-world-applications)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Keywords](#keywords)

---

## Why This Repo?

Most RL tutorials use toy games (CartPole, MountainCar). This guide teaches RL through **real systems** — tool-using agents, verifier-graded reasoning, multi-turn rollouts in sandboxes, response strategy selection, conversation flow optimization, and production personalization. Every concept maps directly to something you can ship or reason about.

**Who this is for:**
- **App and backend engineers** adding RL-powered personalization, ranking, routing, or A/B testing — that is Track B
- **Agent builders** working on tool use, multi-turn rollouts, sandboxes, and verifiers — that is Track A
- ML engineers deciding **whether to train** or to **consume** a post-trained model
- Students learning RL with **practical, runnable examples**
- Anyone curious how **ChatGPT, Claude, and reasoning models are actually trained** (RLHF → DPO → GRPO/RLVR → agentic RL)

**The most common mistake this repo exists to prevent:** using a bandit for a sequential problem, or an MDP for an independent one. Track B's overview opens with that decision because it determines which half of the curriculum matters to you.

---

## What This Repo Is (and Isn't)

**What it is:** A documentation-first curriculum, organized as **two tracks**. Ten self-contained Python guides are live, covering everything from a random baseline up to reasoning-model RL (DPO, GRPO & RLVR) and agentic RL — each with a runnable example, a plain-English explanation of the idea, and a real use case. Three further guides are planned (A5, A6, B6).

**What it isn't:** An application or a library. There is nothing to install and no server to run. The interactive web app in `PLAN.md` is a **proposal** — no `backend/`, `frontend/`, or `start-app.sh` exists yet.

**The core idea this repo is built around:**

```
LLM = WHAT to say           (language generation)
RL  = WHICH STRATEGY to use (optimization from outcomes)
```

You don't reinvent RLHF. You **use** its output (GPT-4, Claude via API) and layer lightweight RL personalization on top of it.

---

## Reinforcement Learning Algorithms Covered

| Track | Guide | Algorithm | Type | Key Concept | Difficulty |
|:-----:|:-----:|-----------|------|-------------|:----------:|
| **B** | B0 | **Random Selection** | Baseline | Establishing a benchmark — no learning | ⭐ |
| **B** | B1 | **Epsilon-Greedy MAB** | Multi-Armed Bandit | Exploration vs exploitation tradeoff | ⭐ |
| **B** | B2 | **LinUCB** | Contextual Bandit | Context-aware personalization, automatic exploration | ⭐⭐ |
| **B** | B3 | **Q-Learning (Tabular)** | Model-free RL | Sequential decision making, Q-table, discount factor | ⭐⭐ |
| **B** | B4 | **Feature-Based Q-Learning** | Linear Approximation | Scaling to large state spaces with feature vectors | ⭐⭐⭐ |
| **B** | B5 | **DQN (Deep Q-Network)** | Deep RL | Neural networks, replay buffer, target network | ⭐⭐⭐⭐ |
| **B** | B6 | *Production Online Systems* | Deployment | Drift, delayed rewards, exploration budget, safety rails | ⭐⭐⭐ *(planned)* |
| **A** | A1 | **PPO (Proximal Policy Optimization)** | Policy Gradient | Direct policy optimization, continuous actions | ⭐⭐⭐⭐⭐ |
| **A** | A2 | **RLHF (Reinforcement Learning from Human Feedback)** | LLM Alignment | Reward modeling, preference learning, ChatGPT/Claude training | ⭐⭐⭐⭐⭐ |
| **A** | A3 | **DPO & RFT** | LLM Alignment | Preference optimization — no reward model, no RL loop | ⭐⭐⭐⭐ |
| **A** | A4 | **GRPO & RLVR** | Reasoning RL | Group-relative advantages, verifiable rewards, DeepSeek-R1/o1-style reasoning | ⭐⭐⭐⭐⭐ |
| **A** | A5 | *Agentic RL & Environments* | Agent RL | Multi-turn rollouts, sandboxes, tool use; the environment as a packaged unit | ⭐⭐⭐⭐ *(planned)* |
| **A** | A6 | *Rewards & Test-Time Compute* | Reward Design | Outcome vs process rewards, generative verifiers, search & reranking | ⭐⭐⭐⭐ *(planned)* |

---

## Algorithm Summaries & Use Cases

Each algorithm solves a **different shape of problem**. This table is the fastest way to find the one you need.

| Guide | Algorithm | What It Does | Reach For It When | Example Use Case |
|:-----:|-----------|--------------|-------------------|------------------|
| B0 | **Random Selection** | Picks actions at random — no memory, no learning | Only as a baseline to measure whether learning helps at all | Measure the floor: "how good is doing nothing smart?" |
| B1 | **Epsilon-Greedy MAB** | Averages reward per action, mostly picks the best, explores ε% of the time | Few actions (2–20) and context doesn't matter | Find the best **global** response style for all users |
| B2 | **LinUCB** | Scores each action from context features plus an automatic exploration bonus | Few actions + context features (mood, time, history) matter | Pick the right response style for **this** user at **this** moment |
| B3 | **Q-Learning (Tabular)** | Scores every (state, action) pair and plans ahead via a discount factor | Discrete states, discrete actions, and the **order** of actions matters | Conversation flow: greeting → ask → troubleshoot → close |
| B4 | **Feature-Based Q-Learning** | Q-learning over a weight vector of features instead of a lookup table | Sequential decisions where the state space is too large for a table | Manage flow across hundreds of context combinations |
| B5 | **DQN** | Neural network approximates Q-values, stabilized by replay buffer + target network | High-dimensional or non-linear state (text embeddings, images) | Learn from raw conversation text, not just categories |
| B6 | *Production Online Systems* | The serving loop around any Track B policy: logging, drift, safety | You are about to serve a learned policy to real traffic | Ship B0–B5 to users without degrading over time |
| A1 | **PPO** | Learns the policy (action probabilities) directly, with clipped updates | Continuous action spaces or token-level LLM optimization | Fine-tune generation — only if you're training your own model |
| A2 | **RLHF** | Trains a reward model from human preferences, then optimizes an LLM with PPO | You are training or fine-tuning a foundation model | You **use** RLHF'd models (GPT-4, Claude) — you don't build them |
| A3 | **DPO & RFT** | Optimizes the policy directly on preference pairs — the reward model cancels out; RFT is generate → filter → fine-tune | You have preference pairs and want alignment without a reward model or RL loop | Fine-tune your own small model on your preference data (cheaper than RLHF) |
| A4 | **GRPO & RLVR** | PPO without a value network (group-relative advantages) + rewards from a programmatic verifier | Reasoning/math/code tasks with automatic correctness checks | Use reasoning models (DeepSeek-R1, o-series) via API — don't train your own |
| A5 | *Agentic RL & Environments* | Multi-turn rollouts against a packaged environment: task → harness → sandbox → verifier | Your task spans many turns, uses tools, and can be machine-checked | Train tool-using and computer-use agents |
| A6 | *Rewards & Test-Time Compute* | Outcome vs process rewards, generative verifiers, search and reranking at inference | You need better answers without (or before) more training | Squeeze more from a fixed model via verifier-guided search |

### When NOT To Use Each

| Algorithm | Don't use when |
|-----------|----------------|
| Random | You have any data at all — even 10 interactions beat random |
| Epsilon-Greedy MAB | Context matters — you'll get one answer for everyone |
| LinUCB | Actions have sequential dependencies — it can't plan ahead |
| Q-Table | State space is huge (10K+ combinations) or continuous |
| Feature-Based Q | Features don't linearly predict reward |
| DQN | You have fewer than ~10K training samples — it overfits |
| PPO | You have discrete actions — Q-learning is simpler and works |
| RLHF | You're not training a foundation model — use a pre-trained one |
| DPO | You need online exploration — DPO is offline and can't beat its data |
| GRPO / RLVR | You have no programmatic verifier (open-ended creativity/empathy) |
| Agentic RL (A5) | The task is single-turn, or you can't sandbox it, or the reward can't be checked by machine |
| Test-time compute (A6) | You need a *training* gain — search and reranking only help at inference |

### The Practical Takeaway

**Two audiences, two sweet spots.** Pick the one that matches you:

| You are… | Sweet spot | Stack |
|----------|-----------|-------|
| An **app/product engineer** | **B0 → B2** (then B3–B4 if the journey is sequential) | LinUCB for strategy selection, Q-learning for flow, LLM via API for generation |
| An **agent builder** | **A3 → A4 → A5** | SFT → light preference optimization → RLVR with a verifier, in a sandboxed environment |

| Component (Track B) | Method | Why |
|---------------------|--------|-----|
| Strategy selection | **LinUCB (B2)** | Context-dependent, few actions, fully online |
| Dialog flow | **Q-Learning (B3)** | Sequential decisions matter |
| Large context spaces | **Feature Q (B4)** | When tabular Q-learning breaks down |
| Response generation | **LLM via API** | Already post-trained — you use it, you don't build it |

**The rule that applies to both tracks:** always run the level below your target as a baseline. B0 is not a warm-up — it is the control condition.

---

## Quick Start

**Prerequisites:** Python 3.9+ and NumPy. No GPU, no framework installs, no API keys required.

```bash
git clone <your-repo-url>
cd rl-testing
```

Each guide contains a **self-contained Python example** — no GPU or external dependencies required. Just `numpy` and the standard library.

```bash
# Example: Run the Epsilon-Greedy Bandit
# 1. Open docs/tracks/b-decision-optimization/b1-epsilon-greedy-bandit.md
# 2. Copy the Python code block
# 3. Save as bandit.py and run:
python bandit.py
```

---

## How to Navigate This Repo

- **New to RL?** Start at [B0](docs/tracks/b-decision-optimization/b0-random-baseline.md) and work up sequentially through Track B.
- **Want the theory?** Read the [Complete RL Methods Guide](docs/RL_METHODS_GUIDE.md).
- **Know what you want to build?** Jump straight to the [decision flowchart](#which-rl-algorithm-should-you-use).

---

## Learning Path

Follow this progression from fundamentals to advanced LLM alignment:

```
TRACK B — Decision Optimization in Systems (online, per-request, CPU)
  B0: Random Baseline            → The control condition — why learning matters at all
  B1: Epsilon-Greedy Bandit      → Exploration vs exploitation, no context
  B2: LinUCB (Contextual)        → Personalization with user context ✅
  B3: Q-Learning (Tabular)       → Sequential decisions, Q-tables
  B4: Feature-Based Q-Learning   → Scale to thousands of states
  B5: DQN                        → Non-linear pattern learning with neural nets
  B6: Production Online Systems  → Drift, delayed rewards, safety rails

TRACK A — Post-Training & Agentic RL (offline batch, GPU, weights)
  A1: PPO                        → Direct policy optimization
  A2: RLHF                       → How ChatGPT & Claude are actually trained
  A3: DPO & RFT                  → Alignment with no reward model, no RL loop
  A4: GRPO & RLVR                → Reasoning models (DeepSeek-R1, o1-style)
  A5: Agentic RL & Environments  → Multi-turn rollouts, tools, sandboxes
  A6: Rewards & Test-Time Compute→ Outcome vs process rewards, search at inference
```

> **For an app or product engineer:** B0 → B2 is the sweet spot, then B3–B4 if the journey is sequential. Use pre-trained LLMs via API for generation and add RL-driven personalization on top.
>
> **For an agent builder:** A3 → A4 → A5. SFT, light preference optimization, then RLVR against a verifier inside a sandboxed environment.

---

## Documentation

### Core Reference

- **[Complete RL Methods Guide](docs/RL_METHODS_GUIDE.md)** — Decision flowcharts, comparison tables, when to use each method, MACHAAO API storage strategy, and OpenRouter integration patterns.

### Hands-On Example Guides

**Track A — Post-Training & Agentic RL** ([track overview](docs/tracks/a-llm-post-training/index.md))

| Guide | What You'll Build | Use Case |
|-------|-------------------|----------|
| [A1 — Policy Gradient / PPO](docs/tracks/a-llm-post-training/a1-policy-gradient-ppo.md) | Policy gradient agent | Continuous action control |
| [A2 — RLHF](docs/tracks/a-llm-post-training/a2-rlhf.md) | Reward model + PPO pipeline | LLM alignment (theory) |
| [A3 — DPO & RFT](docs/tracks/a-llm-post-training/a3-dpo-and-rft.md) | Direct preference optimizer | Alignment without a reward model |
| [A4 — GRPO & RLVR](docs/tracks/a-llm-post-training/a4-grpo-and-rlvr.md) | Group-relative + verifier rewards | Reasoning models (DeepSeek-R1, o1) |
| [A5 — Agentic RL & Environments](docs/tracks/a-llm-post-training/a5-agentic-rl-environments.md) | Multi-turn tool-use rollouts in a sandbox | Training tool-using and computer-use agents |
| [A6 — Rewards & Test-Time Compute](docs/tracks/a-llm-post-training/a6-rewards-and-test-time-compute.md) | Verifier-guided search and reranking | More capability without more training |

**Track B — Decision Optimization in Systems** ([track overview](docs/tracks/b-decision-optimization/index.md))

| Guide | What You'll Build | Use Case |
|-------|-------------------|----------|
| [B0 — Random Baseline](docs/tracks/b-decision-optimization/b0-random-baseline.md) | Baseline reward tracker | A/B testing benchmark |
| [B1 — Epsilon-Greedy MAB](docs/tracks/b-decision-optimization/b1-epsilon-greedy-bandit.md) | Online bandit learner | Best global response style |
| [B2 — LinUCB (Contextual Bandit)](docs/tracks/b-decision-optimization/b2-contextual-bandit-linucb.md) | Context-aware selector | Personalized recommendations |
| [B3 — Q-Learning (Tabular)](docs/tracks/b-decision-optimization/b3-q-learning-tabular.md) | Conversation flow optimizer | Customer support routing |
| [B4 — Feature-Based Q-Learning](docs/tracks/b-decision-optimization/b4-feature-based-q-learning.md) | Linear approximation agent | Large-scale personalization |
| [B5 — DQN](docs/tracks/b-decision-optimization/b5-dqn.md) | Neural network Q-agent | High-dimensional state spaces |
| [B6 — Production Online Systems](docs/tracks/b-decision-optimization/b6-production-online-systems.md) | Serving loop: logging, drift, safety | Shipping any B0–B5 policy to real users |

---

## Which RL Algorithm Should You Use?

```
START: What are you optimizing?
  │
  ├── "Which single response style works best?"
  │    ├── Same for all users? ──────────────► Epsilon-Greedy MAB (Level 1a)
  │    └── Depends on context (mood, time)? ─► LinUCB (Level 1b) ✅
  │
  ├── "What's the best conversation FLOW?"
  │    ├── Small discrete states (< 500)? ───► Q-Table (Level 2)
  │    ├── Many states with features? ───────► Feature-Based Q (Level 3)
  │    └── Raw text as state? ───────────────► DQN (Level 4)
  │
  ├── "Control response at token level?" ────► PPO (Level 5)
  │
  └── "Train my own LLM?"
       ├── Have preference pairs, offline? ──► DPO (Level 7) ✅
       ├── Verifiable task (math/code/logic)? ► GRPO + RLVR (Level 8)
       └── Classic online RL + reward model? ► RLHF (Level 6)
```

---

## Real-World Applications

Each algorithm maps to production use cases:

| Algorithm | Chatbot Application | General Application |
|-----------|---------------------|---------------------|
| **MAB / LinUCB** | Response style personalization | Email subject optimization, ad targeting |
| **Q-Learning** | Dialog policy (conversation flow) | Customer support routing, onboarding wizards |
| **Feature-Based Q** | Context-aware dialog management | Dynamic pricing, hospital patient flow |
| **DQN** | Semantic conversation understanding | Game AI, self-driving, robotics |
| **PPO** | LLM output optimization | Trading bots, music generation |
| **RLHF** | ChatGPT / Claude alignment | Foundation model training |
| **DPO** | Cheap preference alignment of your own model | Open-weight fine-tuning (Mistral, Llama, Zephyr) |
| **GRPO & RLVR** | Reasoning via API (DeepSeek-R1, o-series) | Math/code agents, verifiable-reward training |

---

## Tech Stack

- **Language:** Python 3.9+
- **Dependencies:** NumPy (all examples are dependency-light)
- **No GPU required** — all examples run on CPU
- **No framework lock-in** — pure Python with NumPy for transparency

---

## Project Structure

```
rl-testing/
├── README.md                              ← You are here
├── PLAN.md                                ← Web app architecture plan
├── docs/
│   ├── RL_METHODS_GUIDE.md               ← Complete algorithm reference
│   └── tracks/
│       ├── a-llm-post-training/
│       │   ├── index.md                  ← Track A overview
│       │   ├── a1-policy-gradient-ppo.md
│       │   ├── a2-rlhf.md
│       │   ├── a3-dpo-and-rft.md
│       │   └── a4-grpo-and-rlvr.md
│       └── b-decision-optimization/
│           ├── index.md                  ← Track B overview
│           ├── b0-random-baseline.md
│           ├── b1-epsilon-greedy-bandit.md
│           ├── b2-contextual-bandit-linucb.md
│           ├── b3-q-learning-tabular.md
│           ├── b4-feature-based-q-learning.md
│           └── b5-dqn.md
```

---

## Contributing

Contributions are welcome! Whether it's fixing a typo, adding a new RL algorithm, or improving examples:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-improvement`
3. Make your changes
4. Commit: `git commit -m 'Add feature'`
5. Push: `git push origin feature/my-improvement`
6. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Keywords

`reinforcement-learning` · `q-learning` · `deep-q-network` · `dqn` · `ppo` · `proximal-policy-optimization` · `rlhf` · `reinforcement-learning-from-human-feedback` · `dpo` · `direct-preference-optimization` · `grpo` · `rlvr` · `verifiable-rewards` · `reasoning-models` · `deepseek-r1` · `multi-armed-bandit` · `contextual-bandit` · `linucb` · `epsilon-greedy` · `chatbot` · `personalization` · `recommendation-system` · `ab-testing` · `policy-gradient` · `reward-modeling` · `exploration-exploitation` · `dpo` · `direct-preference-optimization` · `grpo` · `rlvr` · `verifiable-rewards` · `reasoning-models` · `deepseek-r1` · `python` · `machine-learning` · `ai-alignment` · `conversational-ai` · `agentic-rl` · `rlvr` · `grpo` · `dapo` · `gspo` · `process-reward-model` · `test-time-compute` · `simpo` · `orpo` · `kto` · `raft` · `rejection-sampling` · `tool-use` · `environments` · `sandbox` · `verifier` · `credit-assignment`

---

**Built with ❤ using [MACH-AI](https://machai.live)**