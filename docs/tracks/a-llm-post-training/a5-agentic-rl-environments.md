# A5 — Agentic RL & Environments

> **Track A · Guide 6 of 7** — [Track A Index](index.md) · [RL Methods Guide](../../RL_METHODS_GUIDE.md)

## What You'll Learn

A4 taught a model to get a single answer right. A5 teaches a model to **act** — to take a sequence of steps, call tools, observe results, and recover from its own mistakes until a task is done. This is **agentic RL**, and its key artifact is not the model or the algorithm — it is the **environment**: a packaged, resettable, stateful world with tools and a verifier.

> **TL;DR:** A4 = reward the answer. A5 = reward the trajectory. Same verifier idea, but now the model acts over many turns inside an environment that remembers what happened.

---

## The Concept

```
┌─────────────────────────────────────────────────────────────────┐
│                     AGENTIC RL LOOP                              │
│                                                                  │
│   task ─► AGENT (policy) ──► tool call / action                 │
│                ▲                    │                            │
│                │                    ▼                            │
│           observation ◄──── ENVIRONMENT (stateful sandbox)        │
│                │                    │                            │
│                └──── repeat N turns ┘                            │
│                                     │                            │
│                                     ▼                            │
│                          task done? ──► VERIFIER ──► reward       │
│                                                                  │
│   The model is optimized to produce TRAJECTORIES that succeed.   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why Agentic RL Exists

Single-turn RL (A4) trains a model to produce one good output. But a large and growing share of useful work is not one output — it is a sequence of interactions with the world.

```
SINGLE-TURN (A4)                     AGENTIC (A5)
─────────────────                    ─────────────
one prompt                           a task
one response                         a session
verifier checks the ANSWER           environment + verifier check the OUTCOME
reward = 0/1 per output              reward = task success over a trajectory

"Solve 2x + 3 = 7"                   "Fix the failing test in this repo"
    → produce "x = 2"                    → read files, edit, run tests, iterate
```

Three things made this possible — and necessary:

```
1. TOOLS
   Models can call functions (search, run code, edit files, hit APIs).
   The action space is no longer "next token" — it is
   "which tool, with what arguments", chosen every turn.

2. SANDBOXES
   You can run the model's actions in an isolated, resettable world
   (a container, a browser, a repo checkout) so that failure is cheap
   and repeatable.

3. VERIFIERS THAT SCALE PAST ONE ANSWER
   Unit tests, task-completion checks, environment end-state —
   a signal that says "was the JOB done?", not just "was the sentence right?".
```

The result: models that no longer just *answer* — they *operate*. SWE agents that fix bugs, computer-use agents that click through interfaces, web agents that complete multi-step tasks. What trained them is not a bigger A4 loop. It is an environment.

---

## The Environment Is the Product

This is the sentence that matters most in this guide:

> **In agentic RL, the environment is the hard part — not the model, not the algorithm.**

Everyone uses roughly the same algorithm family now (the GRPO/PPO variants from A4). What differentiates agentic training runs is the environment:

```
┌─────────────────────────────────────────────────────────────────┐
│                AN ENVIRONMENT, AS A PACKAGED UNIT                │
│                                                                  │
│   reset()        → a fresh, reproducible starting state           │
│   step(action)   → apply action, advance state, return observation│
│   observation    → what the agent is allowed to see (not the truth)│
│   tools[]        → the action vocabulary (definitions + arguments)│
│   verifier       → did the task get completed? (the reward)       │
│   sandbox        → isolation + reset, so failures stay cheap      │
│   metadata       → task id, difficulty, success criteria          │
│                                                                  │
│   "Ship-ready" means: deterministic reset, safe isolation,        │
│   an honest verifier, and coverage of the tasks you care about.   │
└─────────────────────────────────────────────────────────────────┘
```

An environment is a piece of software you **build, version, test, and maintain** — with all the rigor that implies. When a team says "we trained an agent," what they usually mean is "we spent most of the time building a good environment."

---

## Multi-Turn Rollouts

A **rollout** is one full episode: the agent's path from `reset()` to task completion (or failure). In A4 a rollout is one generation. In A5 a rollout is many turns, each containing many tokens.

```
ROLLOUT = turn 1 → turn 2 → ... → turn N

turn k:
  agent sees observation o_k
  reasons (chain of thought)
  picks tool + arguments = action a_k
  environment executes a_k → new state, observation o_{k+1}

Task reward usually arrives only at the END:
  R = verifier(final_state)
```

The economics are unforgiving:

```
cost per update ≈ (rollouts per prompt)
                × (turns per rollout)      ← new in A5
                × (tokens per turn)
                × (rollout compute)
```

Every turn multiplies cost. This is why agentic RL is where training budgets go to die — and why **environment efficiency** (how fast a `step` runs, how often a task actually terminates) becomes a first-class training concern, not an implementation detail.

---

## Credit Assignment: The Central Hard Problem

Single-turn RLVR gives the whole output one score. In agentic RL, one terminal reward must be attributed across **dozens of turns and thousands of tokens**. Which action actually earned the win? Which one caused the failure?

```
terminal reward: 1   (task succeeded)
   │
   │  was it turn 1's file read?   turn 4's edit?   turn 9's test run?
   ▼
every token in the trajectory receives this same scalar → noisy signal

Options:
  Outcome reward only    → simple, sparse, high variance
  Process reward model   → denser signal, but you must build/train it
  Group-relative (GRPO)  → baseline from sibling rollouts (carried over from A4)
  Turn-level shaping     → per-step rewards (risk: opens the door to hacking)
```

This is exactly where A5 hands off to **[A6 — Rewards & Test-Time Compute](a6-rewards-and-test-time-compute.md)**: process reward models, validators, and search are the tools that make credit assignment tractable. Note that the GRPO trick from A4 — normalize against the group — carries straight into agentic RL, and is what makes the whole thing feasible without a critic.

---

## Where the Reward Comes From

A5 sits at the **far right** of Track A's reward-source axis — a programmatic verifier — but now the verifier observes **environment state**, not just a string.

```
TRUSTWORTHY / EXPENSIVE  ───────────────────────►  CHEAP / SCALABLE

  Human preferences        AI judge           Programmatic verifier
  ────────────────        ────────           ─────────────────────
   A2 (RLHF)               A3 (pairs)          A4 (RLVR on output)
                           RLAIF               A5 (RLVR on environment)
```

```
Task                  Verifier examines                       Reward
─────────────────────────────────────────────────────────────────────
Fix a bug in a repo   the test suite passes                   pass / fail
Complete a web task   end-state matches goal evidence          0 or 1
Operate a UI          target widget reached / task done       shaped
Write code            unit tests + HELD-OUT tests             pass / fail
```

Two dangers are specific to this stage, and both are about the verifier: it must be **held-out** (the agent must not see or edit the tests), and the environment must be **honest** (no shortcut that reaches the goal state without doing the work).

---

## When to Use Agentic RL

| Use Case | Why |
|----------|-----|
| Tool-using assistants | Multi-turn actions against APIs and tools |
| SWE / coding agents | Tasks verified by running a test suite |
| Computer / web agents | Tasks verified by an end-state check |
| Long-horizon tasks | The ORDER of steps and the ability to RECOVER matter |
| You own an environment | You can package `reset` / `step` / `verify` cleanly |
| You have a task verifier | Automatic, held-out, and hard to game |

---

## Key Concepts Visualized

### Single-Turn vs Multi-Turn

```
A4 — single-turn                    A5 — multi-turn
┌──────────┐                        ┌───────────┐
│ 1 output │──► verifier            │ rollout   │──► environment ──► verifier
└──────────┘    reward 0/1          │  N turns  │    (state)         task reward
                                    └───────────

1 generation per update             N turns × tokens per update
```

### The Environment as a Boundary

```
   CONTROLLED BY US               │         CONTROLLED BY THE AGENT
   ────────────────               │         ──────────────────────────
   reset / step / tools           │         which tool, with what args
   observation policy             │         reasoning + recovery
   verifier + held-out tests      │         (this is what we train)
```

### Anatomy of a Turn

```
  observation o_k
       │
       ▼
  ┌─────────────┐
  │  reasoning  │   chain of thought (tokens)
  └─────┬───────┘
        ▼
  ┌─────────────┐
  │ tool + args │   the ACTION (what gets scored, ultimately)
  └─────┬───────┘
        ▼
  ENVIRONMENT.step(action) → new state, observation o_{k+1}
```

### Failure Modes Visualized

```
REWARD HACKING         the agent satisfies the checker, not the task
VERIFIER GAMING        edits tests, asserts pass, prints expected output
SHORTCUT EXPLOIT       reaches goal state without doing the work
CREDIT NOISE           a terminal reward can't tell which turn mattered
SIM-TO-REAL GAP        great in the sandbox, useless in production
ROLLOUT COST BLOWUP    turns × tokens × rollouts eats the budget
```

---

## Agentic RL vs Single-Turn RLVR

| Aspect | A4 (RLVR) | A5 (Agentic RL) |
|--------|-----------|-----------------|
| Horizon | Single turn | Multi-turn episode |
| Action space | One answer | Tools × arguments, chosen per turn |
| Reward source | Verifier on the output | Verifier + environment state |
| Reward timing | Immediate | Terminal / sparse |
| Credit assignment | Trivial (whole output) | The central hard problem |
| Key artifact | A verifier | An **environment** |
| Cost multiplier | 1× | turns per rollout |
| Analogy | A graded exercise | A simulator for a job |

---

## Key Limitations

```
1. YOU MUST BUILD AN ENVIRONMENT
   reset / step / tools / verifier / sandbox — and keep it honest and
   reproducible. For most teams this is the majority of the work.

2. SPARSE, HIGH-VARIANCE REWARD
   One terminal scalar spread across many turns → noisy gradients.
   Mitigations: process rewards, group baselines (GRPO), turn-level shaping.

3. REWARD HACKING — NOW WITH TOOLS
   With tools in the loop, gaming stops being a scoring artifact and
   becomes an action problem: the agent can ACT on the world to fool
   the checker. Mitigations: held-out verifiers, adversarial checks,
   human spot-checks.

4. COST
   Rollouts × turns × tokens. This is the most expensive stage in Track A.

5. SIM-TO-REAL GAP
   Environments are approximations. A policy tuned to the sandbox may not
   transfer to production tools, latency, and failure modes.

6. NOT FOR NON-VERIFIABLE WORK
   No automatic task-success signal? You are back to A2/A3
   (human preferences or AI judges).
```

Reward hacking deserves the same emphasis it gets in the [Track A index](index.md): **it is not a bug you fix once — it is a permanent adversarial dynamic.** Any time a model is optimized against a fixed checker, it will find the checker's weaknesses. With tools available, that search is far more powerful. Design environments adversarially from day one.

---

## Connecting to Real RL Methods Guide

From the [RL Methods Guide](../../RL_METHODS_GUIDE.md):

> **What:** Multi-turn RL inside a stateful, packaged environment, trained on task success.
> **When:** Tool-using agents, coding agents, computer/web agents — tasks with a verifier.
> **Key advantage:** Teaches action, recovery, and long-horizon behavior.
> **Key limitation:** You must build the environment; reward is sparse; cost is high; hacking is easier once tools are involved.

---

## Real-Life Applications

| System | Method | Result |
|--------|--------|--------|
| SWE agents | RLVR on tests inside a repo sandbox | Agents that fix real bugs |
| Computer-use agents | RL on UI end-state | Agents that operate interfaces |
| Web agents | RL on task completion | Multi-step web task completion |
| Tool-using assistants | RL on tool-call success | Reliable multi-tool behavior |
| Reasoning + tools (R1-style) | GRPO + RLVR + tools | Long-horizon, self-correcting agents |

---

## Where You Fit In

```
YOUR ROLE AS AN APP DEVELOPER:
   Don't build an environment and RL-train an agent yourself
     (unless that IS your product)
  ✅ USE tool-calling / agentic models via API
  ✅ If you build agents, your job is the TOOLS and the ENVIRONMENT
     quality — not the training run
  ✅ Your personalization still lives in Track B (LinUCB + Q-Learning)

YOUR PRACTICAL STACK:
  Base model:      a tool-calling / reasoning model via API
  Agent loop:      your tool definitions + your environment
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

For most chatbot applications: Levels 1b–3 are still the sweet spot.
For alignment/fine-tuning: DPO (Level 7) is the pragmatic default.
For reasoning with verifiers: GRPO + RLVR (A4).
For agents that act: A5 — and the environment, not the model, is the work.
```

---

## Further Reading

- **SWE-bench / SWE-agent** — environments for coding agents
- **Computer-use / web-agent benchmarks** — environments for UI and web tasks
- **ReAct / tool-use agent papers** — the observation → reasoning → action loop
- **GRPO / RLVR surveys** (2025) — reward and credit assignment (shared with A4, A6)

---

**Previous:** [A4 — GRPO & RLVR](a4-grpo-and-rlvr.md) · **Next:** *A6 — Rewards & Test-Time Compute* (planned)

**Up:** [Track A Index](index.md) · **Reference:** [RL Methods Guide](../../RL_METHODS_GUIDE.md)