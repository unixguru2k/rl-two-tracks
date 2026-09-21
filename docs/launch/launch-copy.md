# Launch Copy

Ready-to-paste copy for announcing **RL Two Tracks**. Two platforms, two containers — one idea.

> **The line that does the work:** *"Production RL isn't RLHF. It's a contextual bandit in front of a frozen model — a pattern with real deployments and almost no tutorials."*
>
> That sentence is the whole pitch. Everything below is packaging around it.

See [`claims.md`](claims.md) for what is sourced and why the original phrasing was narrowed.

---

## LinkedIn — Main Post

Most "learn RL" content sends app engineers down the wrong path.

It opens with CartPole. It ends with RLHF. And somewhere in the middle, a backend engineer who just wants to pick the better response style for each user gives up, because none of it maps to the code they actually ship.

So I wrote the thing I wished existed: an RL curriculum split into two tracks, because these are two different jobs.

**Track A — Post-Training & Agentic RL.** You optimize the model. Weights, offline, GPU-hours. RLHF, DPO, GRPO, reasoning models, tool-using agents. Unless you own a model, a verifier, and GPUs, you're a consumer of this track, not a builder. That's not a limitation — it's how the ecosystem is supposed to work.

**Track B — Decision Optimization.** You optimize which action your system takes. Your code, online, per-request, on a CPU. Contextual bandits, Q-learning, production guardrails. No GPU, no reward model, no preference data — just a measurable outcome and one key-value store.

The line that separates them:

> Track A optimizes the model that generates the action.
> Track B optimizes which action your system takes.

Both are "RL." They share vocabulary and almost nothing else.

The part I think is most useful: production RL doesn't look like RLHF. It looks like a contextual bandit in front of a frozen model — a pattern with real deployments and almost no tutorials. That's the version that pays rent, and it's the one nobody writes about.

Every guide is self-contained and runs on NumPy. No framework, no API key. One exception — the hands-on post-training guide needs a GPU, and free Colab is enough.

Link in the comments.

---

## LinkedIn — Short Version

Everyone teaching RL to engineers leads with CartPole and ends with RLHF. Neither is what a product engineer needs.

So I split it into two tracks:

**A — Post-Training & Agentic RL:** you optimize the model. Weights, GPU-hours, RLHF/DPO/GRPO. Read it; don't build it unless you own the model.

**B — Decision Optimization:** you optimize which action your system takes. Your code, per-request, on a CPU. Bandits and Q-learning. No GPU, no reward model.

Production RL is B — a contextual bandit in front of a frozen model. It just never gets a tutorial.

Two tracks. NumPy only. Link in the comments.

---

## LinkedIn — First Comment (holds the link)

Repo here: https://github.com/unixguru2k/rl-two-tracks

Start at the track overview, not the master guide — it tells you which half you actually need in about 90 seconds.

---

## X — Thread

**1/**
Production RL isn't RLHF.

It's a contextual bandit in front of a frozen model — updating a few numbers per request on a CPU.

It has real deployments and almost no tutorials. So I wrote two.

---

**2/**
Everyone teaching RL to engineers opens with CartPole and ends with RLHF.

Neither is what a product engineer needs.

So the repo is split into two tracks — because "RL" means two different jobs that share vocabulary and almost nothing else.

---

**3/**
Track A — Post-Training & Agentic RL

You optimize the MODEL.

Weights. Offline. GPU-hours. RLHF, DPO, GRPO, reasoning, tool-using agents.

Unless you own a model, a verifier, and GPUs, you *consume* this track. That's the intended design.

---

**4/**
Track B — Decision Optimization

You optimize WHICH ACTION your system takes.

Your code. Online. Per-request. CPU. Milliseconds. Contextual bandits, Q-learning, production guardrails.

No GPU. No reward model. No preference data.

---

**5/**
The line that separates them:

Track A optimizes the model that generates the action.
Track B optimizes which action your system takes.

Both are "RL." One costs GPU-hours. The other costs a store read.

---

**6/**
The most common mistake this repo exists to prevent:

Using a bandit for a sequential problem. Or an MDP for an independent one.

Do the ORDER of your decisions matter?
No → bandits. Yes → MDP.

That one question routes you to half the curriculum.

---

**7/**
The most valuable rule in Track B: always run the level below your target as a baseline.

B0 (random) isn't a warm-up. It's the control that tells you whether your careful algorithm is doing anything at all.

Most people skip it. Most people can't prove their RL works.

---

**8/**
Same loop shape, opposite economics:

Track A → weights, GPU-hours, weeks
Track B → a few numbers, CPU, microseconds

Most teams need the second one and are busy reading about the first.

---

**9/**
What's in it:

All 14 guides live. Every one self-contained and NumPy-only — except the hands-on post-training guide, which needs a GPU (free Colab is enough).

No framework. No API key. Copy-paste and run.

---

**10/**
Two tracks. One routing table up top so you read one, not both.

Start with the track overview — it tells you which half you actually need in about 90 seconds.

Link below 🧵

---

## X — Reply to Tweet 10 (holds the link)

https://github.com/unixguru2k/rl-two-tracks

Start at the track overview, not the master guide. 90 seconds to find out which half is yours.

---

## X — Alt Hooks

**Single tweet:**

> Production RL isn't RLHF. It's a contextual bandit in front of a frozen model — updating a few numbers per request on a CPU.
>
> So I wrote two tracks: one for optimizing the model (GPU-hours), one for optimizing which action your system takes (milliseconds).
>
> Two tracks. NumPy only. Link below.

**Lead with the controversy:**

> "RL" is two different jobs and almost every tutorial pretends it's one.
>
> One optimizes the model that generates the action. The other optimizes which action your system takes.
>
> Different reward sources. Different runtimes. Different people. 🧵

**Lead with the cost:**

> The RL you need probably runs on a CPU, in under a millisecond, and stores about twenty numbers.
>
> Almost nothing written about RL covers that version.

---

## Practical Notes

- **Link placement:** both platforms suppress reach for links in the main post. Post clean, put the repo URL in the first comment (LinkedIn) or a reply (X).
- **Container per platform:** LinkedIn gets **one block** — its feed shows far more text before truncation and rewards dwell on a single post. X gets a **thread** — multiple entry points, screenshot-able tweets, and a character limit that a single argument can't fit. Posting the LinkedIn block to X (or vice versa) underperforms.
- **Stagger them.** Don't cross-post on the same day — it splits your own audience.
- **Visual that works:** a two-column contrast — left "Track A — model / weights / GPU-hours / offline," right "Track B — action / your code / CPU / per-request." Attach it to X tweet 5 and to the LinkedIn main post. The bandit-vs-MDP fork diagram also works well on X tweet 6.
- **Hashtags:** LinkedIn 3–5 (#ReinforcementLearning #MachineLearning #LLM #SoftwareEngineering #MLOps). X 0–2, on the final tweet only.
- **Repo URL is filled in** (`unixguru2k/rl-two-tracks`) — both in the README Quick Start and in the two link-holding spots above. Re-check if the repo is ever renamed.
- **Strongest standalone line, if you need one:** *"Production RL isn't RLHF. It's a contextual bandit in front of a frozen model."*
