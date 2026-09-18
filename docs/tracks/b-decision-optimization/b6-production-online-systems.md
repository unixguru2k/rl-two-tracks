# B6 — Production Online Systems

> **Track B · Guide 7 of 7** — [Track B Index](index.md) · [RL Methods Guide](../../RL_METHODS_GUIDE.md)

## What You'll Learn

B0–B5 gave you the algorithms. B6 is about **running one in front of real users**, where a notebook's assumptions quietly break. Cold start, drifting tastes, a hard latency budget, exploration that can hurt a real person, and a counterfactual you never get to observe. This is the gap between *"it learns in my notebook"* and *"it helps in production."*

> **TL;DR:** An online learner is a **live system**, not a model. Its hardest problems are not the algorithm — they are cold start, non-stationarity, safety, and evaluating decisions you never got to see.

---

## The Concept

```
┌─────────────────────────────────────────────────────────────────┐
│                   THE ONLINE SERVING LOOP                       │
│                                                                 │
│   request ─► context ─► POLICY ─► guardrail ─► serve            │
│                           │           │          │              │
│                           │           │          ▼              │
│                           │           │     user acts           │
│                           ▼           ▼          │              │
│                    log decision   exposure       ▼              │
│                    + propensity     caps    OBSERVE OUTCOME     │
│                                                  │              │
│                                                  ▼              │
│                                       UPDATE STATE (O(1))       │
│                                                                 │
│   Everything above must fit inside ONE request's latency budget.│
└─────────────────────────────────────────────────────────────────┘
```

Compare this to [Track A's loop](../a-llm-post-training/index.md#the-core-loop). Same shape, opposite economics: Track A updates *weights* in GPU-hours; Track B updates *a few numbers* in microseconds.

### The Latency Budget

```
ONE REQUEST = a fixed budget (say 200 ms end-to-end)

  build context        ~5 ms    (user profile, recent events — already in hand)
  load policy state    ~10 ms   (ONE O(1) read: /app-data/{key})
  score actions        ~1 ms    (a few dot products)
  guardrail check      ~1 ms
  serve + log          ~3 ms
  ────────────────────────────
  the rest of the budget belongs to your application

RULES THAT FALL OUT OF THIS:
  • Policy state is ONE key-value read — never a scan, never a query.
  • No training in the request path — only O(1) state updates.
  • Everything the policy needs must be in the context you already have.
  • If the store read is slow, degrade to the baseline; never block the request.
```

### Cold Start: The Empty-Table Problem

```
NEW USER, EMPTY TABLE:

  per-user-only policy            hybrid (global + personal)
  ┌───────────────────────┐       ┌───────────────────────┐
  │ all Q-values = 0      │       │ global prior, learned │
  │ picks ~randomly       │       │   from everyone else  │
  │ bad for ~20 messages  │       │ + personal, ramping   │
  └───────────────────────┘       └───────────────────────┘

  w_effective = w_global + trust(n) · w_personal
  trust(n)    = min(1, n / warmup)

  n = 0      → pure crowd wisdom
  n = warmup → personal weights dominate
```

A new user is not a stranger if the shared prior already knows what generally works. That is the entire fix — and it is why B2 introduced global weights in the first place.

### Non-Stationarity: The World Moves

```
USER TASTES DRIFT. DATA FROM SIX MONTHS AGO MISLEADS YOU.

  update with NO forgetting:
     every past observation counts equally
     → the policy averages over a user who no longer exists

  update with EXPONENTIAL DECAY (λ per update):
     A ← λ·A + x·xᵀ      b ← λ·b + r·x
     half-life = ln(2) / (−ln λ)
       λ = 0.99  → half-life ≈ 69 updates
       λ = 0.999 → half-life ≈ 693 updates

  Too fast → forgets real signal (fits noise)
  Too slow → lags real drift
  Tune λ to the timescale your users actually change on.
```

### Safety: Exploration Can Hurt a Real Person

```
GUARDRAILS, NOT JUST ALGORITHMS:

  reward floor      never serve an arm predicted below a floor
  fallback arm      when unsure, serve the hand-tuned default
  exposure caps     limit how much traffic a new arm can see per day
  kill switch       a config flag that forces the baseline, instantly
  canary            run a new policy on 1% of traffic before 100%
  monotonic guards  never serve an arm that violates a hard business rule
```

### Evaluation: You Never See the Counterfactual

```
THE FUNDAMENTAL PROBLEM: you only observe the arm you SERVED.

  You served "empathetic" → the user replied.
  You did NOT see what "playful" would have done.
  That missing branch is the COUNTERFACTUAL — and it is gone forever.

  1. LOG PROPENSITY      P(arm served | context). Without it, nothing below works.
  2. A/B / interleaving  the gold standard. Costs traffic, gives clean proof.
  3. Off-policy eval     reweight logged data by 1/propensity (IPS/DR) to
                         estimate a policy you never actually ran.
  4. Reward proxies      a reply is not satisfaction. Pick proxies that cannot
                         be gamed and are not observed too late.
```

---

## Python Example

A LinUCB learner (from B2) hardened for serving: cold-start blending, drift decay, a safety rail, and a decision log.

```python
import numpy as np

ACTIONS = ["empathetic", "playful", "distract", "validate", "question"]
N_FEATURES = 4  # [bias, mood, engagement, time_of_day]


class ProductionBandit:
    """A LinUCB contextual bandit hardened for online serving.

    Four things a notebook version lacks:
      1. Cold start   — blend SHARED global weights with per-user weights
      2. Drift        — exponentially forget stale observations
      3. Safety rail  — never serve an arm predicted below the reward floor
      4. Decision log — record every serve for off-policy evaluation
    """

    def __init__(self, actions, n_features, alpha=1.0, decay=0.99,
                 warmup=20, reward_floor=-0.5, use_global=True):
        self.actions = list(actions)
        self.n_features = n_features
        self.alpha = alpha                # exploration bonus
        self.decay = decay                # forgetting factor (non-stationarity)
        self.warmup = warmup              # obs before personal weights count fully
        self.reward_floor = reward_floor
        self.use_global = use_global
        self.fallback = self.actions[0]

        # GLOBAL (shared) learner — the cold-start prior for everyone
        self.gA = {a: np.eye(n_features) for a in self.actions}
        self.gb = {a: np.zeros(n_features) for a in self.actions}

        # PERSONAL (per-user) learner
        self.pA = {a: np.eye(n_features) for a in self.actions}
        self.pb = {a: np.zeros(n_features) for a in self.actions}
        self.pn = {a: 0 for a in self.actions}

        # In production this is written ASYNC to /content, never in the hot path
        self.decisions = []

    def _weights(self, action):
        """Blend global + personal weights; personal ramps in with data."""
        if self.use_global:
            global_w = np.linalg.solve(self.gA[action], self.gb[action])
        else:
            global_w = np.zeros(self.n_features)
        personal_w = np.linalg.solve(self.pA[action], self.pb[action])
        trust = min(1.0, self.pn[action] / self.warmup)
        return global_w + trust * personal_w

    def select(self, x):
        """Score every arm, apply the safety rail, log the decision."""
        predicted, scores = {}, {}
        for action in self.actions:
            predicted[action] = float(self._weights(action) @ x)
            a_inv = np.linalg.inv(self.pA[action])
            bonus = self.alpha * np.sqrt(max(0.0, float(x @ a_inv @ x)))
            scores[action] = predicted[action] + bonus

        action = max(scores, key=scores.get)

        # SAFETY RAIL: if the best arm is predicted to be bad, serve the default
        if predicted[action] < self.reward_floor:
            action = self.fallback

        self.decisions.append((action, dict(scores)))
        return action

    def update(self, action, x, reward):
        """O(1) state update — the only thing allowed in the request path."""
        for a in self.actions:                    # 1) forget stale evidence
            self.gA[a] *= self.decay
            self.gb[a] *= self.decay
            self.pA[a] *= self.decay
            self.pb[a] *= self.decay
        outer = np.outer(x, x)                    # 2) fold in new observation
        self.gA[action] += outer
        self.gb[action] += reward * x
        self.pA[action] += outer
        self.pb[action] += reward * x
        self.pn[action] += 1


def simulate(n_users=20, n_steps=500, drift_at=250, seed=7):
    """Compare three policies against users whose tastes DRIFT mid-run."""
    rng = np.random.default_rng(seed)

    # Each user has their own true weights (unknown to the bandit)
    true_w = rng.normal(0, 1, size=(n_users, len(ACTIONS), N_FEATURES))

    def context():
        return np.array([1.0,
                         rng.uniform(-1, 1),   # mood
                         rng.uniform(0, 1),    # engagement
                         rng.uniform(0, 1)])   # time of day

    policies = {
        "random":     None,   # the B0 control
        "personal":   ProductionBandit(ACTIONS, N_FEATURES, use_global=False),
        "production": ProductionBandit(ACTIONS, N_FEATURES, use_global=True),
    }
    totals = {name: 0.0 for name in policies}
    history = {name: [] for name in policies}

    for step in range(n_steps):
        if step == drift_at:
            true_w *= -1.0            # tastes flip → the world is non-stationary

        user = int(rng.integers(n_users))
        x = context()
        noise = rng.normal(0, 0.5)    # shared noise → a fair comparison

        for name, policy in policies.items():
            if policy is None:
                action = ACTIONS[int(rng.integers(len(ACTIONS)))]
            else:
                action = policy.select(x)

            reward = float(true_w[user, ACTIONS.index(action)] @ x) + noise
            totals[name] += reward
            history[name].append(reward)

            if policy is not None:
                policy.update(action, x, reward)

    return totals, history, drift_at, n_steps


totals, history, drift_at, n_steps = simulate()

print("=== CUMULATIVE REWARD (higher is better) ===")
for name, total in totals.items():
    print(f"  {name:<11} total={total:8.1f}   avg/step={total / n_steps:+.3f}")

print("\n=== COLD START  ·  DRIFT  ·  STEADY STATE ===")
for name, h in history.items():
    h = np.asarray(h)
    early = h[:100].mean()
    post = h[drift_at:drift_at + 100].mean()
    late = h[-100:].mean()
    print(f"  {name:<11} first100={early:+.3f}   post-drift={post:+.3f}   last100={late:+.3f}")

# Expected: production > personal > random overall.
# "personal" pays the cold-start tax (no shared prior) and never catches up.
# Both learners dip after the drift, then recover — the world moved.
```

---

## When to Use Production Online Systems

| Signal | Why B6 matters |
|--------|----------------|
| Real users, policy in the request path | Latency budget and safety rails |
| New users are a large share of traffic | Cold start is your biggest tax |
| Preferences change over months | Non-stationarity needs forgetting |
| A bad decision has real cost | Guardrails, not just algorithms |
| You must prove the policy helped | Off-policy evaluation and holdouts |
| You are deciding whether to keep learning online | Fallback discipline, kill switch |

---

## Key Concepts Visualized

### The Rollout Ladder

```
  SHADOW  ──►  CANARY  ──►   A/B   ──►  FULL
  log only     1% traffic   50/50      100%
  no user      small blast  clean      default
  impact       radius       proof      path
```

Never move a new policy from "written" straight to "100% of traffic."

### The Counterfactual Gap

```
  what happened:        serve "empathetic" ──► reply ──► reward 1
  what you wanted:      what would "playful" have scored?
                                    ▲
                                    └── unobservable. Forever.

  Propensity logging is the only thing that lets you ESTIMATE the gap later.
```

### Decay vs. No Decay

```
  NO DECAY (λ = 1.0)              DECAY (λ = 0.99)
  ┌──────────────────────┐        ┌──────────────────────┐
  │ user drifts ────────►│        │ user drifts ────────►│
  │ policy still averages│        │ policy re-learns in  │
  │ over the OLD user    │        │ ~69 updates (half-life)│
  └──────────────────────┘        └──────────────────────┘
```

### Where the State Lives

```
  MUTABLE LEARNING STATE          APPEND-ONLY EVENT LOG
  (read every request)            (written async, read offline)
  ┌──────────────────────┐        ┌──────────────────────┐
  │ /app-data/{key}      │        │ /content             │
  │ Q-table, weights     │        │ reward events,       │
  │ O(1) GET / PUT       │        │ interactions,        │
  │                      │        │ decisions + propensity│
  └──────────────────────┘        └──────────────────────┘

  Keep them SEPARATE. Mixing them means either slow reads or no history.
```

---

## Key Limitations

```
1. NO FREE LUNCH ON EVALUATION
   Off-policy estimates are high-variance and need logged propensities.
   The only clean proof is a holdout — and a holdout costs traffic.

2. LATENCY IS A HARD CONSTRAINT
   A policy that needs a second store read or a model call is not online.
   Keep the request-path read count at exactly one.

3. SAFETY IS A PRODUCT DECISION, NOT A MATH ONE
   A reward floor and a kill switch encode your risk appetite.
   They are set with the people who own the user experience.

4. DRIFT TUNING IS EMPIRICAL
   λ and warmup have no correct value. You find them by watching
   production, not by reading a paper.

5. FEEDBACK LOOPS
   Logging only what you served biases future data toward what you
   already chose. Exploration is what keeps the data honest.

6. COLD START IS NEVER FULLY SOLVED
   The global prior helps; it does not make a first impression good.
   Route brand-new users to a hand-tuned default, not to the learner.
```

---

## Connecting to Real RL Methods Guide

From the [RL Methods Guide](../../RL_METHODS_GUIDE.md):

> **Progression Path**
> MONTH 1: Classic MAB → MONTH 2: LinUCB → MONTH 4: Q-Learning + LinUCB → ...

B6 is the layer that makes that progression survivable in production. The algorithms live in B0–B5; the constraints live here. The master guide's storage appendix maps directly onto the serving loop:

| Data | Endpoint | In the request path? |
|------|----------|----------------------|
| Policy state (Q-table, weights) | `/app-data/{key}` | ✅ one O(1) read |
| Reward events, interaction history | `/content` | ❌ written async, read offline |

That split — **mutable state in app-data, append-only events in content** — is the single most important layout decision for an online learner.

---

## Real-Life Applications

| Scenario | Cold start | Drift | Safety need |
|----------|-----------|-------|-------------|
| Response-style personalization | New users use the global default | Tone preferences shift seasonally | Never serve a style that failed a brand check |
| Feed / content ranking | Prior from popular items | Interests change weekly | Exposure caps so no item dominates |
| Support routing | Route new issues by the global prior | Issue mix changes with releases | Fallback to the standard queue |
| Pricing / promotion | Anchor on the safe price | Demand shifts with season | Floor price, kill switch |
| Onboarding flow | Default flow for new cohorts | Funnel behavior evolves | Never experiment on payment steps |

---

## Next Step

That is the end of Track B. You now have the whole ladder: a random control (B0), the 80/20 (B1–B2), sequences (B3–B5), and the production layer (B6). Two directions from here:

- **If your reward needs a judge** — you cannot measure the outcome directly ("was this empathetic?") — that is the [Cross-Track Bridge](../a-llm-post-training/index.md): use a Track A model as your scorer, and its score becomes your Track B reward.
- **If you want the vocabulary behind the models you consume**, read [Track A](../a-llm-post-training/index.md).

---

**Previous:** [B5 — DQN](b5-dqn.md)

**Up:** [Track B Index](index.md) · **Reference:** [RL Methods Guide — Progression Path](../../RL_METHODS_GUIDE.md#progression-path)
