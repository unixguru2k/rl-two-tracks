# B1 — Epsilon-Greedy Bandit

> **Track B · Guide 2 of 6** — [Track B Index](index.md) · [RL Methods Guide](../../RL_METHODS_GUIDE.md)

## What You'll Learn

Epsilon-Greedy is the simplest *learning* algorithm. It tries all options, remembers what works, and mostly picks the best one — while occasionally exploring to make sure it's not missing something better.

---

## The Concept

```
┌─────────────────────────────────────────────────────────┐
│               EPSILON-GREEDY BANDIT                     │
│                                                         │
│   Actions: [sweet, playful, flirty, caring]             │
│                                                         │
│   Message comes in                                      │
│        │                                                │
│        ├── Roll dice: random() < ε (10% of time)?      │
│        │       └── YES → Pick random action (EXPLORE)   │
│        │                                                │
│        └── NO → Pick best known action (EXPLOIT)        │
│                                                         │
│   Track reward → Update averages → Learn over time      │
└─────────────────────────────────────────────────────────┘
```

---

## How It Works

```
The ε (epsilon) = exploration rate

ε = 0.1  →  10% explore, 90% exploit (good default)
ε = 0.3  →  30% explore, 70% exploit (more curious)
ε = 0.01 →   1% explore, 99% exploit (very confident)

START: All actions have equal chance

After 10 messages:
  sweet:   4 rewards → avg reward = 6.2  ← BEST
  playful: 3 rewards → avg reward = 4.1
  flirty:  2 rewards → avg reward = 3.8
  caring:  1 reward  → avg reward = 5.5

With ε=0.1:
  90% of the time → pick "sweet" (highest avg reward)
  10% of the time → pick any random action (to keep exploring)

After 100 messages, estimates are more accurate:
  sweet:   avg reward = 6.5 (confident)
  playful: avg reward = 4.2 (confident)
  flirty:  avg reward = 3.5 (confident)
  caring:  avg reward = 5.8 (confident)
```

---

## Python Example

```python
import random

class EpsilonGreedyBandit:
    """Classic Multi-Armed Bandit with Epsilon-Greedy strategy."""
    
    def __init__(self, actions: list[str], epsilon: float = 0.1):
        self.actions = actions
        self.epsilon = epsilon
        
        # Track average reward per action
        self.rewards = {action: 0.0 for action in actions}
        self.counts = {action: 0 for action in actions}
        self.total_plays = 0
    
    def select(self) -> str:
        """Pick an action: exploit best known OR explore randomly."""
        self.total_plays += 1
        
        # Explore: random action with probability ε
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        
        # Exploit: pick action with highest average reward
        return max(self.rewards, key=lambda a: self.rewards[a])
    
    def update(self, action: str, reward: float):
        """Update average reward for an action."""
        self.counts[action] += 1
        count = self.counts[action]
        
        # Running average formula: new_avg = old_avg + (1/n)(reward - old_avg)
        self.rewards[action] += (reward - self.rewards[action]) / count
    
    def get_stats(self) -> dict:
        """See current knowledge."""
        return {
            action: {
                "avg_reward": round(self.rewards[action], 2),
                "times_played": self.counts[action],
                "percentage": round(self.counts[action] / self.total_plays * 100, 1) if self.total_plays else 0
            }
            for action in self.actions
        }


# --- Simulated User Rewards ---
def simulate_reward(action: str) -> float:
    """Simulate user reactions to response styles."""
    true_rewards = {
        "sweet":   6.5,   # Users like sweet responses
        "playful": 4.2,   # Moderate engagement
        "flirty":  3.5,   # Sometimes too forward
        "caring":  5.8,   # Good for sad moments
    }
    # Add some noise
    return true_rewards[action] + random.gauss(0, 1.0)


# Usage
actions = ["sweet", "playful", "flirty", "caring"]
bandit = EpsilonGreedyBandit(actions, epsilon=0.1)

# Simulate 500 messages
for i in range(500):
    chosen = bandit.select()
    reward = simulate_reward(chosen)
    bandit.update(chosen, reward)

# See what it learned
stats = bandit.get_stats()
for action, info in stats.items():
    print(f"{action}: avg_reward={info['avg_reward']}, played {info['times_played']} times ({info['percentage']}%)")

# After 500 plays, "sweet" and "caring" should dominate (they have highest true rewards)
```

---

## When to Use Epsilon-Greedy

| Use Case | Why |
|----------|-----|
| Finding best global default | Works when context doesn't matter |
| Simple A/B/C/D testing | Better than random, tracks rewards |
| Response style selection | Which tone works best across ALL users |
| Quick optimization | Easy to implement, fast to learn |

---

## Key Concepts Visualized

```
Exploration vs Exploitation Over Time:

Early (few messages):
┌──────────────────────────────────┐
│ ████████████████████░░░░░░░░░░░ │  Lots of exploration
│         ε = 0.5 (curious)       │  (not enough data to trust)
└──────────────────────────────────┘

Mid (100+ messages):
┌──────────────────────────────────┐
│ ████████████████████████████░░░ │  Mostly exploit
│         ε = 0.1 (balanced)      │  (some exploration)
└──────────────────────────────────┘

Late (500+ messages):
┌──────────────────────────────────┐
│ ██████████████████████████████░ │  Mostly exploit
│         ε = 0.01 (confident)    │  (rare exploration)
```

---

## Key Limitation: No Personalization

```
The Bandit's Blind Spot:

User A (happy, morning, engaged)  ──┐
                                    ├── Same action chosen!
User B (sad, night, lonely)      ──┘

The bandit learns:
  "sweet gets avg reward 6.5"
  But it doesn't know WHO gets sweet at what time.

It finds the BEST GLOBAL default, not the best per-user.
```

---

## Connecting to Real RL Methods Guide

From the [RL Methods Guide](../../RL_METHODS_GUIDE.md#level-1a-classic-multi-armed-bandit-epsilon-greedy):

> **What:** Try all options, remember what works, mostly pick the best.
> **When:** Few actions, no context matters, simple optimization.
> **Key limitation:** Same answer for everyone — no personalization.

---

## Real-Life Applications

| Scenario | Actions | Reward | Notes |
|----------|---------|--------|-------|
| Email subject lines | 5 subject lines | Open rate | Find best subject for ALL subscribers |
| Homepage banners | 4 designs | Click-through | Best banner for all visitors |
| Game difficulty | Easy/Med/Hard | Session length | Sweet spot for retention |
| Default response style | sweet/playful/flirty | Engagement | Best style for ALL users |

---

## What the Bandit Learns Over Time

```
After 50 plays (estimates noisy):
  sweet:   5.8  (uncertain)
  playful: 3.9  (uncertain)
  flirty:  4.2  (uncertain)
  caring:  6.1  (uncertain)

After 500 plays (estimates solid):
  sweet:   6.4  (confident)
  playful: 4.1  (confident)
  flirty:  3.5  (confident)
  caring:  5.7  (confident)

After 5000 plays (very sure):
  sweet:   6.5  (very confident)
  playful: 4.2  (very confident)
  flirty:  3.5  (very confident)
  caring:  5.8  (very confident)
```

---

## Comparison to Random Selection

| Metric | Random | Epsilon-Greedy |
|--------|--------|----------------|
| Learning | ❌ None | ✅ Learns best action |
| Personalization | ❌ None | ❌ Same for everyone |
| Exploration | Random always | Adaptive |
| Complexity | Zero | Low |
| Long-term performance | Poor | Good |

---

## Next Step

Once you find the best GLOBAL default, move to **Level 1b: Contextual Bandit (LinUCB)** to personalize decisions based on context (mood, time, user).

---

**See Also**: [RL Methods Guide - Level 1a](../../RL_METHODS_GUIDE.md#level-1a-classic-multi-armed-bandit-epsilon-greedy)
