# B0 — Random Baseline

> **Track B · Guide 1 of 7** — [Track B Index](index.md) · [RL Methods Guide](../../RL_METHODS_GUIDE.md)

## What You'll Learn

Random selection is the simplest "algorithm" - it picks actions with no intelligence. This serves as a **baseline** to measure whether any learning algorithm actually helps.

---

## The Concept

```
┌─────────────────────────────────────────────────────────┐
│                   RANDOM SELECTION                      │
│                                                         │
│   Actions: [sweet, playful, flirty, caring]             │
│                                                         │
│   Message comes in → Pick randomly → Send response      │
│                                                         │
│   No memory. No learning. No optimization.              │
└─────────────────────────────────────────────────────────┘
```

---

## Python Example

```python
import random

class RandomSelector:
    """Selects actions randomly with no learning."""
    
    def __init__(self, actions: list[str]):
        self.actions = actions
        self.history = []
    
    def select(self) -> str:
        """Pick a random action."""
        action = random.choice(self.actions)
        self.history.append(action)
        return action
    
    def get_stats(self) -> dict:
        """See distribution of selections."""
        stats = {}
        for action in self.actions:
            count = self.history.count(action)
            stats[action] = {
                "count": count,
                "percentage": round(count / len(self.history) * 100, 1) if self.history else 0
            }
        return stats


# Usage
actions = ["sweet", "playful", "flirty", "caring"]
selector = RandomSelector(actions)

# Simulate 100 messages
for _ in range(100):
    chosen = selector.select()
    # No reward tracking, no updates

print(selector.get_stats())
# Expected: roughly 25% each (random)
```

---

## When to Use Random Selection

| Use Case | Why |
|----------|-----|
| A/B testing baseline | Measure if optimization helps at all |
| Initial data collection | Gather some data before training |
| Debugging | Verify your reward pipeline works |
| Exploration fallback | Last resort when no data exists |

---

## What Random Selection Teaches You

1. **Baseline metrics** - What's the "average" performance without learning?
2. **Reward distribution** - Are some actions genuinely better?
3. **System validation** - Does your reward calculation work?

---

## Key Limitation

```
Random selection treats every user the same.

User A (happy, morning) → Random action
User B (sad, night)     → Same random action

No personalization. No context. No memory.
```

---

## Measuring Against Random

```python
def compare_to_random(random_rewards: list, learned_rewards: list):
    """Measure improvement over random baseline."""
    random_avg = sum(random_rewards) / len(random_rewards)
    learned_avg = sum(learned_rewards) / len(learned_rewards)
    
    improvement = ((learned_avg - random_avg) / abs(random_avg)) * 100
    
    return {
        "random_average": round(random_avg, 2),
        "learned_average": round(learned_avg, 2),
        "improvement_pct": round(improvement, 1)
    }

# If learned_avg > random_avg, your RL is helping!
```

---

## Next Step

Once you have a baseline, move to **Level 1a: Classic Multi-Armed Bandit** to start actually learning from rewards.

---

**Next:** [B1 — Epsilon-Greedy Bandit](b1-epsilon-greedy-bandit.md)

**Up:** [Track B Index](index.md) · **Reference:** [RL Methods Guide — Level 0](../../RL_METHODS_GUIDE.md#level-0-random-selection)
