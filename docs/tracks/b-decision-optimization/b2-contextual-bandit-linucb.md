# B2 — Contextual Bandit (LinUCB)

> **Track B · Guide 3 of 7** — [Track B Index](index.md) · [RL Methods Guide](../../RL_METHODS_GUIDE.md)

## What You'll Learn

LinUCB is a *smart* bandit that uses **context** (user features) to make different decisions for different situations. It also explores automatically — no epsilon to tune.

---

## The Concept

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONTEXTUAL BANDIT (LinUCB)                   │
│                                                                 │
│   Actions: [sweet, playful, flirty, caring, distract]           │
│                                                                 │
│   Message comes in                                              │
│        │                                                        │
│        ├── Extract context: [mood, time, engagement, topic]     │
│        │                                                        │
│        ├── For EACH action, compute:                            │
│        │     score = predicted_reward + uncertainty_bonus        │
│        │                                                        │
│        └── Pick action with HIGHEST score                       │
│                                                                 │
│   Track reward → Update weights → Learn per-context patterns    │
└─────────────────────────────────────────────────────────────────┘
```

---

## How It Works

```
The KEY difference from Classic MAB:

Classic MAB:
  "sweet gets avg reward 6.5"  (same for ALL users)

LinUCB:
  "sweet gets reward 8.2 WHEN mood=sad AND time=night"
  "sweet gets reward 3.1 WHEN mood=happy AND time=morning"

It learns a WEIGHT VECTOR per action that maps features → reward.

Score(action) = w · features + confidence_bonus
                ─────────────   ─────────────────
                "what I expect"  "how sure I am"
                                  (auto-exploration!)
```

### Automatic Exploration

```
New action with few observations:
  confidence_bonus = LARGE → gets tried automatically

After 30 observations:
  confidence_bonus = small → only picked if reward is high

No epsilon to tune. Exploration happens naturally.
```

---

## Python Example

```python
import numpy as np

class LinUCB:
    """Linear Upper Confidence Bound - Contextual Bandit."""
    
    def __init__(self, actions: list[str], n_features: int, alpha: float = 1.0):
        self.actions = actions
        self.n_features = n_features
        self.alpha = alpha  # Exploration parameter
        
        # Per-action parameters
        self.A = {}  # d×d matrix per action (feature correlation)
        self.b = {}  # d×1 vector per action (reward correlation)
        
        for action in actions:
            self.A[action] = np.eye(n_features)  # Identity matrix
            self.b[action] = np.zeros(n_features)  # Zero vector
    
    def select(self, features: np.ndarray) -> str:
        """Pick action with highest UCB score given context."""
        best_action = None
        best_score = -float('inf')
        
        for action in self.actions:
            A_inv = np.linalg.inv(self.A[action])
            theta = A_inv @ self.b[action]  # Learned weights
            
            # Predicted reward
            predicted = theta @ features
            
            # Uncertainty bonus (exploration)
            uncertainty = self.alpha * np.sqrt(features @ A_inv @ features)
            
            score = predicted + uncertainty
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action
    
    def update(self, action: str, features: np.ndarray, reward: float):
        """Update weights for the chosen action."""
        self.A[action] += np.outer(features, features)
        self.b[action] += reward * features
    
    def get_stats(self) -> dict:
        """See learned weights per action."""
        stats = {}
        for action in self.actions:
            A_inv = np.linalg.inv(self.A[action])
            theta = A_inv @ self.b[action]
            stats[action] = {
                "weights": theta.tolist(),
                "observations": int(np.trace(self.A[action]) - self.n_features)
            }
        return stats


# --- Feature Extraction ---
def extract_features(mood: str, time_of_day: str, engagement: float) -> np.ndarray:
    """Convert context to feature vector."""
    mood_map = {"happy": 1.0, "neutral": 0.5, "sad": 0.0}
    time_map = {"morning": 0.0, "afternoon": 0.5, "night": 1.0}
    
    return np.array([
        1.0,  # Bias term
        mood_map.get(mood, 0.5),
        time_map.get(time_of_day, 0.5),
        engagement,
        mood_map.get(mood, 0.5) * time_map.get(time_of_day, 0.5)  # Interaction
    ])


# --- Simulated Rewards ---
def simulate_reward(action: str, mood: str, time_of_day: str) -> float:
    """Simulate user reactions based on context."""
    # True reward patterns
    if mood == "sad" and time_of_day == "night":
        rewards = {"sweet": 7.0, "playful": 2.0, "flirty": 1.0, "caring": 9.0, "distract": 5.0}
    elif mood == "happy" and time_of_day == "morning":
        rewards = {"sweet": 5.0, "playful": 8.0, "flirty": 6.0, "caring": 4.0, "distract": 3.0}
    else:
        rewards = {"sweet": 6.0, "playful": 5.0, "flirty": 4.0, "caring": 6.0, "distract": 4.0}
    
    return rewards[action] + np.random.normal(0, 0.5)


# Usage
actions = ["sweet", "playful", "flirty", "caring", "distract"]
bandit = LinUCB(actions, n_features=5, alpha=1.0)

# Simulate 500 interactions
contexts = [
    ("sad", "night", 0.3),
    ("happy", "morning", 0.8),
    ("neutral", "afternoon", 0.5),
]

for i in range(500):
    mood, time, engagement = contexts[i % len(contexts)]
    features = extract_features(mood, time, engagement)
    
    chosen = bandit.select(features)
    reward = simulate_reward(chosen, mood, time)
    bandit.update(chosen, features, reward)

# See what it learned
stats = bandit.get_stats()
for action, info in stats.items():
    print(f"{action}: observations={info['observations']}, weights={[round(w, 2) for w in info['weights']]}")

# After 500 plays:
# - "caring" should dominate for sad+night
# - "playful" should dominate for happy+morning
```

---

## When to Use LinUCB

| Use Case | Why |
|----------|-----|
| Personalized recommendations | Context matters (who, when, where) |
| Response style selection | Different users need different approaches |
| A/B testing with segments | Better than random within segments |
| Content ranking | User features predict preferences |
| Ad targeting | User context determines best ad |

---

## Key Concepts Visualized

```
Context → Features → Score per Action → Best Action

User: sad, night, low engagement
Features: [1.0, 0.0, 1.0, 0.3, 0.0]

  sweet:   7.2 + 0.3 = 7.5
  playful: 2.1 + 0.3 = 2.4
  flirty:  1.0 + 0.3 = 1.3
  caring:  8.9 + 0.2 = 9.1  ← PICKED (highest)
  distract: 5.0 + 0.3 = 5.3

User: happy, morning, high engagement
Features: [1.0, 1.0, 0.0, 0.8, 0.0]

  sweet:   5.0 + 0.2 = 5.2
  playful: 7.8 + 0.2 = 8.0  ← PICKED (highest)
  flirty:  6.0 + 0.3 = 6.3
  caring:  4.0 + 0.3 = 4.3
  distract: 3.0 + 0.3 = 3.3
```

---

## Key Limitation: Linear Assumption

```
LinUCB assumes: reward = w₁×mood + w₂×time + w₃×engagement + ...

What if the TRUE relationship is:
  reward = mood² × sin(time) × log(engagement)

LinUCB can't learn this! It only sees linear patterns.

Solution: Use DQN (Level 4) for non-linear relationships.
```

---

## Connecting to Real RL Methods Guide

From the [RL Methods Guide](../../RL_METHODS_GUIDE.md#level-1b-contextual-bandit-linucb):

> **What:** Same as MAB, but uses CONTEXT to make different decisions for different situations. Smart automatic exploration.
> **When:** 2-20 actions + 5-15 context features matter.
> **Key advantage:** Automatic exploration, no epsilon tuning.
> **Key limitation:** Assumes linear relationship between features and reward.

---

## Real-Life Applications

| Scenario | Actions | Context Features | Reward |
|----------|---------|------------------|--------|
| News recommendation | 10 categories | reading history, time, device | read time, shares |
| Treatment selection | 5 medications | age, weight, lab results | symptom improvement |
| Music playlist | 8 genres | mood, time, activity | listen duration |
| Support routing | 5 specializations | issue type, customer tier | resolution time |

---

## LinUCB vs Classic MAB

| Aspect | Classic MAB | LinUCB |
|--------|-------------|--------|
| Context | ❌ None | ✅ Yes |
| Personalization | ❌ Same for all | ✅ Per-context |
| Exploration | Manual (epsilon) | Automatic (UCB) |
| New users | Blank slate | Generalizes from features |
| Storage | ~20 numbers | ~500 numbers |
| Complexity | Very low | Medium |

---

## What LinUCB Learns Over Time

```
After 50 plays (learning patterns):
  sweet:   weights = [0.5, 2.1, -0.3, 1.2, 0.1]
  caring:  weights = [0.8, 3.5, -0.5, 2.0, 0.2]
  (Starting to see mood matters)

After 500 plays (confident):
  sweet:   weights = [0.4, 2.3, -0.2, 1.1, 0.1]
  caring:  weights = [0.9, 4.1, -0.8, 2.5, 0.3]
  (Clear: caring works best when mood is sad)

After 5000 plays (very sure):
  sweet:   weights = [0.4, 2.2, -0.2, 1.1, 0.1]
  caring:  weights = [0.9, 4.2, -0.8, 2.5, 0.3]
  (Weights stabilized, minimal exploration)
```

---

## Next Step

Once you master single-step decisions with context, move to **Level 2: Q-Learning (Tabular)** to handle sequential decisions where the order of actions matters.

---

**Previous:** [B1 — Epsilon-Greedy Bandit](b1-epsilon-greedy-bandit.md) · **Next:** [B3 — Q-Learning (Tabular)](b3-q-learning-tabular.md)

**Up:** [Track B Index](index.md) · **Reference:** [RL Methods Guide — Level 1b](../../RL_METHODS_GUIDE.md#level-1b-contextual-bandit-linucb)