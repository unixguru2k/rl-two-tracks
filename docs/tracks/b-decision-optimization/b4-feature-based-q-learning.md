# B4 — Feature-Based Q-Learning

> **Track B · Guide 5 of 7** — [Track B Index](index.md) · [RL Methods Guide](../../RL_METHODS_GUIDE.md)

## What You'll Learn

Feature-Based Q-Learning solves Q-Learning's biggest problem: **state explosion**. Instead of memorizing every (state, action) pair, it learns a *weight vector* that maps features to rewards — generalizing to states it's never seen.

---

## The Concept

```
┌─────────────────────────────────────────────────────────────────┐
│              FEATURE-BASED Q-LEARNING                           │
│                                                                 │
│   Instead of: Q(state, action) = lookup value                   │
│   Use:        Q(state, action) = w₁×f₁ + w₂×f₂ + w₃×f₃ + ... │
│                                                                 │
│   Features: mood, time, engagement, topic, relationship_stage   │
│   Weights:  Learned numbers that map features → reward          │
│                                                                 │
│   Key Insight: States with SIMILAR features get SIMILAR values  │
│                even if we've never seen that exact state before! │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why Features Instead of Tables

```
Q-TABLE APPROACH (Level 2):
  States: mood × topic × time × engagement × relationship_stage
        = 5 × 50 × 48 × 10 × 5 = 600,000 cells!
  Actions: 5
  Total: 3,000,000 Q-values to store and learn
  Most states never visited → Can't learn!

FEATURE APPROACH (Level 3):
  Features: [mood_score, time_sin, time_cos, engagement, topic_hash, rel_stage, ...]
         = 15 features
  Actions: 5
  Total: 75 weights to store and learn
  ANY state gets a Q-value from features!
```

---

## How It Works

```
Traditional Q-Learning:
  Q(sad+night+low+food, comfort) = 7.2  (memorized)
  Q(happy+morning+high+food, comfort) = ???  (never seen!)

Feature-Based Q-Learning:
  Q(state, action) = w₀ + w₁×mood + w₂×time + w₃×engagement + ...

  Q(sad+night+low+food, comfort) = 0.5 + 2.3×0.0 + 1.8×1.0 + 1.2×0.3 + ...
                                 = 4.66

  Q(happy+morning+high+food, comfort) = 0.5 + 2.3×1.0 + 1.8×0.0 + 1.2×0.8 + ...
                                       = 3.96  (generalized!)

The weights are shared across ALL states for that action.
One update teaches the agent about MANY similar states.
```

---

## Python Example

```python
import numpy as np
from collections import defaultdict

class FeatureQLearner:
    """Q-Learning with linear function approximation."""
    
    def __init__(
        self,
        actions: list[str],
        n_features: int,
        alpha: float = 0.01,
        gamma: float = 0.9,
        epsilon: float = 0.1
    ):
        self.actions = actions
        self.n_features = n_features
        self.alpha = alpha      # Learning rate (smaller for function approximation)
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate
        
        # Weight vectors: one per action
        # w[action] = array of weights for each feature
        self.weights = {
            action: np.zeros(n_features) for action in actions
        }
        
        # Track stats
        self.history = []
    
    def q_value(self, features: np.ndarray, action: str) -> float:
        """Compute Q-value: w · features (dot product)."""
        return float(np.dot(self.weights[action], features))
    
    def select_action(self, features: np.ndarray) -> str:
        """Pick action using ε-greedy policy."""
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions)
        
        # Pick action with highest Q-value
        q_values = {a: self.q_value(features, a) for a in self.actions}
        return max(q_values, key=q_values.get)
    
    def update(
        self,
        features: np.ndarray,
        action: str,
        reward: float,
        next_features: np.ndarray,
        done: bool
    ):
        """Update weights using TD learning."""
        current_q = self.q_value(features, action)
        
        if done:
            target = reward
        else:
            # Best Q-value for next state
            max_future_q = max(
                self.q_value(next_features, a) for a in self.actions
            )
            target = reward + self.gamma * max_future_q
        
        # TD error
        td_error = target - current_q
        
        # Gradient update: w ← w + α × error × features
        self.weights[action] += self.alpha * td_error * features
        
        # Track
        self.history.append({
            "q_value": current_q,
            "td_error": td_error,
            "reward": reward
        })
    
    def get_policy_summary(self) -> dict:
        """See learned weights per action."""
        return {
            action: {
                "weights": self.weights[action].tolist(),
                "magnitude": float(np.linalg.norm(self.weights[action]))
            }
            for action in self.actions
        }


# --- Feature Extraction ---
def extract_features(mood: str, time_of_day: str, engagement: float, topic: str) -> np.ndarray:
    """Convert conversation context to feature vector."""
    # Mood encoding
    mood_map = {"happy": 1.0, "neutral": 0.5, "sad": 0.0, "angry": -0.5}
    
    # Time encoding (cyclical using sin/cos)
    time_map = {"morning": 0.25, "afternoon": 0.5, "evening": 0.75, "night": 1.0}
    time_val = time_map.get(time_of_day, 0.5)
    
    # Topic encoding (simple hash to 0-1)
    topic_hash = hash(topic) % 100 / 100.0
    
    return np.array([
        1.0,                              # Bias term
        mood_map.get(mood, 0.5),          # Mood feature
        np.sin(2 * np.pi * time_val),     # Time sin encoding
        np.cos(2 * np.pi * time_val),     # Time cos encoding
        engagement,                        # Engagement level
        topic_hash,                        # Topic encoding
        mood_map.get(mood, 0.5) * engagement,  # Mood × engagement interaction
    ])


# --- Simulated Conversation Environment ---
class ConversationFeatureEnv:
    """Conversation environment with continuous features."""
    
    def __init__(self):
        self.actions = ["empathetic", "playful", "distract", "question", "validate"]
        
        # True reward function (what we're trying to learn)
        self.true_weights = {
            "empathetic": np.array([5.0, -3.0, 0.5, -0.5, -2.0, 0.1, -1.5]),
            "playful":    np.array([3.0, 2.0, 0.3, 0.8, 1.5, 0.2, 1.0]),
            "distract":   np.array([4.0, -1.0, 0.2, -0.3, 0.5, 0.0, -0.5]),
            "question":   np.array([2.0, 0.5, 0.4, 0.2, 0.8, -0.1, 0.3]),
            "validate":   np.array([6.0, -2.0, 0.1, -0.2, -1.0, 0.0, -0.8]),
        }
        
        self.current_features = None
        self.steps = 0
        self.max_steps = 20
    
    def reset(self) -> np.ndarray:
        """Start new conversation with random context."""
        mood = np.random.choice(["happy", "neutral", "sad"])
        time_of_day = np.random.choice(["morning", "afternoon", "evening", "night"])
        engagement = np.random.uniform(0.2, 0.9)
        topic = np.random.choice(["work", "family", "hobbies", "health", "goals"])
        
        self.current_features = extract_features(mood, time_of_day, engagement, topic)
        self.steps = 0
        return self.current_features
    
    def step(self, action: str) -> tuple:
        """Take action, get reward."""
        # Compute true reward from features
        true_reward = float(np.dot(self.true_weights[action], self.current_features))
        
        # Add noise
        reward = true_reward + np.random.normal(0, 1.0)
        
        # New features (conversation continues)
        mood = np.random.choice(["happy", "neutral", "sad"], p=[0.3, 0.4, 0.3])
        time_of_day = np.random.choice(["morning", "afternoon", "evening", "night"])
        engagement = np.clip(
            self.current_features[4] + np.random.normal(0, 0.1), 0.1, 0.9
        )
        topic = np.random.choice(["work", "family", "hobbies", "health", "goals"])
        
        next_features = extract_features(mood, time_of_day, engagement, topic)
        
        self.steps += 1
        done = self.steps >= self.max_steps or np.random.random() < 0.1
        
        self.current_features = next_features
        return reward, next_features, done


# --- Training ---
def train(n_episodes: int = 1000):
    """Train feature-based Q-learner."""
    env = ConversationFeatureEnv()
    agent = FeatureQLearner(
        actions=env.actions,
        n_features=7,  # Number of features from extract_features
        alpha=0.01,    # Smaller learning rate for stability
        gamma=0.9,
        epsilon=0.15
    )
    
    rewards_per_episode = []
    
    for episode in range(n_episodes):
        features = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(features)
            reward, next_features, done = env.step(action)
            
            agent.update(features, action, reward, next_features, done)
            
            total_reward += reward
            features = next_features
        
        rewards_per_episode.append(total_reward)
        
        if (episode + 1) % 200 == 0:
            avg = np.mean(rewards_per_episode[-200:])
            print(f"Episode {episode + 1}: avg reward = {avg:.2f}")
    
    return agent, rewards_per_episode


# Run training
agent, rewards = train(1000)

# See what it learned
print("\n=== LEARNED WEIGHTS ===")
summary = agent.get_policy_summary()
for action, info in summary.items():
    print(f"{action}: magnitude={info['magnitude']:.2f}, weights={[round(w, 2) for w in info['weights']]}")

# Test predictions
print("\n=== POLICY PREDICTIONS ===")
test_cases = [
    ("sad", "night", 0.3, "health"),
    ("happy", "morning", 0.8, "hobbies"),
    ("neutral", "afternoon", 0.5, "work"),
]

for mood, time, eng, topic in test_cases:
    features = extract_features(mood, time, eng, topic)
    q_vals = {a: agent.q_value(features, a) for a in agent.actions}
    best = max(q_vals, key=q_vals.get)
    print(f"  {mood}+{time}+{eng:.1f}+{topic}: {best} ({q_vals[best]:.2f})")
    print(f"    All Q-values: {', '.join(f'{a}={v:.2f}' for a, v in q_vals.items())}")
```

---

## When to Use Feature-Based Q-Learning

| Use Case | Why |
|----------|-----|
| Conversation with many contexts | Can't store all (mood × topic × time) combos |
| Continuous features | Engagement = 0.347, not just "low"/"high" |
| Personalization at scale | Thousands of users, each with unique contexts |
| When tabular Q-learning fails | State space too large for lookup table |

---

## Key Concepts Visualized

### Generalization

```
Q-TABLE: Each state is an island
┌─────────────────────────────────────────────────┐
│  sad+night+low  │  sad+night+med  │  sad+night+high │
│  Q = 7.2        │  Q = ???        │  Q = ???         │
│  (visited)      │  (never seen)   │  (never seen)    │
└─────────────────────────────────────────────────┘

FEATURE-BASED: States are connected by features
┌─────────────────────────────────────────────────┐
│  sad+night+low   →  features: [0.0, 1.0, 0.3, ...]  │
│  sad+night+med   →  features: [0.0, 1.0, 0.5, ...]  │
│  sad+night+high  →  features: [0.0, 1.0, 0.8, ...]  │
│                                                      │
│  Similar features → Similar Q-values!                │
│  Update one → improves predictions for ALL similar!  │
└─────────────────────────────────────────────────┘
```

### Weight Learning

```
Episode 1: All weights = 0
  Q(sad+night, empathetic) = 0.0 (blind guess)

Episode 100: Starting to learn
  empathetic weights = [3.2, -1.5, 0.3, ...]
  Q(sad+night, empathetic) = 3.2 + (-1.5 × 0.0) + (0.3 × 1.0) + ...
                           = 3.5 (mood matters!)

Episode 1000: Confident
  empathetic weights = [5.1, -3.0, 0.5, ...]
  Q(sad+night, empathetic) = 5.1 + (-3.0 × 0.0) + (0.5 × 1.0) + ...
                           = 5.6 (strong: empathetic works when sad)

  playful weights = [3.0, 2.0, 0.3, ...]
  Q(happy+morning, playful) = 3.0 + (2.0 × 1.0) + (0.3 × 0.0) + ...
                            = 5.0 (strong: playful works when happy)
```

---

## Linear Approximation vs Tabular Q-Learning

| Aspect | Tabular Q-Learning | Feature-Based Q-Learning |
|--------|-------------------|--------------------------|
| Storage | State × Action table | Weight vector per action |
| New states | Zero knowledge | Generalizes from features |
| Continuous states | ❌ Can't handle | ✅ Works naturally |
| Update cost | One table entry | One vector multiply |
| Stability | Very stable | Can be unstable |
| Feature engineering | Not needed | Critical for performance |

---

## Key Limitation: Linear Assumption

```
Feature-Based Q-Learning assumes:
  Q(s, a) = w₁×mood + w₂×time + w₃×engagement + ...

What if the TRUE reward is:
  reward = mood² × sin(time) × log(engagement + 1)

Linear approximation can't learn this!

Solution: Use DQN (Level 4) which uses a neural network
to learn non-linear feature interactions automatically.
```

---

## Connecting to Real RL Methods Guide

From the [RL Methods Guide](../../RL_METHODS_GUIDE.md#level-3-feature-based-q-learning-linear-approximation):

> **What:** Q-learning but uses features instead of a lookup table.
> **When:** Sequential decisions + state space too large for a table.
> **Key advantage:** Generalizes to unseen states.
> **Key limitation:** Assumes linear feature-reward relationship.

---

## Real-Life Applications

| Scenario | Features | Why Linear Works |
|----------|----------|------------------|
| Robotic assembly | Sensor readings | Continuous features, fast updates |
| Dynamic pricing | Demand, supply, time | Many continuous values |
| Hospital patient flow | Condition, beds, staff | Complex but predictable |
| Conversation management | Mood, topic, engagement | Linear relationships exist |

---

## Next Step

When your features have complex non-linear relationships, move to **Level 4: Deep Q-Network (DQN)** which uses neural networks to learn any function automatically.

---

**Previous:** [B3 — Q-Learning (Tabular)](b3-q-learning-tabular.md) · **Next:** [B5 — DQN](b5-dqn.md)

**Up:** [Track B Index](index.md) · **Reference:** [RL Methods Guide — Level 3](../../RL_METHODS_GUIDE.md#level-3-feature-based-q-learning-linear-approximation)
