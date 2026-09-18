# B5 — DQN (Deep Q-Network)

> **Track B · Guide 6 of 7** — [Track B Index](index.md) · [RL Methods Guide](../../RL_METHODS_GUIDE.md)

## What You'll Learn

DQN replaces the Q-table or linear function with a **neural network** that can learn complex, non-linear patterns. It handles raw inputs like text, images, or high-dimensional data that simpler methods can't process.

---

## The Concept

```
┌─────────────────────────────────────────────────────────────────┐
│                    DEEP Q-NETWORK (DQN)                        │
│                                                                 │
│   Instead of: Q(state) = lookup table                           │
│   Or:         Q(state) = w · features  (linear)                │
│   Use:        Q(state) = neural_network(state)  (any function) │
│                                                                 │
│   Input: Raw features (text embeddings, pixels, sensor data)   │
│   Output: Q-value for each action                              │
│                                                                 │
│   Key Power: Can learn non-linear patterns automatically!      │
└─────────────────────────────────────────────────────────────────┘
```

---

## How It's Different From Linear Methods

```
LINEAR (Level 3):
  Q(s, a) = w₁×mood + w₂×time + w₃×engagement + ...
  Can ONLY learn: "more mood → higher reward" (linear)
  Can't learn: "reward peaks at medium mood, drops at extremes" (non-linear)

DQN (Level 4):
  Q(s, a) = neural_network([mood, time, engagement, ...])
  CAN learn ANY pattern: linear, quadratic, sinusoidal, interactions
  Automatically discovers: "mood² × time matters, not mood × time"
```

### The Neural Network

```
Input Layer          Hidden Layers         Output Layer
(features)          (learn patterns)       (Q-values)

[ mood     ]──┐
[ time     ]──┼──→ [ 32 neurons ] ──→ [ 16 neurons ] ──→ [ Q(sweet)   ]
[ engage   ]──┤        ↑                    ↑              [ Q(playful) ]
[ topic    ]──┘    Learns simple         Learns complex   [ Q(flirty)  ]
[ history  ]──┘    patterns like         interactions     [ Q(caring)  ]
                    "time matters"       like "sad × night"
```

---

## The Replay Buffer (Key Innovation)

```
PROBLEM: Training on sequential experiences is unstable
  Experience 1: (sad, night) → caring → reward +8
  Experience 2: (sad, night) → playful → reward -3
  
  If we train immediately, network overfits to recent experiences.

SOLUTION: Store experiences in a buffer, sample random batches

┌────────────────────────────────────────────────────────────┐
│                    REPLAY BUFFER                           │
│                                                            │
│  Buffer stores last 10,000 experiences:                    │
│  ┌───────┬───────┬───────┬───────┬───────┐                │
│  │ exp 1 │ exp 2 │ exp 3 │  ...  │exp10K │                │
│  └───────┴───────┴───────┴───────┴───────┘                │
│                                                            │
│  Each experience = (state, action, reward, next_state)     │
│                                                            │
│  Training: Sample random batch of 32 from buffer           │
│  → Breaks correlation between sequential experiences       │
│  → More stable learning                                    │
└────────────────────────────────────────────────────────────┘
```

---

## Python Example

```python
import numpy as np
from collections import deque
import random

class ReplayBuffer:
    """Stores experiences for stable training."""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> tuple:
        """Sample random batch of experiences."""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class SimpleDQN:
    """Simplified Deep Q-Network using numpy (no PyTorch/TensorFlow needed)."""
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 64,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = learning_rate
        
        # Neural network weights (2 hidden layers)
        self.W1 = np.random.randn(state_size, hidden_size) * np.sqrt(2.0 / state_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(hidden_size)
        self.W3 = np.random.randn(hidden_size, action_size) * np.sqrt(2.0 / hidden_size)
        self.b3 = np.zeros(action_size)
        
        # Target network (for stability)
        self.target_W1 = self.W1.copy()
        self.target_b1 = self.b1.copy()
        self.target_W2 = self.W2.copy()
        self.target_b2 = self.b2.copy()
        self.target_W3 = self.W3.copy()
        self.target_b3 = self.b3.copy()
        
        self.update_target()
    
    def forward(self, state: np.ndarray) -> np.ndarray:
        """Forward pass: state → Q-values for all actions."""
        h1 = np.maximum(0, state @ self.W1 + self.b1)  # ReLU
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)     # ReLU
        return h2 @ self.W3 + self.b3                   # Linear output
    
    def target_forward(self, state: np.ndarray) -> np.ndarray:
        """Forward pass through target network."""
        h1 = np.maximum(0, state @ self.target_W1 + self.target_b1)
        h2 = np.maximum(0, h1 @ self.target_W2 + self.target_b2)
        return h2 @ self.target_W3 + self.target_b3
    
    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        q_values = self.forward(state)
        return int(np.argmax(q_values))
    
    def train_batch(self, states, actions, rewards, next_states, dones):
        """Train on a batch of experiences."""
        batch_size = len(states)
        
        # Current Q-values
        current_q = self.forward(states)
        
        # Target Q-values (from target network)
        next_q = self.target_forward(next_states)
        max_next_q = np.max(next_q, axis=1)
        
        # Compute targets
        targets = current_q.copy()
        for i in range(batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * max_next_q[i]
        
        # Backpropagation (simplified)
        # Layer 3
        h2 = np.maximum(0, states @ self.W1 + self.b1)
        h2 = np.maximum(0, h2 @ self.W2 + self.b2)
        
        d_output = (current_q - targets) / batch_size
        d_W3 = h2.T @ d_output
        d_b3 = np.sum(d_output, axis=0)
        
        # Layer 2
        d_h2 = d_output @ self.W3.T
        d_h2[h2 <= 0] = 0  # ReLU gradient
        
        h1 = np.maximum(0, states @ self.W1 + self.b1)
        d_W2 = h1.T @ d_h2
        d_b2 = np.sum(d_h2, axis=0)
        
        # Layer 1
        d_h1 = d_h2 @ self.W2.T
        d_h1[h1 <= 0] = 0  # ReLU gradient
        
        d_W1 = states.T @ d_h1
        d_b1 = np.sum(d_h1, axis=0)
        
        # Update weights
        self.W3 -= self.lr * d_W3
        self.b3 -= self.lr * d_b3
        self.W2 -= self.lr * d_W2
        self.b2 -= self.lr * d_b2
        self.W1 -= self.lr * d_W1
        self.b1 -= self.lr * d_b1
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target(self):
        """Copy weights to target network."""
        self.target_W1 = self.W1.copy()
        self.target_b1 = self.b1.copy()
        self.target_W2 = self.W2.copy()
        self.target_b2 = self.b2.copy()
        self.target_W3 = self.W3.copy()
        self.target_b3 = self.b3.copy()


# --- Simulated Environment ---
class ConversationDNQEnv:
    """Conversation environment with complex feature interactions."""
    
    def __init__(self):
        self.actions = ["empathetic", "playful", "distract", "validate", "question"]
        self.state_size = 10  # More features than linear methods
        self.steps = 0
        self.max_steps = 30
    
    def _get_state(self) -> np.ndarray:
        """Generate state with complex non-linear patterns."""
        mood = np.random.uniform(-1, 1)
        time = np.random.uniform(0, 1)
        engagement = np.random.uniform(0, 1)
        sentiment = np.random.uniform(-1, 1)
        topic_complexity = np.random.uniform(0, 1)
        
        return np.array([
            mood,
            time,
            engagement,
            sentiment,
            topic_complexity,
            mood ** 2,                    # Non-linear: mood squared
            np.sin(2 * np.pi * time),     # Cyclical time
            engagement * sentiment,       # Interaction
            mood * topic_complexity,      # Another interaction
            1.0 if engagement < 0.3 else 0.0  # Low engagement flag
        ])
    
    def reset(self) -> np.ndarray:
        self.steps = 0
        return self._get_state()
    
    def step(self, action: int) -> tuple:
        """Take action, get reward with non-linear patterns."""
        state = self._get_state()
        
        # Non-linear reward function (this is what DQN can learn but linear can't!)
        mood = state[0]
        time = state[1]
        engagement = state[2]
        mood_sq = state[5]
        time_sin = state[6]
        
        # True reward has non-linear interactions
        if action == 0:  # empathetic
            reward = 5.0 * (1 - mood) * engagement + 2.0 * time_sin
        elif action == 1:  # playful
            reward = 3.0 * mood * engagement - 1.0 * (1 - mood_sq)
        elif action == 2:  # distract
            reward = 2.0 * (1 - engagement) * time
        elif action == 3:  # validate
            reward = 4.0 * (1 - mood) * (1 + time_sin) + 1.0
        else:  # question
            reward = 1.5 * engagement * topic_complexity
        
        # Add noise
        reward += np.random.normal(0, 0.5)
        
        self.steps += 1
        done = self.steps >= self.max_steps
        next_state = self._get_state()
        
        return reward, next_state, done


# --- Training ---
def train_dqn(n_episodes: int = 500, batch_size: int = 32):
    """Train DQN agent."""
    env = ConversationDNQEnv()
    agent = SimpleDQN(
        state_size=env.state_size,
        action_size=len(env.actions),
        hidden_size=64,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    buffer = ReplayBuffer(capacity=10000)
    rewards_per_episode = []
    target_update_freq = 10  # Update target network every N episodes
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            reward, next_state, done = env.step(action)
            
            # Store experience
            buffer.push(state, action, reward, next_state, done)
            
            # Train on batch
            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                agent.train_batch(states, actions, rewards, next_states, dones)
            
            total_reward += reward
            state = next_state
        
        rewards_per_episode.append(total_reward)
        
        # Update target network periodically
        if (episode + 1) % target_update_freq == 0:
            agent.update_target()
        
        if (episode + 1) % 50 == 0:
            avg = np.mean(rewards_per_episode[-50:])
            print(f"Episode {episode + 1}: avg_reward = {avg:.2f}, epsilon = {agent.epsilon:.3f}")
    
    return agent, rewards_per_episode


# Run training
print("Training DQN agent...")
agent, rewards = train_dqn(500, batch_size=32)

# Test the learned policy
print("\n=== TESTING LEARNED POLICY ===")
action_names = ["empathetic", "playful", "distract", "validate", "question"]
test_states = [
    ("Sad user at night", np.array([-0.8, 0.9, 0.3, -0.5, 0.5, 0.64, 0.59, -0.24, -0.4, 1.0])),
    ("Happy user in morning", np.array([0.8, 0.2, 0.8, 0.6, 0.3, 0.64, 0.95, 0.48, 0.24, 0.0])),
    ("Neutral user, low engagement", np.array([0.0, 0.5, 0.2, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 1.0])),
]

for label, state in test_states:
    q_values = agent.forward(state)
    best_action = action_names[np.argmax(q_values)]
    print(f"\n{label}:")
    print(f"  Best action: {best_action}")
    print(f"  Q-values: {', '.join(f'{a}={v:.2f}' for a, v in zip(action_names, q_values))}")
```

---

## When to Use DQN

| Use Case | Why |
|----------|-----|
| High-dimensional states | Text embeddings, images, sensor data |
| Non-linear patterns | Reward depends on complex interactions |
| Large state spaces | Millions of possible states |
| Raw input processing | Learn features automatically |

---

## Key Concepts Visualized

### Neural Network Architecture

```
INPUT (10 features)          HIDDEN (64)           OUTPUT (5 actions)

[mood        ]──────┐
[time        ]──────┤
[engagement  ]──────┼──→ [64 neurons] ──→ [64 neurons] ──→ [Q(empathetic) ]
[sentiment   ]──────┤      ↑                   ↑           [Q(playful)    ]
[complexity  ]──────┘   Learns edges       Learns          [Q(distract)   ]
[mood²       ]──────┘   like "mood         complex         [Q(validate)   ]
[sin(time)   ]──────┘   matters"           patterns        [Q(question)   ]
[mood×engage ]──────┘                      like "sad×night"
[mood×topic  ]──────┘
[low_engage  ]──────┘
```

### Training Process

```
Episode 1-50: Random exploration (epsilon = 1.0)
  Agent tries random actions, fills replay buffer
  Q-values are unreliable

Episode 50-200: Learning (epsilon decaying)
  Agent uses learned Q-values more often
  Buffer has diverse experiences
  Network learns basic patterns

Episode 200-500: Exploitation (epsilon = 0.01)
  Agent mostly exploits learned policy
  Q-values are accurate
  Consistent high rewards
```

### Target Network (Stability Trick)

```
PROBLEM: Network chases its own tail
  Q-values change → targets change → Q-values change → ...

SOLUTION: Use a frozen "target network" for computing targets

Every 10 episodes:
  target_network = main_network  (copy weights)

During training:
  target = reward + gamma × target_network(next_state)
  
  main_network learns toward STABLE targets
  → Much more stable training!
```

---

## DQN vs Linear Methods

| Aspect | Feature-Based Q-Learning | DQN |
|--------|-------------------------|-----|
| Function | Linear: w · features | Non-linear: neural network |
| Pattern learning | Linear only | Any pattern |
| Training stability | Very stable | Can be unstable |
| Compute cost | Very low | High (GPU recommended) |
| Data needed | Hundreds of episodes | Thousands of episodes |
| Feature engineering | Critical | Less critical (learns features) |

---

## Key Limitations

```
1. DATA HUNGRY
   Needs thousands of experiences to learn
   Small datasets → overfitting
   
2. UNSTABLE TRAINING
   Neural networks can "forget" old patterns
   Replay buffer + target network help, but not perfect
   
3. HYPERPARAMETER SENSITIVE
   Learning rate, epsilon decay, hidden size all matter
   Poor choices → no learning or collapse

4. COMPUTE EXPENSIVE
   Training requires many forward/backward passes
   GPU recommended for large networks

5. EXPLORATION CHALLENGE
   Epsilon-greedy is simple but inefficient
   Better methods exist (e.g., curiosity-driven exploration)
```

---

## Connecting to Real RL Methods Guide

From the [RL Methods Guide](../../RL_METHODS_GUIDE.md#level-4-deep-q-network-dqn):

> **What:** Replace the Q-table/linear function with a neural network.
> **When:** States are high-dimensional (images, text, complex data) or have non-linear relationships.
> **Key advantage:** Learns non-linear patterns automatically.
> **Key limitation:** Unstable, needs careful tuning, expensive.

---

## Real-Life Applications

| Scenario | State Size | Why DQN Works |
|----------|-----------|---------------|
| Atari games | 100,800 pixels | Raw image input, non-linear patterns |
| Self-driving | Camera + lidar + GPS | Multi-sensor fusion |
| Robot manipulation | Camera + joints | Vision-based control |
| Game AI (Go, Chess) | Board positions | Complex strategic patterns |
| Advanced chatbot | Text embeddings | Semantic understanding |

---

## What DQN Learns Over Time

```
After 50 episodes (random exploration):
  Q-values: All similar (~0)
  Behavior: Random actions
  Buffer: 1,500 experiences

After 200 episodes (starting to learn):
  Q-values: Different per action
  Behavior: Prefers some actions
  Buffer: 6,000 experiences
  Learns: "empathetic works when mood is low"

After 500 episodes (confident):
  Q-values: Accurate predictions
  Behavior: Consistent policy
  Buffer: 15,000 experiences
  Learns: Complex non-linear patterns like "sad × night × low_engagement → validate"
```

---

## DQN vs Q-Table

```
Q-TABLE (Level 2):
  Stores: Explicit value for each (state, action) pair
  Learns: Only from states it has SEEN
  Size: Grows with number of states
  Good: Small, discrete state spaces

DQN (Level 4):
  Stores: Neural network weights (fixed size)
  Learns: Generalizes to UNSEEN states
  Size: Fixed (network architecture)
  Good: Large, continuous, high-dimensional state spaces
```

---

## Next Step

When you need continuous actions (not just discrete choices) or want to train LLM outputs, move to **Level 5: Policy Gradient / PPO** which learns the policy directly.

---

**Previous:** [B4 — Feature-Based Q-Learning](b4-feature-based-q-learning.md) · **Next:** *B6 — Production Online Systems* (planned)

**Up:** [Track B Index](index.md) · **Reference:** [RL Methods Guide — Level 4](../../RL_METHODS_GUIDE.md#level-4-deep-q-network-dqn)
