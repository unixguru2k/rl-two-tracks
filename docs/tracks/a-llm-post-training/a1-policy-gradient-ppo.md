# A1 — Policy Gradient / PPO

> **Track A · Guide 2 of 7** — [Track A Index](index.md) · [RL Methods Guide](../../RL_METHODS_GUIDE.md)

## What You'll Learn

Policy Gradient methods learn the **policy directly** — a probability distribution over actions — instead of learning Q-values. PPO (Proximal Policy Optimization) is the most popular variant, used to train ChatGPT, Claude, and other modern AI systems.

---

## The Concept

```
┌─────────────────────────────────────────────────────────────────┐
│               POLICY GRADIENT / PPO                             │
│                                                                 │
│   Instead of: Learn Q-values → derive policy                    │
│   Use:        Learn POLICY directly (probabilities)             │
│                                                                 │
│   Input: State (context, features)                              │
│   Output: Probability distribution over actions                 │
│                                                                 │
│   Policy(state) = [0.7 empathetic, 0.2 playful, 0.1 distract]  │
│   → Sample action from distribution                             │
│   → Get reward                                                  │
│   → Update policy to make good actions MORE likely              │
└─────────────────────────────────────────────────────────────────┘
```

---

## How It's Different From Value-Based Methods

```
VALUE-BASED (Q-Learning, DQN):
  1. Learn Q(state, action) for each action
  2. Pick action with highest Q-value
  3. Policy is DERIVED from Q-values
  
  Problem: Can't handle continuous actions!
  "What's the optimal temperature? 0.7? 0.73? 0.731?"

POLICY GRADIENT (PPO):
  1. Learn policy directly: π(action | state) = probability
  2. Sample action from distribution
  3. Update policy based on reward
  
  Advantage: Works with continuous actions!
  "Output temperature = 0.73 directly"
```

---

## The Policy Network

```
Instead of outputting Q-values for each action,
output PROBABILITIES for each action:

State → Neural Network → [0.7, 0.2, 0.1]
                           ↑    ↑    ↑
                        empathetic  playful  distract

Then SAMPLE from this distribution:
  - 70% chance: empathetic
  - 20% chance: playful
  - 10% chance: distract

After good reward for "empathetic":
  Policy shifts → [0.8, 0.15, 0.05]
  
  Empathetic becomes MORE likely!
```

---

## PPO: The Key Innovation

```
PROBLEM with vanilla Policy Gradient:
  One bad experience can drastically change policy
  → Policy oscillates, never converges

PPO SOLUTION: "Don't change too much at once"

  Old policy: π_old(action | state) = 0.7
  New policy: π_new(action | state) = 0.8
  
  Ratio = π_new / π_old = 0.8 / 0.7 = 1.14
  
  PPO clips this ratio to [0.8, 1.2] (example)
  
  → Policy can only change a LITTLE per update
  → Stable, reliable training
```

### The PPO Update Rule

```
L_CLIP = E[min(ratio × advantage, clip(ratio, 1-ε, 1+ε) × advantage)]

Translation:
  ratio = π_new / π_old (how much policy changed)
  advantage = reward - baseline (how much better than expected)
  ε = 0.2 (clip range)
  
  If ratio > 1+ε: Clamp to 1+ε (don't increase too much)
  If ratio < 1-ε: Clamp to 1-ε (don't decrease too much)
  
  → Policy updates are SMALL and STABLE
```

---

## Python Example

```python
import numpy as np

class PolicyNetwork:
    """Simple policy network for discrete actions."""
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 64,
        learning_rate: float = 0.001
    ):
        self.action_size = action_size
        self.lr = learning_rate
        
        # Network weights
        self.W1 = np.random.randn(state_size, hidden_size) * np.sqrt(2.0 / state_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, action_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(action_size)
        
        # Store trajectory for PPO update
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
    
    def forward(self, state: np.ndarray) -> np.ndarray:
        """Forward pass: state → action probabilities."""
        h = np.maximum(0, state @ self.W1 + self.b1)  # ReLU
        logits = h @ self.W2 + self.b2
        
        # Softmax to get probabilities
        logits = logits - np.max(logits)  # Numerical stability
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits)
    
    def select_action(self, state: np.ndarray) -> tuple:
        """Sample action from policy distribution."""
        probs = self.forward(state)
        
        # Sample action
        action = np.random.choice(self.action_size, p=probs)
        log_prob = np.log(probs[action] + 1e-8)
        
        # Store for training
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        
        return action, log_prob
    
    def compute_returns(self, gamma: float = 0.99) -> np.ndarray:
        """Compute discounted returns."""
        returns = []
        R = 0
        
        for r in reversed(self.rewards):
            R = r + gamma * R
            returns.insert(0, R)
        
        returns = np.array(returns)
        # Normalize returns (stabilizes training)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        
        return returns
    
    def ppo_update(
        self,
        old_log_probs: np.ndarray,
        clip_epsilon: float = 0.2,
        epochs: int = 4
    ):
        """PPO update with clipping."""
        states = np.array(self.states)
        actions = np.array(self.actions)
        returns = self.compute_returns()
        
        for _ in range(epochs):
            # Forward pass
            probs = self.forward(states)
            
            # New log probabilities
            new_log_probs = np.log(probs[np.arange(len(actions)), actions] + 1e-8)
            
            # Ratio
            ratio = np.exp(new_log_probs - old_log_probs)
            
            # Clipped surrogate
            surr1 = ratio * returns
            surr2 = np.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * returns
            loss = -np.mean(np.minimum(surr1, surr2))
            
            # Backpropagation (simplified)
            self._backward(states, actions, probs, returns, ratio, clip_epsilon)
        
        # Clear trajectory
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
    
    def _backward(self, states, actions, probs, returns, ratio, clip_epsilon):
        """Simplified backpropagation."""
        batch_size = len(states)
        
        # Gradient of clipped loss
        clipped = np.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
        grad = np.where(
            ratio * returns > clipped * returns,
            returns,
            clipped * returns
        )
        
        # Softmax gradient
        d_logits = probs.copy()
        d_logits[np.arange(batch_size), actions] -= 1
        d_logits *= grad[:, np.newaxis]
        
        # Backprop through layers
        h = np.maximum(0, states @ self.W1 + self.b1)
        d_W2 = h.T @ d_logits / batch_size
        d_b2 = np.mean(d_logits, axis=0)
        
        d_h = d_logits @ self.W2.T
        d_h[h <= 0] = 0  # ReLU gradient
        
        d_W1 = states.T @ d_h / batch_size
        d_b1 = np.mean(d_h, axis=0)
        
        # Update
        self.W2 -= self.lr * d_W2
        self.b2 -= self.lr * d_b2
        self.W1 -= self.lr * d_W1
        self.b1 -= self.lr * d_b1


class ValueNetwork:
    """Baseline value network (reduces variance)."""
    
    def __init__(self, state_size: int, hidden_size: int = 64, lr: float = 0.001):
        self.lr = lr
        self.W1 = np.random.randn(state_size, hidden_size) * np.sqrt(2.0 / state_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, 1) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(1)
    
    def predict(self, state: np.ndarray) -> float:
        h = np.maximum(0, state @ self.W1 + self.b1)
        return float(h @ self.W2 + self.b2)
    
    def update(self, state: np.ndarray, target: float):
        h = np.maximum(0, state @ self.W1 + self.b1)
        prediction = h @ self.W2 + self.b2
        
        error = prediction - target
        
        d_W2 = np.outer(h, error)
        d_b2 = error
        
        d_h = error * self.W2.flatten()
        d_h[h <= 0] = 0
        
        d_W1 = np.outer(state, d_h)
        d_b1 = d_h
        
        self.W2 -= self.lr * d_W2
        self.b2 -= self.lr * d_b2
        self.W1 -= self.lr * d_W1
        self.b1 -= self.lr * d_b1


# --- Simulated Environment ---
class ConversationPPOEnv:
    """Conversation environment for policy gradient."""
    
    def __init__(self):
        self.actions = ["empathetic", "playful", "distract", "validate", "question"]
        self.state_size = 8
        self.steps = 0
        self.max_steps = 50
    
    def _get_state(self) -> np.ndarray:
        return np.array([
            np.random.uniform(-1, 1),   # mood
            np.random.uniform(0, 1),    # time
            np.random.uniform(0, 1),    # engagement
            np.random.uniform(-1, 1),   # sentiment
            np.random.uniform(0, 1),    # topic_complexity
            np.random.uniform(0, 1),    # relationship_stage
            np.random.uniform(0, 1),    # conversation_length
            np.random.uniform(-1, 1),   # user_energy
        ])
    
    def reset(self) -> np.ndarray:
        self.steps = 0
        return self._get_state()
    
    def step(self, action: int) -> tuple:
        state = self._get_state()
        
        # Reward function
        mood = state[0]
        engagement = state[2]
        sentiment = state[3]
        
        if action == 0:  # empathetic
            reward = 3.0 * (1 - mood) * engagement + 1.0
        elif action == 1:  # playful
            reward = 2.0 * mood * engagement
        elif action == 2:  # distract
            reward = 1.5 * (1 - engagement)
        elif action == 3:  # validate
            reward = 2.5 * (1 - mood) * (1 + sentiment)
        else:  # question
            reward = 1.0 * engagement
        
        reward += np.random.normal(0, 0.5)
        
        self.steps += 1
        done = self.steps >= self.max_steps
        
        return reward, self._get_state(), done


# --- Training ---
def train_ppo(
    n_episodes: int = 300,
    gamma: float = 0.99,
    clip_epsilon: float = 0.2,
    ppo_epochs: int = 4
):
    """Train PPO agent."""
    env = ConversationPPOEnv()
    
    policy = PolicyNetwork(
        state_size=env.state_size,
        action_size=len(env.actions),
        hidden_size=64,
        learning_rate=0.0003
    )
    
    value_net = ValueNetwork(
        state_size=env.state_size,
        hidden_size=64,
        lr=0.001
    )
    
    rewards_per_episode = []
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        # Collect trajectory
        while not done:
            action, log_prob = policy.select_action(state)
            reward, next_state, done = env.step(action)
            
            policy.rewards.append(reward)
            total_reward += reward
            state = next_state
        
        rewards_per_episode.append(total_reward)
        
        # Store old log probs for PPO
        old_log_probs = np.array(policy.log_probs)
        
        # PPO update
        policy.ppo_update(old_log_probs, clip_epsilon, ppo_epochs)
        
        if (episode + 1) % 50 == 0:
            avg = np.mean(rewards_per_episode[-50:])
            print(f"Episode {episode + 1}: avg_reward = {avg:.2f}")
    
    return policy, rewards_per_episode


# Run training
print("Training PPO agent...")
policy, rewards = train_ppo(300)

# Test the learned policy
print("\n=== TESTING LEARNED POLICY ===")
action_names = ["empathetic", "playful", "distract", "validate", "question"]

test_states = [
    ("Sad user", np.array([-0.8, 0.9, 0.3, -0.5, 0.5, 0.3, 0.2, -0.3])),
    ("Happy user", np.array([0.8, 0.2, 0.8, 0.6, 0.3, 0.7, 0.5, 0.5])),
    ("Neutral, low engagement", np.array([0.0, 0.5, 0.2, 0.0, 0.7, 0.5, 0.3, 0.0])),
]

for label, state in test_states:
    probs = policy.forward(state)
    best_action = action_names[np.argmax(probs)]
    print(f"\n{label}:")
    print(f"  Best action: {best_action}")
    print(f"  Probabilities: {', '.join(f'{a}={p:.2f}' for a, p in zip(action_names, probs))}")
```

---

## When to Use Policy Gradient / PPO

| Use Case | Why |
|----------|-----|
| Continuous actions | Temperature, length, intensity |
| LLM fine-tuning | Token probabilities are continuous |
| Stochastic policies | Need probability distributions |
| Complex action spaces | High-dimensional or structured actions |

---

## Key Concepts Visualized

### Policy vs Value Methods

```
VALUE-BASED (Q-Learning, DQN):
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   State     │───→│ Q-values    │───→│ Argmax      │───→ Action
│             │    │ [2.1, 5.3,  │    │ Pick highest│
│             │    │  1.2, 3.8]  │    │             │
└─────────────┘    └─────────────┘    └─────────────┘

POLICY GRADIENT (PPO):
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   State     │───→│ Probabilities│───→│ Sample      │───→ Action
│             │    │ [0.1, 0.6,  │    │ Random draw │
│             │    │  0.05, 0.25]│    │ from dist.  │
└─────────────┘    └─────────────┘    └─────────────┘
```

### PPO Clipping

```
WITHOUT CLIPPING:
  Good reward → Large policy shift → Unstable
  
  Episode 1: π(empathetic) = 0.3
  Episode 2: Reward +10! → π(empathetic) = 0.9 (huge shift!)
  Episode 3: Reward -5 → π(empathetic) = 0.1 (overcorrected!)
  → Policy oscillates forever

WITH PPO CLIPPING (ε = 0.2):
  Good reward → Small policy shift → Stable
  
  Episode 1: π(empathetic) = 0.3
  Episode 2: Reward +10! → π(empathetic) = 0.36 (clipped!)
  Episode 3: Reward +8 → π(empathetic) = 0.43 (steady increase)
  → Policy converges smoothly
```

### Advantage Estimation

```
REWARD ALONE:
  "I got +5 reward" → Good? Bad? Depends on context!

ADVANTAGE:
  "I got +5 reward, but expected +3" → Advantage = +2 (better than expected!)
  "I got +5 reward, but expected +8" → Advantage = -3 (worse than expected!)

  advantage = reward - baseline
  
  → Updates are RELATIVE to expectations
  → Reduces variance, stabilizes training
```

---

## PPO vs DQN

| Aspect | DQN | PPO |
|--------|-----|-----|
| Learns | Q-values (action values) | Policy (action probabilities) |
| Action selection | Argmax of Q-values | Sample from distribution |
| Continuous actions | ❌ Difficult | ✅ Natural |
| Training stability | Moderate (replay buffer) | High (clipped updates) |
| On-policy/Off-policy | Off-policy | On-policy |
| Data efficiency | Higher (reuses data) | Lower (new data each update) |
| Used for | Games, discrete control | Robotics, LLM training |

---

## Key Limitations

```
1. ON-POLICY (data hungry)
   Can't reuse old experiences!
   Each update needs FRESH data from current policy
   → Need many episodes to train

2. HIGH VARIANCE
   Policy gradient estimates are noisy
   Need many samples to get reliable gradients
   → Slow learning

3. LOCAL OPTIMA
   Can get stuck in suboptimal policies
   No exploration bonus like UCB
   → Need careful exploration scheduling

4. HYPERPARAMETER SENSITIVE
   Clip epsilon, learning rate, batch size all matter
   Poor choices → no learning or collapse

5. COMPUTE EXPENSIVE
   Multiple forward passes per update (PPO epochs)
   GPU recommended for large networks
```

---

## Connecting to Real RL Methods Guide

From the [RL Methods Guide](../../RL_METHODS_GUIDE.md#level-5-policy-gradient--ppo):

> **What:** Learn the POLICY directly (probability distribution over actions).
> **When:** Continuous action spaces, or you want to optimize LLM output.
> **Key advantage:** Handles continuous/complex action spaces.
> **Key limitation:** Slow, expensive, complex to debug.

---

## Real-Life Applications

| Scenario | Action Space | Why PPO |
|----------|-------------|---------|
| Robotic arm | Joint torques (continuous) | Can't discretize "0.73 Nm" |
| Trading bot | Allocation percentages | Continuous (0-100%) |
| Music generation | Note frequency, duration | Continuous output |
| ChatGPT training | Token probabilities | 50K+ vocabulary, continuous |
| Game AI | Continuous control | Smooth movements |

---

## What PPO Learns Over Time

```
After 50 episodes (exploring):
  Policy: Roughly uniform [0.2, 0.2, 0.2, 0.2, 0.2]
  Behavior: Mostly random
  Rewards: Low, high variance

After 150 episodes (learning):
  Policy: Shifting [0.4, 0.3, 0.1, 0.15, 0.05]
  Behavior: Prefers empathetic and playful
  Rewards: Improving, less variance

After 300 episodes (converged):
  Policy: Confident [0.7, 0.2, 0.03, 0.05, 0.02]
  Behavior: Mostly empathetic, sometimes playful
  Rewards: High, stable
```

---

## PPO vs Other Policy Methods

```
VANILLA POLICY GRADIENT (REINFORCE):
  + Simple
  - High variance
  - Unstable
  = Rarely used in practice

ACTOR-CRITIC (A2C):
  + Lower variance (uses value baseline)
  - Still unstable with large updates
  = Better, but fragile

PPO:
  + Low variance (advantage)
  + Stable (clipped updates)
  + Simple to implement
  = Industry standard (OpenAI, Anthropic, DeepMind)
```

---

## The Full RL Stack

```
Level 0: Random        → No learning
Level 1a: Classic MAB   → Best global default
Level 1b: LinUCB        → Personalize with context
Level 2: Q-Learning     → Sequential decisions
Level 3: Feature Q      → Scale to many states
Level 4: DQN            → Non-linear patterns
Level 5: PPO            → Continuous actions, LLM training ← YOU ARE HERE
Level 6: RLHF           → Train foundation models (uses PPO!)
```

---

## Next Step

PPO is the engine behind **Level 6: RLHF**. If you want to understand how ChatGPT and Claude are trained, see how PPO combines with reward models from human feedback.

---

**Previous:** [A0 — Hands-On Post-Training](a0-hands-on-post-training.md) · **Next:** [A2 — RLHF](a2-rlhf.md)

**Up:** [Track A Index](index.md) · **Reference:** [RL Methods Guide — Level 5](../../RL_METHODS_GUIDE.md#level-5-policy-gradient--ppo)
