# B3 — Q-Learning (Tabular)

> **Track B · Guide 4 of 6** — [Track B Index](index.md) · [RL Methods Guide](../../RL_METHODS_GUIDE.md)

## What You'll Learn

Q-Learning is the first "real" RL algorithm. Unlike bandits that pick one action per message, Q-Learning handles **sequences** — where what you do NOW affects what happens NEXT.

---

## The Concept

```
┌─────────────────────────────────────────────────────────────────┐
│                     Q-LEARNING (Tabular)                        │
│                                                                 │
│   States: [greeting, ask_info, troubleshoot, confirm, close]    │
│   Actions: [ask, answer, confirm, escalate, close]              │
│                                                                 │
│   Agent is in a STATE                                           │
│        │                                                        │
│        ├── Choose action (ε-greedy: explore OR exploit)         │
│        │                                                        │
│        ├── Take action → Observe REWARD + NEW STATE             │
│        │                                                        │
│        └── Update Q-table: "How good was that move?"            │
│                                                                 │
│   Q-Table learns: "In state X, action Y leads to reward Z"     │
└─────────────────────────────────────────────────────────────────┘
```

---

## How It's Different From Bandits

```
BANDITS (Levels 0-1b):
  Each decision is INDEPENDENT.
  "What's the best action RIGHT NOW?"
  No concept of "what happens next."

Q-LEARNING (Level 2+):
  Decisions are SEQUENTIAL.
  "What action leads to the BEST FUTURE?"
  What you do NOW affects what's available NEXT.

Example:
  Bandit: "Should I be caring or playful?"
          (one-shot, doesn't matter what came before)

  Q-Learn: "I'm in 'sad_user' state. If I'm caring now,
           user might open up (new state). From 'opened_up',
           I can comfort them (big reward). If I'm playful
           now, user leaves (dead end, no reward)."
```

---

## The Q-Table

```
The Q-Table is a cheat sheet:

                    ask    answer  confirm  escalate  close
                ┌───────┬────────┬────────┬─────────┬───────┐
  greeting      │  2.1  │  0.5   │  0.0   │  0.0    │ -5.0  │
  ask_info      │  1.0  │  3.5   │  0.2   │ -1.0    │ -5.0  │
  troubleshoot  │  0.8  │  5.2   │  2.0   │  1.5    │ -3.0  │
  confirm       │ -1.0  │  1.0   │  8.0   │  0.5    │  2.0  │
  close         │ -5.0  │ -5.0   │  1.0   │ -5.0    │  9.0  │
                └───────┴────────┴────────┴─────────┴───────┘

In "troubleshoot" state:
  answer = 5.2 (best!) → Pick "answer"
  confirm = 2.0 → Good fallback
  escalate = 1.5 → Ok if stuck
```

---

## The Q-Learning Update Rule

```
Q(s, a) ← Q(s, a) + α × [reward + γ × max(Q(s', a')) - Q(s, a)]

Translation:
  "Move my estimate toward what ACTUALLY happened,
   weighted by how much I trust new information (α),
   and considering the BEST possible future (γ × max Q)."

  α (alpha) = learning rate (0.1 = conservative, 0.5 = aggressive)
  γ (gamma) = discount factor (0.9 = value future, 0.1 = only care about now)
```

### Step-by-Step Update

```
Current state: "troubleshoot"
Action taken: "answer"
Reward received: +3 (user provided info)
New state: "confirm"

Old Q(troubleshoot, answer) = 5.0
α = 0.1, γ = 0.9
max Q(confirm, *) = 8.0 (from "confirm" action)

New Q = 5.0 + 0.1 × [3.0 + 0.9 × 8.0 - 5.0]
      = 5.0 + 0.1 × [3.0 + 7.2 - 5.0]
      = 5.0 + 0.1 × 5.2
      = 5.0 + 0.52
      = 5.52

The Q-value increased because the future looks good!
```

---

## Python Example

```python
import random
from collections import defaultdict

class QLearningAgent:
    """Tabular Q-Learning Agent."""
    
    def __init__(
        self,
        states: list[str],
        actions: list[str],
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1
    ):
        self.states = states
        self.actions = actions
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate
        
        # Q-table: Q[state][action] = value
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Track stats
        self.history = []
    
    def select_action(self, state: str) -> str:
        """Pick action using ε-greedy policy."""
        if random.random() < self.epsilon:
            # Explore: random action
            return random.choice(self.actions)
        
        # Exploit: best known action
        q_values = self.q_table[state]
        if not q_values:
            return random.choice(self.actions)
        
        return max(self.actions, key=lambda a: q_values.get(a, 0.0))
    
    def update(self, state: str, action: str, reward: float, next_state: str, done: bool):
        """Q-Learning update rule."""
        current_q = self.q_table[state][action]
        
        if done:
            # Terminal state: no future reward
            target = reward
        else:
            # Best future Q-value
            max_future_q = max(
                self.q_table[next_state].get(a, 0.0) for a in self.actions
            )
            target = reward + self.gamma * max_future_q
        
        # Update Q-value
        self.q_table[state][action] = current_q + self.alpha * (target - current_q)
        
        # Track history
        self.history.append({
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "q_value": self.q_table[state][action]
        })
    
    def get_policy(self) -> dict:
        """Extract best action per state."""
        policy = {}
        for state in self.states:
            q_values = self.q_table[state]
            if q_values:
                best_action = max(q_values, key=q_values.get)
                policy[state] = {
                    "best_action": best_action,
                    "q_values": dict(q_values)
                }
            else:
                policy[state] = {"best_action": "unknown", "q_values": {}}
        return policy
    
    def print_q_table(self):
        """Pretty-print the Q-table."""
        print(f"\n{'State':<15}", end="")
        for action in self.actions:
            print(f"{action:<12}", end="")
        print("\n" + "-" * 75)
        
        for state in self.states:
            print(f"{state:<15}", end="")
            for action in self.actions:
                val = self.q_table[state].get(action, 0.0)
                print(f"{val:<12.2f}", end="")
            print()


# --- Simulated Environment ---
class ConversationEnv:
    """Simple conversation flow environment."""
    
    def __init__(self):
        self.states = ["greeting", "ask_info", "troubleshoot", "confirm", "close"]
        self.actions = ["ask", "answer", "confirm", "escalate", "close"]
        
        # Transition rewards (state, action) -> (reward, next_state, done)
        self.transitions = {
            ("greeting", "ask"):        (1.0, "ask_info", False),
            ("greeting", "answer"):     (0.5, "greeting", False),
            ("greeting", "close"):      (-5.0, "close", True),
            
            ("ask_info", "ask"):        (-1.0, "ask_info", False),  # Repeated question
            ("ask_info", "answer"):     (3.0, "troubleshoot", False),
            ("ask_info", "confirm"):    (0.0, "ask_info", False),
            ("ask_info", "close"):      (-5.0, "close", True),
            
            ("troubleshoot", "ask"):    (-2.0, "troubleshoot", False),
            ("troubleshoot", "answer"): (5.0, "confirm", False),
            ("troubleshoot", "confirm"): (2.0, "confirm", False),
            ("troubleshoot", "escalate"): (1.0, "close", True),
            ("troubleshoot", "close"):  (-3.0, "close", True),
            
            ("confirm", "confirm"):    (8.0, "close", True),  # User confirms!
            ("confirm", "ask"):         (-1.0, "confirm", False),
            ("confirm", "close"):       (2.0, "close", True),
        }
        
        self.current_state = "greeting"
    
    def reset(self) -> str:
        """Start new conversation."""
        self.current_state = "greeting"
        return self.current_state
    
    def step(self, action: str) -> tuple:
        """Take action, return (reward, next_state, done)."""
        key = (self.current_state, action)
        
        if key in self.transitions:
            reward, next_state, done = self.transitions[key]
        else:
            # Invalid action: small penalty
            reward = -1.0
            next_state = self.current_state
            done = False
        
        # Add noise to rewards
        reward += random.gauss(0, 0.5)
        
        self.current_state = next_state
        return reward, next_state, done


# --- Training ---
def train(n_episodes: int = 500):
    """Train Q-learning agent on conversation environment."""
    env = ConversationEnv()
    agent = QLearningAgent(
        states=env.states,
        actions=env.actions,
        alpha=0.1,
        gamma=0.9,
        epsilon=0.2  # More exploration during training
    )
    
    rewards_per_episode = []
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        max_steps = 20  # Prevent infinite loops
        
        while not done and steps < max_steps:
            action = agent.select_action(state)
            reward, next_state, done = env.step(action)
            
            agent.update(state, action, reward, next_state, done)
            
            total_reward += reward
            state = next_state
            steps += 1
        
        rewards_per_episode.append(total_reward)
        
        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = sum(rewards_per_episode[-100:]) / 100
            print(f"Episode {episode + 1}: avg reward = {avg_reward:.2f}")
    
    return agent, rewards_per_episode


# Run training
agent, rewards = train(500)

# See what it learned
print("\n=== LEARNED Q-TABLE ===")
agent.print_q_table()

print("\n=== LEARNED POLICY ===")
policy = agent.get_policy()
for state, info in policy.items():
    print(f"  {state}: {info['best_action']} (Q={info['q_values'].get(info['best_action'], 0):.2f})")

# Expected learned policy:
#   greeting → ask        (start conversation)
#   ask_info → answer     (get info from user)
#   troubleshoot → answer (solve problem)
#   confirm → confirm     (get confirmation)
#   close → close         (end gracefully)
```

---

## When to Use Q-Learning

| Use Case | Why |
|----------|-----|
| Conversation flow | Sequence of decisions matters |
| Customer support routing | Greeting → troubleshoot → resolve → close |
| Game AI | Board games, turn-based strategy |
| Onboarding flows | Step 1 → step 2 → step 3 → complete |
| Escalation timing | When to escalate vs. keep trying |

---

## Key Concepts Visualized

### Exploration vs Exploitation in Sequences

```
Episode 1 (random, exploring):
  greeting → close → Done! (bad: -5.0 reward)
  Learns: "Don't close immediately"

Episode 50 (mostly exploring):
  greeting → ask → ask_info → close → Done! (bad: -4.0 reward)
  Learns: "Don't close during troubleshooting"

Episode 200 (mostly exploiting):
  greeting → ask → answer → answer → confirm → Done! (good: +15.0)
  Learns: "This sequence works!"

Episode 500 (confident):
  greeting → ask → answer → answer → confirm → confirm → Done! (good: +19.0)
  Optimal policy learned!
```

### Discount Factor (γ) Effect

```
γ = 0.9 (value future rewards):
  "I'll take a small hit now (-1 ask) to get a big reward later (+8 confirm)"

γ = 0.1 (only care about immediate):
  "I only care about THIS step's reward, ignore future"
  → Behaves like a bandit!

γ = 0.0 (purely greedy):
  "Immediate reward only, no planning"
```

---

## Q-Learning vs Bandits

| Aspect | Bandits (Levels 0-1b) | Q-Learning |
|--------|----------------------|------------|
| Decisions | Independent | Sequential |
| State | None or context only | Explicit state tracking |
| Planning | None (myopic) | Looks ahead (γ) |
| Storage | One value per action | One value per (state, action) |
| Complexity | Very low | Low-medium |

---

## What Q-Learning Learns Over Time

```
Episode 1 (blank Q-table):
  All Q-values = 0.0
  Agent picks randomly, gets mostly negative rewards

Episode 100 (starting to learn):
  greeting → ask:     1.5  (positive: leads to info gathering)
  greeting → close:  -3.2  (negative: too early to close)
  ask_info → answer:  2.8  (positive: getting info works)
  confirm → confirm:  5.1  (very positive: leads to resolution)

Episode 500 (confident):
  greeting → ask:     2.1
  ask_info → answer:  5.2
  troubleshoot → answer: 7.8
  confirm → confirm: 9.0  (optimal!)
  close → close:      9.0
```

---

## Key Limitation: State Explosion

```
With few states:
  5 states × 5 actions = 25 Q-values → Easy!

With many states:
  (mood × topic × time × engagement × relationship_stage) = thousands
  thousands × 5 actions = tens of thousands of Q-values
  Most never visited → Agent can't learn!

Solution: Use Feature-Based Q-Learning (Level 3) or DQN (Level 4)
```

---

## Connecting to Real RL Methods Guide

From the [RL Methods Guide](../../RL_METHODS_GUIDE.md#level-2-q-learning-tabular):

> **What:** Score every (state, action) pair. Learn from transitions.
> **When:** Discrete states + discrete actions + you care about SEQUENCES.
> **Key advantage:** Learns multi-step strategies.
> **Key limitation:** Only works with discrete, small state spaces.

---

## Real-Life Applications

| Scenario | States | Actions | Why Q-Learning Works |
|----------|--------|---------|---------------------|
| Customer service chatbot | 5 flow stages | 5 response types | Clear sequential flow |
| Game AI (chess) | Board positions | Legal moves | Each move affects future |
| Traffic signals | Traffic density levels | Green/red timing | Cycle affects next cycle |
| Onboarding wizard | Progress steps | Skip/tip/require | Step sequence matters |

---

## Next Step

When your state space grows too large for a table, move to **Level 3: Feature-Based Q-Learning** which uses features instead of exhaustive lookup.

---

**See Also**: [RL Methods Guide - Level 2](../../RL_METHODS_GUIDE.md#level-2-q-learning-tabular)
