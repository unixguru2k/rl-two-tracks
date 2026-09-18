# A2 — RLHF (Reinforcement Learning from Human Feedback)

> **Track A · Guide 3 of 6** — [Track A Index](index.md) · [RL Methods Guide](../../RL_METHODS_GUIDE.md)

## What You'll Learn

RLHF is the technique that transformed raw language models into helpful, harmless assistants like ChatGPT and Claude. It trains a **reward model** from human preferences, then uses PPO to optimize the LLM to maximize that reward.

---

## The Concept

```
┌─────────────────────────────────────────────────────────────────┐
│                    RLHF PIPELINE                                 │
│                                                                 │
│   Phase 1: Collect human preferences                            │
│     "Response A vs Response B — which is better?"               │
│     Human picks A → A > B preference data                       │
│                                                                 │
│   Phase 2: Train reward model                                   │
│     Learn a model that SCORES responses like humans would       │
│     Input: (prompt, response) → Output: quality score           │
│                                                                 │
│   Phase 3: Optimize LLM with PPO                                │
│     Generate responses → Score with reward model → Update LLM   │
│     LLM learns to produce high-scoring responses                │
│                                                                 │
│   Result: LLM that is HELPFUL, HARMLESS, HONEST                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why RLHF Exists

```
PROBLEM: Pre-trained LLMs predict the NEXT TOKEN
  Training data: internet text
  Goal: "What word comes next?"
  
  This produces text that is:
    ✅ Fluent and grammatical
    ✅ Factually knowledgeable
    ❌ Not necessarily helpful
    ❌ Not necessarily safe
    ❌ Not aligned with what users WANT

SOLUTION: RLHF aligns the model with human preferences
  Instead of "predict next token"
  Train for "generate response humans PREFER"

  Before RLHF: "To make a bomb, you need..." (completes the pattern)
  After RLHF:  "I can't help with that. Here's something useful instead..."
```

---

## The Three Phases

### Phase 1: Supervised Fine-Tuning (SFT)

```
Start with a pre-trained LLM (GPT-4 base, Llama, etc.)

Fine-tune on high-quality human-written demonstrations:
  Prompt: "Explain quantum computing"
  Response: "Quantum computing uses quantum bits (qubits) that can..."

This teaches the model the FORMAT of good responses.
But we can't write demonstrations for every scenario.
```

### Phase 2: Reward Model Training

```
Show humans pairs of responses:
  Prompt: "How do I handle a sad friend?"
  
  Response A: "Just tell them to cheer up."
  Response B: "Listen without judgment. Let them know you're there."
  
  Human preference: B > A

Collect thousands of these comparisons.
Train a reward model to predict human preferences:
  reward_model(prompt, response) → score
  
  reward_model("sad friend", "tell them to cheer up") → 2.1
  reward_model("sad friend", "listen without judgment") → 8.7
```

### Phase 3: PPO Optimization

```
Use the reward model as the reward signal for PPO:

1. LLM generates a response
2. Reward model scores it
3. PPO updates LLM to produce higher-scoring responses

   LLM → "Just tell them to cheer up" → reward: 2.1
   PPO: "That was bad. Adjust weights to make this less likely."

   LLM → "Listen without judgment" → reward: 8.7
   PPO: "That was good. Adjust weights to make this more likely."

After thousands of iterations: LLM consistently produces preferred responses.
```

---

## Python Example (Simplified RLHF Pipeline)

```python
import numpy as np
from collections import deque

# ============================================================
# This is a SIMPLIFIED illustration of the RLHF concept.
# Real RLHF uses transformer models with billions of parameters.
# Here we use small networks to show the PIPELINE, not the scale.
# ============================================================


# --- Phase 1: Simulated Preference Data Collection ---
class PreferenceCollector:
    """Simulates collecting human preferences between response pairs."""
    
    def __init__(self):
        # Simulated "human preference" function
        # In reality, this is done by human annotators
        self.quality_signals = {
            "helpful": 2.0,
            "empathetic": 1.5,
            "honest": 1.8,
            "concise": 0.8,
            "creative": 0.5,
            "harmful": -5.0,
            "misleading": -3.0,
            "rude": -4.0,
        }
    
    def score_response(self, response_features: dict) -> float:
        """Simulate human quality judgment."""
        score = 0
        for feature, value in response_features.items():
            if feature in self.quality_signals:
                score += value * self.quality_signals[feature]
        return score + np.random.normal(0, 0.5)  # Human inconsistency
    
    def collect_preference(self, prompt: str, response_a_features: dict, response_b_features: dict) -> tuple:
        """Collect a preference: which response is better?"""
        score_a = self.score_response(response_a_features)
        score_b = self.score_response(response_b_features)
        
        preferred = "a" if score_a > score_b else "b"
        return preferred, score_a, score_b


# --- Phase 2: Reward Model ---
class RewardModel:
    """Learns to predict human quality preferences."""
    
    def __init__(self, input_size: int, lr: float = 0.001):
        self.lr = lr
        # Simple 2-layer network
        self.W1 = np.random.randn(input_size, 32) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(32)
        self.W2 = np.random.randn(32, 1) * np.sqrt(2.0 / 32)
        self.b2 = np.zeros(1)
    
    def score(self, features: np.ndarray) -> float:
        """Predict reward score for a response."""
        h = np.maximum(0, features @ self.W1 + self.b1)
        return float(h @ self.W2 + self.b2)
    
    def train_on_preference(self, features_a: np.ndarray, features_b: np.ndarray, preferred: str):
        """Train on a single human preference (A vs B)."""
        score_a = self._forward(features_a)
        score_b = self._forward(features_b)
        
        # We want: preferred score > non-preferred score
        if preferred == "a":
            target_a, target_b = 1.0, -1.0
        else:
            target_a, target_b = -1.0, 1.0
        
        # Gradient descent on both
        self._backward(features_a, score_a, target_a)
        self._backward(features_b, score_b, target_b)
    
    def _forward(self, features):
        h = np.maximum(0, features @ self.W1 + self.b1)
        return float(h @ self.W2 + self.b2)
    
    def _backward(self, features, predicted, target):
        error = predicted - target
        h = np.maximum(0, features @ self.W1 + self.b1)
        
        dW2 = np.outer(h, np.array([error]))
        db2 = np.array([error])
        
        d_h = error * self.W2.flatten()
        d_h[h <= 0] = 0
        
        dW1 = np.outer(features, d_h)
        db1 = d_h
        
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2


# --- Phase 3: Policy (LLM surrogate) optimized with PPO ---
class RLHFPolicy:
    """Simplified policy network representing the LLM's response generation."""
    
    def __init__(self, state_size: int, action_size: int, lr: float = 0.0003):
        self.action_size = action_size
        self.lr = lr
        self.W1 = np.random.randn(state_size, 32) * np.sqrt(2.0 / state_size)
        self.b1 = np.zeros(32)
        self.W2 = np.random.randn(32, action_size) * np.sqrt(2.0 / 32)
        self.b2 = np.zeros(action_size)
    
    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        """Get response strategy probabilities."""
        h = np.maximum(0, state @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits)
    
    def select_action(self, state: np.ndarray) -> tuple:
        probs = self.get_action_probs(state)
        action = np.random.choice(self.action_size, p=probs)
        return action, probs[action]
    
    def ppo_update(self, state, action, advantage, old_prob, clip_epsilon=0.2):
        """PPO update step."""
        probs = self.get_action_probs(state)
        new_prob = probs[action]
        
        ratio = new_prob / (old_prob + 1e-8)
        clipped = np.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
        
        # Simplified gradient
        grad = -advantage * (np.eye(self.action_size)[action] - probs)
        dW2 = np.outer(np.maximum(0, state @ self.W1 + self.b1), grad)
        db2 = grad
        
        self.W2 += self.lr * dW2
        self.b2 += self.lr * db2


# --- Response Feature Simulator ---
def generate_response_features(strategy: int, prompt_context: dict) -> np.ndarray:
    """Simulate generating response features for a given strategy and context."""
    # Each strategy produces different quality signals
    strategy_profiles = {
        0: {"helpful": 0.8, "empathetic": 0.9, "honest": 0.7, "concise": 0.6, "creative": 0.3, "harmful": 0.0, "misleading": 0.0, "rude": 0.0},
        1: {"helpful": 0.6, "empathetic": 0.3, "honest": 0.8, "concise": 0.9, "creative": 0.4, "harmful": 0.0, "misleading": 0.0, "rude": 0.1},
        2: {"helpful": 0.9, "empathetic": 0.7, "honest": 0.9, "concise": 0.5, "creative": 0.7, "harmful": 0.0, "misleading": 0.0, "rude": 0.0},
        3: {"helpful": 0.3, "empathetic": 0.2, "honest": 0.5, "concise": 0.4, "creative": 0.8, "harmful": 0.1, "misleading": 0.1, "rude": 0.2},
    }
    
    profile = strategy_profiles.get(strategy, strategy_profiles[0])
    
    # Context influences quality (e.g., empathetic strategy scores higher on empathetic context)
    context_modifier = prompt_context.get("needs_empathy", 0) * 0.3
    
    features = np.array([
        profile["helpful"] + context_modifier,
        profile["empathetic"] + context_modifier,
        profile["honest"],
        profile["concise"],
        profile["creative"],
        profile["harmful"],
        profile["misleading"],
        profile["rude"],
    ])
    
    # Add noise (LLM generation variance)
    features += np.random.normal(0, 0.05, size=features.shape)
    return np.clip(features, 0, 1)


def features_to_quality_vector(features: np.ndarray) -> np.ndarray:
    """Convert response features to input for reward model."""
    return features


# --- Full RLHF Pipeline ---
def run_rlhf_pipeline(n_preference_pairs: int = 500, n_rl_episodes: int = 300):
    """Run the complete RLHF pipeline."""
    
    print("=" * 60)
    print("PHASE 1: Collecting Human Preferences")
    print("=" * 60)
    
    collector = PreferenceCollector()
    preference_data = []
    
    for i in range(n_preference_pairs):
        context = {"needs_empathy": np.random.uniform(0, 1)}
        
        # Generate two random responses
        strategy_a = np.random.randint(0, 4)
        strategy_b = np.random.randint(0, 4)
        
        features_a = generate_response_features(strategy_a, context)
        features_b = generate_response_features(strategy_b, context)
        
        preferred, score_a, score_b = collector.collect_preference("prompt", 
            {k: v for k, v in zip(["helpful", "empathetic", "honest", "concise", "creative", "harmful", "misleading", "rude"], features_a)},
            {k: v for k, v in zip(["helpful", "empathetic", "honest", "concise", "creative", "harmful", "misleading", "rude"], features_b)}
        )
        
        preference_data.append((features_a, features_b, preferred))
    
    print(f"  Collected {n_preference_pairs} preference pairs")
    
    print("\n" + "=" * 60)
    print("PHASE 2: Training Reward Model")
    print("=" * 60)
    
    reward_model = RewardModel(input_size=8, lr=0.005)
    
    # Train reward model on preferences
    for epoch in range(10):
        correct = 0
        for features_a, features_b, preferred in preference_data:
            reward_model.train_on_preference(features_a, features_b, preferred)
            
            # Check accuracy
            score_a = reward_model.score(features_a)
            score_b = reward_model.score(features_b)
            predicted = "a" if score_a > score_b else "b"
            if predicted == preferred:
                correct += 1
        
        accuracy = correct / len(preference_data) * 100
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch + 1}: preference prediction accuracy = {accuracy:.1f}%")
    
    print(f"  Reward model trained!")
    
    # Show reward model scores
    print("\n  Reward model quality scores:")
    test_profiles = [
        ("High quality (helpful+honest)", np.array([0.9, 0.7, 0.9, 0.6, 0.5, 0.0, 0.0, 0.0])),
        ("Low quality (harmful+rude)", np.array([0.2, 0.1, 0.3, 0.4, 0.5, 0.8, 0.5, 0.7])),
        ("Medium quality", np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0])),
    ]
    for label, features in test_profiles:
        score = reward_model.score(features)
        print(f"    {label}: {score:.2f}")
    
    print("\n" + "=" * 60)
    print("PHASE 3: PPO Optimization (Training LLM Policy)")
    print("=" * 60)
    
    state_size = 5  # [empathy_need, complexity, urgency, topic_familiarity, conversation_length]
    action_size = 4  # [empathetic, concise, comprehensive, creative]
    
    policy = RLHFPolicy(state_size, action_size)
    
    rewards_per_episode = []
    strategy_counts = [0] * action_size
    
    for episode in range(n_rl_episodes):
        # Simulate a prompt context
        state = np.array([
            np.random.uniform(0, 1),  # empathy need
            np.random.uniform(0, 1),  # complexity
            np.random.uniform(0, 1),  # urgency
            np.random.uniform(0, 1),  # topic familiarity
            np.random.uniform(0, 1),  # conversation length
        ])
        
        # Policy selects response strategy
        action, prob = policy.select_action(state)
        strategy_counts[action] += 1
        
        # Generate response features for this strategy
        context = {"needs_empathy": state[0]}
        response_features = generate_response_features(action, context)
        
        # Score with reward model (proxy for human preference)
        reward = reward_model.score(response_features)
        
        # Simple advantage estimate
        baseline = np.mean(rewards_per_episode[-50:]) if rewards_per_episode else 0
        advantage = reward - baseline
        
        # PPO update
        policy.ppo_update(state, action, advantage, prob)
        
        rewards_per_episode.append(reward)
        
        if (episode + 1) % 100 == 0:
            avg = np.mean(rewards_per_episode[-100:])
            print(f"  Episode {episode + 1}: avg_reward = {avg:.2f}")
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    action_names = ["empathetic", "concise", "comprehensive", "creative"]
    total = sum(strategy_counts)
    print(f"\n  Strategy distribution after RLHF training:")
    for name, count in zip(action_names, strategy_counts):
        bar = "█" * int(count / total * 40)
        print(f"    {name:<15} {count/total*100:.1f}% {bar}")
    
    print(f"\n  Final average reward: {np.mean(rewards_per_episode[-50:]):.2f}")
    print(f"  (Higher = more aligned with human preferences)")
    
    return reward_model, policy


# Run the full pipeline
reward_model, policy = run_rlhf_pipeline()
```

---

## When to Use RLHF

| Use Case | Why |
|----------|-----|
| Training foundation models | This is literally what it's for |
| Aligning AI with human values | Reward model captures preferences |
| Improving safety | Penalize harmful outputs |
| Improving helpfulness | Reward helpful, complete answers |

---

## Key Concepts Visualized

### The Full RLHF Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  PHASE 1: DATA COLLECTION                                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ Prompt A  │  │ Prompt B  │  │ Prompt C  │  │  ...     │        │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────────┘        │
│       │              │              │                             │
│  ┌────▼─────┐  ┌────▼─────┐  ┌────▼─────┐                      │
│  │Response 1│  │Response 1│  │Response 1│  LLM generates         │
│  │Response 2│  │Response 2│  │Response 2│  multiple responses     │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                      │
│       │              │              │                             │
│  ┌────▼─────┐  ┌────▼─────┐  ┌────▼─────┐                      │
│  │Human:    │  │Human:    │  │Human:    │  Humans rank           │
│  │ 2 > 1    │  │ 1 > 2    │  │ 2 > 1    │  preferences           │
│  └──────────┘  └──────────┘  └──────────┘                      │
│                                                                  │
│  PHASE 2: REWARD MODEL TRAINING                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Reward Model learns: (prompt, response) → quality score │   │
│  │  Training signal: human preferences (A preferred over B) │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  PHASE 3: PPO OPTIMIZATION                                       │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐                   │
│  │   LLM   │────►│Response │────►│ Reward  │                   │
│  │ (policy)│     │         │     │  Model  │                   │
│  └────┬────┘     └─────────┘     └────┬────┘                   │
│       │                                │                         │
│       │    "Adjust weights to          │                         │
│       │     produce higher-scoring     │                         │
│       │     responses"                 │                         │
│       │◄───────────────────────────────┘                         │
│                                                                  │
│  REPEAT for thousands of iterations                              │
└──────────────────────────────────────────────────────────────────┘
```

### Why Not Just Supervised Learning?

```
SUPERVISED APPROACH:
  Collect perfect responses → Train LLM to copy them
  Problem: You can't write perfect responses for EVERY scenario
  Problem: LLM can't IMPROVE beyond the demonstrations

RLHF APPROACH:
  Collect COMPARISONS (easier!) → Train reward model → Optimize LLM
  Advantage: Comparisons are easier than writing perfect responses
  Advantage: LLM can DISCOVER responses better than any single demonstration
  Advantage: Reward model generalizes to new scenarios
```

### Preference Data is Easier to Collect

```
TASK 1: "Write the perfect response to this message"
  Difficulty: ★★★★★ (requires expert writing)
  Time: 5-10 minutes per prompt
  
TASK 2: "Which of these two responses is better?"
  Difficulty: ★★☆☆☆ (anyone can compare)
  Time: 10-30 seconds per pair
  
RLHF uses Task 2 → achieves results of Task 1 at scale!
```

---

## RLHF vs Previous Methods

| Aspect | DQN (L4) | PPO (L5) | RLHF (L6) |
|--------|----------|----------|-----------|
| Reward source | Environment | Environment | Human preferences |
| Training data | Interactions | Interactions | Comparisons |
| What's optimized | Q-values | Policy | LLM weights |
| Scale | Thousands of steps | Thousands of episodes | Millions of comparisons |
| Infrastructure | CPU/GPU | GPU | GPU cluster |
| Time | Hours | Hours-days | Weeks-months |
| Who does this | App developers | App developers | AI labs (OpenAI, Anthropic) |

---

## Key Limitations

```
1. MASSIVE COMPUTE REQUIREMENT
   Training reward model: days on GPU
   PPO optimization of LLM: weeks on GPU cluster
   Cost: thousands to millions of dollars
   → NOT something you do yourself. Use pre-trained models.

2. REWARD HACKING
   LLM finds shortcuts to get high reward without being truly helpful
   Example: "I understand your pain" repeated 10 times = high empathy score?
   Solution: KL divergence penalty (don't stray too far from base model)

3. REWARD MODEL IMPERFECTIONS
   Reward model is an approximation of human preferences
   Garbage in → garbage out
   Solution: Multiple reward models, regular human evaluation

4. ANNOTATOR BIAS
   Human raters have their own biases
   "Helpful" means different things to different people
   Solution: Diverse annotator pools, clear guidelines

5. ALIGNMENT TAX
   RLHF can make models MORE cautious (refuse reasonable requests)
   Over-optimization → generic, bland responses
   Solution: Balance helpfulness vs harmlessness
```

---

## Where You Fit In

```
YOUR ROLE AS AN APP DEVELOPER:
  ❌ Don't train your own LLM from scratch
  ❌ Don't run RLHF yourself
  ✅ USE pre-trained, RLHF'd models via OpenRouter
  ✅ ADD personalization on top using Levels 0-4

YOUR STACK:
  Base model: GPT-4, Claude, Llama (already RLHF'd)
  Personalization: LinUCB / Q-Learning (Levels 1b-2)
  Strategy selection: Contextual bandit
  Conversation flow: Q-learning or feature-based

  You don't need Level 6. You USE the output of Level 6.
```

---

## Connecting to Real RL Methods Guide

From the [RL Methods Guide](../../RL_METHODS_GUIDE.md#level-6-rlhf-reinforcement-learning-from-human-feedback):

> **What:** Train a reward model from human preferences, then use PPO to optimize an LLM to maximize that reward.
> **When:** You're training/fine-tuning a foundation model.
> **Key advantage:** Aligns AI with human preferences at scale.
> **Key limitation:** Requires massive compute and data.

---

## Real-Life Applications

| System | What RLHF Did | Scale |
|--------|---------------|-------|
| ChatGPT (OpenAI) | Made GPT-4 helpful & safe | Millions of comparisons |
| Claude (Anthropic) | Constitutional AI + RLHF | Preference + rule-based |
| Llama 2 (Meta) | Community preference data | Open-weight RLHF |
| Gemini (Google) | Multi-modal alignment | Text + image preferences |

---

## Summary: The Complete RL Journey

```
Level 0: Random         → No learning, establish baseline
Level 1a: Classic MAB   → Learn best GLOBAL default
Level 1b: LinUCB        → Personalize with context
Level 2: Q-Learning     → Handle sequential decisions
Level 3: Feature Q      → Scale to large state spaces
Level 4: DQN            → Handle raw inputs, non-linear patterns
Level 5: PPO            → Continuous actions, direct policy optimization
Level 6: RLHF           → Train foundation models (you USE, not BUILD)
Level 7: DPO            → Direct preference optimization (no reward model)
Level 8: GRPO + RLVR    → Group-relative, verifiable rewards (reasoning)

For most chatbot applications: Levels 1b-3 are the sweet spot.
Use Level 6's output (ChatGPT, Claude) as your LLM via OpenRouter.
Add personalization with Levels 0-4 in your backend.
```

---

## Modern Successors: DPO and GRPO

RLHF (PPO + reward model) is the classic pipeline, but the field has moved on:

- **Level 7 — DPO:** the same alignment goal **without** a reward model or RL loop. One supervised-style pass on preference pairs. Now the default for open-weight fine-tuning.
- **Level 8 — GRPO & RLVR:** PPO **without a critic** (group-relative advantages) trained on **verifiable** rewards. This is the paradigm behind reasoning models (DeepSeek-R1, o-series).

---

## Next Step

If you fine-tune your **own** model from preference pairs, go to **Level 7: DPO** — it's simpler, cheaper, and usually enough. For tasks with automatic correctness checks (math, code), continue to **Level 8: GRPO & RLVR**.

---

**See Also**: [Track A Index](index.md) · [RL Methods Guide - Level 6](../../RL_METHODS_GUIDE.md#level-6-rlhf-reinforcement-learning-from-human-feedback) · [A3: DPO & RFT](a3-dpo-and-rft.md) · [A4: GRPO & RLVR](a4-grpo-and-rlvr.md)
