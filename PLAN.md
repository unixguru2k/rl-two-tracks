# RL Learning Lab for Chatbot Development

## Project Overview

An interactive web application for learning and experimenting with Reinforcement Learning concepts through visual environments, with direct application to chatbot development.

---

## Technology Stack

| Component | Technology | Reason |
|-----------|------------|--------|
| Backend | Python (Flask) | Best for numerical computation, clean API design |
| Frontend | HTML + Tailwind/DaisyUI | Beautiful, responsive UI |
| Visualizations | Chart.js + Custom Canvas | Real-time RL visualizations |
| Data Storage | MACHAAO API | Save experiments, track progress |
| Real-time Updates | SSE | Live training progress |

---

## Core Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RL Learning Lab                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │ Environments │    │  Algorithms  │    │ Experiments  │          │
│  │              │    │              │    │              │          │
│  │ • GridWorld  │    │ • Q-Learning │    │ • Save/Load  │          │
│  │ • Bandits    │    │ • SARSA      │    │ • Compare    │          │
│  │ • Maze       │    │ • Policy     │    │ • History    │          │
│  │ • Custom     │    │ • DQN        │    │ • Analytics  │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Interactive Dashboard                     │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │   │
│  │  │ Environment │  │  Q-Table    │  │   Learning  │         │   │
│  │  │  Renderer   │  │  Heatmap    │  │   Curves    │         │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Hyperparameter Controls                   │   │
│  │  Learning Rate │ Discount │ Epsilon │ Episodes │ Speed      │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## RL Environments

### Base Environments (Learning Fundamentals)

| Environment | Concept Taught | Difficulty |
|-------------|---------------|------------|
| **GridWorld** | Basic RL, States, Actions, Rewards | Beginner |
| **Multi-Armed Bandit** | Exploration vs Exploitation | Beginner |
| **Maze Solver** | Pathfinding, Value Iteration | Intermediate |

### Chatbot-Specific Environments (Applied RL)

| Environment | What It Teaches | Chatbot Application |
|-------------|-----------------|---------------------|
| **Intent Classification Bandit** | Multi-armed bandits | Selecting which intent classifier to trust |
| **Dialog Policy GridWorld** | State-action-reward | Conversation flow management |
| **Response Strategy MAB** | Exploration/exploitation | Testing different response styles |
| **Slot Filling MDP** | Sequential decisions | Gathering required information step-by-step |
| **Escalation Timing** | Optimal stopping | When to hand off to human agent |

---

## RL Algorithms

| Algorithm | Type | Use Case |
|-----------|------|----------|
| Q-Learning | Model-free, Off-policy | General purpose, learns optimal actions |
| SARSA | Model-free, On-policy | Safer learning, conservative |
| Expected SARSA | Hybrid | Combines Q-learning and SARSA benefits |
| Monte Carlo | Episode-based | When full episodes are available |
| Policy Gradient | Direct optimization | Continuous action spaces |
| Contextual Bandit | Bandit variant | Personalized response selection |

---

## Chatbot RL Concepts

### Dialog Policy Learning

```
States (S):
┌─────┬─────┬─────┬─────┬─────┐
│ GRT │ ASK │INFO │CONF │ END │  GRT = Greeting
├─────┼─────┼─────┼─────┼─────┤  ASK = Ask for info
│     │WAIT │EXPL │CNCL │     │  INFO = Provide information
│     │     │     │     │     │  CONF = Confirmation
└─────┴─────┴─────┴─────┴─────┘  END = Conversation end

Actions (A) per state:
• ask_question    • provide_answer   • confirm_intent
• clarify         • escalate_human   • close_conversation

Rewards (R):
• +10  User says "thanks" / problem resolved
• +5   User provides requested information
• -5   User repeats question (didn't understand)
• -10  User abandons conversation
• -20  User asks for human agent
```

### Response Strategy Bandit

```
Arms (strategies to try):
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│  Formal  │ │ Friendly │ │ Concise  │ │ Detailed │
│  Tone    │ │  Tone    │ │ Response │ │ Response │
│   🎩     │ │   😊     │ │   ⚡     │ │   📚     │
└──────────┘ └──────────┘ └──────────┘ └──────────┘

Context features:
• Time of day
• Query type
• User history
• Sentiment

Reward signals:
• Reply length
• Follow-up questions
• Satisfaction ratings
• Task completion
```

---

## Project Structure

```
rl-testing/
├── backend/
│   ├── app.py                    # Flask application
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py          # Centralized config
│   ├── environments/
│   │   ├── __init__.py
│   │   ├── base.py              # Base environment class
│   │   ├── gridworld.py         # GridWorld environment
│   │   ├── bandit.py            # Multi-armed bandit
│   │   ├── maze.py              # Maze environment
│   │   ├── dialog_policy.py     # Conversation flow MDP
│   │   ├── response_bandit.py   # Response strategy selection
│   │   ├── slot_filling.py      # Information gathering MDP
│   │   └── registry.py          # Environment registry
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── base.py              # Base agent class
│   │   ├── q_learning.py        # Q-Learning
│   │   ├── sarsa.py             # SARSA
│   │   ├── policy_gradient.py   # Policy Gradient
│   │   ├── contextual_bandit.py # Context-aware bandit
│   │   └── registry.py          # Algorithm registry
│   ├── store/
│   │   ├── __init__.py          # Factory
│   │   ├── base.py              # Store interface
│   │   └── machaao_store.py     # MACHAAO implementation
│   ├── services/
│   │   ├── __init__.py
│   │   ├── experiment_service.py
│   │   └── training_service.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── api.py               # REST API endpoints
│   │   └── sse.py               # SSE for real-time training
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       └── sse_helper.py
├── frontend/
│   ├── index.html               # Main dashboard
│   ├── css/
│   │   └── styles.css           # Tailwind config
│   ├── js/
│   │   ├── app.js               # Main application
│   │   ├── environments.js      # Environment renderers
│   │   ├── charts.js            # Chart visualizations
│   │   ├── training.js          # Training controls
│   │   └── sse_client.js        # SSE client helper
│   └── components/
│       ├── grid-renderer.js     # GridWorld canvas renderer
│       ├── q-table-heatmap.js   # Q-value visualization
│       └── learning-curve.js    # Training progress chart
├── requirements.txt
├── start-app.sh
├── .env.template
└── README.md
```

---

## Data Flow

### General Architecture

```
┌─────────────┐      REST/SSE       ┌─────────────┐      Store       ┌─────────────┐
│   Browser   │ ◄──────────────────► │   Flask     │ ◄──────────────► │  MACHAAO    │
│   (UI)      │                      │   Backend   │                  │  API        │
└─────────────┘                      └─────────────┘                  └─────────────┘
      │                                    │
      │ Canvas Rendering                   │ RL Training Loop
      ▼                                    ▼
┌─────────────┐                      ┌─────────────┐
│ Visualize   │                      │ Environment │
│ Agent/World │                      │ + Agent     │
└─────────────┘                      └─────────────┘
```

### MACHAAO API Usage

| Data | MACHAAO Endpoint | Why |
|------|-----------------|-----|
| Conversation history | `/ai/conversations` | Store full chat context |
| Q-table / Policy weights | `/app-data/{key}` | O(1) lookup for agent decisions |
| User feedback events | `/content` (type: feedback) | Searchable for training |
| Experiment results | `/content` (type: experiment) | Compare strategies over time |
| RL model config | `/app-data/{key}` | Hyperparameters per user segment |

---

## Learning Path

```
Step 1: GridWorld Basics
   └── Understand states, actions, rewards, Q-table
   └── Visualize agent learning in a simple grid

Step 2: Multi-Armed Bandits
   └── Learn exploration vs exploitation
   └── Apply to response strategy selection

Step 3: Dialog Policy MDP
   └── Model conversation as sequential decision problem
   └── Train agent to manage conversation flow

Step 4: Contextual Bandits
   └── Incorporate user context into decisions
   └── Personalize response selection

Step 5: Reward Modeling
   └── Learn reward function from human feedback
   └── This is the core of RLHF!

Step 6: Integration
   └── Connect trained policy to live chatbot
   └── Collect feedback and iterate
```

---

## Concept Mapping

| Concept | GridWorld (Learn) | Chatbot (Apply) |
|---------|-------------------|-----------------|
| **State** | Grid position | Conversation context, intent, slots |
| **Action** | Up/Down/Left/Right | Ask/Answer/Confirm/Escalate |
| **Reward** | +1 at goal, -1 at pit | User satisfaction, task completion |
| **Policy** | Best move per cell | Best response per conversation state |
| **Q-table** | Value of each move | Value of each dialog action |

---

## Key Interactions

| Action | What Happens |
|--------|--------------|
| Select Environment | Loads GridWorld/Bandit/Dialog with visualization |
| Choose Algorithm | Sets learning strategy (Q-learning, SARSA, etc.) |
| Adjust Hyperparameters | Tune α, γ, ε in real-time |
| Start Training | Agent begins learning, live visualization |
| Step Mode | Execute one step at a time for learning |
| Save Experiment | Stores results to MACHAAO for comparison |
| View History | See past experiments, compare learning curves |

---

## Environment Variables

```bash
# MACHAAO Platform (auto-injected by platform on deploy)
MACHAAO_API_TOKEN=
MACHAAO_APP_ID=
MACHAAO_DEVELOPER_TOKEN=
MACHAAO_API_BASE_URL=https://api.machaao.com
MACHAAO_API_VERSION=v2

# Application
PORT=5000
DEBUG=false
STORE_BACKEND=machaao
```

---

## Status

**Plan saved**: 2026-09-04

**Next step**: User to say "implement" or "build" when ready to create the project.
