import numpy as np


class GridEnv:
    """A tiny stateful environment with four movement tools."""

    N_ACTIONS = 4

    def __init__(self, size=5, goal=(4, 4), max_steps=20, seed=0):
        self.size = size
        self.goal = goal
        self.max_steps = max_steps

    def reset(self):
        self.pos = (0, 0)
        self.steps = 0
        return self._obs()

    def _obs(self):
        return np.array([self.pos[0] / (self.size - 1),
                         self.pos[1] / (self.size - 1)])

    def step(self, action):
        r, c = self.pos
        if action == 0:
            r = max(0, r - 1)
        elif action == 1:
            r = min(self.size - 1, r + 1)
        elif action == 2:
            c = max(0, c - 1)
        else:
            c = min(self.size - 1, c + 1)
        self.pos = (r, c)
        self.steps += 1
        done = self.pos == self.goal or self.steps >= self.max_steps
        reward = 1.0 if self.pos == self.goal else 0.0
        return self._obs(), reward, done


class LinearSoftmaxPolicy:
    def __init__(self, obs_size, n_actions, lr=0.05, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0.0, 0.01, (n_actions, obs_size))
        self.b = np.zeros(n_actions)
        self.lr = lr

    def probs(self, obs):
        logits = self.W @ obs + self.b
        z = logits - np.max(logits)
        e = np.exp(z)
        return e / e.sum()

    def update(self, trajectory, advantage):
        for obs, action, _ in trajectory:
            p = self.probs(obs)
            one_hot = np.zeros_like(p)
            one_hot[action] = 1.0
            grad_logits = advantage * (one_hot - p)
            self.W += self.lr * np.outer(grad_logits, obs)
            self.b += self.lr * grad_logits


def rollout(policy, env):
    obs = env.reset()
    trajectory = []
    done = False
    while not done:
        probs = policy.probs(obs)
        action = int(np.random.choice(len(probs), p=probs))
        next_obs, reward, done = env.step(action)
        trajectory.append((obs, action, reward))
        obs = next_obs
    total_reward = sum(step[2] for step in trajectory)
    return trajectory, total_reward


def train(n_iterations=400, group_size=16, lr=0.05, seed=1):
    np.random.seed(seed)
    env = GridEnv(seed=seed)
    policy = LinearSoftmaxPolicy(obs_size=2, n_actions=GridEnv.N_ACTIONS,
                                 lr=lr, seed=seed)
    window = []
    for it in range(n_iterations):
        trajectories, returns = [], []
        for _ in range(group_size):
            traj, R = rollout(policy, env)
            trajectories.append(traj)
            returns.append(R)
        returns = np.array(returns)
        if returns.std() > 1e-8:
            advantages = (returns - returns.mean()) / (returns.std() + 1e-8)
        else:
            advantages = returns - returns.mean()
        for traj, adv in zip(trajectories, advantages):
            policy.update(traj, adv)
        window.append(returns.mean())
        if (it + 1) % 100 == 0:
            print(f"Iter {it+1}: avg return (last 100) = {np.mean(window[-100:]):.3f}")
    return policy, window


def greedy_path(policy, env):
    obs = env.reset()
    path = [env.pos]
    done = False
    while not done:
        action = int(np.argmax(policy.probs(obs)))
        obs, _, done = env.step(action)
        path.append(env.pos)
    return path


print("Training agentic policy on a gridworld environment...")
policy, history = train()
env = GridEnv()
path = greedy_path(policy, env)
print("\nGreedy path from start to goal:")
print("  " + " -> ".join(str(p) for p in path))
print(f"  Reached goal: {path[-1] == env.goal}")
print(f"  Steps taken:  {len(path) - 1}")
