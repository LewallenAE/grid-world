# GridWorld Design Document
## Formal Problem Definition, Architecture, Mathematics

---

## 1. Problem Statement

**Goal:** Build a production-grade reinforcement learning infrastructure that:
1. Defines a simple but complete RL environment (GridWorld)
2. Trains agents using policy gradients (REINFORCE)
3. Optimizes policy with preference learning (DPO)
4. Fine-tunes efficiently (LoRA, QLoRA)
5. Measures reliability (evaluation harness)
6. Deploys reproducibly (Docker, K8s)

**Why this problem matters:**
- RL is fundamental to agent systems (the focus of your interviews)
- Post-training (DPO, LoRA) is how models are adapted and aligned
- Evaluation is how you know if anything actually works
- Production infrastructure is how you scale from toy to real systems

---

## 2. Formal Problem Definition

### 2.1 GridWorld as an MDP

A Markov Decision Process is a tuple:
$$
\mathcal{M} = \langle S, A, P, R, \gamma, \mu_0 \rangle
$$

where:

| Symbol | Meaning | Definition |
|--------|---------|------------|
| $S$ | State space | $S = \{(x, y) : x, y \in [0, 4]\}$ (25 states) |
| $A$ | Action space | $A = \{\text{up}, \text{down}, \text{left}, \text{right}, \text{stay}\}$ (5 actions) |
| $P(s' \mid s, a)$ | Transition probability | Deterministic: $P(s'(s,a) \mid s, a) = 1$, else 0 |
| $R(s, a, s')$ | Reward function | See Section 2.2 |
| $\gamma$ | Discount factor | $\gamma = 0.99$ (value of future rewards) |
| $\mu_0(s_0)$ | Initial state dist. | Uniform random: $\mu_0(s) = 1/25$ for all $s$ |

### 2.2 Reward Function

$$
R(s, a, s') = \begin{cases}
+10 & \text{if } s' = s_{\text{goal}} \text{ (reached goal)} \\
-1 & \text{if } \text{action is invalid (wall or bounds)} \\
+0.5 & \text{if } s' \notin V(s_t) \text{ (new tile visited)} \\
-0.1 & \text{otherwise (standard step)}
\end{cases}
$$

where:
- $s_{\text{goal}}$ = fixed goal position (set per episode)
- $V(s_t)$ = set of states visited up to timestep $t$ (for exploration bonus)

### 2.3 Episode Definition

An episode is a trajectory:
$$
\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T, a_T, r_T)
$$

**Episode termination condition:**
$$
T = \min(50, \text{step at which } s_T = s_{\text{goal}})
$$

**Initial state:**
$$
s_0 \sim \text{Uniform}(S \setminus \{s_{\text{goal}}\})
$$

(Random position, not the goal)

### 2.4 Return (Cumulative Discounted Reward)

For a trajectory $\tau = (s_0, a_0, r_0, \ldots, s_T, a_T, r_T)$:

$$
G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}
$$

The return at timestep $t$ is the cumulative future reward, discounted by $\gamma^k$.

**Key insight:** A step in the future is worth $\gamma$ times less than a step now.

---

## 3. Policy and Value Functions

### 3.1 Stochastic Policy

A policy maps state to probability distribution over actions:

$$
\pi_\theta(a \mid s) : S \to \Delta(A)
$$

where $\theta$ are the parameters we'll train.

**Implementation:** Neural network that outputs logits, then softmax:

$$
\pi_\theta(a \mid s) = \frac{\exp(f_\theta(s)[a])}{\sum_{a'} \exp(f_\theta(s)[a'])}
$$

where $f_\theta(s) \in \mathbb{R}^{|A|}$ is the network output.

### 3.2 Value Function Approximation

$$
V_\phi(s) : S \to \mathbb{R}
$$

Maps state to estimated return. Also implemented as a neural network.

(Note: We won't use value functions for REINFORCE, but will for baselines in later phases.)

---

## 4. Training Algorithms

### 4.1 REINFORCE (Policy Gradient)

**Algorithm:**

```
Repeat for each training iteration:
  1. Collect N episodes using current policy π_θ
  2. For each trajectory τ_i in the batch:
     - Compute return G_t for each timestep t
  3. Compute policy gradient loss
  4. Update parameters θ using gradient descent
```

**Loss Function:**

$$
\mathcal{L}_{\text{REINFORCE}}(\theta) = -\mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_t \log \pi_\theta(a_t \mid s_t) \cdot G_t \right]
$$

**Why this works:**
- We increase log probability of actions that lead to high returns
- We decrease log probability of actions that lead to low returns
- Gradient ascent on expected return

**Parameter Update:**

$$
\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}_{\text{REINFORCE}}(\theta)
$$

where $\alpha$ is the learning rate.

### 4.2 DPO: Direct Preference Optimization

**Motivation:** REINFORCE optimizes raw reward. DPO optimizes preferences ("trajectory A is better than trajectory B"), which is closer to alignment.

**Preference Pair:**
$$
(\tau_w, \tau_l)
$$

where $\tau_w$ is "winning" (better) and $\tau_l$ is "losing" (worse).

**Bradley-Terry Preference Model:**

The probability that $\tau_w$ is preferred over $\tau_l$ is:

$$
P(\tau_w \succ \tau_l) = \frac{\exp(\beta r_\phi(\tau_w))}{\exp(\beta r_\phi(\tau_w)) + \exp(\beta r_\phi(\tau_l))}
$$

where:
- $r_\phi(\tau)$ = reward model score for trajectory $\tau$
- $\beta$ = temperature (strength of preference)

**DPO removes the reward model and optimizes policy directly:**

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(\tau_w, \tau_l)} \left[ \log \sigma\left( \beta \left[ \log \frac{\pi_\theta(\tau_w)}{\pi_{\text{ref}}(\tau_w)} - \log \frac{\pi_\theta(\tau_l)}{\pi_{\text{ref}}(\tau_l)} \right] \right) \right]
$$

where:
- $\pi_\theta$ = policy to optimize
- $\pi_{\text{ref}}$ = reference policy (frozen, usually $\pi$ from Phase 3)
- $\sigma(x) = 1/(1 + e^{-x})$ = sigmoid
- $\beta = 0.5$ (typical value)

**Intuition:**
- If $\pi_\theta$ assigns higher probability to $\tau_w$ than $\pi_{\text{ref}}$, and lower to $\tau_l$, the loss decreases (good)
- If $\pi_\theta$ violates the preference, the loss increases (bad)

### 4.3 LoRA: Low-Rank Adaptation

**Standard Fine-tuning:**
$$
W' = W_0 + \Delta W
$$

where all of $\Delta W$ is trainable (millions of parameters).

**LoRA:**
$$
W' = W_0 + AB
$$

where:
- $W_0 \in \mathbb{R}^{n \times m}$ = original weight (frozen)
- $A \in \mathbb{R}^{n \times r}$ = down-projection (trainable)
- $B \in \mathbb{R}^{r \times m}$ = up-projection (trainable)
- $r$ = rank (typically 8-64, much smaller than $n, m$)

**Trainable parameters:**
$$
|A| + |B| = nr + rm = r(n + m)
$$

vs. full fine-tuning:
$$
|W| = nm
$$

**Ratio:** $r(n+m) / nm \approx r / \max(n, m)$, so if $r = 8$ and $\max(n, m) = 1000$, LoRA uses ~0.8% of parameters.

**Forward Pass:**

$$
y = (W_0 + AB) x = W_0 x + A(Bx)
$$

Only $A, B$ backpropagate; $W_0$ is frozen.

### 4.4 QLoRA: Quantized LoRA

**Key idea:** Quantize $W_0$ to 4-bit, keep $A, B$ in full precision.

$$
W_0 \approx Q(W_0) \quad \text{(4-bit quantized)}
$$

$$
y = Q(W_0) x + AB x
$$

**Benefits:**
- Reduces memory of frozen weights by ~75%
- $A, B$ still trained in full precision (learning doesn't suffer)
- Total memory ≈ 4-bit weights + full-precision LoRA adapters

---

## 5. Evaluation Metrics

### 5.1 Success Rate

$$
\text{Success Rate} = \frac{N_{\text{success}}}{N_{\text{total}}}
$$

where:
- $N_{\text{success}}$ = # episodes where agent reached goal
- $N_{\text{total}}$ = total eval episodes

**Interpretation:** What fraction of the time does the agent solve the problem?

### 5.2 Average Steps to Goal

$$
\text{Avg Steps} = \mathbb{E}[\text{episode length} \mid \text{goal reached}]
$$

**Interpretation:** On average, how efficient is the solution?

### 5.3 Average Return per Episode

$$
\text{Avg Return} = \mathbb{E}_{\tau \sim \pi_{\text{eval}}}[G_0]
$$

where $G_0 = \sum_t \gamma^t r_t$ is the return from the start.

**Interpretation:** How good is the total cumulative reward? (Accounts for both success and efficiency.)

### 5.4 Preference Alignment

$$
\text{Alignment} = \frac{1}{N} \sum_i \mathbb{1}[\text{action matches preference}]
$$

(Used after DPO training to verify policy respects learned preferences.)

---

## 6. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      gridworld/                             │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  environment/                                                 │
│  ├── gridworld.py          (MDP implementation)              │
│  ├── state.py              (State representation)            │
│  └── reward.py             (Reward computation)              │
│                                                               │
│  agent/                                                       │
│  ├── policy.py             (π_θ neural network)              │
│  ├── value.py              (V_φ neural network)              │
│  ├── lora.py               (LoRA layer)                      │
│  └── qlora.py              (QLoRA layer)                     │
│                                                               │
│  trainer/                                                     │
│  ├── training_loop.py      (Data collection + update)        │
│  ├── reinforce.py          (REINFORCE update)               │
│  └── dpo.py                (DPO loss & update)               │
│                                                               │
│  evaluator/                                                   │
│  ├── eval_harness.py       (Run eval, compute metrics)       │
│  └── metrics.py            (Metric definitions)              │
│                                                               │
│  config/                                                      │
│  ├── base_config.py        (Default hyperparams)             │
│  └── config.yaml           (User-facing config)              │
│                                                               │
│  logger.py                 (Structured logging)              │
│                                                               │
└─────────────────────────────────────────────────────────────┘

Data Flow:
  train.py
    ├─→ environment.py      (sample episode τ)
    ├─→ policy.py           (π_θ(a|s))
    ├─→ trainer.py          (collect τ, compute G_t, compute ∇)
    └─→ checkpoint          (save model weights)

  eval.py
    ├─→ environment.py      (new goals)
    ├─→ policy.py           (π_θ(a|s) deterministic)
    ├─→ evaluator.py        (compute metrics)
    └─→ metrics.yaml        (log results)
```

---

## 7. Data Structures

### 7.1 State

```python
@dataclass
class State:
    x: int          # x coordinate [0, 4]
    y: int          # y coordinate [0, 4]
    visited: set    # visited positions (for exploration bonus)
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))
```

### 7.2 Transition

```python
@dataclass
class Transition:
    state: State
    action: int         # 0=up, 1=down, 2=left, 3=right, 4=stay
    reward: float
    next_state: State
    done: bool          # episode termination?
```

### 7.3 Episode / Trajectory

```python
@dataclass
class Episode:
    transitions: List[Transition]
    returns: List[float]  # G_t for each t (computed after episode)
    
    @property
    def length(self) -> int:
        return len(self.transitions)
    
    @property
    def total_reward(self) -> float:
        return sum(t.reward for t in self.transitions)
```

### 7.4 Batch

```python
@dataclass
class EpisodeBatch:
    episodes: List[Episode]
    
    def states(self) -> torch.Tensor:
        """Stack all states: shape (N_total, state_dim)"""
    
    def actions(self) -> torch.Tensor:
        """Stack all actions: shape (N_total,)"""
    
    def returns(self) -> torch.Tensor:
        """Stack all returns: shape (N_total,)"""
```

---

## 8. Configuration

All hyperparameters externalized in YAML:

```yaml
# config.yaml

# Environment
grid_size: 5
max_steps_per_episode: 50
num_goal_positions: 5

# Rewards
reward_goal: 10.0
reward_step: -0.1
reward_invalid: -1.0
reward_explore: 0.5

# RL
discount_factor: 0.99
learning_rate: 1e-3
batch_size: 32
num_training_episodes: 1000

# DPO
dpo_beta: 0.5
dpo_learning_rate: 5e-5

# LoRA
lora_rank: 8
lora_alpha: 16

# Evaluation
eval_episodes: 100
eval_goals: 10  # new unseen goals
```

---

## 9. Interfaces

### 9.1 Environment Interface

```python
class GridWorld:
    def reset(self) -> State:
        """Reset episode, return initial state"""
    
    def step(self, state: State, action: int) -> Tuple[State, float, bool]:
        """Take action, return (next_state, reward, done)"""
    
    def render(self, state: State) -> str:
        """ASCII representation"""
```

### 9.2 Policy Interface

```python
class Policy(nn.Module):
    def forward(self, state: State) -> torch.Tensor:
        """Return logits over actions, shape (batch, 5)"""
    
    def sample_action(self, state: State) -> int:
        """Sample action from π_θ(·|state)"""
    
    def log_prob(self, state: State, action: int) -> float:
        """Return log π_θ(action | state)"""
```

### 9.3 Trainer Interface

```python
class Trainer:
    def collect_episodes(self, num_episodes: int) -> EpisodeBatch:
        """Run agent, return batch of episodes"""
    
    def compute_returns(self, episodes: EpisodeBatch):
        """Fill in returns for each transition"""
    
    def update_policy(self, episodes: EpisodeBatch):
        """Gradient step on loss"""
```

### 9.4 Evaluator Interface

```python
class Evaluator:
    def eval(self, policy: Policy, num_episodes: int) -> Dict[str, float]:
        """Run policy on new goals, return metrics:
        {
            'success_rate': float,
            'avg_steps': float,
            'avg_return': float
        }
        """
```

---

## 10. Training/Eval Loop Pseudocode

### Phase 3: REINFORCE

```
for epoch in range(num_epochs):
    # Collect data
    episodes = trainer.collect_episodes(batch_size=32)
    
    # Compute returns (bootstrap with 0)
    for episode in episodes:
        G = 0
        for t in reversed(range(len(episode.transitions))):
            r = episode.transitions[t].reward
            G = r + gamma * G
            episode.returns[t] = G
    
    # Update policy
    loss = trainer.update_policy(episodes)  # -E[log π(a|s) * G]
    
    # Evaluate
    if epoch % 10 == 0:
        metrics = evaluator.eval(policy, 100)
        print(f"Epoch {epoch}: SR={metrics['success_rate']}")
```

### Phase 4: DPO

```
for epoch in range(num_epochs):
    # Collect preferences: for each winning τ_w, sample losing τ_l
    preferences = generate_preference_pairs(episodes)
    
    # Compute DPO loss
    losses = []
    for τ_w, τ_l in preferences:
        log_pi_theta_w = policy.log_prob(τ_w)
        log_pi_theta_l = policy.log_prob(τ_l)
        
        log_pi_ref_w = ref_policy.log_prob(τ_w)
        log_pi_ref_l = ref_policy.log_prob(τ_l)
        
        logits = beta * (log_pi_theta_w - log_pi_theta_l) - \
                 beta * (log_pi_ref_w - log_pi_ref_l)
        
        loss = -log_sigmoid(logits)
        losses.append(loss)
    
    # Backprop
    total_loss = mean(losses)
    total_loss.backward()
    optimizer.step()
```

---

## 11. Success Criteria

- [ ] Environment is deterministic and rewards are correctly computed
- [ ] REINFORCE trains agent to solve GridWorld (success rate → 1.0)
- [ ] Preference pairs can be generated from episodes
- [ ] DPO training respects preferences without catastrophic divergence from reference
- [ ] LoRA reduces parameters while maintaining performance
- [ ] QLoRA reduces memory without degrading learning
- [ ] Evaluation harness shows metrics improving over training
- [ ] Code runs in Docker identically to local
- [ ] K8s manifests deploy training job successfully
- [ ] You can explain the entire system to an interviewer

---

## 12. References

1. Sutton & Barto (2018) — "Reinforcement Learning: An Introduction" (Sections 13.1-13.2)
2. Rafailov et al. (2023) — "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
3. Hu et al. (2021) — "LoRA: Low-Rank Adaptation of Large Language Models"
4. Dettmers et al. (2023) — "QLoRA: Efficient Finetuning of Quantized LLMs"
5. Chip Huyen (2023) — "Designing Machine Learning Systems"

---

## Next: Project Setup

See `setup.md` for creating the project structure with `uv`.
