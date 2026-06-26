# GridWorld: Production RL + Post-Training Infrastructure
## Rigorous Learning Path

This is a ground-up curriculum for building production-grade RL infrastructure with post-training optimization. Every concept is taught with math, every assignment is verified against what was taught.

---

## Learning Structure

Each phase has:
1. **Theory** — Raw math with defined notation (from papers/textbooks)
2. **Implementation** — Production-grade code in PyTorch/scikit-learn
3. **Verification** — Challenges that test what was taught, not beyond
4. **Checkpoint** — Milestones before moving forward

---

## Phase 0: Foundation & Setup
**Goal:** Proper project structure, environment, reproducibility

### What You'll Learn
- Python project structure (package organization, `uv` dependency management)
- Configuration-driven design (externalized hyperparameters)
- Structured logging (container-friendly, JSON output)
- Docker/K8s thinking (but build it second, not first)

### Resources
- "Designing Machine Learning Systems" — Chip Huyen (Ch. 1-3: ML systems design, data flow)
- "AI Engineering" — Chip Huyen (Chapters on infrastructure, reproducibility)

### Code Deliverables
- [ ] `gridworld/` package structure
- [ ] `pyproject.toml` with uv configuration
- [ ] Config system (YAML-based hyperparameters)
- [ ] Structured logging module (JSON output)
- [ ] Dockerfile (multi-stage, production-ready)
- [ ] K8s manifests (Deployment, ConfigMap)

### Checkpoint
You can:
- [ ] Run `uv sync` and have a clean environment
- [ ] Explain why each file in the project exists
- [ ] Change hyperparameters via config without touching code
- [ ] Run code in Docker identically to local

---

## Phase 1: RL Fundamentals
**Goal:** Understand MDPs, state/action/reward, policy gradients from first principles

### What You'll Learn
- **Markov Decision Process (MDP)** definition
- **States, Actions, Rewards** — formal definitions with notation
- **Policy** — deterministic and stochastic policies
- **Value Functions** — V(s) and Q(s,a)
- **Policy Gradient Theorem** — why we can optimize log π(a|s) × return
- **Discount Factor** — why γ matters, how to set it

### Math Notation You'll Master
$$
S = \text{state space}, \quad A = \text{action space}
$$
$$
P(s' | s, a) = \text{transition probability}
$$
$$
R(s, a, s') = \text{reward function}
$$
$$
V_\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^\infty \gamma^t r_t | S_0 = s\right]
$$
$$
\pi(a|s) = \text{policy (probability of action given state)}
$$
$$
\nabla_\theta J(\theta) = \mathbb{E}_\tau\left[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]
$$

where $G_t = \sum_{k=0}^\infty \gamma^k r_{t+k}$ is the return.

### Resources
- **"Reinforcement Learning: An Introduction"** — Sutton & Barto (Ch. 1-3: MDPs, Ch. 13: Policy Gradient Methods)

### Code Deliverables
- [ ] `gridworld/core/mdp.py` — MDP class with formal definitions
- [ ] State/Action/Reward type definitions
- [ ] Policy representation (both deterministic and stochastic)
- [ ] Return computation with discount factor

### Checkpoint: Write from Scratch
You are given:
- A sequence of (state, action, reward) tuples
- A discount factor γ

You must:
- [ ] Compute returns $G_t$ for every timestep
- [ ] Explain why $\gamma$ affects how we value future rewards
- [ ] Implement value function approximation (linear function of state features)

---

## Phase 2: GridWorld Environment
**Goal:** Build a deterministic, measurable RL environment

### What You'll Learn
- Deterministic vs stochastic environments
- How to structure environment code (reset, step, render)
- Reward shaping and its effects on learning
- Episode termination conditions

### GridWorld Specification
```
Grid:        5×5 deterministic grid
State:       (x, y) position, x, y ∈ [0, 4]
Actions:     {up, down, left, right, stay} (5 actions)

Reward Function:
  R_goal = +10        (reach goal position)
  R_step = -0.1       (per timestep, encourages efficiency)
  R_invalid = -1      (hit wall or out of bounds)
  R_explore = +0.5    (visit new tile for first time)

Episode:
  Max steps: 50
  Start: random position (not goal)
  End: reach goal OR 50 steps exceeded
```

### Math Definition
$$
\text{Episode trajectory}: \tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T)
$$

$$
r_t = \begin{cases}
+10 & \text{if } s_{t+1} = s_{\text{goal}} \\
-1 & \text{if } a_t \text{ invalid (wall/bounds)} \\
+0.5 & \text{if } s_{t+1} \text{ never visited before} \\
-0.1 & \text{otherwise}
\end{cases}
$$

### Resources
- "Building Agentic AI" — Chapters on environment design and agent interfaces
- Your own MDP definitions from Phase 1

### Code Deliverables
- [ ] `gridworld/environment/gridworld.py` — Full environment
- [ ] Deterministic grid with obstacles (walls)
- [ ] Reward computation (all cases above)
- [ ] Episode tracking (visited tiles, trajectory logging)
- [ ] Rendering (ASCII grid visualization)

### Checkpoint: Test Environment
Run 10 episodes manually (no learning). Verify:
- [ ] Episode resets properly
- [ ] Rewards are computed correctly for each action type
- [ ] Exploration bonus works
- [ ] Episodes terminate at goal or 50 steps
- [ ] Deterministic (same seed = same trajectory)

---

## Phase 3: Training Loop & Policy Gradients
**Goal:** Collect data, compute returns, update policy using REINFORCE

### What You'll Learn
- **Episode Collection** — run agent, store trajectories
- **Return Computation** — cumulative discounted rewards
- **Policy Gradient Update** — REINFORCE algorithm
- **Batching** — gradient computation over multiple episodes
- **Learning rate and optimization** — Adam optimizer

### Algorithm: REINFORCE
$$
\text{Collect } N \text{ episodes: } \tau_1, \tau_2, \ldots, \tau_N
$$

$$
\text{For each episode } \tau_i = (s_0, a_0, r_0, \ldots, s_T, a_T, r_T):
$$

$$
G_t^{(i)} = \sum_{k=t}^{T} \gamma^{k-t} r_k^{(i)}
$$

$$
\mathcal{L}(\theta) = -\frac{1}{NT} \sum_{i=1}^N \sum_{t=0}^{T_i} \log \pi_\theta(a_t^{(i)} | s_t^{(i)}) \cdot G_t^{(i)}
$$

$$
\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}(\theta)
$$

### Resources
- Sutton & Barto, Ch. 13 (Policy Gradient Methods)
- "Machine Learning with PyTorch and Scikit-Learn" (Optimization, gradient descent)

### Code Deliverables
- [ ] `gridworld/agent/policy.py` — Neural network policy $\pi_\theta(a|s)$
- [ ] `gridworld/trainer/training_loop.py` — Data collection + return computation
- [ ] `gridworld/trainer/optimizer.py` — REINFORCE policy gradient update
- [ ] Metrics tracking (episode reward, success rate, loss)

### Checkpoint: Train an Agent
- [ ] Run 100 episodes of training
- [ ] Verify success rate increases over time
- [ ] Log metrics (episode rewards, returns, loss)
- [ ] Agent should learn to reach goal more efficiently

---

## Phase 4: Preference Learning & DPO
**Goal:** Learn from preference pairs (A better than B), not just raw rewards

### What You'll Learn
- **Reward Modeling** — predict reward from trajectory
- **Preference Data** — pairs of trajectories with preference labels
- **DPO (Direct Preference Optimization)** — optimize policy directly from preferences
- **Why DPO matters** — avoids reward hacking, aligns with human preference

### Math: Direct Preference Optimization
$$
\text{Preference pair: } (\tau_w, \tau_l) \quad \text{(winning trajectory, losing trajectory)}
$$

$$
\text{Probability of preference:}
$$

$$
P(\tau_w \succ \tau_l) = \frac{\exp(\beta r_\phi(\tau_w))}{\exp(\beta r_\phi(\tau_w)) + \exp(\beta r_\phi(\tau_l))}
$$

$$
\text{DPO Loss (direct policy optimization):}
$$

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta, \pi_{\text{ref}}) = -\mathbb{E}_{(\tau_w, \tau_l)} \left[ \log \sigma\left(\beta \log \frac{\pi_\theta(\tau_w)}{\pi_{\text{ref}}(\tau_w)} - \beta \log \frac{\pi_\theta(\tau_l)}{\pi_{\text{ref}}(\tau_l)}\right) \right]
$$

where:
- $\pi_\theta$ = policy to optimize
- $\pi_{\text{ref}}$ = reference policy (frozen copy of $\pi_\theta$ from Phase 3)
- $\beta$ = temperature (controls preference strength)
- $\sigma$ = sigmoid function

### Resources
- **"Direct Preference Optimization: Your Language Model is Secretly a Reward Model"** — Rafailov et al. (Full paper)
- "Constitutional AI: Harmlessness from AI Feedback" — Anthropic (Context on preference labels)

### Code Deliverables
- [ ] `gridworld/data/preference_pairs.py` — Generate preference pairs from trajectories
- [ ] `gridworld/agent/dpo_trainer.py` — DPO loss and optimization
- [ ] Preference labeling logic (which trajectory is "better"?)
- [ ] Reference policy management (keep frozen copy)

### Checkpoint: DPO Training
- [ ] Generate 100 preference pairs from Phase 3 trajectories
- [ ] Train policy with DPO
- [ ] Verify policy still solves GridWorld (doesn't diverge from reference)
- [ ] Compare DPO vs REINFORCE agent behavior

---

## Phase 5: Fine-Tuning with LoRA & QLoRA
**Goal:** Efficient adaptation using low-rank updates and quantization

### What You'll Learn
- **LoRA (Low-Rank Adaptation)** — parameter-efficient fine-tuning
- **QLoRA (Quantized LoRA)** — reduced memory via 4-bit quantization
- **Why they matter** — reduce trainable parameters, memory, without hurting performance
- **Rank selection** — trade-off between model size and expressiveness

### Math: LoRA
$$
W' = W_0 + \Delta W = W_0 + AB
$$

where:
- $W_0$ = original weight matrix (frozen)
- $A \in \mathbb{R}^{n \times r}$ = down-projection matrix
- $B \in \mathbb{R}^{r \times m}$ = up-projection matrix
- $r$ = rank (typically $r \ll \min(n, m)$)

Forward pass:
$$
y = (W_0 + AB) x = W_0 x + AB x
$$

Only $A$ and $B$ are trained; $W_0$ is frozen.

### QLoRA Extension
$$
W_{\text{4-bit}} = \text{Quantize}(W_0) \quad \text{(4-bit quantization)}
$$
$$
W' = \text{Dequantize}(W_{\text{4-bit}}) + AB
$$

Only $A, B$ in full precision; original weights in 4-bit.

### Resources
- **"LoRA: Low-Rank Adaptation of Large Language Models"** — Hu et al. (Full paper)
- **"QLoRA: Efficient Finetuning of Quantized LLMs"** — Dettmers et al. (Full paper)
- "AI Engineering" — Chip Huyen (Ch. on efficient training)

### Code Deliverables
- [ ] `gridworld/agent/lora.py` — LoRA layer implementation
- [ ] `gridworld/agent/qlora.py` — QLoRA with 4-bit quantization
- [ ] Policy with LoRA adapter
- [ ] Fine-tuning loop using LoRA

### Checkpoint: LoRA vs Full Training
- [ ] Train policy with LoRA (rank=8)
- [ ] Train policy with full parameters
- [ ] Compare:
  - [ ] Final performance (success rate)
  - [ ] Number of trainable parameters
  - [ ] Memory usage
  - [ ] Training time

---

## Phase 6: Evaluation Harness & Metrics
**Goal:** Measure agent reliability, success rate, alignment with preferences

### What You'll Learn
- **Evaluation vs Training** — separate data, separate goals
- **Metrics** — what to measure and why
- **Continuous evaluation** — ongoing monitoring for regressions
- **Safety/alignment metrics** — does agent do what we prefer?

### Metrics Definition
$$
\text{Success Rate} = \frac{\text{# episodes where goal reached}}{\text{# total eval episodes}}
$$

$$
\text{Avg Steps to Goal} = \mathbb{E}[\text{episode length} | \text{goal reached}]
$$

$$
\text{Avg Reward per Episode} = \mathbb{E}[G_0] \quad \text{(average return)}
$$

$$
\text{Preference Alignment} = \frac{\text{# preferred actions taken}}{\text{# total actions}}
$$

### Evaluation Protocol
```
For each eval round:
  1. Sample N new goal positions (unseen during training)
  2. Run agent deterministically (no exploration randomness)
  3. Collect metrics over N episodes
  4. Track metrics over time (do they improve?)
```

### Resources
- "Designing Machine Learning Systems" — Chip Huyen (Ch. 4-5: Evaluation, metrics)
- DPO paper (Evaluation section)

### Code Deliverables
- [ ] `gridworld/evaluator/eval_harness.py` — Run evaluation
- [ ] Metric computation (success rate, steps, reward, alignment)
- [ ] Eval dataset (new, unseen goals)
- [ ] Continuous evaluation loop (run during training)
- [ ] Metric logging and visualization

### Checkpoint: Evaluate All Phases
- [ ] Run eval harness on Phase 3 agent (REINFORCE)
- [ ] Run eval harness on Phase 4 agent (DPO)
- [ ] Run eval harness on Phase 5 agent (LoRA)
- [ ] Compare metrics
- [ ] Verify metrics improve over training

---

## Phase 7: Production Infrastructure
**Goal:** Container, orchestration, logging, reproducibility at scale

### What You'll Learn
- **Containerization** — Docker multi-stage builds
- **Configuration Management** — env-specific configs
- **Structured Logging** — JSON logs for observability
- **Orchestration** — K8s Deployments, Jobs, ConfigMaps
- **Artifact Management** — save/load trained models, logs

### Code Deliverables
- [ ] Production Dockerfile (multi-stage)
- [ ] docker-compose.yml for local development
- [ ] K8s Deployment manifest (training job)
- [ ] K8s ConfigMap (hyperparameter injection)
- [ ] Logging integration (JSON, structured)
- [ ] Model serialization (save/load checkpoints)
- [ ] CI/CD hooks (automated testing, linting)

### Checkpoint: Full Production Workflow
- [ ] Build Docker image
- [ ] Run training in Docker locally
- [ ] Deploy to K8s (local or mock)
- [ ] Verify logs are structured and readable
- [ ] Load trained model and run inference

---

## Phase 8: Integration & Interview Readiness
**Goal:** Tie it all together, tell the story, defend the design

### What You'll Learn
- How each phase connects (data flow)
- Trade-offs and design decisions
- Why you made architectural choices

### You Should Be Able To Explain
- [ ] "Here's the RL problem we're solving and why it matters"
- [ ] "Here's how we structure environments, training, and evaluation"
- [ ] "Here's how DPO improves over simple policy gradients"
- [ ] "Here's why LoRA is more efficient than full fine-tuning"
- [ ] "Here's how we measure if training actually works"
- [ ] "Here's how we package and deploy this at scale"

### Code Deliverables
- [ ] End-to-end training pipeline (train.py)
- [ ] End-to-end evaluation pipeline (eval.py)
- [ ] Complete PROGRESSION.md updated with what you built
- [ ] design.md with final architecture
- [ ] README explaining the full system

---

## Challenge Structure

For each phase:

1. **Teaching** — We teach the math, show examples
2. **Implementation** — You implement using only what was taught
3. **Verification** — You prove it works (tests, metrics, demos)
4. **Challenge** — You solve a problem that requires the knowledge, nothing more

**Critical Rule:** Before I give you a challenge, I check:
- Does it use only concepts we taught?
- Does it use only libraries/tools we've shown?
- If it requires something new, I either:
  - Teach it first, OR
  - Simplify the challenge to match what we taught, OR
  - Don't give it and reassess

---

## Tech Stack

- **Language:** Python 3.10+
- **Package Manager:** `uv` (faster, cleaner than pip)
- **ML Frameworks:** PyTorch (deep learning), scikit-learn (traditional ML)
- **Configuration:** YAML (externalized hyperparameters)
- **Logging:** Python `logging` module (structured JSON output)
- **Containerization:** Docker
- **Orchestration:** Kubernetes (YAML manifests)
- **Testing:** pytest (unit tests for each phase)

---

## Success Criteria

By the end:
- [ ] You understand MDPs and policy gradients from first principles
- [ ] You've built a working RL environment and training loop
- [ ] You understand why DPO is better than raw policy gradients
- [ ] You know how to make fine-tuning efficient (LoRA/QLoRA)
- [ ] You can measure whether learning actually works (evaluation harness)
- [ ] You can package and deploy it (Docker, K8s)
- [ ] **You can explain the entire system in 5 minutes to an interviewer**

---

## Next Steps

1. Review `design.md` (problem definition, architecture, math)
2. Review `setup.md` (project initialization with uv)
3. We start Phase 0: Setting up the project structure
4. Then Phase 1: RL fundamentals (teaching, then coding)
5. Then onward...

You teach me nothing I haven't learned. I challenge you with nothing you haven't been taught. We go methodically.

Ready?
