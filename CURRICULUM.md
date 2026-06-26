# GridWorld: Complete Learning Curriculum
## Rigorous, Ground-Up RL + Post-Training Infrastructure

This is the full curriculum for learning to build production-grade RL systems. Each phase builds on the previous. If the session disconnects, this is your roadmap.

---

## Phase Progression: 0 → 8 (+9 Optional)

```
Phase 0: Foundation           (30 min)     Config + Logging + Project Structure
   ↓
Phase 1: RL Fundamentals      (2 hrs)      MDPs, Policy Gradients, Math
   ↓
Phase 2: GridWorld Env        (1 hr)       5×5 Grid, State/Action/Reward
   ↓
Phase 3: REINFORCE Training   (2 hrs)      Policy Gradient Update Loop
   ↓
Phase 4: DPO Optimization     (2 hrs)      Preference Learning, Post-training
   ↓
Phase 5: LoRA/QLoRA           (2 hrs)      Efficient Fine-tuning
   ↓
Phase 6: Evaluation Harness   (1.5 hrs)    Metrics, Continuous Eval
   ↓
Phase 7: Production Infra     (2 hrs)      Docker, K8s, Deployment
   ↓
Phase 8: Interview Ready      (1 hr)       Integration, Story, Defense
   ↓
Phase 9: Advanced (Optional)  (Variable)   Multi-objective, Safe RL, Meta-learning

────────────────────────────────────────
TOTAL: ~15 hours (2-hour blocks = 8 sessions)
```

### What You'll Have Built By Phase:

| Phase | Deliverable | What It Does |
|-------|-------------|--------------|
| **0** | Config + Logger | Externalized hyperparams, structured JSON logging |
| **1** | MDP definitions | Formal RL math (states, actions, returns, policy gradient) |
| **2** | GridWorld env | Working 5×5 grid environment with rewards |
| **3** | REINFORCE agent | Policy learns to solve GridWorld (50%+ success) |
| **4** | DPO post-training | Policy optimized for preferences (production-grade) |
| **5** | LoRA fine-tuning | Efficient adaptation (99% fewer parameters) |
| **6** | Eval harness | Metrics prove learning works on unseen goals |
| **7** | Docker + K8s | Reproducible, deployable, production-ready |
| **8** | Full system + story | Everything connected, explainable, interview-ready |

### Prerequisites

- **Phase 0**: Just Python basics (imports, functions)
- **Phase 1+**: Phase N-1 must be complete (each depends on previous)

### Time Commitment

- **Best case**: 15 hours total, 2-3 hours per session
- **Realistic**: 20 hours (includes debugging, re-reading, thinking)
- **Not a bootcamp**: No 12-hour/day pressure. Work at your pace.

---

---

## Philosophy

**Teaching Approach:**
- No code handed to you. You write every line.
- Every concept taught before it's used.
- Every implementation verified against what was taught.
- Math is the blueprint; code is the execution.

**Success Metric:**
You can explain the entire system in 5 minutes to an interviewer AND implement any component from scratch.

---

## Phase 0: Project Foundation & Configuration
**Duration:** ~30 min | **Status:** IN PROGRESS

### What Gets Taught
- Why dataclasses over hardcoded variables (modularity, type safety, reusability)
- Configuration design: what's tunable vs. hardcoded
- Structured logging for containers/observability
- Project layout for production systems

### What You Implement
1. `gridworld/config/base_config.py` — Four dataclasses with defaults
2. `gridworld/logger.py` — JSON structured logging to stdout
3. `scripts/train.py` — Entry point that imports config + logger
4. Create initial test to verify imports work

### Key References
- "Designing Machine Learning Systems" — Chip Huyen (Ch. 1-3)
- `design.md` Section 5 (hyperparameters)
- `Journal.md` Entries 1-3 (questions you've already answered)

### Code Deliverables
- [ ] `gridworld/config/base_config.py` (4 dataclasses, typed, defaults)
- [ ] `gridworld/logger.py` (JSON formatter, get_logger function)
- [ ] `scripts/train.py` (loads config, initializes logger, runs)
- [ ] `tests/test_config.py` (verify config instantiates)
- [ ] `tests/test_logger.py` (verify JSON output format)

### Success Criteria
- [ ] `python -c "from gridworld.config import GridWorldConfig; print(GridWorldConfig())"` works
- [ ] `python scripts/train.py` outputs valid JSON logs to stdout
- [ ] Config can be modified in code without breaking imports
- [ ] You can explain why each file exists and what it does

### Journal Entry
- [ ] Update Journal.md Entry 4: Hyperparameter selection feedback
- [ ] Update Journal.md Entry 5: Evaluation metrics feedback

---

## Phase 1: RL Fundamentals & Math
**Duration:** ~1-2 hours | **Status:** PENDING

### What Gets Taught
- Markov Decision Process (formal definition)
- State space $S$, action space $A$, reward function $R$
- Policy $\pi(a|s)$ (stochastic and deterministic)
- Value function $V_\pi(s)$ and $Q_\pi(s,a)$
- Return $G_t = \sum_k \gamma^k r_{t+k}$ and discount factor $\gamma$
- Policy gradient theorem and why we optimize $\nabla \log \pi(a|s) \cdot G_t$

### What You Implement
1. `gridworld/core/mdp.py` — Formal MDP class with type definitions
2. `gridworld/core/types.py` — State, Action, Transition, Episode dataclasses
3. Unit tests for return computation with different discount factors
4. Whiteboard: Draw the MDP for GridWorld (state transitions, rewards)

### Key References
- **Sutton & Barto** — "Reinforcement Learning: An Introduction" (Ch. 1-3, Ch. 13.1-13.2)
- `design.md` Section 2 (formal MDP definition)
- LaTeX equations for $V_\pi(s)$, $G_t$, policy gradient

### Code Deliverables
- [ ] `gridworld/core/mdp.py` (MDP class, formal definitions)
- [ ] `gridworld/core/types.py` (State, Action, Transition, Episode)
- [ ] `tests/test_mdp.py` (test return computation)
- [ ] Whiteboard drawing (take a photo, add to repo)

### Success Criteria
- [ ] You can write the return computation formula by hand
- [ ] You can explain why $\gamma = 0.99$ vs $\gamma = 0.5$ changes agent behavior
- [ ] Return computation tests pass
- [ ] You can draw the GridWorld MDP from memory

### Journal Entry
- [ ] Entry 6: Understanding returns and discount factor

---

## Phase 2: GridWorld Environment Implementation
**Duration:** ~1 hour | **Status:** PENDING

### What Gets Taught
- Deterministic environments (our case)
- Environment interface: `reset()`, `step()`, `render()`
- State representation for discrete grid
- Reward computation logic
- Episode tracking (visited states for exploration bonus)

### What You Implement
1. `gridworld/environment/gridworld.py` — Full environment class
2. `gridworld/environment/state.py` — State representation (x, y, visited set)
3. `gridworld/environment/reward.py` — Reward computation (all cases)
4. ASCII rendering for debugging
5. Tests: run 10 episodes, verify rewards are correct

### Key References
- `design.md` Section 2.2-2.3 (reward function, episode definition)
- "Building Agentic AI" — Environment design chapter
- Your own MDP definitions from Phase 1

### Code Deliverables
- [ ] `gridworld/environment/gridworld.py` (GridWorld class, reset/step/render)
- [ ] `gridworld/environment/state.py` (State dataclass with visited tracking)
- [ ] `gridworld/environment/reward.py` (Reward computation function)
- [ ] `tests/test_environment.py` (determinism, rewards, episodes)

### Success Criteria
- [ ] 10 test episodes run without errors
- [ ] All reward cases work: goal, step, invalid, explore, stay
- [ ] Episodes terminate at goal or max steps
- [ ] Same seed = same trajectory (deterministic)
- [ ] ASCII render shows grid clearly

### Journal Entry
- [ ] Entry 7: Environment design trade-offs

---

## Phase 3: Training Loop & REINFORCE
**Duration:** ~2 hours | **Status:** PENDING

### What Gets Taught
- Episode collection (run agent, store trajectories)
- Return computation (bootstrap with 0)
- Policy network (neural network outputting action probabilities)
- REINFORCE loss: $\mathcal{L} = -\mathbb{E}[\log \pi(a|s) \cdot G_t]$
- Gradient descent on policy parameters
- Batching and averaging losses

### What You Implement
1. `gridworld/agent/policy.py` — Neural network policy $\pi_\theta(a|s)$
2. `gridworld/trainer/training_loop.py` — Episode collection + return computation
3. `gridworld/trainer/reinforce.py` — REINFORCE loss and update step
4. Metrics tracking (episode reward, returns, loss)
5. Training loop: 100 epochs, log metrics

### Key References
- Sutton & Barto, Ch. 13 (Policy Gradient Methods)
- "Machine Learning with PyTorch and Scikit-Learn" (Neural networks, gradients)
- `design.md` Section 4.1 (REINFORCE algorithm pseudocode)

### Code Deliverables
- [ ] `gridworld/agent/policy.py` (Neural network, sample_action, log_prob)
- [ ] `gridworld/trainer/training_loop.py` (collect_episodes, compute_returns)
- [ ] `gridworld/trainer/reinforce.py` (loss computation, update)
- [ ] `gridworld/metrics.py` (track episode rewards, avg return, loss)
- [ ] `scripts/train.py` updated to run REINFORCE training
- [ ] `tests/test_trainer.py` (verify gradient computation)

### Success Criteria
- [ ] Run 100 epochs of training
- [ ] Success rate increases from ~0% to >50%
- [ ] Loss decreases over time
- [ ] JSON logs show metrics improving
- [ ] Agent reaches goal more efficiently (steps decrease)

### Journal Entry
- [ ] Entry 8: Why REINFORCE works (gradient ascent intuition)

---

## Phase 4: Preference Learning & DPO
**Duration:** ~2 hours | **Status:** PENDING

### What Gets Taught
- Preference pairs: (trajectory_A, trajectory_B) with label "A is better"
- Bradley-Terry preference model
- Why DPO is better than reward modeling (no extra model to train)
- DPO loss: direct optimization of policy from preferences
- Reference policy (frozen copy from Phase 3)

### What You Implement
1. `gridworld/data/preference_pairs.py` — Generate preference pairs from episodes
2. `gridworld/trainer/dpo.py` — DPO loss and training loop
3. Preference labeling logic (which trajectory is "better"? based on returns)
4. Reference policy management (initialize, freeze, keep fixed)
5. Tests: verify preferences are generated correctly

### Key References
- **DPO Paper** — Rafailov et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (2023)
- "Constitutional AI: Harmlessness from AI Feedback" — Anthropic
- `design.md` Section 4.2 (DPO algorithm, equations)

### Code Deliverables
- [ ] `gridworld/data/preference_pairs.py` (generate pairs from trajectories)
- [ ] `gridworld/trainer/dpo.py` (DPO loss, training loop)
- [ ] Reference policy initialization in training script
- [ ] `tests/test_dpo.py` (verify loss computation)
- [ ] `scripts/train_dpo.py` (DPO training entry point)

### Success Criteria
- [ ] Generate 100 preference pairs from Phase 3 trajectories
- [ ] Train with DPO for 50 epochs
- [ ] Agent still solves GridWorld (doesn't diverge from reference)
- [ ] Success rate comparable or better than Phase 3
- [ ] Can explain DPO loss formula by hand

### Journal Entry
- [ ] Entry 9: Preference learning vs. raw rewards

---

## Phase 5: Fine-Tuning with LoRA & QLoRA
**Duration:** ~2 hours | **Status:** PENDING

### What Gets Taught
- LoRA: Low-rank adaptation ($W' = W_0 + AB$ where $r \ll \min(n,m)$)
- Why LoRA is efficient (fewer parameters, less memory)
- Rank selection (trade-off: higher rank = more expressive, but more params)
- QLoRA: Quantize $W_0$ to 4-bit, keep $A, B$ full precision
- When to use LoRA vs. full fine-tuning

### What You Implement
1. `gridworld/agent/lora.py` — LoRA layer implementation
2. `gridworld/agent/qlora.py` — QLoRA with 4-bit quantization
3. Policy with LoRA adapter (wrap existing policy)
4. Fine-tuning loop using LoRA
5. Compare: LoRA vs. full training (parameters, memory, performance)

### Key References
- **LoRA Paper** — Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
- **QLoRA Paper** — Dettmers et al. "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)
- "AI Engineering" — Chip Huyen (Ch. on efficient training)
- `design.md` Section 4.3-4.4 (LoRA, QLoRA math)

### Code Deliverables
- [ ] `gridworld/agent/lora.py` (LoRA layer, forward pass)
- [ ] `gridworld/agent/qlora.py` (QLoRA with quantization)
- [ ] Policy wrapper to use LoRA adapters
- [ ] `scripts/train_lora.py` (LoRA training entry point)
- [ ] `scripts/train_qlora.py` (QLoRA training entry point)
- [ ] Benchmark script comparing parameter counts, memory, time

### Success Criteria
- [ ] LoRA with rank=8 achieves same success rate as full training
- [ ] Parameter count: LoRA < 1% of full model
- [ ] QLoRA memory usage significantly lower than LoRA
- [ ] Training time comparable (slight overhead for quantization)
- [ ] Can explain rank selection trade-off

### Journal Entry
- [ ] Entry 10: LoRA efficiency gains (parameter count analysis)

---

## Phase 6: Evaluation Harness & Metrics
**Duration:** ~1.5 hours | **Status:** PENDING

### What Gets Taught
- Evaluation vs. training (separate goals, separate data)
- Metrics: success rate, avg steps, avg return, preference alignment
- Continuous evaluation (running during training to catch regressions)
- Unseen goals (eval on new goal positions not seen during training)

### What You Implement
1. `gridworld/evaluator/eval_harness.py` — Run evaluation, compute metrics
2. `gridworld/evaluator/metrics.py` — Metric definitions (functions)
3. Eval dataset: new unseen goal positions
4. Continuous eval loop (runs every N training epochs)
5. Visualization: plot metrics over time

### Key References
- "Designing Machine Learning Systems" — Chip Huyen (Ch. 4-5: Evaluation)
- DPO paper, Evaluation section
- `design.md` Section 5 (metrics definitions)

### Code Deliverables
- [ ] `gridworld/evaluator/eval_harness.py` (run_eval function)
- [ ] `gridworld/evaluator/metrics.py` (success_rate, avg_steps, avg_return)
- [ ] Eval dataset (10 new goal positions)
- [ ] Continuous eval integration (call eval every N epochs)
- [ ] `tests/test_evaluator.py` (verify metrics computation)

### Success Criteria
- [ ] Run eval harness on Phase 3 agent (REINFORCE)
- [ ] Run eval harness on Phase 4 agent (DPO)
- [ ] Run eval harness on Phase 5 agents (LoRA, QLoRA)
- [ ] Metrics improve over training (success rate ↑, steps ↓)
- [ ] Eval data separate from training data

### Journal Entry
- [ ] Entry 11: Why evaluation is harder than you think

---

## Phase 7: Production Infrastructure
**Duration:** ~2 hours | **Status:** PENDING

### What Gets Taught
- Docker multi-stage builds (keep images small)
- Configuration injection (environment variables, ConfigMaps)
- Artifact management (save/load models, logs, metrics)
- Kubernetes Deployments and Jobs
- Reproducibility (same code + same config = same results)

### What You Implement
1. `Dockerfile` (multi-stage, production-ready)
2. `docker-compose.yml` (local dev, run training)
3. `kubernetes/deployment.yaml` (training job manifest)
4. `kubernetes/configmap.yaml` (inject hyperparams)
5. Model checkpointing (save best model)
6. Metrics export (JSON to persistent storage)

### Key References
- "AI Engineering" — Chip Huyen (Infrastructure, deployment)
- Docker best practices
- K8s Job documentation
- Your own logging setup from Phase 0

### Code Deliverables
- [ ] `Dockerfile` (multi-stage, runs `scripts/train.py`)
- [ ] `docker-compose.yml` (mounts code, persistent volumes)
- [ ] `kubernetes/deployment.yaml` (K8s Job for training)
- [ ] `kubernetes/configmap.yaml` (hyperparameter injection)
- [ ] Model serialization (save/load checkpoints)
- [ ] `data/logs/`, `data/models/`, `data/metrics/` (artifact dirs)

### Success Criteria
- [ ] Build Docker image: `docker build -t gridworld:latest .`
- [ ] Run locally: `docker run -v $(pwd)/data:/app/data gridworld:latest`
- [ ] Logs appear in `data/logs/` (JSON format)
- [ ] Models saved to `data/models/`
- [ ] K8s deployment can start training job (or mock test)
- [ ] Same hyperparams → same results (reproducible)

### Journal Entry
- [ ] Entry 12: Why Docker and K8s matter (not just CI/CD, but science)

---

## Phase 8: Integration & Interview Readiness
**Duration:** ~1 hour | **Status:** PENDING

### What Gets Taught
- How each phase connects (data flow from env → train → eval)
- Trade-offs and why you made architectural choices
- Explaining complex systems simply
- Practice talking through code

### What You Implement
1. End-to-end training pipeline (env → REINFORCE → DPO → LoRA → eval)
2. Update `PROGRESSION.md` with what you built
3. Write `README.md` (architecture, quick start, results)
4. Practice interview explanation (5 min script)

### Code Deliverables
- [ ] `scripts/end_to_end.py` (full pipeline)
- [ ] Updated `PROGRESSION.md`
- [ ] `README.md` (what this is, how to use it, results)
- [ ] Interview notes (key talking points)

### Success Criteria
- [ ] Can run full pipeline: env → train → eval in one command
- [ ] Can explain each phase in 2-3 sentences
- [ ] Can defend architecture choices (why this way, not that way?)
- [ ] Can walk an interviewer through the code in 5 minutes
- [ ] Metrics show clear improvement (phase → phase)

### Journal Entry
- [ ] Entry 13: Lessons learned and what you'd do differently

---

## Phase 9 (Optional): Advanced Topics
**Duration:** Variable | **Status:** PENDING

### Possible Extensions
- **Multi-objective RL**: Agent optimizes for multiple rewards simultaneously
- **Safe RL**: Add constraints (don't hit walls too often)
- **Meta-learning**: Train agent to learn new goal positions faster
- **Inverse RL**: Given good trajectories, infer the reward function
- **Distributed training**: Multiple agents training in parallel (K8s)

Choose one based on interview needs.

---

## Interview Story

Once you complete Phase 8, here's your 5-minute pitch:

> "I built a production-grade RL infrastructure from scratch. Here's how it connects:
>
> **Environment**: GridWorld is a 5×5 deterministic grid. Agent starts randomly, tries to reach goal within 50 steps. Rewards for goal (+10), steps (-0.1), invalid moves (-1), exploration (+0.5).
>
> **Training Loop**: I implemented REINFORCE—collect episodes, compute returns with discount factor γ=0.99, update policy using policy gradient. Over 100 epochs, success rate goes from 0% to 70%+.
>
> **Post-training**: I then optimized with DPO (Direct Preference Optimization), which learns from preference pairs instead of raw rewards. No separate reward model—policy optimizes directly. Success rate improved and agent became more aligned with preferences.
>
> **Efficiency**: I added LoRA fine-tuning—low-rank adapters on top of frozen weights. Reduces trainable parameters by 99% while maintaining performance. Then QLoRA quantizes the frozen weights to 4-bit.
>
> **Evaluation**: Continuous evaluation harness runs on unseen goals. Metrics: success rate, average steps, average return. Shows clear improvement across all phases.
>
> **Production**: The whole thing containerizes in Docker and deploys to K8s. Configuration externalized to YAML. Structured JSON logging for observability. Reproducible: same config = same results.
>
> This demonstrates: RL fundamentals, post-training techniques, production infrastructure, and how to think about scaling from toy problems to real systems."

---

## Dependencies & Tools

**Python Libraries:**
- `torch` — Neural networks
- `scikit-learn` — Traditional ML utilities
- `pyyaml` — Configuration files
- `pydantic` — Config validation
- `pytest` — Unit tests

**Infrastructure:**
- `uv` — Package management (already set up)
- `Docker` — Containerization (Phase 7)
- `Kubernetes` — Orchestration (Phase 7, optional)

---

## Success Metrics (Overall)

By the end:
- [ ] You understand every line of code you wrote
- [ ] You can explain the math (equations on whiteboard)
- [ ] You can defend architecture choices
- [ ] You can implement any component from scratch
- [ ] You can deploy and run in production
- [ ] You can interview about this confidently
- [ ] You have a GitHub portfolio piece

---

## If Session Disconnects

1. Read this file (you're reading it now ✓)
2. Check `Journal.md` for what you've learned so far
3. Read `PROGRESSION.md` (why you're building this)
4. Read `design.md` (what you're building)
5. Read `instructions.md` (how to build it)
6. Check the todo list (what's next)
7. Jump back in at the last in_progress task

---

## Time Estimate

- Phase 0: 30 min
- Phase 1: 2 hours
- Phase 2: 1 hour
- Phase 3: 2 hours
- Phase 4: 2 hours
- Phase 5: 2 hours
- Phase 6: 1.5 hours
- Phase 7: 2 hours
- Phase 8: 1 hour

**Total: ~15 hours** of focused learning.

If you work in 2-hour blocks, that's ~8 sessions. Completely doable.

---

## This Is Your Roadmap

You now have:
- ✅ Project structure (`/Full-Stack/gridworld/`)
- ✅ Learning progression (`CURRICULUM.md` — you're reading it)
- ✅ Math definitions (`design.md`)
- ✅ Teaching approach (`instructions.md`)
- ✅ Learning journal (`Journal.md`)
- ✅ Progress tracker (todo list)
- ✅ Interview story (Phase 8)

Everything you need to go from "I want to learn RL" to "I built production RL infrastructure and can explain it in interviews."

Ready?
