# GridWorld Learning Journal
## Questions, Answers, and Insights

This documents the learning journey building production RL infrastructure from scratch. Each entry captures a key concept tested and mastered.

---

## Entry 1: Why Dataclasses Over Hardcoded Variables?

**Question:** Why use dataclasses instead of just setting variables like `grid_size = 5` in a file?

**Your Answer:**
> Dataclasses decouple the logic and allow us to import and make it reusable. Just defining it as hardcoded—everything would need to be in that singular file for access. Having one gigantic file would be a mess and brittle for production.

**What You Got Right:**
- ✅ Modularity and reusability
- ✅ Decoupling logic from configuration
- ✅ Scalability concern (avoiding monolithic files)
- ✅ Production thinking (brittleness)

**Refinement:**
Dataclasses also give us:
- **Type safety**: `grid_size: int` prevents accidentally setting it to a string
- **Defaults**: `grid_size: int = 5` means it has a sensible default
- **Composability**: You can nest dataclasses (GridWorldConfig contains EnvironmentConfig, TrainingConfig, etc.)
- **Serialization**: Easy to convert to/from YAML, JSON
- **Validation**: Can add checks (e.g., learning_rate must be > 0)

**Lesson:** Configuration is not just data—it's a contract between your code and the user. Dataclasses enforce that contract.

---

## Entry 2: Hardcoded vs. Tunable MDP Components

**Question:** In the MDP definition, which components are hardcoded (never change) vs. tunable (you might change them)?

**Your Answer:**
> State space: changeable. Action space: hardcoded unless we want diagonal movement later (but that's technically two moves). Transition probability: hardcoded—1 and 0, the agent either moved or it didn't. Discount factor: changeable. μ (initial distribution): changeable since it's uniform random distribution.

**What You Got Right:**
- ✅ State space is tunable (grid size, obstacles)
- ✅ Action space insight: recognized it could be extended
- ✅ Transition probability deterministic (key insight for this domain)
- ✅ Discount factor tunable (affects learning behavior)
- ✅ Initial distribution tunable (could weight towards certain start positions)

**Refinement:**
The deeper insight: **A component is tunable if changing it changes the learning problem, not the infrastructure.**
- Discount factor γ: changing it makes the agent value future rewards differently → tunable
- Transition probability for a deterministic grid: always 1.0 or 0.0 → hardcoded
- But later, if we add **stochastic** environments (wind, slip), transition probability becomes tunable

**Lesson:** Separating configuration from code means you need to ask: "Will someone reasonably want to change this?" If yes, make it tunable. If no, hardcode it.

---

## Entry 3: Reward Shaping Design

**Question:** What rewards would you assign to each action type in GridWorld?

**Your Answer:**
- Reach goal: +10
- Take a step in wrong direction: -0.5 (penalizes inefficiency)
- Hit wall: -5 (strong penalty)
- Visit new tile: +0.5 (exploration bonus)
- Stay stationary: -1 (prevents laziness)
- Max steps per episode: 50

**What You Got Right:**
- ✅ Differentiated rewards by action type (not all penalties are equal)
- ✅ Intuition about magnitude (wall is -5, worse than step is -0.5)
- ✅ Exploration bonus (+0.5) encourages discovery
- ✅ Lazy prevention (-1 for staying) pushes exploration
- ✅ Thought about episode length (max 50 steps)

**Refinement from Literature:**
DPO papers (Rafailov et al.) use smaller per-step penalties (typically -0.1) because:
- The preference signal (which trajectory is better) matters more than absolute reward magnitude
- Smaller penalties keep the agent from getting stuck in local minima
- Your -0.5 for "wrong direction" is clever but needs refinement: how does the agent know direction is "wrong"? This requires additional state info (current position + goal position).

**Better approach (Sutton & Barto aligned):**
- Reach goal: +10 (terminal reward)
- Normal step: -0.1 (cost of time, encourages efficiency)
- Invalid move (wall/bounds): -1 (strong discourage)
- New tile visited: +0.5 (exploration bonus)
- Stay stationary: -0.1 (same as normal step, or ban it)

**Lesson:** Reward shaping is an art and science. Reference the literature for magnitudes, but think through whether the agent can actually learn what you're rewarding.

---

## Entry 4: Hyperparameter Selection

**Question:** What learning rate, batch size, and number of epochs would you choose for training?

**Your Answer:**
- Learning rate: 1e-3
- Batch size: 32
- Number of epochs: minimum 16, probably 150+

**What You Got Right:**
- ✅ Learning rate 1e-3 is standard (not aggressive, not conservative)
- ✅ Batch size 32 is practical (good gradient estimates, reasonable compute)
- ✅ 150 epochs shows understanding that training continues until plateaus
- ✅ Thought about iteration depth (not just "run once")

**Refinement from Literature:**
- **DPO vs REINFORCE**: When doing preference learning (Phase 4), learning rate drops to 5e-5 (200× smaller) because preferences are finer-grained than raw rewards. Check the paper, not the intuition.
- **Batch size trade-off**: 32 is good for small grids. Larger problems: 64-128. Smaller GPUs: 8-16.
- **Epochs**: 150 is solid. Diminishing returns after 50-100, but continuing helps. Monitor loss curves.
- **Warmup**: Some papers use learning rate warmup (start low, ramp up) but not critical for small problems.

**Lesson:** Hyperparameters follow patterns from literature. REINFORCE ≠ DPO in learning rate. When you move to a new algorithm, always check the reference paper's hyperparams first.

---

## Entry 5: Evaluation Metrics

**Question:** What metrics matter for evaluating if the agent learned?

**Your Answer:**
- `num_episodes`: How many episodes to evaluate on (100)
- `num_goal_positions`: How many new (unseen) goals to test (10)

**What You Got Right:**
- ✅ Separation of concerns: eval data ≠ train data (critical)
- ✅ Multiple unseen goals test generalization, not memorization
- ✅ 100 episodes gives statistical significance (rule: >30, use 100+)
- ✅ Understood that you need fresh goals to know if agent actually learned

**Refinement from Literature:**
Beyond just counting episodes, you track:
- **Success Rate** = % episodes reaching goal (target: 70%+ after training)
- **Avg Steps to Goal** = efficiency (lower = better, max 50)
- **Avg Return** = total cumulative reward (higher = better)
- **Preference Alignment** = after DPO, % of preferred actions taken (Phase 4)

These form a dashboard:
- Success ↑ = learning works
- Steps ↓ = efficiency improves
- Return ↑ = reward structure understood
- Alignment ↑ = preferences respected

**Lesson:** Metrics are how you prove success. Choose metrics that answer the question: "Did the agent learn what I wanted?" Not just "did it run."

---

## Meta: How to Use This Journal

Each entry follows:
1. **Question** — What concept are we testing?
2. **Your Answer** — What you said/wrote
3. **What You Got Right** — Affirm the good thinking
4. **Refinement** — What we can improve or deepen from literature/practice
5. **Lesson** — The broader principle, applicable elsewhere

This document becomes your learning artifact. Later, you can point to it in interviews: "Here's how I learned RL infrastructure—starting from first principles, validating against literature, implementing."

---

## Next Entry
Entry 6: MDP Fundamentals - Understanding States, Actions, Returns (Phase 1)
