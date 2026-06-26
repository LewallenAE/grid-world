# GridWorld: RL Infrastructure Learning Project

## What This Is
GridWorld is a production-minded reinforcement learning training infrastructure built to demonstrate understanding of RL systems design, data pipeline architecture, and deployment infrastructure.

It is **not** a complex ML research project. It is a **clean, explainable implementation** that shows how RL systems fit together at scale.

## Why We're Building This

**Context:**
- You have real ML/RL experience: hex binary classifiers, DV eval harness, RLHF harness
- Two recent interviews went poorly not because your thinking was wrong, but because you were unclear/hesitant explaining it
- You need a concrete, production-grade artifact you can reference with confidence

**Interview Signal:**
This gives you a talking point for the $500K+ RL/ML infrastructure roles currently recruiting you:
- "I've built RL training infrastructure from scratch—environment, training loop, eval harness, containerized and deployed"
- You can explain the full pipeline confidently because you built it
- Shows you think about systems, not just algorithms

## The Four-Part Progression

### Part 1: Environment
- Simple 5x5 deterministic grid
- Agent position state (x, y)
- Actions: up, down, left, right, stay
- Rewards: goal +10, step -0.1, invalid -1, exploration +0.5
- Max 50 steps per episode

**Why:** Defines the problem boundary. You need to be able to explain state/action/reward.

### Part 2: Training Loop
- Episode collection (run n episodes, collect trajectories)
- Return computation (cumulative discounted rewards, γ=0.99)
- Policy update (simple gradient ascent on log-prob × return)
- Metric tracking throughout

**Why:** Shows data flow: collect → compute → update. This is the core of any RL system.

### Part 3: Evaluation Harness
- Separate eval entry point
- New unseen goal positions
- Metrics: success rate, avg steps, total reward
- Track improvement over training

**Why:** Demonstrates you understand train/test separation and measurable progress.

### Part 4: Production Integration
- Proper Python package structure
- YAML configuration (externalized hyperparams)
- Structured logging (JSON, container-friendly)
- Model serialization (save/load trained agents)
- **Containerization** (Dockerfile)
- **Orchestration** (K8s manifests for training jobs)

**Why:** Shows you think about scalability and reproducibility from day one.

## Code Architecture

```
gridworld/
├── gridworld/              # Python package
│   ├── __init__.py
│   ├── environment.py      # Grid world definition
│   ├── agent.py            # Policy & learning
│   ├── trainer.py          # Training loop orchestration
│   ├── evaluator.py        # Eval harness
│   ├── config.py           # Config management
│   └── logger.py           # Structured logging
├── train.py                # Training entry point
├── eval.py                 # Evaluation entry point
├── config.yaml             # Training configuration
├── requirements.txt
├── setup.py
├── Dockerfile
├── docker-compose.yml
├── kubernetes/
│   ├── deployment.yaml
│   └── configmap.yaml
└── PROGRESSION.md          # This file
```

## Success Criteria

- [ ] Environment runs deterministically and produces expected rewards
- [ ] Training loop improves agent success rate over episodes
- [ ] Eval harness shows measurable progress (success rate increases)
- [ ] Configuration is externalized (no hardcoded values)
- [ ] Logging is structured and container-friendly
- [ ] Code runs in Docker without modification
- [ ] K8s manifests define a training job
- [ ] You can explain the full pipeline in 5 minutes

## Interview Story

Once complete, you can say:

> "I built a RL training infrastructure from scratch. It has four components:
> - An environment that defines state, action, reward
> - A training loop that collects trajectories, computes returns, updates policy
> - An eval harness that measures success rate and tracks improvement
> - Production infrastructure: configuration management, structured logging, containerization, K8s deployment
> 
> Here's how it works..." [walk them through code]

This demonstrates: systems thinking, clean architecture, scalability mindset, AND understanding of the actual RL pipeline.

## Notes

- Keep it simple. This is not a research project.
- Focus on clarity over optimization.
- Every piece should be explainable in 30 seconds.
- The K8s part shows you think about scaling, not that you need complex orchestration for GridWorld.
