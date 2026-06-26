#!/usr/bin/env python3
"""
GridWorld Environment Configuration: (Editable)

# MDP the abstract math definition:
-----------------------------------
# M       = <S, A, P, R, y, u_o> where:    
# S       = State space   S = (x, y) E [0, 4] (25 states) aka this is the 5x5 grid.
# A       = Action space  {up, down, left, right, stay}
# P       = Transition probability function P(s'| (s,a) \\n
#           in this state moving or not moving = 1,0; P(s'(s,a) | s, a) = 1, 0 (Deterministic)
# R       = Reward  R(s, a , s')
# y/gamma = The discount factor usually 0.99
# u mu    = The initial state uniform random u_0(s) = 1/25 for all (s).

# Current Configuration layout schematic

# EnvironmentConfig grid_size, max_step_per_episode, reward_goal, reward_step, reward_invalid, reward_explore, reward_stay
# TrainingConfig learning_rate, batch_size, num_epochs, discount_factor (can be updates with LoRA, QLoRA, DPO, ExGRPO etc. later)
# EvaluationConfig num_episodes, num_goal_positions
# GridWorldConfig env, training, evaluation, log_dir, model_dir, metrics_dir (Path objects)
"""

# ---------- Future Import ----------
from __future__ import annotations

# ---------- Standard Library Imports ----------
from dataclasses import dataclass, field
from pathlib import Path

# ---------- Third Party Imports ----------

# ---------- Project Level Imports ----------

# ---------- Begin File ----------

@dataclass
class EnvironmentConfig:
    """GridWorld Environment Hyperparameters"""
    grid_size: int = 5
    max_steps_per_episode: int = 50
    reward_goal: int = +10
    reward_step: float = -0.1
    reward_invalid: int = -1
    reward_explore: float = +0.5
    reward_stay: float = -0.1


@dataclass
class TrainingConfig:
    """Training Configuration Reinforcement for HyperParameters"""
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 100
    discount_factor: float = 0.99


@dataclass
class EvaluationConfig:
    """Evaluation Harness Hyperparameters"""
    num_episodes: int = 100
    num_goal_positions: int = 10


@dataclass
class GridWorldConfig:
    """Full configuration"""
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    log_dir: Path = Path("data/logs")
    model_dir: Path = Path("data/models")
    metrics_dir: Path = Path("data/metrics")
    