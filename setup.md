# GridWorld: Project Setup Guide
## Using `uv` for Python Project Management

---

## Prerequisites

Install `uv` (fast Python package manager):

```bash
# On macOS
brew install uv

# Or follow https://docs.astral.sh/uv/getting-started/installation/
```

Verify installation:
```bash
uv --version
```

---

## Project Structure

After setup, your project will look like:

```
gridworld/
├── gridworld/                  # Main Python package
│   ├── __init__.py
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── gridworld.py       # GridWorld MDP
│   │   ├── state.py           # State representation
│   │   └── reward.py          # Reward computation
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── policy.py          # Policy network
│   │   ├── value.py           # Value network
│   │   ├── lora.py            # LoRA layer
│   │   └── qlora.py           # QLoRA layer
│   ├── trainer/
│   │   ├── __init__.py
│   │   ├── training_loop.py   # Training orchestration
│   │   ├── reinforce.py       # REINFORCE update
│   │   └── dpo.py             # DPO update
│   ├── evaluator/
│   │   ├── __init__.py
│   │   ├── eval_harness.py    # Evaluation pipeline
│   │   └── metrics.py         # Metric definitions
│   ├── config/
│   │   ├── __init__.py
│   │   ├── base_config.py     # Default hyperparams
│   │   └── config.yaml        # User config
│   └── logger.py              # Structured logging
│
├── scripts/
│   ├── train.py               # Training entry point
│   └── eval.py                # Evaluation entry point
│
├── tests/
│   ├── __init__.py
│   ├── test_environment.py    # Environment tests
│   ├── test_agent.py          # Agent tests
│   └── test_trainer.py        # Training tests
│
├── configs/
│   ├── default.yaml           # Default config
│   └── debug.yaml             # Debug (small, fast)
│
├── data/
│   ├── models/                # Saved model checkpoints
│   ├── logs/                  # Training logs (JSON)
│   └── metrics/               # Evaluation metrics
│
├── kubernetes/
│   ├── deployment.yaml        # K8s training job
│   ├── configmap.yaml         # Config injection
│   └── service.yaml           # (if needed)
│
├── .dockerignore               # Docker ignore rules
├── Dockerfile                  # Container image
├── docker-compose.yml          # Local dev with Docker
│
├── pyproject.toml             # uv project config
├── uv.lock                    # Locked dependencies
├── requirements.txt           # (legacy, if needed)
├── setup.py                   # (for pip compatibility)
│
├── PROGRESSION.md             # Original progress tracking
├── design.md                  # Problem definition (this file)
├── instructions.md            # Learning path (this file)
├── setup.md                   # This file
└── README.md                  # High-level overview
```

---

## Step 1: Initialize Project with `uv`

```bash
cd /Users/anthonylewallen/Full-Stack/gridworld

# Initialize a new Python project
uv init --name gridworld --python 3.10

# This creates:
# - pyproject.toml (project metadata)
# - .python-version (Python 3.10)
# - src/ (optional, we'll use gridworld/ instead)
```

If you already have the directory:

```bash
cd /Users/anthonylewallen/Full-Stack/gridworld

# Create pyproject.toml manually (see below)
```

---

## Step 2: Configure `pyproject.toml`

Create or edit `pyproject.toml`:

```toml
[project]
name = "gridworld"
version = "0.1.0"
description = "Production RL infrastructure: environment, training, evaluation"
authors = [
    { name = "Anthony Lewallen", email = "lewallenae@gmail.com" }
]
readme = "README.md"
requires-python = ">=3.10"

dependencies = [
    "torch>=2.0.0",
    "scikit-learn>=1.3.0",
    "numpy>=1.24.0",
    "pyyaml>=6.0",
    "pydantic>=2.0",  # Config validation
    "python-dotenv>=1.0",  # Environment variables
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "black>=23.0",
    "ruff>=0.1.0",
    "mypy>=1.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.black]
line-length = 100

[tool.ruff]
line-length = 100
select = ["E", "F", "W", "I"]  # pycodestyle, pyflakes, warnings, isort

[tool.mypy]
python_version = "3.10"
strict = true
```

---

## Step 3: Create Dependencies Lock File

```bash
# Create uv.lock (pinned dependency versions)
uv sync

# This will:
# 1. Create .venv/ (virtual environment)
# 2. Install dependencies
# 3. Create uv.lock (reproducible, deterministic)
```

Verify:
```bash
ls -la
# Should see: .venv/, pyproject.toml, uv.lock

source .venv/bin/activate  # Enter venv
python --version           # Should be 3.10+
pip list                   # Should see torch, scikit-learn, etc.
```

---

## Step 4: Create Package Structure

```bash
# Create directories
mkdir -p gridworld/{environment,agent,trainer,evaluator,config}
mkdir -p tests data/{models,logs,metrics} kubernetes

# Create __init__.py files
touch gridworld/__init__.py
touch gridworld/environment/__init__.py
touch gridworld/agent/__init__.py
touch gridworld/trainer/__init__.py
touch gridworld/evaluator/__init__.py
touch gridworld/config/__init__.py

# Create entry points
touch scripts/{train.py,eval.py}
touch tests/__init__.py
```

---

## Step 5: Create Initial Files

### 5.1 `gridworld/__init__.py`

```python
"""GridWorld: Production RL + Post-Training Infrastructure"""

__version__ = "0.1.0"
```

### 5.2 `gridworld/logger.py` (Structured Logging)

```python
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

class JSONFormatter(logging.Formatter):
    """Output logs as JSON for container/observability tools"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)

def get_logger(name: str, log_dir: Path = None) -> logging.Logger:
    """Get logger with JSON output to stdout + optional file"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Stdout handler (JSON)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / f"{name}.jsonl")
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)
    
    return logger
```

### 5.3 `gridworld/config/base_config.py`

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class EnvironmentConfig:
    """GridWorld environment hyperparameters"""
    grid_size: int = 5
    max_steps_per_episode: int = 50
    reward_goal: float = 10.0
    reward_step: float = -0.1
    reward_invalid: float = -1.0
    reward_explore: float = 0.5

@dataclass
class TrainingConfig:
    """REINFORCE training hyperparameters"""
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 100
    discount_factor: float = 0.99

@dataclass
class DPOConfig:
    """DPO fine-tuning hyperparameters"""
    learning_rate: float = 5e-5
    beta: float = 0.5

@dataclass
class LoRAConfig:
    """LoRA adapter hyperparameters"""
    rank: int = 8
    alpha: float = 16.0

@dataclass
class EvaluationConfig:
    """Evaluation harness hyperparameters"""
    num_episodes: int = 100
    num_goal_positions: int = 10

@dataclass
class GridWorldConfig:
    """Full configuration"""
    env: EnvironmentConfig = EnvironmentConfig()
    training: TrainingConfig = TrainingConfig()
    dpo: DPOConfig = DPOConfig()
    lora: LoRAConfig = LoRAConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    log_dir: Path = Path("data/logs")
    model_dir: Path = Path("data/models")
    metrics_dir: Path = Path("data/metrics")
```

### 5.4 `scripts/train.py` (Training Entry Point)

```python
#!/usr/bin/env python3
"""Training entry point"""

import sys
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gridworld.config.base_config import GridWorldConfig
from gridworld.logger import get_logger

def main():
    config = GridWorldConfig()
    logger = get_logger("gridworld.train", config.log_dir)
    logger.info("Training started", extra={"config": str(config)})
    print("✓ Training setup complete. Ready for Phase 0.")

if __name__ == "__main__":
    main()
```

### 5.5 `Dockerfile`

```dockerfile
# Multi-stage build

FROM python:3.10-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY pyproject.toml uv.lock ./
COPY gridworld/ ./gridworld/
COPY scripts/ ./scripts/

# Install uv
RUN pip install --no-cache-dir uv

# Create venv and install dependencies
RUN uv sync --frozen

# Production stage
FROM python:3.10-slim

WORKDIR /app

# Copy venv from base
COPY --from=base /app/.venv /app/.venv
COPY --from=base /app/gridworld /app/gridworld
COPY --from=base /app/scripts /app/scripts

# Set PATH
ENV PATH="/app/.venv/bin:$PATH"

# Default command
CMD ["python", "scripts/train.py"]
```

---

## Step 6: Verify Setup

```bash
# Activate venv
source .venv/bin/activate

# Run training entry point
python scripts/train.py

# Expected output:
# {"timestamp": "...", "level": "INFO", "logger": "gridworld.train", "message": "Training started..."}
# ✓ Training setup complete. Ready for Phase 0.

# Build Docker image
docker build -t gridworld:latest .

# Run in Docker
docker run --rm gridworld:latest
```

---

## Step 7: Add to Git

```bash
# Create .gitignore
cat > .gitignore << 'EOF'
.venv/
.python-version
__pycache__/
*.pyc
*.egg-info/
dist/
build/
.DS_Store
data/logs/*.jsonl
data/models/*.pt
.mypy_cache/
.ruff_cache/
.pytest_cache/
EOF

# Initialize git
git init
git add pyproject.toml uv.lock Dockerfile .gitignore
git commit -m "Initial GridWorld project setup"
```

---

## Key `uv` Commands

```bash
# Sync dependencies (install)
uv sync

# Add new dependency
uv add torch scikit-learn

# Add dev dependency
uv add --group dev pytest black

# Run command in venv
uv run python scripts/train.py

# Update dependencies
uv lock --upgrade

# Install package in editable mode (for local development)
uv pip install -e .

# Clean venv
rm -rf .venv uv.lock
uv sync
```

---

## Next Steps

1. ✅ Complete Step 1-7 above
2. 📖 Read `design.md` (problem definition)
3. 📋 Read `instructions.md` (learning path)
4. 🚀 Start Phase 0: Implement `gridworld/environment/gridworld.py`

You're ready to begin.
