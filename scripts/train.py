#!/usr/bin/env python3
"""
 Enter module docstring here
"""

# ---------- Future Import ----------
from __future__ import annotations

# ---------- Standard Library Imports ----------


# ---------- Third Party Imports ----------


# ---------- Project Level Imports ----------

from gridworld.logger import get_logger
from gridworld.config.base_config import GridWorldConfig

# ---------- Begin File ----------
def main():
    gwc = GridWorldConfig()
    logger = get_logger("gridworld.train")
    logger.info("Training started", extra={
        "grid_size": gwc.env.grid_size,
        "learning_rate": gwc.training.learning_rate,
        "max_epochs": gwc.training.num_epochs
    })
    
if __name__ == "__main__":
    main()


