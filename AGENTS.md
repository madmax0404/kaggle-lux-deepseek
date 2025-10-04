# Repository Guidelines

## Project Structure & Module Organization
Lux_DeepSeek_Portfolio1 is notebook-first. Core PPO experiments live in `Notebooks/Agent_Development/DeepSeek-R1-Distill-Qwen-1.5B_*.ipynb`, timestamped per run. Custom trainer logic resides in `Notebooks/Agent_Development/Modified_PPO_Trainer`; keep `ppo_trainer.py` as the main reference and stash experiments in dated copies. Exploratory analysis sits in `Notebooks/EDA`, while static assets land in `images/`. Place replays and lightweight artifacts beside the notebooks they document, and describe any new entry point in a top-cell markdown note.

## Build, Test, and Development Commands
Work from a GPU-ready Python 3.12 env. Useful commands:
- `python -m venv .venv && source .venv/bin/activate` creates an isolated environment.
- `pip install -r requirements.txt` installs training and simulation dependencies.
- `jupyter lab Notebooks` opens the notebook workspace (use `jupyter notebook` if lighter).
- `pytest Notebooks/Agent_Development/Modified_PPO_Trainer -q` runs unit checks on PPO helpers.
- `python -m luxai_s3.viewer Notebooks/Agent_Development/replay_my_agent.html` inspects saved matches.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation and `snake_case` identifiers. Preserve the notebook pattern `DeepSeek-R1-Distill-<Model>_<Method>_YYYYMMDD_##.ipynb` so runs sort lexically. Move reusable helpers into `Modified_PPO_Trainer`, maintain existing type hints, and keep line length close to the current 120-character style.

## Testing Guidelines
Prefer pytest-based coverage for trainer modules; create `tests/` beside `Modified_PPO_Trainer` when adding new suites and name cases clearly (e.g., `test_rewards_clamped_when_invalid_logprob`). For notebooks, include a quick smoke cell that instantiates the trainer on a minimal rollout (≤2 turns, CPU fallback) and rerun it before opening a PR. Use `pytest --maxfail=1 --disable-warnings --cov=Notebooks/Agent_Development/Modified_PPO_Trainer` to track regressions.

## Commit & Pull Request Guidelines
History shows short, present-tense subjects (`Update README.md`, `added english readme`). Keep commits focused with ≤72-character subjects and note run context or config changes in the body. PR descriptions should outline intent, link Kaggle or WandB runs, enumerate validation commands, and attach new replays or screenshots under `images/`. Highlight reviewers when touching reward shaping, dependency pins, or notebook entry points.

## Agent-Specific Workflow Tips
Log PPO runs with WandB when possible; otherwise record seeds, map IDs, and reward deltas in the lead markdown cell. Save new replays as `replay_<scenario>.html` under `Agent_Development` and reference them from README updates. Before exporting an agent, confirm notebook paths point to the intended checkpoint and store large binaries outside Git.
