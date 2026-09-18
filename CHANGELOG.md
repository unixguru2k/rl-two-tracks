# Changelog

All notable changes to the project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- add B6 guide: Production Online Systems (docs/tracks/b-decision-optimization/b6-production-online-systems.md) — cold start, non-stationarity, latency budget, safety rails, and off-policy evaluation; completes Track B (7 of 7)
- resolve broken links and add missing project files (CONTRIBUTING.md, LICENSE, docs/tracks/a-llm-post-training/a5-agentic-rl-environments.md, +2 more) [via commit]
- add navigation footers and update guide counts for seamless reading (docs/tracks/a-llm-post-training/a0-hands-on-post-training.md, docs/tracks/a-llm-post-training/a1-policy-gradient-ppo.md, docs/tracks/a-llm-post-training/a2-rlhf.md, +8 more) [via commit]
- add navigation breadcrumbs and update status in documentation (README.md, docs/tracks/a-llm-post-training/a0-hands-on-post-training.md, docs/tracks/a-llm-post-training/a1-policy-gradient-ppo.md, +10 more) [via commit]
- add A0 hands-on post-training guide (CHANGELOG.md, README.md, docs/tracks/a-llm-post-training/a0-hands-on-post-training.md, +1 more) [via commit]
- add A0 guide: Hands-On Post-Training (docs/tracks/a-llm-post-training/a0-hands-on-post-training.md) — runnable SFT → DPO → GRPO pipeline (M4 + Colab); the only guide requiring a GPU
- add A5 guide: Agentic RL & Environments (docs/tracks/a-llm-post-training/a5-agentic-rl-environments.md) — conceptual only, no code
- add RL chatbot learning lab project plan (PLAN.md) [via commit]

### Fixed

- standardize navigation footers and fix guide numbering (docs/tracks/a-llm-post-training/a0-hands-on-post-training.md, docs/tracks/a-llm-post-training/a2-rlhf.md, docs/tracks/a-llm-post-training/a5-agentic-rl-environments.md, +1 more) [via commit]
### Changed

- ignore CHANGELOG.md in .gitignore (.gitignore) [via commit]
- rename project to RL Two Tracks (CHANGELOG.md, CONTRIBUTING.md, LICENSE, +2 more) [via commit]
- rename project from "RL Learning Lab" to "RL Two Tracks" (README.md, CONTRIBUTING.md, PLAN.md, LICENSE) — repo name now names the repo's core differentiator (the two-track A/B split)
- update README and docs to reflect A5 guide status (CHANGELOG.md, README.md, docs/tracks/a-llm-post-training/a5-agentic-rl-environments.md) [via commit]
- restructure README.md for linear readability: single routing table up top, then Track B and Track A as self-contained blocks with no interleaved guide tables; each track's guides, paradigm, ladder, and paths now sit together
- update README, Track A index, and status line to reflect A0 guide (12 of 13 guides live); A0 noted as the only non-NumPy, GPU-required guide

### Removed

- README.md: duplicate A/B guide tables and repeated Track A "post-training stack" block that interleaved the two tracks

### Security
