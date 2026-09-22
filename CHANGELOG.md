# Changelog

All notable changes to the project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- fix Presentation section missing from README body: add section with slides link after Quick Start (README.md) [via commit]
- add project presentation slides and link in README (README.md, slides/index.html) [via commit]
- add A6 guide: Rewards & Test-Time Compute (docs/tracks/a-llm-post-training/a6-rewards-and-test-time-compute.md) — outcome vs process rewards, test-time compute (best-of-N, self-consistency, search), and a NumPy comparison of four inference strategies; completes Track A (7 of 7) and the full 14-guide curriculum
- add launch materials and narrow production RL claim scope (CHANGELOG.md, README.md, docs/launch/claims.md, +2 more) [via commit]
- add launch copy and claim receipts under `docs/launch/` — LinkedIn post, X thread, and a sourced note on the production-RL claim
- add .gitignore to project structure tree in README (README.md) [via commit]
- add B6 guide and complete Track B (CHANGELOG.md, CONTRIBUTING.md, README.md, +3 more) [via commit]
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

- mark all 14 guides live: update status lines in README.md, CONTRIBUTING.md, and docs/launch/launch-copy.md; link A6 from the Track A index, README guide table, and A5 footer (removes the last `(planned)` marker)
- finalize launch readiness by replacing placeholders and updating metadata (CHANGELOG.md, README.md, docs/launch/launch-copy.md, +2 more) [via commit]
- update RL guides with 2025-2026 findings on scaling and alignment (README.md, docs/tracks/a-llm-post-training/a3-dpo-and-rft.md, docs/tracks/a-llm-post-training/a4-grpo-and-rlvr.md, +1 more) [via commit]
- narrow the unsupported "most production RL" claim in README.md and docs/tracks/b-decision-optimization/index.md to a scoped, checkable population (ads, ranking, recommendation); receipts recorded in docs/launch/claims.md
- update project structure tree in README (README.md) [via commit]
- clarify A0 GPU requirement in README tech stack (README.md) [via commit]
- update last updated date in README (README.md) [via commit]
- ignore CHANGELOG.md in .gitignore (.gitignore) [via commit]
- rename project to RL Two Tracks (CHANGELOG.md, CONTRIBUTING.md, LICENSE, +2 more) [via commit]
- rename project from "RL Learning Lab" to "RL Two Tracks" (README.md, CONTRIBUTING.md, PLAN.md, LICENSE) — repo name now names the repo's core differentiator (the two-track A/B split)
- update README and docs to reflect A5 guide status (CHANGELOG.md, README.md, docs/tracks/a-llm-post-training/a5-agentic-rl-environments.md) [via commit]
- restructure README.md for linear readability: single routing table up top, then Track B and Track A as self-contained blocks with no interleaved guide tables; each track's guides, paradigm, ladder, and paths now sit together
- update README, Track A index, and status line to reflect A0 guide (12 of 13 guides live); A0 noted as the only non-NumPy, GPU-required guide

### Removed

- remove MACHAAO and vendor-specific API references from PLAN.md (CHANGELOG.md, PLAN.md) [via commit]
- remove MACHAAO and vendor-specific API references from guides (docs/RL_METHODS_GUIDE.md, docs/tracks/a-llm-post-training/a2-rlhf.md, docs/tracks/b-decision-optimization/b6-production-online-systems.md, +1 more) [via commit]
- remove remaining MACHAAO and vendor-specific API references from PLAN.md — retargeted storage to a generic key-value store + append-only log (`MACHAAO` → `Data Store`, `/app-data/{key}` → `key-value store`, `/content` → `append-only log`, `machaao_store.py` → `local_store.py`, `STORE_BACKEND=machaao` → `local`, and dropped the `MACHAAO_*` env vars)
- README.md: duplicate A/B guide tables and repeated Track A "post-training stack" block that interleaved the two tracks

### Security
