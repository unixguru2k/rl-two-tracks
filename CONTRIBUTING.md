# Contributing

Thanks for helping improve RL Two Tracks. This repo is **documentation-first**: every guide is self-contained and copy-paste runnable, and there is nothing to install.

## Good First Contributions

The one planned guide is the highest-value addition:

- **A6 — Rewards & Test-Time Compute** → `docs/tracks/a-llm-post-training/a6-rewards-and-test-time-compute.md`

Until it is written it is referenced as plain italic text with a `(planned)` marker — never as a link — so no page dead-ends.

## Guide Conventions

To keep the tracks seamless to read and navigate, every guide follows the same shape.

**1. Breadcrumb header (first line):**

```
> **Track A · Guide 4 of 7** — [Track A Index](index.md) · [RL Methods Guide](../../RL_METHODS_GUIDE.md)
```

**2. Section order:**

What You'll Learn → The Concept → Python Example → When to Use → Key Concepts Visualized → Key Limitations → Connecting to Real RL Methods Guide → Real-Life Applications → Next Step

**3. Footer navigation (last lines):**

```
**Previous:** [<prev title>](<prev file>) · **Next:** [<next title>](<next file>)

**Up:** [Track X Index](index.md) · **Reference:** [RL Methods Guide — Level N](../../RL_METHODS_GUIDE.md#<anchor>)
```

- The first guide in a track omits **Previous**. The last guide omits **Next** — unless a further guide is planned, in which case point **Next** at it as italic text with a `(planned)` marker (as A5 does for A6).
- Always keep `Guide X of 7` in sync across both tracks (14 guides total: 7 per track).

## Pull Requests

- Keep examples **NumPy-only** — the one exception is **A0**, which needs a GPU (free Colab is enough).
- Verify every link and anchor resolves before opening a PR.
- When adding or completing a guide, update the matching `index.md` **and** the status line in `README.md`.

## License

By contributing, you agree that your contributions are licensed under the [MIT License](LICENSE).
