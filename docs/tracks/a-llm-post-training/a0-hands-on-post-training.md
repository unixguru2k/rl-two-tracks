# A0 — Hands-On Post-Training (SFT → DPO → GRPO)

> **Track A · Guide 1 of 7** — [Track A Index](index.md) · [RL Methods Guide](../../RL_METHODS_GUIDE.md)

## What You'll Learn

The other Track A guides explain *why* post-training works. This one makes you *do* it — end to end, on hardware you already own. You will run the full pipeline: **SFT → DPO → GRPO**, starting with a small model on your laptop and finishing on a free Colab GPU.

> **TL;DR:** Climb the pipeline in order. SFT teaches *format*, DPO teaches *preference*, GRPO teaches *reasoning*. Most people stop at DPO — that is the intended design.
>
> ⚠ **This is the only guide in the repo that needs more than NumPy.** Every other guide runs on CPU with a single dependency. This one needs a GPU (the free Colab tier is enough) and a real library (TRL).

---

## Prerequisites

- An Apple Silicon Mac (for the fast local loop) and a Google account (for Colab)
- Comfort with Python and a terminal
- Read [A3](a3-dpo-and-rft.md) and [A4](a4-grpo-and-rlvr.md) first — this guide assumes you already know what DPO and GRPO *are*

---

## The Pipeline You're Climbing

```
   SFT            DPO              GRPO
   ───            ───              ────
 teach format   align to pref    learn from a verifier
 cross-entropy  sigmoid on       group-relative, 0/1 reward
 on targets     winner/loser
   │              │                │
   │              │                └── A4: silent failures, 8× compute
   │              └─────────────────── A3: no reward model, no RL loop
   └────────────────────────────────── the foundation everything sits on

 Most people stop at DPO. That is the intended design.
```

| Stage | Guide | Objective | Data | Signal |
|-------|-------|-----------|------|--------|
| **SFT** | [A1](a1-policy-gradient-ppo.md) context | Match demonstrations | Instruction / answer pairs | Cross-entropy loss |
| **DPO** | [A3](a3-dpo-and-rft.md) | Prefer winner over loser | Preference pairs | Sigmoid log-ratio loss |
| **GRPO** | [A4](a4-grpo-and-rlvr.md) | Maximize verifiable reward | Problems + a verifier | Group-relative advantage |

---

## Hardware Reality

What actually fits where. Be honest with yourself about this table before you start.

| Machine | Memory | What fits |
|---------|--------|-----------|
| **Colab free** | T4, 16 GB | QLoRA on ≤3B models. SFT + DPO. 12 h sessions, disconnects. |
| **Colab Pro** | L4 24 GB / A100 40 GB | 7B QLoRA, small GRPO runs |
| **M4 (16–32 GB)** | unified | MLX inference + LoRA. Great for learning, not for large training. |

**The rule for this guide:** QLoRA + a **0.5B–3B** model. You are learning the *mechanics*, not chasing a frontier model.

---

## The Stack

```
TRL           SFTTrainer / DPOTrainer / GRPOTrainer   ← learn this one
PEFT          LoRA adapters
bitsandbytes  4-bit quantization (QLoRA)
Unsloth       2× faster, half the memory (Colab-friendly)
MLX / mlx-lm  Apple-Silicon-native inference + LoRA
```

Learn TRL. Reach for Unsloth on Colab when you want the run to actually finish.

---

## Part 1 — Baseline Inference (M4)

Before you train anything, run a model and watch it behave. You cannot tell whether training worked if you don't know the baseline.

**1.1 — Isolate your environment**

```bash
cd ~/rl-testing
python3 -m venv .venv
source .venv/bin/activate        # ← required in EVERY new terminal
pip install -U pip mlx-lm
```

**1.2 — Run inference**

Use the module form — it works regardless of PATH:

```bash
python -m mlx_lm generate \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --prompt "Explain RLHF in one sentence"
```

> **CLI version note:** newer `mlx-lm` uses subcommands (`mlx_lm generate`); older versions use dotted modules (`mlx_lm.generate`). The `python -m` form above works on both. The first run downloads ~3 GB.

**1.3 — Record your baseline**

```bash
python -m mlx_lm generate \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --prompt "You are a pirate. Explain RLHF in one sentence."
```

Write the output down. Every training run from here gets compared against this.

---

## Part 2 — Local LoRA SFT (M4)

This is the most valuable part for learning: a full **train → evaluate** cycle in minutes, on your laptop. MLX has LoRA built in.

**2.1 — Make a tiny dataset**

```bash
mkdir -p ~/rl-testing/data
cat > ~/rl-testing/data/train.jsonl << 'EOF'
{"messages": [{"role": "user", "content": "Say hi"}, {"role": "assistant", "content": "Ahoy! Ready to set sail."}]}
{"messages": [{"role": "user", "content": "What is RL?"}, {"role": "assistant", "content": "Ahoy! It be learnin' from rewards, matey."}]}
{"messages": [{"role": "user", "content": "Explain PPO"}, {"role": "assistant", "content": "Ahoy! A clipped policy update, savvy?"}]}
EOF
cp ~/rl-testing/data/train.jsonl ~/rl-testing/data/valid.jsonl
```

Three examples is enough to *see* the mechanics. It will overfit — that is the point.

**2.2 — Train the adapter**

```bash
python -m mlx_lm lora \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --train \
  --data ~/rl-testing/data \
  --iters 200 \
  --batch-size 2 \
  --learning-rate 1e-5 \
  --adapter-path ~/rl-testing/adapters
```

**2.3 — Generate with the adapter**

```bash
python -m mlx_lm generate \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-path ~/rl-testing/adapters \
  --prompt "Say hi"
```

Compare to Part 1. If it now talks like a pirate, **you just did SFT and watched the weights change.**

---

## Part 3 — SFT on Colab

Now scale up to a GPU. Open a new Colab notebook → **Runtime → Change runtime type → T4 GPU**.

**3.1 — Install**

```python
!pip install -q -U trl peft transformers accelerate bitsandbytes datasets
```

**3.2 — Train**

```python
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
dataset = load_dataset("trl-lib/Capybara", split="train[:2000]")

trainer = SFTTrainer(
    model=model_id,
    train_dataset=dataset,
    peft_config=LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear",
    ),
    args=SFTConfig(
        output_dir="/content/qwen-sft",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
    ),
)
trainer.train()
trainer.save_model("/content/qwen-sft")
```

**3.3 — Save to Drive immediately**

Colab wipes `/content` on disconnect.

```python
from google.colab import drive
drive.mount('/content/drive')
!cp -r /content/qwen-sft /content/drive/MyDrive/qwen-sft
```

**3.4 — Compare**

Generate the same prompt with the base model and the trained adapter. The format should have shifted. That difference *is* what you trained.

---

## Part 4 — DPO on Colab

Start from your SFT output.

```python
from datasets import load_dataset
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig

dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:2000]")

trainer = DPOTrainer(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear"),
    train_dataset=dataset,
    args=DPOConfig(
        output_dir="/content/qwen-dpo",
        beta=0.1,                        # KL strength to the reference model
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        num_train_epochs=1,
        bf16=True,
        logging_steps=10,
    ),
)
trainer.train()
```

**The one knob that matters: `beta`.**

```
beta too low   → the model drifts off the reference → incoherent output
beta too high  → it barely learns
beta = 0.1     → the standard starting point
```

**Measure it properly.** Take ~50 prompts, generate with the SFT model and the DPO model, and pick the winner with a judge (or by hand). That is **win-rate** — the real metric. Training loss is not.

---

## Part 5 — GRPO on Colab

The hardest stage. Do it last, on the smallest model, with the simplest possible verifier.

```python
import re
from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig

dataset = load_dataset("openai/gsm8k", "main", split="train[:1000]")

def reward_correct(completions, answer, **kwargs):
    """The verifier — this is RLVR. Reward is 0 or 1. No human, no reward model."""
    rewards = []
    for completion, gold in zip(completions, answer):
        text = completion[0]["content"] if isinstance(completion, list) else completion
        m = re.search(r"####\s*([\d,\.\-]+)", text)
        pred = m.group(1).replace(",", "") if m else None
        rewards.append(1.0 if pred == gold.replace(",", "") else 0.0)
    return rewards

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    reward_funcs=reward_correct,
    train_dataset=dataset,
    peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear"),
    args=GRPOConfig(
        output_dir="/content/qwen-grpo",
        num_generations=8,               # the GROUP size from A4
        per_device_train_batch_size=8,
        learning_rate=1e-6,
        num_train_epochs=1,
        bf16=True,
        logging_steps=10,
    ),
)
trainer.train()
```

**Watch the reward column, not the loss.** It should climb from ~0.0 toward ~0.3+ on a 1.5B model. If it stays flat at 0.0, **every sample in the group scored the same** — that is the vanishing-advantage failure, not a learning-rate problem.

---

## What to Measure at Each Stage

| Stage | Metric | Not this |
|-------|--------|----------|
| SFT | Does the format change? Eval loss | Training loss |
| DPO | **Win-rate** vs the SFT model | Training loss |
| GRPO | **pass@1** on held-out problems | Training loss |

The trap that catches everyone: **loss going down is not the goal.** Behavior change is the goal. Always compare against the *previous stage's model*, not just the loss number.

---

## Why Part 5 Is the Hard One

GRPO is hard for one reason: **it is the first stage where the signal can silently be zero.** Every stage before it feeds the model *correct data*. GRPO makes the model *discover* correctness from a 0/1 signal that is usually 0.

```
The advantage, and what happens when the model is weak:

  A_i = (r_i − mean(r)) / std(r)

  rewards = [0, 0, 0, 0, 0, 0, 0, 0]
  mean = 0,  std = 0
  → A_i = 0 for every sample
  → gradient = 0
  → the model learns NOTHING, and the loss looks "fine"
```

That is the **vanishing-advantage** failure from the [Track A index](index.md), and it is the default state early in training.

The multipliers stacked on top of it:

| Why it's hard | Detail |
|---------------|--------|
| **Group cost** | 8 full generations per prompt before a single gradient step |
| **Long outputs** | Reasoning lengthens sequences → memory ≈ group × seq × hidden |
| **Verifier fragility** | The model must emit the exact format your regex expects; early on it doesn't |
| **Two models** | Policy + frozen reference for the KL penalty |
| **KL leash** | Too strong → learns nothing; too weak → KL collapse |
| **Unforgiving LR** | `1e-6`, 50–200× smaller than SFT, or it collapses |
| **Newest tooling** | `GRPOTrainer` has more sharp edges and fewer examples |
| **Slow loop** | 30–60 min per run on a T4 → 2–3 shots before Colab disconnects |

**How to make it not-hard:**

```
1. Verifier FIRST. Unit-test reward_correct on hand-made strings.
   Confirm it returns 1.0 on a correct answer BEFORE training.
2. Make the task trivially easy. Tiny arithmetic, not GSM8K.
   You want a base pass-rate of ~30–50% so groups have variance.
3. Smallest model that can sometimes succeed (0.5B–1.5B).
4. num_generations=4, max_completion_length=128. Shrink everything.
5. Watch the REWARD column. Flat at 0 → vanishing advantage.
6. Run it 3 times and confirm reward moves before you believe anything.
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `command not found: mlx_lm.generate` | venv not active / wrong CLI form | `source .venv/bin/activate`, use `python -m mlx_lm ...` |
| Garbage / repeating output | Chat template mismatch | Use the model's own template |
| Loss flat at 0 (GRPO) | All group rewards identical | Bigger model, easier problems, larger group |
| Incoherent after DPO | Drifted off the reference | Raise `beta`, lower LR |
| Colab killed mid-run | Session timeout | Save to Drive every epoch |
| OOM on T4 | Model too big | Use a 0.5B model, or QLoRA (4-bit) |

---

## Key Limitations

```
1. YOU ARE LEARNING MECHANICS, NOT TRAINING A FRONTIER MODEL
   A 1.5B model on a free T4 will not produce a reasoning model.
   The point is to run the pipeline yourself, once.

2. LoRA ≠ full fine-tune
   Adapters are cheap and swappable, but they cannot teach a new
   capability the way a full fine-tune can.

3. Colab is not a training platform
   12 h sessions, disconnects, no persistent disk. Fine for learning,
   wrong for production.

4. GRPO needs a verifier
   Works only where correctness is machine-checkable. Open-ended
   creativity or empathy has no automatic verifier.

5. Reward hacking is real
   A model optimized against a fixed verifier will find its weaknesses.
   Always hold out test problems.
```

---

## Connecting to Real RL Methods Guide

From the [RL Methods Guide](../../RL_METHODS_GUIDE.md):

> **SFT** teaches the shape of a good answer. **DPO** aligns to preferences with no reward model. **GRPO** optimizes against a verifier with no critic.
> **This guide** is the hands-on path through all three.

---

## Real-Life Applications

| Stage | What it's used for in practice |
|-------|-------------------------------|
| SFT | Instruction-following, format compliance, domain tone |
| DPO | Open-weight alignment — the default recipe for Zephyr, Llama, and Mistral fine-tunes |
| GRPO | Reasoning and code, where a verifier exists (the DeepSeek-R1 lineage) |

---

## Where You Fit In

```
YOUR ROLE AS AN APP DEVELOPER:
  ❌ You will almost never post-train for a product feature
  ✅ Personalization lives in Track B (a bandit in front of a frozen model)
  ✅ Post-train only when the behavior cannot be expressed as an instruction

RUN THIS GUIDE TO UNDERSTAND, NOT TO SHIP.
  It makes A1–A5 concrete: you'll have watched loss curves,
  win-rates, and a verifier move — with your own hands.
```

---

## Summary: The Complete RL Journey

```
Level 0: Random         → No learning, establish baseline
Level 1a: Classic MAB   → Learn best GLOBAL default
Level 1b: LinUCB        → Personalize with context
Level 2: Q-Learning     → Handle sequential decisions
Level 3: Feature Q      → Scale to large state spaces
Level 4: DQN            → Raw inputs, non-linear patterns
Level 5: PPO            → Direct policy optimization (has critic)
Level 6: RLHF           → Reward model + PPO (classic alignment)
Level 7: DPO            → Direct preference optimization (no RM, offline)
Level 8: GRPO + RLVR    → Group-relative, verifiable rewards (reasoning)

A0 is the hands-on path through Levels 5–8:
  SFT → DPO (Level 7) → GRPO (Level 8)

For most chatbot applications: Levels 1b–3 are still the sweet spot.
For alignment/fine-tuning: DPO is the pragmatic default.
For reasoning with verifiers: GRPO + RLVR.
```

---

## Further Reading

- **TRL documentation** — SFTTrainer / DPOTrainer / GRPOTrainer
- **PEFT / LoRA** — Hu et al., 2021
- **QLoRA** — Dettmers et al., 2023
- **Unsloth** — memory-efficient fine-tuning
- **mlx-lm** — Apple Silicon inference and LoRA

---

**Next:** [A1 — Policy Gradient / PPO](a1-policy-gradient-ppo.md)

**Up:** [Track A Index](index.md) · **Reference:** [RL Methods Guide](../../RL_METHODS_GUIDE.md)
