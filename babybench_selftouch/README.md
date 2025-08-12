# Baby Robot — Curiosity‑Driven Multimodal Self‑Exploration

A clean, reproducible implementation of our developmentally‑inspired framework that lets a simulated infant robot learn its **body schema** (a predictive sensorimotor model of its own body) with **no external tasks** and **no extrinsic rewards**. Learning is driven by:

* A **dual‑channel VAE** that fuses proprioception and touch into a compact latent state.
* A **latent‑space forward dynamics model** used for curiosity.
* A **dynamic, diversity‑seeking touch reward** that simulates habituation and interest shift.
* A simple curriculum that shifts emphasis from “discover touch” → “understand dynamics”.

> TL;DR: From random motor babbling to structured, infant‑like self‑touch sequences (head/torso → lower body) using only intrinsic motivation.

---

## Repository Layout

```
babybench_selftouch/
├─ train_selftouch.py          # Entry point: sets up env, wrapper, ICM, PPO, training loop
├─ config_selftouch.yml        # BabyBench environment configuration (scene, sensors, episode length…)
├─ selftouch_wrapper.py        # TouchRewardWrapper: time‑gated, diversity‑seeking touch rewards (+ hand logic)
├─ rewards.py                  # SoftmaxTouchReward: habituation + interest‑shift via softmax of decayed counts
├─ icm/
│  ├─ icm_module.py            # ICMModule: dual VAEs + latent forward model (+ optional inverse, unused)
│  ├─ vae.py                   # VAE encoder/decoder with σ‑loss reconstruction
│  ├─ forward.py               # Residual forward model Δz prediction in latent space
│  └─ (other helpers)
└─ icm_callback.py             # On‑policy rollout hook: computes ICM reward, mixes weights, saves models, logs

paper/
└─ babyrobot_paper.pdf         # Method & results write‑up
```

---

## Method Overview

**State model (Body Schema)**

* **Dual‑VAE** encodes two streams separately: high‑dimensional **proprioception** and sparse **touch**, then concatenates latent means to form $z_t$.
* **Forward model** (MLP, residual) predicts $z_{t+1}$ from $[z_t, a_t]$. Curiosity = prediction error.
* **Stabilized training** uses a σ‑loss (log‑MSE‑style) to smooth gradients on large errors.

**Intrinsic reward (no extrinsic rewards used)**

* **ICM curiosity**: forward‑model prediction error in latent space.
* **Touch reward**:

  * **General body**: time‑gated (reward window + cooldown), diversity‑seeking via **softmax** over exponentially decayed touch counts (habituation → interest shift to under‑explored parts).
  * **Hand touches**: separate path with its own windows/cooldowns and an over‑hold penalty (discourages static contacts).
* **Curriculum weighting**: linearly ramp weights over the first \~1M steps: high touch‑drive early, then curiosity dominates.

**Training loop**

* PPO (SB3) with `MultiInputPolicy` for proprioception + touch.
* After each rollout: compute ICM forward loss from transitions; wrapper injects unweighted touch/hand components; callback mixes dynamic weights and writes the final reward into PPO.

---

## Requirements

* Python 3.10+
* \[stable‑baselines3]
* PyTorch (CUDA/MPS optional)
* MuJoCo runtime
* **BabyBench** environment (MIMo infant model); install per BabyBench docs.

> Tip: Ensure MuJoCo and BabyBench are discoverable on your `PYTHONPATH` and that system packages for MuJoCo are installed.

---

## Quick Start

1. **Install dependencies** (pseudo‑steps)

```bash
# Create and activate env (example)
conda create -n babyrobot python=3.10 -y && conda activate babyrobot
pip install torch stable-baselines3 gymnasium mujoco
# Install BabyBench per its README (provides `babybench.utils.make_env`)
```

2. **Configure the environment**

* Edit `config_selftouch.yml` to choose scene, sensors, and episode length. Defaults:

  * `scene: crib`
  * `max_episode_steps: 4000`
  * touch + proprioception enabled; vision off by default.
  * results saved under `results/self_touch/`.

3. **Train (4M steps baseline)**

```bash
python -m babybench_selftouch.train_selftouch \
  --config babybench_selftouch/config_selftouch.yml \
  --train_for 4000000
```

Expected console notes:

* Saves PPO and ICM checkpoints every 4096 env‑steps under `results/self_touch/{ppo_model,icm_model}`.
* TensorBoard logs under `results/self_touch/logs`.
* Prints used random seed.

---

## Key Hyperparameters (baseline)

* **PPO**: `n_steps=4096`, `ent_coef=3e-5`, `lr=1e-4`, policy=`MultiInputPolicy`.
* **VAE latents**: proprio=64, touch=24; hidden=512; `vae_beta=0.01`; optimizer lr=3e-4.
* **Curriculum** (linear 0→1 until 1M steps):

  * `λ_icm: 0.005 → 0.1`
  * `λ_touch: 10.0 → 2.5`
  * `λ_hand: 80.0 → 8.0`
* **Touch wrapper** (defaults in script):

  * General: `reward_window=80`, `cooldown=200`
  * Hand: `reward_value=1`, `reward_window=60`, `cooldown=30`, `overhold_threshold=300`, `overhold_penalty=1`
* **Seed**: `88` (also set for NumPy & PyTorch).

> You can change all of these in `train_selftouch.py` or pass a different YAML and tweak the wrapper init.

---

## Outputs & Logging

* **Checkpoints** (every 4096 steps):

  * `results/self_touch/ppo_model/p_model_<steps>_steps.zip`
  * `results/self_touch/icm_model/icm_model_<steps>_steps.pth`
* **TensorBoard**: training curves, including per‑rollout behavior stats logged by the callback:

  * `behavior/touch_diversity_by_hand`, individual part frequencies & durations, and reward component means.

Launch TensorBoard:

```bash
tensorboard --logdir results/self_touch/logs
```

---

## How it Works (Implementation Notes)

* **TouchRewardWrapper** parses BabyBench touch arrays by body‑part, tracks per‑part contact durations, applies **window/cooldown** gates, then:

  * awards **diversity‑weighted** reward for non‑hand body parts (softmax over decayed counts),
  * awards a **separate hand‑touch** bonus/penalty, and
  * exposes **unweighted** components through `info['reward_components']`.
* **ICMCallback** pulls those components + computes **latent forward loss** on‑the‑fly; mixes them using the **dynamic λ‑schedule** and **replaces** the env reward for PPO. It also:

  * auto‑discovers MIMo’s hand/body geoms,
  * logs touch diversity, per‑part counts/durations, and per‑rollout reward means,
  * checkpoints PPO & ICM.
* **ICMModule** holds two VAEs and a residual forward model. After each rollout it runs batched updates with σ‑loss reconstruction and KL (β‑VAE style). The inverse model is included but **not used** in this version.

---

## Reproducing Paper‑Style Behaviors

* Train for ≥4M steps with the default curriculum.
* Inspect TensorBoard for:

  * falling VAE/forward losses,
  * rising exploration diversity and a shift from touch‑dominated to curiosity‑dominated reward.
* Render episodes with a saved PPO checkpoint to qualitatively observe self‑touch patterns (e.g., head/torso → lower body). (A small `eval_*.py` script is easy to add; see TODO below.)

---

## Tips & Gotchas

* **Episode vs. rollout length**: `max_episode_steps=4000` and `n_steps=4096` are intentionally close; PPO handles partial episodes in the buffer—this setting worked robustly in our runs.
* **Sensitivity**: behaviors are somewhat seed/hyper‑sensitive (common for curiosity setups). Keep the curriculum and wrapper gates intact to avoid degenerate reward‑hacking.
* **Hands vs. non‑hand parts**: Hand geoms are auto‑detected from MuJoCo body names containing `"hand"`; if you edit the MIMo model, confirm naming conventions still match.

---

## Extend / Modify

* **Add vision**: enable `vision_active: True` in YAML and extend the policy/encoders.
* **Change curriculum**: tweak `LAMBDA_*_SCHEDULE` and `DYNAMIC_WEIGHT_STOP_STEP` in `train_selftouch.py`.
* **Experiment**: try ablations—remove touch reward, freeze weights, or replace the VAE with a direct MLP (expect collapse/degeneracy).

---

## Citation

If you use this code, please cite the accompanying paper (see `paper/babyrobot_paper.pdf`).

---

## License

TBD.

---

## TODOs

* Evaluation/visualization script for replaying checkpoints and exporting contact maps.
* Optional inverse‑model path (currently unused).
* Packaging + unit tests.
