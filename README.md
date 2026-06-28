# LituanicaX — Isaac Sim Challenge

Deep reinforcement learning (PPO) for a **MuSHR nano v2** RC car navigating a
cone-lined track in **NVIDIA Isaac Sim 2025.1** using **Isaac Lab + RSL-RL**.

Trained on an **RTX 3070 (8 GB)** with **200 parallel environments** in
approximately **6 hours** (1000 PPO iterations). The policy runs at ~60 Hz
(8× decimation from 480 Hz physics).

## Task

The car must drive around a closed track at high speed (~6.7 m/s), staying
between the walls, without flipping over, and ideally completing laps. Each
episode lasts up to 90 seconds (10 800 steps). Termination triggers: wall
collision, car flipped (roll >~73°), or moving too slow after 45 steps.

## Observation (23 dims)

| Indices | Signal | Normalisation |
|---------|--------|---------------|
| 0 | Mean wheel ω → speed | [0, 1] via `max_velocity_m_s` |
| 1 | Forward body velocity | [0, 1] via `max_velocity_m_s` |
| 2 | Lateral body velocity | [-1, 1] via `max_velocity_m_s` |
| 3 | Yaw rate | [-1, 1] via 10 rad/s |
| 4 | Distance to centerline (cross-track error) | [0, 1] via 0.3 m |
| 5 | Signed relative heading to track tangent | [-1, 1] via π |
| 6 | Arc distance to next corner | [0, 1] via 5 m |
| 7 | Curvature of next corner | [0, 1] via 10 m⁻¹ |
| 8–22 | 5 lookahead points (body-frame XY + curvature) | [-1, 1] via 5 m |

Lookahead offsets: ~0.5, 1.0, 2.0, 3.5, and 5.0 m ahead along the direction
of travel. The direction of travel is detected from the car's world-frame
velocity — lookahead and corner metrics automatically invert when the car
drives the opposite way around the track.

## Action (2 dims)

- **Throttle**: `[-1, 1]` → positive accelerates forward, negative brakes
- **Steering**: `[-1, 1]` → scales to ±0.488 rad steering angle

The reward encourages the car to drive as fast as possible, penalising
excessive steering, rapid control changes, wheel slip, and body roll.

## Rewards

| Component | Weight | Calculation |
|-----------|--------|-------------|
| Forward speed | 4.0 | `lin_vel_b_x / max_velocity * dt` — higher is better |
| Alive bonus | 0.1 | Constant +0.1 per step |
| Lap completion | 4.0 | `4.0 / lap_time_s` — faster laps get larger bonus |
| Steering usage | 0.003 | Negative penalty for steering beyond deadzone (±0.05) |
| Steering rate | 0.003 | Negative penalty for rapid steering changes |
| Throttle rate | 0.002 | Negative penalty for rapid throttle changes |
| Wheel slip | 0.03 | Negative penalty for |wheel_vel − speed| / speed |
| Body roll | 0.1 | Negative penalty for |roll| / 15° |

## Termination

- **Wall collision**: car center within 0.15 m of any wall segment
- **Flipped**: up-axis projection < 0.3 (roll >~73°)
- **Stopped**: forward speed < 0.27 m/s after step 45
- **Time out**: episode reaches 90 s

## Training

### Start training

```bash
train --num_envs 200 --max_iterations 1000
```

Data is logged to `logs/rsl_rl/cone_track/<timestamp>/`. The
`train.py` script automatically launches TensorBoard in the background
and prints the URL (default port 6006, falls back to the next available
port if taken). Open it in your browser to monitor live metrics.

TensorBoard logs are at `logs/rsl_rl/cone_track/` — you can also
launch manually:

```bash
tensorboard --logdir logs/rsl_rl/cone_track
```

Key metrics to watch:

| Metric | What it means |
|--------|---------------|
| `Train/mean_reward` | Average episode reward (should rise) |
| `Train/mean_episode_length` | How long agents survive (→ 10 800 = full episode) |
| `Lap/lap_time_s` | Lap completion times (→ ~15–20 s is fast) |
| `Lap/laps_per_min` | Combined lap rate across all envs |
| `Lap/fastest_lap_s` | Global fastest lap so far |
| `Train/mean_lr` | Adaptive learning rate (decreases with KL) |
| `Info/kl` | Policy KL divergence (should stay near `desired_kl=0.01`) |

### Play a checkpoint

```bash
play --checkpoint logs/rsl_rl/cone_track/<timestamp>/model_<iter>.pt
```

## Installation

Requires **NVIDIA driver 550–580**, **uv** (Python package manager),
and an **NVIDIA GPU** with at least 8 GB VRAM (tested on RTX 3070).

```bash
# 1. Install uv if you don't have it yet
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone the repo
git clone git@github.com:MCiuzelis/LituanicaX_IsaacSimChallenge.git
cd LituanicaX_IsaacSimChallenge

# 3. Run the install script
chmod +x install.sh
./install.sh
```

`install.sh` does the following:

1. **Driver check** — verifies the NVIDIA driver version is in the
   compatible range (550–580). Exits early with a message if the
   driver is ≥ 590 (known incompatibility with Isaac Sim 5.1).
2. **Python 3.11** — installs via `uv python install 3.11` if not
   already available.
3. **Isaac Lab submodule** — syncs and initialises the `IsaacLab/`
   submodule at the pinned tag `v2.3.0` (only version compatible
   with Isaac Sim 5.1's dependency tree).
4. **Virtual environment** — runs `uv venv --python 3.11` to create
   `.venv/`, then `uv sync` to install all dependencies from the
   lockfile.
5. **Patches** — force-reinstalls `opencv-python` (GUI-capable build
   replacing the headless variant pulled by Isaac Sim) and pins
   `numpy<2.0.0` and `setuptools<82.0.0` for ABI and import
   compatibility.
6. **Helper scripts** — writes `train`, `play`, and `visualize`
   wrappers to `~/.local/bin/` that `cd` to the project directory
   and run `uv run python <name>.py "$@"`. On dual-GPU systems also
   sets `__NV_PRIME_RENDER_OFFLOAD=1` to force Vulkan onto the
   NVIDIA card.

After the script finishes:

```bash
# Activate the venv (optional — the helper scripts handle this)
source .venv/bin/activate

# Verify it works
train --help
```

## Spawn & direction balancing

Agents spawn at 5 preset positions, each duplicated with a 180° yaw flip so
they learn to drive both directions. An adaptive spawn balancer tracks rolling
mean episode length per direction — if one direction underperforms, a higher
fraction of envs spawn there to keep gradient contributions balanced. As both
directions improve, spawns converge back to 50/50.

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Actor hidden dims | [512, 256] (Mish) |
| Critic hidden dims | [512, 256] (Mish) |
| Init noise std | 0.8 |
| Learning rate | 1e−4 (adaptive) |
| PPO epochs | 6 |
| Mini-batches | 6 |
| GAE λ | 0.95 |
| Discount γ | 0.9965 |
| Entropy coef | 0.004 |
| Clip parameter | 0.2 |
| Desired KL | 0.01 |
| Max grad norm | 0.75 |
| Steps per env per iter | 1600 |

## Project structure

```
├── train.py              Training entry point
├── play.py               Playback / deployment entry point
├── cli_args.py           Shared CLI argument helpers
├── pyproject.toml        UV project config
├── uv.lock
├── install.sh             One-shot setup script (see Installation)
├── Track.blend            Track Blender source file
├── TrackPoints.csv        Centerline waypoints (Blender export)
├── assets/
│   ├── mushr_nano_v2.usd  MuSHR nano v2 robot
│   ├── Track.usdc         Track mesh
│   └── Walls.usdc         Wall collision geometry
├── source/tasks/
│   └── cone_track/
│       ├── env.py          Environment (ConeTrackEnv)
│       ├── cfg.py          Task registration
│       └── agents/
│           └── ppo_cfg.py  PPO runner & algorithm config
├── logs/rsl_rl/
│   └── cone_track/         Training logs, checkpoints, videos
└── .gitignore
```
