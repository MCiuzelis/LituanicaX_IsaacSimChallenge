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
play --num_envs 200 --max_iterations 1000
```

Data is logged to `logs/rsl_rl/cone_track/<timestamp>/`.

### Launch TensorBoard

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
| `Train/mean_lr` | Adaptive learning rate (decreases with KL) |
| `Info/kl` | Policy KL divergence (should stay near `desired_kl=0.01`) |

### Play a checkpoint

```bash
play --checkpoint logs/rsl_rl/cone_track/<timestamp>/model_<iter>.pt
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
├── TrackPoints.csv       Centerline waypoints (Blender export)
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
└── .opencode/              Plan tracking
```
