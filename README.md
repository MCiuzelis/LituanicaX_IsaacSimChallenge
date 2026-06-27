# LituanicaX — Isaac Sim Challenge

Deep reinforcement learning (PPO) for a **MuSHR nano v2** RC car navigating a cone-lined track in **NVIDIA Isaac Sim**.

The agent observes a 23-dimensional state (speed, yaw rate, centerline error, heading, corner distance/curvature, plus 5 body-frame lookahead track points) and learns to drive at high speed (~6.7 m/s) using a 2-layer MLP actor-critic (512→256, mish activation) trained via RSL-RL.

## Structure

- `train.py` / `play.py` — entry points for training and deployment
- `source/tasks/cone_track/` — environment (`env.py`) and PPO config (`agents/ppo_cfg.py`)
- `assets/` — USD assets: car, track mesh, walls
- `TrackPoints.csv` — centerline waypoints
