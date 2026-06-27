"""RSL-RL PPO configuration for the cone-track task."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class ConeTrackPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env: int = 1200
    max_iterations: int = 1000
    save_interval: int = 10
    experiment_name: str = "cone_track"

    # Observation group mapping required by RslRlVecEnvWrapper
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}

    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256],
        critic_hidden_dims=[512, 256],
        activation="mish",
    )

    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        learning_rate=1e-4,
        schedule="adaptive",
        num_learning_epochs=6,
        num_mini_batches=6,
        clip_param=0.2,
        entropy_coef=0.005,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        gamma=0.9965,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.75,
    )
