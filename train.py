import wandb
from wandb.integration.sb3 import WandbCallback

import sys
sys.path.insert(0, 'pcg-agents')

from pcgrl import AsteroidsPCGEnvWithAStar
from stable_baselines3 import PPO


def main():
    wandb.init(project="asteroids-pcg", name="basic-pcg-run")
    env = AsteroidsPCGEnvWithAStar(
        render_mode=None, max_steps=600, spawn_limit=3)

    # Create a PPO model
    model = PPO("MlpPolicy", env, verbose=1, device="cpu")

    # Train
    model.learn(total_timesteps=200000, callback=WandbCallback(
        gradient_save_freq=1000,
        model_save_freq=5000,
        model_save_path="models/",
        verbose=2
    ))

    # Save
    model.save("outputs/asteroids_pcg_model")

    env.close()


if __name__ == "__main__":
    main()
