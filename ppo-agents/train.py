import wandb
from pathlib import Path
from wandb.integration.sb3 import WandbCallback

from pcgrl_koster_powerups import AsteroidsPCGEnvKoster
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "ppo" / "outputs"
MODEL_DIR = ROOT / "ppo" / "models"
TB_DIR = OUTPUT_DIR / "ppo" / "tb"


for p in (OUTPUT_DIR, MODEL_DIR, TB_DIR):
    p.mkdir(parents=True, exist_ok=True)


def main():
    wandb.init(
        project="asteroids-pcg",
        name="basic-pcg-run",
        dir=OUTPUT_DIR,
        config={
            "algo": "PPO",
            "total_timesteps": 200_000,
            "env_max_steps": 600,
            "spawn_limit": 3,
            "gamma": 0.99,
            "learning_rate": 3e-4,
            "gae_lambda": 0.95,
            "n_steps": 2048,
            "ent_coef": 0.0,
            "clip_range": 0.2,
        },
        sync_tensorboard=True,
    )
    env = AsteroidsPCGEnvKoster(render_mode=None, max_steps=600, spawn_limit=3)

    # Create a PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cpu",
        tensorboard_log=TB_DIR,
    )

    # Train
    model.learn(
        total_timesteps=500000,
        callback=WandbCallback(
            gradient_save_freq=1000,
            model_save_freq=5000,
            model_save_path=MODEL_DIR,
            verbose=2,
        ),
    )

    # Save
    model.save(MODEL_DIR / "asteroids_pcg_model")

    env.close()
    wandb.finish()


if __name__ == "__main__":
    main()
