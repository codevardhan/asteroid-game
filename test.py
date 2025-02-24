import time
import pygame
from stable_baselines3 import PPO
from pcgrl import AsteroidsPCGEnvWithAStar


def main():
    model = PPO.load("outputs/asteroids_pcg_model", device="cpu") 
    env = AsteroidsPCGEnvWithAStar(
        render_mode="human", max_steps=600, spawn_limit=3)

    # model = PPO.load("pcg_koster_model")
    # env = AsteroidsPCGEnvKoster(
    #     render_mode="human", max_steps=600, spawn_limit=3)

    obs, info = env.reset()
    done = False
    truncated = False

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        time.sleep(0.02)

    env.close()


if __name__ == "__main__":
    main()
