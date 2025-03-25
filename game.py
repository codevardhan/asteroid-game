import sys
import threading
import time

sys.path.insert(0, "game/original")
sys.path.insert(0, "pcg-agents/")

import pygame
import pygame.freetype
from constants import *
from player import Player
from asteroid import Asteroid
from asteroidfield import AsteroidField
from shot import Shot
from stable_baselines3 import PPO
from pcgrl_koster import AsteroidsPCGEnvKoster

# Shared variables and synchronization primitives
current_obs = None
predicted_action = None
obs_lock = threading.Lock()
stop_prediction = False


# offload predicition to a diff thread
def prediction_worker(model):
    global current_obs, predicted_action, stop_prediction
    while not stop_prediction:
        # Copy the current observation safely
        with obs_lock:
            obs_copy = current_obs
        if obs_copy is not None:
            # Run the model prediction
            action, _ = model.predict(obs_copy, deterministic=False)
            with obs_lock:
                predicted_action = action
        # Slight sleep to yield control
        time.sleep(0.005)


def main():
    global current_obs, predicted_action, stop_prediction
    pygame.init()
    clock = pygame.time.Clock()

    # Create the environment with rendering enabled
    env = AsteroidsPCGEnvKoster(render_mode="human", max_steps=600, spawn_limit=3)
    obs, info = env.reset()

    model = PPO.load("outputs/asteroids_pcg_model", device="cpu")

    env.agent.update = lambda dt, player, asteroids: None

    stop_prediction = False
    pred_thread = threading.Thread(target=prediction_worker, args=(model,))
    pred_thread.start()

    terminated = False
    while not terminated:
        # Process events (quit and restart game)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if terminated and event.key == pygame.K_r:
                    # Restart game if game over and R is pressed
                    obs, info = env.reset()

        with obs_lock:
            current_obs = obs
            action_to_use = predicted_action

        if action_to_use is None:
            action_to_use = [0, 0, 0, 0]

        obs, reward, terminated, truncated, info = env.step(action_to_use)
        env.render(info)
        clock.tick(60)

    stop_prediction = True
    pred_thread.join()
    
    env.game_over_state()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
