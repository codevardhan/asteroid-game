from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from ray.rllib.core.rl_module import RLModule
from environment import AsteroidsRLLibEnv

# Create only the neural network (RLModule) from our algorithm checkpoint.
# See here (https://docs.ray.io/en/master/rllib/checkpoints.html)
# to learn more about checkpointing and the specific "path" used.
checkpoint_path = "/home/codevardhan/ray_results/PPO_2025-04-07_13-00-25/PPO_AsteroidsRLLibEnv_cceb4_00000_0_2025-04-07_13-00-25/checkpoint_000004"
rl_module_a = RLModule.from_checkpoint(
    Path(checkpoint_path)
    / "learner_group"
    / "learner"
    / "rl_module"
    / "asteroid_policy"
)
env = AsteroidsRLLibEnv()
episolon = 0.94
for _ in range(1000):
    episode_return = 0.0
    done = False
    obs, _ = env.reset()
    while not done:
        # Uncomment this line to render the env.
        env.render()

        # Compute the next action from a batch (B=1) of observations.
        obs_batch_asteroid = torch.from_numpy(obs.get("asteroid")).unsqueeze(
            0
        )  # add batch B=1 dimension
        print(torch.from_numpy(obs.get("asteroid")).unsqueeze(0))
        # obs_batch_player = torch.from_numpy(obs.get("player")).unsqueeze(0)  # add batch B=1 dimension
        model_outputs = rl_module_a.forward_inference({"obs": obs_batch_asteroid})
        # Extract the action distribution parameters from the output and dissolve batch dim.
        print(model_outputs["action_dist_inputs"][0].numpy())
        action_dist_params = model_outputs["action_dist_inputs"]
        # We have continuous actions -> take the mean (max likelihood).
        # greedy_action = np.clip(
        # action_dist_params[0:1],  # 0=mean, 1=log(stddev), [0:1]=use mean, but keep shape=(1,)
        # a_min=env.action_space.asteroid[0],
        # a_max=env.action_space.asteroid[0],
        # )
        # For discrete actions, you should take the argmax over the logits:
        greedy_action = np.argmax(action_dist_params)
        action_dict = {"player": None, "asteroid": greedy_action}
        # Send the action to the environment for the next step.
        obs_dict, rew_dict, terminated, truncated, info_dict = env.step(action_dict)
        obs = obs_dict
        # Perform env-loop bookkeeping.
        episode_return += rew_dict.get("asteroid")
        done = terminated.get("asteroid") or truncated.get("asteroid")

print(f"Reached episode return of {episode_return}.")
