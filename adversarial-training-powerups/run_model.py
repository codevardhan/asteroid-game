from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from ray.rllib.core.rl_module import RLModule
from environment import AsteroidsRLLibEnv
# Create only the neural network (RLModule) from our algorithm checkpoint.
# See here (https://docs.ray.io/en/master/rllib/checkpoints.html)
# to learn more about checkpointing and the specific "path" used.
checkpoint_path = "/Users/charlescarlson/ray_results/PPO_2025-04-01_11-17-23/PPO_AsteroidsRLLibEnv_69d4e_00000_0_2025-04-01_11-17-23/checkpoint_000000/"
rl_module_a = RLModule.from_checkpoint(
    Path(checkpoint_path)
    / "learner_group"
    / "learner"
    / "rl_module"
    / "asteroid_policy"
)
rl_module_p = RLModule.from_checkpoint(
    Path(checkpoint_path)
    / "learner_group"
    / "learner"
    / "rl_module"
    / "player_policy"
)
env = AsteroidsRLLibEnv()
episode_return = 0.0
done = False
obs,_ = env.reset()
while not done:
    # Uncomment this line to render the env.
    env.render()

    # Compute the next action from a batch (B=1) of observations.
    obs_batch_asteroid = torch.from_numpy(obs.get("asteroid")).unsqueeze(0)  # add batch B=1 dimension
    obs_batch_player = torch.from_numpy(obs.get("player")).unsqueeze(0)  # add batch B=1 dimension
    print(obs_batch_player)
    print(torch.from_numpy(obs.get("asteroid")).unsqueeze(0))
    #obs_batch_player = torch.from_numpy(obs.get("player")).unsqueeze(0)  # add batch B=1 dimension
    model_outputs = rl_module_a.forward_inference({"obs": obs_batch_asteroid})
    print()
    model_outputs_player = rl_module_p.forward_inference({"obs": obs_batch_player})
    print(model_outputs)
    # Extract the action distribution parameters from the output and dissolve batch dim.
    print(model_outputs["action_dist_inputs"][0].numpy())
    action_dist_params = model_outputs["action_dist_inputs"]
    a_dist = model_outputs_player["action_dist_inputs"]
    print(action_dist_params)
    # We have continuous actions -> take the mean (max likelihood).
    # greedy_action = np.clip(
    #     action_dist_params[0:1],  # 0=mean, 1=log(stddev), [0:1]=use mean, but keep shape=(1,)
    #     a_min=env.action_space.low[0],
    #     a_max=env.action_space.high[0],
    # )
    # For discrete actions, you should take the argmax over the logits:
    greedy_action = np.argmax(action_dist_params)
    g_a = np.argmax(a_dist)
    print(greedy_action,g_a)
    action_dict = {"player": g_a, "asteroid": 1}
    # Send the action to the environment for the next step.
    obs_dict, rew_dict, terminated, truncated, info_dict= env.step(action_dict)

    # Perform env-loop bookkeeping.
    episode_return += rew_dict.get("asteroid")
    done = terminated.get("asteroid") or truncated.get("asteroid")

print(f"Reached episode return of {episode_return}.")