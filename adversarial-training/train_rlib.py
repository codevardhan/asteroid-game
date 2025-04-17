import ray
import os
import torch
import numpy as np
import torch.nn as nn
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.callbacks.callbacks import RLlibCallback
from environment import AsteroidsRLLibEnv

# from env2 import AsteroidsRLLibEnv
import matplotlib.pyplot as plt
import pandas as pd
from ray.rllib.algorithms.callbacks import DefaultCallbacks

# class TrainableCheckpoint(tune.Trainable):
#     def setup(self, config):
#         self.model = nn.Sequential(
#             nn.Linear(config.get("input_size", 32), 32), nn.ReLU(), nn.Linear(32, 10)
#         )

#     def step(self):
#         return {}

#     def save_checkpoint(self, tmp_checkpoint_dir):
#         checkpoint_path = os.path.join(tmp_checkpoint_dir, "multi-agent-asteroid.pth")
#         torch.save(self.model.state_dict(), checkpoint_path)
#         return tmp_checkpoint_dir

#     def load_checkpoint(self, tmp_checkpoint_dir):
#         checkpoint_path = os.path.join(tmp_checkpoint_dir, "multi-agent-asteroid.pth")
#         self.model.load_state_dict(torch.load(checkpoint_path))


class SelectiveTrainingCallback(RLlibCallback):
    def on_train_result(self, *, trainer, result, **kwargs):
        # Update every 2 iterations
        iteration = result["training_iteration"]

        if iteration % 3 == 0:  # Train the player every 3 iterations
            trainer.workers.foreach_worker(
                lambda worker: worker.set_policies_to_train(
                    ["player_policy", "asteroid_policy"]
                )
            )
        else:
            trainer.workers.foreach_worker(
                lambda worker: worker.set_policies_to_train(["asteroid_policy"])
            )

        return result


if __name__ == "__main__":
    ray.init()

    def policy_mapping_fn(agent_id, *args, **kwargs):
        return f"{agent_id}_policy"

    # Create a base PPOConfig:
    config = PPOConfig()

    # 1) Environment + Env Config
    config = config.environment(
        env=AsteroidsRLLibEnv,
        env_config={"render_mode": True},
    )
    # 2) Framework
    config = config.framework("torch")

    # 3) Resources (GPUs etc.)
    config = config.resources(num_gpus=1)

    # 4) Multi-agent setup
    env_example = AsteroidsRLLibEnv({"render_mode": True})

    # config = config.callbacks(VideoSaveCallback)

    config = config.multi_agent(
        policies={
            "player_policy": (
                None,
                env_example.observation_space_player,
                env_example.action_space_player,
                {
                    "model": {
                        "fcnet_hiddens": [32, 64],
                        "fcnet_activation": "relu",
                    },
                },
            ),
            "asteroid_policy": (
                None,
                env_example.observation_space_asteroid,
                env_example.action_space_asteroid,
                {
                    "model": {
                        "fcnet_hiddens": [32, 64],
                        "fcnet_activation": "tanh",
                    },
                },
            ),
        },
        policy_mapping_fn=lambda agent_id, *args, **kw: f"{agent_id}_policy",
        policies_to_train=["player_policy", "asteroid_policy"],
    )
    config["callbacks"] = SelectiveTrainingCallback

    # 5) RLModule configuration for your model architecture
    #    (the new place to put 'fcnet_hiddens', CNN sizes, etc.)
    # config = config.rl_module(
    #     model_config={
    #         "fcnet_hiddens": [
    #             64,
    #             64,
    #         ],
    #         # "fcnet_activation": "relu",
    #     }
    # )

    #     config = config.rl_module(
    #     model_config={
    #         "fcnet_hiddens": [64, 64],
    #         "fcnet_activation": "relu",
    #     }
    # )

    # 6) Training hyperparameters
    #    (still set gamma, lr, etc. via .training)
    config = config.training(gamma=0.99, lr=1e-3, entropy_coeff=0.01)
    config = config.training(gamma=0.99, lr=1e-3, entropy_coeff=0.01)
    config.rollout_fragment_length = 500
    # 7) Rollout/worker config. The new API uses direct fields:
    #    Typically: config.num_rollout_workers, not config.num_env_runners

    config.num_env_runners = 1

    # Now run with Ray Tune’s Tuner
    tuner = tune.Tuner(
            "PPO",
            run_config=tune.RunConfig(
            stop={"training_iteration": 50},
            checkpoint_config=tune.CheckpointConfig(checkpoint_frequency=5),                                 # Stops after 300 training iterations
        ),
        param_space=config.to_dict()
    )
    results = tuner.fit()
    print("Training completed!")
    latest_result = results.get_best_result()
    checkpoint_dir = latest_result.checkpoint.path
    trained_algo = PPO.from_checkpoint(checkpoint_dir)
    save_path = os.path.join(checkpoint_dir, "trained_model")
    trained_algo.save(save_path)

    print(f"Model saved at: {save_path}")
