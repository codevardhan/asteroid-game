import ray
import os
import torch
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.callbacks.callbacks import RLlibCallback
from environment import AsteroidsRLLibEnv
import matplotlib.pyplot as plt
import pandas as pd

class SelectiveTrainingCallback(RLlibCallback):
    def on_train_result(self, *, trainer, result, **kwargs):
        # Update every 2 iterations
        iteration = result["training_iteration"]
        
        if iteration % 3 == 0:  # Train the player every 2 iterations
            trainer.workers.foreach_worker(
                lambda worker: worker.set_policies_to_train(["player_policy", "asteroid_policy"])
            )
        else:
            trainer.workers.foreach_worker(
                lambda worker: worker.set_policies_to_train(["asteroid_policy"])
            )
            
        return result
    
if __name__ == "__main__":
    ray.init()
    def policy_mapping_fn(agent_id, *args, **kwargs):
        # agent_id is either "player" or "asteroid"
        return f"{agent_id}_policy"

    # Create a base PPOConfig:
    config = PPOConfig()

    # 1) Environment + Env Config
    config = config.environment(
        env=AsteroidsRLLibEnv,
        env_config={"render_mode": True},  # or True to see the window
    )
    # 2) Framework
    config = config.framework("torch")

    # 3) Resources (GPUs etc.)
    config = config.resources(num_gpus=1)

    # 4) Multi-agent setup
    env_example = AsteroidsRLLibEnv({"render_mode": True})
    config = config.multi_agent(
        policies={
            "player_policy": (
                None,  # default policy class
                env_example.observation_space_player,
                env_example.action_space_player,
                {},
            ),
            "asteroid_policy": (
                None,
                env_example.observation_space_asteroid,
                env_example.action_space_asteroid,
                {},
            ),
        },
        policy_mapping_fn=policy_mapping_fn,
        policies_to_train=["player_policy", "asteroid_policy"],
    )
    config["callbacks"] = SelectiveTrainingCallback


    # 5) RLModule configuration for your model architecture
    #    (the new place to put 'fcnet_hiddens', CNN sizes, etc.)
    config = config.rl_module(
        model_config={
            "fcnet_hiddens": [
                64,
                64,
            ],  # previously under model={"fcnet_hiddens": [...]}
            # Additional new-API fields as needed (e.g. activation_fn, etc.)
        }
    )

    # 6) Training hyperparameters
    #    (still set gamma, lr, etc. via .training)
    config = config.training(
        gamma=0.99,
        lr=1e-3,
        entropy_coeff=0.01
    )
    # 7) Rollout/worker config. The new API uses direct fields:
    #    Typically: config.num_rollout_workers, not config.num_env_runners
    config.num_env_runners = 1
    # config.num_envs_per_worker = 1
    # config.create_env_on_local_worker = True
    # etc.

    # Now run with Ray Tune’s Tuner
    tuner = tune.Tuner(
        "PPO",
    run_config=tune.RunConfig(
    stop={"training_iteration": 300}  # Stops after 300 training iterations
    ),
        param_space=config.to_dict(),
    )
    results = tuner.fit()
    print("Training completed!")
    latest_result = results.get_best_result()
    checkpoint_dir = latest_result.checkpoint.path
    trained_algo = PPO.from_checkpoint(checkpoint_dir)
    save_path = os.path.join(checkpoint_dir, "trained_model")
    trained_algo.save(save_path)
    print(f"Model saved at: {save_path}")
    # # Assuming 'results' is the Tune ExperimentAnalysis object returned by tuner.fit()
    # df = results.get_dataframe()

    # # Inspect column names to see what metrics were logged.
    # print(df.columns)

    # # Plot overall mean episode reward
    # plt.figure(figsize=(10,6))
    # plt.plot(df['episode_reward_mean'], label='Mean Episode Reward')

    # # If your multi-agent metrics are logged separately, you might have columns like:
    # # 'policy_reward_player_mean', 'policy_reward_asteroid_mean'
    # if 'policy_reward_player_mean' in df.columns:
    #     plt.plot(df['policy_reward_player_mean'], label='Player Mean Reward')
    # if 'policy_reward_asteroid_mean' in df.columns:
    #     plt.plot(df['policy_reward_asteroid_mean'], label='Asteroid Mean Reward')

    # plt.xlabel('Iteration')
    # plt.ylabel('Reward')
    # plt.title('Training Progress')
    # plt.legend()
    # plt.show()
