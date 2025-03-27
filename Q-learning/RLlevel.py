import numpy as np
from collections import deque
import random
import csv
import os
import matplotlib.pyplot as plt

from constants import ASTEROID_KINDS, ASTEROID_SPAWN_RATE


class RLDifficultyManager:
    def __init__(self):
        self.level = 1
        self.max_level = 20
        self.time_since_last_action = 0
        self.episode_length = 0

        self.engagement_score = 0.5  # <- important: needs to come early

        self.difficulty_params = {
            "spawn_rate": {"min": 0.2, "max": 1.5, "current": ASTEROID_SPAWN_RATE},
            "asteroid_speed_min": {"min": 40, "max": 200, "current": 40},
            "asteroid_speed_max": {"min": 100, "max": 350, "current": 100},
            "asteroid_kinds": {"min": ASTEROID_KINDS, "max": 5, "current": ASTEROID_KINDS}
        }

        self.survival_time = []
        self.hit_ratio = []
        self.score_rate = []
        self.near_miss_count = 0

        self.state = self._get_state()
        self.previous_state = self.state
        self.action_space = [
            "increase_spawn_rate", "decrease_spawn_rate",
            "increase_speed", "decrease_speed",
            "increase_kinds", "decrease_kinds",
            "no_change"
        ]

        self.q_table = {}
        self.learning_rate = 0.1
        self.learning_rate = max(0.1, self.learning_rate * 0.99)
        self.discount_factor = 0.95
        self.exploration_rate = 1.0
        self.exploration_decay = 0.995
        self.min_exploration_rate = 0.01

        self.state_history = deque(maxlen=10)
        self.training_log = []
        self.verbose = False

        for state in self._get_all_possible_states():
            self.q_table[state] = {action: 0 for action in self.action_space}

    def _get_all_possible_states(self):
        player_skills = ["low", "medium", "high"]
        difficulties = ["very_easy", "easy", "medium", "hard", "very_hard"]
        engagements = ["low", "medium", "high"]
        return [f"{s}_{d}_{e}" for s in player_skills for d in difficulties for e in engagements]

    def _get_state(self):
        if len(self.survival_time) < 2:
            player_skill = "medium"
        else:
            avg_survival = np.mean(self.survival_time[-3:])
            if avg_survival < 30:
                player_skill = "low"
            elif avg_survival < 90:
                player_skill = "medium"
            else:
                player_skill = "high"

        spawn_rate_norm = (self.difficulty_params["spawn_rate"]["current"] - self.difficulty_params["spawn_rate"]["min"]) / (self.difficulty_params["spawn_rate"]["max"] - self.difficulty_params["spawn_rate"]["min"])
        speed_norm = (self.difficulty_params["asteroid_speed_max"]["current"] - self.difficulty_params["asteroid_speed_max"]["min"]) / (self.difficulty_params["asteroid_speed_max"]["max"] - self.difficulty_params["asteroid_speed_max"]["min"])
        difficulty_value = (spawn_rate_norm + speed_norm) / 2

        if difficulty_value < 0.2:
            difficulty = "very_easy"
        elif difficulty_value < 0.4:
            difficulty = "easy"
        elif difficulty_value < 0.6:
            difficulty = "medium"
        elif difficulty_value < 0.8:
            difficulty = "hard"
        else:
            difficulty = "very_hard"

        if self.engagement_score < 0.3:
            engagement = "low"
        elif self.engagement_score < 0.7:
            engagement = "medium"
        else:
            engagement = "high"

        return f"{player_skill}_{difficulty}_{engagement}"

    def update(self, dt, player_data):
        self.time_since_last_action += dt
        self.episode_length += dt

        if player_data.get("near_misses", 0) > 0:
            self.near_miss_count += player_data["near_misses"]

        self._update_engagement_score(player_data)

        if self.time_since_last_action >= 5.0:
            self.time_since_last_action = 0
            self._take_action()
            self.previous_state = self.state
            self.state = self._get_state()
            self.state_history.append((self.previous_state, self.state))

        if not player_data.get("player_alive", True) and self.previous_state != self.state:
            self._end_episode(player_data)
            return True
        return False

    def _update_engagement_score(self, player_data):
        self.engagement_score *= 0.99
        self.engagement_score += 0.05 * player_data.get("near_misses", 0)
        if player_data.get("shots_fired", 0) > 0:
            self.engagement_score += 0.02
        self.engagement_score += 0.03 * player_data.get("shots_hit", 0)
        self.engagement_score = max(0, min(1, self.engagement_score))

    def _take_action(self):
        if random.random() < self.exploration_rate:
            action = random.choice(self.action_space)
        else:
            if self.state not in self.q_table:
                self.q_table[self.state] = {action: 0 for action in self.action_space}
            max_q = max(self.q_table[self.state].values())
            best_actions = [a for a, q in self.q_table[self.state].items() if q == max_q]
            action = random.choice(best_actions)

        self._execute_action(action)
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)

        if self.verbose:
            print(f"[RL] Action taken: {action}, State: {self.state}, epsilon: {self.exploration_rate:.3f}")

    def _execute_action(self, action):
        p = self.difficulty_params
        if action == "increase_spawn_rate":
            p["spawn_rate"]["current"] = max(p["spawn_rate"]["current"] * 0.9, p["spawn_rate"]["min"])
        elif action == "decrease_spawn_rate":
            p["spawn_rate"]["current"] = min(p["spawn_rate"]["current"] * 1.1, p["spawn_rate"]["max"])
        elif action == "increase_speed":
            p["asteroid_speed_min"]["current"] = min(p["asteroid_speed_min"]["current"] + 5, p["asteroid_speed_min"]["max"])
            p["asteroid_speed_max"]["current"] = min(p["asteroid_speed_max"]["current"] + 10, p["asteroid_speed_max"]["max"])
        elif action == "decrease_speed":
            p["asteroid_speed_min"]["current"] = max(p["asteroid_speed_min"]["current"] - 5, p["asteroid_speed_min"]["min"])
            p["asteroid_speed_max"]["current"] = max(p["asteroid_speed_max"]["current"] - 10, p["asteroid_speed_max"]["min"])
        elif action == "increase_kinds":
            p["asteroid_kinds"]["current"] = min(p["asteroid_kinds"]["current"] + 1, p["asteroid_kinds"]["max"])
        elif action == "decrease_kinds":
            p["asteroid_kinds"]["current"] = max(p["asteroid_kinds"]["current"] - 1, p["asteroid_kinds"]["min"])

    def _end_episode(self, player_data):
        self.survival_time.append(self.episode_length)
        score_rate = player_data.get("score", 0) / max(1, self.episode_length)
        self.score_rate.append(score_rate)

        for prev_state, next_state in self.state_history:
            reward = self._calculate_reward(prev_state, next_state, player_data)
            self._update_q_value(prev_state, next_state, reward)

        self.training_log.append({
            "episode": len(self.survival_time),
            "survival_time": self.episode_length,
            "engagement": self.engagement_score,
            "score": player_data.get("score", 0),
            "exploration_rate": self.exploration_rate
        })

        self.episode_length = 0
        self.near_miss_count = 0
        self.state_history.clear()

        if len(self.survival_time) % 3 == 0 and self.level < self.max_level:
            self.level += 1

    def _calculate_reward(self, prev_state, next_state, player_data):
        prev_eng = prev_state.split('_')[2]
        next_eng = next_state.split('_')[2]

        engagement_reward = 1.0 if next_eng == "high" else 0.5 if next_eng == "medium" and prev_eng == "low" else -0.5
        survival_reward = -0.5 if self.episode_length < 15 else min(1.0, self.episode_length / 120.0)
        return engagement_reward + survival_reward

    def _update_q_value(self, prev_state, curr_state, reward):
        if prev_state not in self.q_table:
            self.q_table[prev_state] = {a: 0 for a in self.action_space}
        if curr_state not in self.q_table:
            self.q_table[curr_state] = {a: 0 for a in self.action_space}

        best_action = max(self.q_table[curr_state], key=self.q_table[curr_state].get)
        td_target = reward + self.discount_factor * self.q_table[curr_state][best_action]

        for action in self.action_space:
            current_q = self.q_table[prev_state][action]
            self.q_table[prev_state][action] += self.learning_rate * (td_target - current_q)

    def save_model(self, filename):
        import json
        with open(filename, 'w') as f:
            json.dump(self.q_table, f)

    def load_model(self, filename):
        import json
        try:
            with open(filename, 'r') as f:
                self.q_table = json.load(f)
        except FileNotFoundError:
            print(f"Model file {filename} not found. Starting with a new model.")

    def save_training_log(self, filename="training_log.csv"):
        keys = ["episode", "survival_time", "engagement", "score", "exploration_rate"]
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.training_log)
        if self.verbose:
            print(f"[RL] Training log saved to {filename}")

    def plot_training_progress(self, save_path=None):
        if not self.training_log:
            print("[RL] No training data to plot.")
            return

        episodes = [d["episode"] for d in self.training_log]
        engagement = [d["engagement"] for d in self.training_log]
        survival = [d["survival_time"] for d in self.training_log]
        score = [d["score"] for d in self.training_log]
        exploration = [d["exploration_rate"] for d in self.training_log]

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(episodes, engagement)
        plt.title("Engagement Over Time")

        plt.subplot(2, 2, 2)
        plt.plot(episodes, survival, color='orange')
        plt.title("Survival Time")

        plt.subplot(2, 2, 3)
        plt.plot(episodes, score, color='green')
        plt.title("Score Per Episode")

        plt.subplot(2, 2, 4)
        plt.plot(episodes, exploration, color='purple')
        plt.title("Exploration Rate")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"[RL] Training plot saved to {save_path}")
        else:
            plt.show()

    def get_spawn_rate(self):
        return self.difficulty_params["spawn_rate"]["current"]

    def get_speed_range(self):
        return (
            self.difficulty_params["asteroid_speed_min"]["current"],
            self.difficulty_params["asteroid_speed_max"]["current"]
        )

    def get_asteroid_kinds(self):
        return self.difficulty_params["asteroid_kinds"]["current"]

    def get_engagement_score(self):
        return self.engagement_score

    def get_exploration_rate(self):
        return self.exploration_rate
