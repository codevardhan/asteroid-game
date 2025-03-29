import sys

sys.path.insert(0, "asteroid-random")
sys.path.insert(0, "agents")

from a_star import AStarAgent
from constants import *

from asteroidfield import AsteroidField
from asteroid import Asteroid
from shot import Shot
from player import Player
from powerups import PowerUp
from powerup_manager import PowerUpManager
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import random
import math


class AsteroidsPCGEnvKoster(gym.Env):
    """
    An Asteroids environment:
    - RL agent spawns asteroids (PCG).
    - A* agent flies/attacks as surrogate player.
    - Reward shaped to encourage 'fun' per Koster's principles.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        agent=None,
        render_mode=None,
        max_steps=600,
        spawn_limit=3,
        replan_interval=0.5,
        near_miss_radius=40.0,
        diversity_window=20,
    ):
        """
        :param render_mode: 'human' or None
        :param max_steps: max steps per episode
        :param spawn_limit: max number of asteroids to spawn each RL step
        :param replan_interval: how often the A* agent replans
        :param near_miss_radius: distance threshold for awarding 'near-miss' events
        :param diversity_window: how many recent actions to track for 'variety' reward
        """
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.spawn_limit = spawn_limit
        self.font = pygame.freetype.SysFont(None, 36)
        # "Near miss" threshold: if an asteroid passes within this distance of player
        # but does not collide, it counts as a near miss (challenge).
        self.near_miss_radius = near_miss_radius

        # We'll track variety in spawning. Keep last N actions.
        self.diversity_window = diversity_window
        self.last_spawns = []

        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        else:
            pygame.display.init()
            self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        self.updatables = pygame.sprite.Group()
        self.drawables = pygame.sprite.Group()
        self.asteroids = pygame.sprite.Group()
        self.shots = pygame.sprite.Group()
        self.powerups = pygame.sprite.Group()

        Asteroid.containers = (self.asteroids, self.updatables, self.drawables)
        Shot.containers = (self.shots, self.updatables, self.drawables)
        PowerUp.containers = (self.powerups,self.updatables,self.drawables)
        AsteroidField.containers = self.updatables
        PowerUpManager.containers = self.updatables

        # A* agent = surrogate player
        if agent is None:
            self.agent = AStarAgent(
                grid_size=(64, 36),
                safe_distance=50,
                replan_interval=replan_interval,
                shoot_distance=300,
                shoot_angle_thresh=15,
            )
        else:
            self.agent = agent

        self.asteroid_field = AsteroidField()
        self.powerup_manager = PowerUpManager()

        Player.containers = (self.updatables, self.drawables)
        self.player = Player(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)

        self.score = 0  # how many hits / splits
        self.steps_elapsed = 0
        self.game_over = False

        self.near_miss_count = 0

        # Gym action/observation space
        # Action: (count, radius, speed, angle) -
        #   - count in [0..spawn_limit]
        #   - radius in [ASTEROID_MIN_RADIUS, ASTEROID_MAX_RADIUS]
        #   - speed in [40..120], angle in [0..360)
        # Discrete radius, speed, angle.

        self.action_space = spaces.Discrete(3)

        # Observation space: [num_asteroids, player_x, player_y, near_miss_count]
        low = np.array([0, 0, 0, 0], dtype=np.float32)
        high = np.array([100, SCREEN_WIDTH, SCREEN_HEIGHT, 100], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.MAX_ASTEROIDS = 40

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        random.seed(seed)
        np.random.seed(seed if seed is not None else None)

        # Clear sprite groups
        self.updatables.empty()
        self.drawables.empty()
        self.asteroids.empty()
        self.shots.empty()
        self.powerups.empty()

        self.asteroid_field = AsteroidField()
        self.powerup_manager = PowerUpManager()
        Player.containers = (self.updatables, self.drawables)
        Shot.containers = (self.shots, self.updatables, self.drawables)
        Asteroid.containers = (self.asteroids, self.updatables, self.drawables)
        PowerUp.containers = (self.powerups,self.updatables,self.drawables)

        self.player = Player(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)

        # A* agent fresh start
        self.agent.current_path = []
        self.agent.time_since_replan = 0.0

        self.score = 0
        self.steps_elapsed = 0
        self.game_over = False
        self.last_spawns.clear()

        return self._get_obs(), {}

    def game_over_state(self):
        self.font.render_to(
            self.screen,
            (SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2),
            "GAME OVER",
            (255, 255, 255),
        )
        self.font.render_to(
            self.screen,
            (SCREEN_WIDTH // 2 - 140, SCREEN_HEIGHT // 2 + 50),
            "Press R to restart",
            (255, 255, 255),
        )
        self.player.kill()

    def step(self, action):
        """
        Action = (spawn_count, radius_idx, speed_idx, angle_idx)
        We'll spawn `spawn_count` asteroids, each with the given radius, speed, angle.
        """
        # Unpack action
        spawn_count, radius_idx, speed_idx, angle_idx = action

        available_spawns = max(0, self.MAX_ASTEROIDS - len(self.asteroids))
        spawn_count = min(spawn_count, available_spawns)

        # Convert discrete indices to actual in-game values
        spawn_radius = ASTEROID_MIN_RADIUS + radius_idx
        spawn_speed = 40 + speed_idx  # from 40..120 if speed_idx in [0..80]
        spawn_angle = angle_idx * 10  # 36 steps => 0..350 in increments of 10

        # Spawn asteroids
        for _ in range(spawn_count):
            self._spawn_asteroid(
                radius=spawn_radius, speed=spawn_speed, angle=spawn_angle
            )
            # track the spawns for diversity
            self.last_spawns.append((spawn_radius, spawn_speed, spawn_angle))

        if len(self.last_spawns) > self.diversity_window:
            self.last_spawns = self.last_spawns[-self.diversity_window :]

        # Run one step of the simulation
        dt = 1.0 / 60.0
        self.near_miss_count = 0  # reset near-miss counter for this step
        self._update_game(dt)

        # Reward
        reward = self._compute_reward()

        # Obs
        obs = self._get_obs()

        debug_info = {
            "score": self.score,
            "steps_elapsed": self.steps_elapsed,
            "num_asteroids": len(self.asteroids),
            "near_miss_count": self.near_miss_count,
            "last_spawns": self.last_spawns,
            "action": action,
            "spawn_params": {
                "radius": spawn_radius,
                "speed": spawn_speed,
                "angle": spawn_angle,
            },
        }

        # check if done?
        self.steps_elapsed += 1
        terminated = self.game_over
        truncated = self.steps_elapsed >= self.max_steps

        return obs, reward, terminated, truncated, debug_info

    def render(self, info):
        if self.render_mode == "human":
            self.screen.fill((0, 0, 0))
            for d in self.drawables:
                d.draw(self.screen)
            self.font.render_to(
                self.screen, (10, 10), f"Score: {info['score']}", (255, 255, 255)
            )

            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        pygame.quit()

    def _get_obs(self):
        """
        Observation includes near-miss count to inform the agent about 'challenge' level.
        Changing buckets to discretize state space
        """
        num_asts = len(self.asteroids)
        if num_asts < 10:
            num_asts_bucket = 0
        elif num_asts >= 10 and num_asts < 20:
            num_asts_bucket = 1
        elif num_asts >= 20 and num_asts < 40:
            num_asts_bucket = 2
        elif num_asts >= 40:
            num_asts_bucket = 3

        px, py = self.player.position.x, self.player.position.y

        num_pup = len(self.powerups)
        if num_pup < 3:
            num_pup_bucket = 0
        elif num_pup >=3 and num_pup < 5:
            num_pup_bucket = 1
        elif num_pup >=5:
            num_pup_bucket = 2

        #player lives bucket
        p_lives = self.player.player_lives
        if p_lives == 1:
            p_lives_bucket = 0
        elif p_lives > 2 and p_lives < 5:
            p_lives_bucket = 1
        else:
            p_lives_bucket = 2

        #bucket for near miss count
        if self.near_miss_count < 5:
            near_miss_bucket = 0  # Low risk
        elif self.near_miss_count < 15:
            near_miss_bucket = 1  # Medium risk
        else:
            near_miss_bucket = 2 

        obs = np.array([num_asts, px, py, self.near_miss_count], dtype=np.float32)
        return obs

    def _update_game(self, dt):
        # Minimal event loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        # A* agent update
        self.agent.update(dt, self.player, self.asteroids)

        # Update sprites
        for upd in self.updatables:
            upd.update(dt)

        self._handle_collisions()
        self._wrap_sprites()

    def _handle_collisions(self):
        # We'll also detect near misses: if asteroid passes near player but doesn't collide
        player_pos = self.player.position

        for asteroid in self.asteroids:
            dist = (asteroid.position - player_pos).length()
            if dist < asteroid.radius + self.near_miss_radius:
                # If it's truly colliding, game over
                if asteroid.collision_check(self.player):
                    self.game_over = True
                else:
                    # near-miss
                    self.near_miss_count += 1

            # Shots
            for shot in self.shots:
                if asteroid.collision_check(shot):
                    asteroid.split()
                    shot.kill()
                    self.score += 1

    def _wrap_sprites(self):
        for sprite in self.updatables:
            if hasattr(sprite, "position"):
                if sprite.position.x < 0:
                    sprite.position.x = SCREEN_WIDTH
                elif sprite.position.x > SCREEN_WIDTH:
                    sprite.position.x = 0
                if sprite.position.y < 0:
                    sprite.position.y = SCREEN_HEIGHT
                elif sprite.position.y > SCREEN_HEIGHT:
                    sprite.position.y = 0

    def _spawn_asteroid(self, radius, speed, angle):
        """
        Spawn an asteroid at a random edge location but with given radius, speed, and direction.
        """
        edges = AsteroidField.edges
        edge = random.choice(edges)
        position = edge[1](random.uniform(0, 1))

        # Velocity from angle + speed
        # in pygame angle 0 is right
        vx = speed * math.sin(math.radians(angle))
        vy = -speed * math.cos(math.radians(angle))
        velocity = pygame.Vector2(vx, vy)

        asteroid = Asteroid(position.x, position.y, radius)
        asteroid.velocity = velocity

    # ----------------------------------------------------------------
    # Koster-inspired reward design
    # ----------------------------------------------------------------
    def _compute_reward(self):
        """
        A "fun"-oriented reward function:
        1) + for destroying asteroids => sense of mastery
        2) + for near misses => challenge without death
        3) variety bonus => if the RL agent spawns different radius/speed/angles
        4) - if player dies quickly or if no asteroids appear (boredom)
        5) small step cost to encourage interesting pacing
        """
        reward = 0.0

        # Mastery: Reward from self.score (each destroyed asteroid)
        #    scale it small to avoid overshadowing other factors
        reward += 0.1 * self.score

        # Challenge: near misses
        #    more near_miss => more excitement
        reward += 0.05 * self.near_miss_count

        # Variety in recent spawns
        #    measure distinct (radius, speed, angle) combos in self.last_spawns
        unique_spawns = len(set(self.last_spawns))
        diversity_factor = unique_spawns / max(1, len(self.last_spawns))
        # diversity_factor => 1.0 if all spawns are distinct, near 0 if identical
        reward += 0.1 * diversity_factor

        # If the player is dead => big penalty
        if self.game_over:
            # bigger penalty if it happens too early => environment was too hard
            early_factor = 1.0 - (self.steps_elapsed / self.max_steps)
            reward -= 5.0 + 5.0 * early_factor

        # If no asteroids for too long => negative for boredom
        #    e.g., if we see < 1 asteroid on screen frequently, environment is dull
        if len(self.asteroids) < 1:
            reward -= 0.05

        # Penalty for too many asteroids
        if len(self.asteroids) > self.MAX_ASTEROIDS:
            reward -= 0.1 * (len(self.asteroids) - self.MAX_ASTEROIDS)

        # Small negative step cost
        reward -= 0.01

        return reward
