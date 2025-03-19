
import sys
sys.path.insert(0, 'game/original')
sys.path.insert(0, 'agents')

from a_star import AStarAgent
from constants import *

from asteroidfield import AsteroidField
from asteroid import Asteroid
from shot import Shot
from player import Player

from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import random
import math


class AsteroidsPCGEnvWithAStar(gym.Env):
    """
    An Asteroids environment where:
    - The RL agent spawns asteroids (PCG).
        -velocity, frequency, and size of the asteroids
    - The RL agent spawns powerups (PCG).
        -spawn rate, frequency, what kinds of powerups
    - The A* agent controls the player ship (movement + combat).
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        render_mode=None,
        max_steps=600,
        spawn_limit=3,
        replan_interval=0.5,
    ):
        """
        :param render_mode: None or 'human'
        :param max_steps: max steps per episode
        :param spawn_limit: max number of asteroids the RL can spawn each step
        :param replan_interval: how often the AStarAgent replans
        """
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.spawn_limit = spawn_limit

        # Pygame init
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(
                (SCREEN_WIDTH, SCREEN_HEIGHT))
        else:
            pygame.display.init()
            self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        # Sprite groups
        self.updatables = pygame.sprite.Group()
        self.drawables = pygame.sprite.Group()
        self.asteroids = pygame.sprite.Group()
        self.shots = pygame.sprite.Group()

        # Link containers
        Asteroid.containers = (self.asteroids, self.updatables, self.drawables)
        Shot.containers = (self.shots, self.updatables, self.drawables)
        AsteroidField.containers = self.updatables

        # A* agent (the "surrogate player" logic)
        self.astar_agent = AStarAgent(
            grid_size=(64, 36),
            safe_distance=50,
            replan_interval=replan_interval,
            shoot_distance=300,
            shoot_angle_thresh=15
        )

        self.asteroid_field = AsteroidField()

        # Create the player
        Player.containers = (self.updatables, self.drawables)
        self.player = Player(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)

        # Env state
        self.score = 0
        self.steps_elapsed = 0
        self.game_over = False

        # RL action space: integer from 0..spawn_limit
        # => how many asteroids to spawn this step
        self.action_space = spaces.Discrete(self.spawn_limit + 1)

        # RL observation space:
        # e.g. [num_asteroids, player's x, player's y, ???]
        # actions up down, left right, shoot, 
        low = np.array([0, 0, 0], dtype=np.float32)
        high = np.array([100, SCREEN_WIDTH, SCREEN_HEIGHT], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        # acitons for RL agent would be
        #spawning asteroids, spawning powerups, enemy?????
        #spawning RL created effects, possibly

    def _get_obs(self):
        """
        Example observation: number of asteroids, player's position.
        Extend as needed (player health, velocities, etc.).
        - health, velocity, position, bullets shot at given state, # of powerups collected
        """
        num_asts = len(self.asteroids)
        px, py = self.player.position.x, self.player.position.y
        obs = np.array([num_asts, px, py], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        random.seed(seed)
        np.random.seed(seed if seed is not None else None)

        # Clear sprite groups
        self.updatables.empty()
        self.drawables.empty()
        self.asteroids.empty()
        self.shots.empty()

        # Create new
        self.asteroid_field = AsteroidField()
        Player.containers = (self.updatables, self.drawables)
        Shot.containers = (self.shots, self.updatables, self.drawables)
        Asteroid.containers = (self.asteroids, self.updatables, self.drawables)
        self.player = Player(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)

        # A* agent fresh start
        self.astar_agent.current_path = []
        self.astar_agent.time_since_replan = 0.0

        self.score = 0
        self.steps_elapsed = 0
        self.game_over = False

        return self._get_obs(), {}

    def step(self, action):
        """
        RL agent's action => how many new asteroids to spawn.
        Then we let the A* agent control the player for 1 step of the simulation.
        """
        # 1) Spawn asteroids based on the RL action
        spawn_count = int(action)  # 0..spawn_limit
        for _ in range(spawn_count):
            self._spawn_asteroid()

        # 2) Let the A* agent update (which controls the player for one tick).
        dt = 1.0 / 60.0
        self._update_game(dt)

        # 3) Compute reward
        reward = self._compute_reward()

        # 4) Build observation
        obs = self._get_obs()

        # 5) Check if done
        self.steps_elapsed += 1
        terminated = self.game_over
        truncated = (self.steps_elapsed >= self.max_steps)

        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            self.screen.fill((0, 0, 0))
            for d in self.drawables:
                d.draw(self.screen)
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        pygame.quit()

    # ----------------------------------------------------------------
    # Internal Helpers
    # ----------------------------------------------------------------
    def _update_game(self, dt):
        """
        One step of game update:
        - A* agent decides how to rotate/move/shoot
        - We step all sprites, handle collisions, etc.
        """
        # Minimal event loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        # A* controls the player
        self.astar_agent.update(dt, self.player, self.asteroids)

        # Now update all sprites (position, collisions, etc.)
        for upd in self.updatables:
            upd.update(dt)

        self._handle_collisions()
        self._wrap_sprites()

    def _handle_collisions(self):
        for asteroid in self.asteroids:
            # Player collision
            if asteroid.collision_check(self.player):
                # Player destroyed
                self.game_over = True
            # Shots
            for shot in self.shots:
                if asteroid.collision_check(shot):
                    asteroid.split()
                    shot.kill()
                    self.score += 1

    def _wrap_sprites(self):
        """
        Classic Asteroids wrapping
        """
        for sprite in self.updatables:
            if hasattr(sprite, 'position'):
                if sprite.position.x < 0:
                    sprite.position.x = SCREEN_WIDTH
                elif sprite.position.x > SCREEN_WIDTH:
                    sprite.position.x = 0
                if sprite.position.y < 0:
                    sprite.position.y = SCREEN_HEIGHT
                elif sprite.position.y > SCREEN_HEIGHT:
                    sprite.position.y = 0

    def _spawn_asteroid(self):
        """
        Spawn an asteroid in a random edge location with random velocity.
        Similar to AsteroidField logic, but done manually for PCG control.
        """
        edges = AsteroidField.edges
        edge = random.choice(edges)
        speed = random.randint(40, 100)
        velocity = edge[0] * speed
        velocity = velocity.rotate(random.randint(-30, 30))
        position = edge[1](random.uniform(0, 1))

        radius = random.randint(ASTEROID_MIN_RADIUS, ASTEROID_MAX_RADIUS)
        asteroid = Asteroid(position.x, position.y, radius)
        asteroid.velocity = velocity

    def _compute_reward(self):
        """
        Simple reward function:
        + small reward for each asteroid destroyed (reflected in self.score)
        - penalty if player dies, including losing lives
        - small negative per step to limit spamming or drawn-out play
        - small reward for powerups
        - large reward for near miss, for dodging
        """
        reward = 0.0

        # Each time we come here, the total self.score might have increased by some hits.
        # We can do a simple scaling:
        reward += 0.1 * self.score

        if self.game_over:
            reward -= 5.0

        # small negative step cost
        reward -= 0.01

        return reward
