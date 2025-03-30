import sys

sys.path.insert(0, "player_agents")
sys.path.insert(0, "asteroid_agents")

from human import HumanPlayerAgent
from random_asteroids import RandomAsteroidAgent
import random
import pygame
import pygame.freetype
from constants import *

from ray.rllib.env import MultiAgentEnv
import gymnasium as gym

from gymnasium.spaces import Box, Discrete, Dict
import math
import numpy as np


class CircleShape(pygame.sprite.Sprite):
    def __init__(self, x, y, radius):
        if hasattr(self, "containers"):
            super().__init__(self.containers)
        else:
            super().__init__()
        self.position = pygame.Vector2(x, y)
        self.velocity = pygame.Vector2(0, 0)
        self.radius = radius

    def draw(self, screen):
        # Should be overridden by subclass
        pass

    def update(self, dt):
        # Should be overridden by subclass
        pass

    def collision_check(self, other):
        r1 = self.radius
        r2 = other.radius
        if self.position.distance_to(other.position) <= r1 + r2:
            return True
        return False

class Asteroid(CircleShape):
    def __init__(self, x, y, radius):
        super().__init__(x, y, radius)

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 255, 255), self.position, self.radius, 2)

    def update(self, dt):
        self.position += self.velocity * dt

    def split(self):
        """
        Splits an asteroid into two smaller pieces (if large enough)
        at some random angle offsets.
        """
        self.kill()
        if self.radius <= ASTEROID_MIN_RADIUS:
            return

        random_degrees = random.uniform(20, 50)
        vector1 = pygame.math.Vector2.rotate(self.velocity, random_degrees)
        vector2 = pygame.math.Vector2.rotate(self.velocity, -random_degrees)

        new_radius = self.radius - ASTEROID_MIN_RADIUS

        asteroid1 = Asteroid(self.position.x, self.position.y, new_radius)
        asteroid2 = Asteroid(self.position.x, self.position.y, new_radius)

        asteroid1.velocity = vector1 * 1.5
        asteroid2.velocity = vector2 * 1.5


class Shot(CircleShape):
    def __init__(self, x, y, radius):
        super().__init__(x, y, radius)

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 255, 255), self.position, self.radius, 2)

    def update(self, dt):
        self.position += self.velocity * dt


class Player(CircleShape):
    def __init__(self, x, y):
        super().__init__(x, y, PLAYER_RADIUS)
        self.rotation = 0
        self.timer = 0

    def triangle(self):
        """
        Renders the ship as a triangle, but
        for collision detection we still use the circle radius.
        """
        forward = pygame.Vector2(0, 1).rotate(self.rotation)
        right = pygame.Vector2(0, 1).rotate(self.rotation + 90) * self.radius / 1.5
        a = self.position + forward * self.radius
        b = self.position - forward * self.radius - right
        c = self.position - forward * self.radius + right
        return [a, b, c]

    def draw(self, screen):
        pygame.draw.polygon(screen, (255, 255, 255), self.triangle(), 2)

    def update(self, dt):
        """
        By default, we do nothing here. If using HumanPlayerAgent,
        we'll check keys in a separate step method. If using a policy agent,
        that agent will explicitly set commands for rotate/move.
        """
        # If we want to handle the default "human" logic inside the environment,
        # we’ll do it there. This function can remain an empty stub if the env
        # calls movement/rotation directly.
        pass

    def move(self, dt, forward_factor=1.0):
        forward = pygame.Vector2(0, 1).rotate(self.rotation)
        self.position += forward * PLAYER_SPEED * dt * forward_factor

    def rotate(self, dt, direction=1.0):
        self.rotation += PLAYER_TURN_SPEED * dt * direction

    def shoot(self):
        shot = Shot(self.position.x, self.position.y, SHOT_RADIUS)
        shot.velocity = pygame.Vector2(0, 1).rotate(self.rotation) * PLAYER_SHOOT_SPEED
        shot.ttl = 2.0  # time-to-live for shot
        return shot

class AsteroidsRLLibEnv(MultiAgentEnv):
    """
    RLlib-compatible multi-agent environment.
    Agents:
      "player" -> discrete action ∈ [0..17]
      "asteroid" -> discrete action ∈ [0..108]
    Observations:
      - For the player: e.g. (playerX, playerY, num_asteroids)
      - For the asteroid: e.g. (num_asteroids, playerX, playerY)

    Each step => returns obs, rew, done, info for both agents.
    """

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = {}

        # Rendering
        self.render_mode = config.get("render_mode", False)
        pygame.init()
        self.clock = pygame.time.Clock()
        if self.render_mode:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        else:
            self.screen = None
        self.font = pygame.freetype.SysFont(None, 24)

        # State
        self.player = None
        self.asteroids = pygame.sprite.Group()
        self.shots = pygame.sprite.Group()
        self.updatables = pygame.sprite.Group()

        self.score = 0
        self.game_over = False
        self.steps_elapsed = 0

        self.near_miss_count = 0
        self.last_spawns = []
        self.dt = 0.016

        self.max_steps = MAX_STEPS  # you can override in config

        # RLlib requires specifying the spaces
        # (could be gym.spaces.Box, Discrete, etc.)
        self.observation_space_player = Box(
            low=-1e5, high=1e5, shape=(3,), dtype=np.float32
        )
        self.observation_space_asteroid = Box(
            low=-1e5, high=1e5, shape=(3,), dtype=np.float32
        )

        self.action_space_player = Discrete(PLAYER_ACTION_SIZE)  # 18
        self.action_space_asteroid = Discrete(ASTEROID_ACTION_SIZE)  # 109
        
        self.possible_agents = ["player", "asteroid"]
        self.agents = self.possible_agents.copy()

    def reset(self, seed=None, options=None):
        # Clear old state
        self.asteroids.empty()
        self.shots.empty()
        self.updatables.empty()

        # Create player at center
        self.player = Player(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)
        self.updatables.add(self.player)

        self.score = 0
        self.game_over = False
        self.steps_elapsed = 0
        self.near_miss_count = 0
        self.last_spawns.clear()

        # Return the initial observation and an empty info dict
        obs = {"player": self._get_player_obs(), "asteroid": self._get_asteroid_obs()}
        return obs, {}

    def step(self, action_dict):
        """
        action_dict => {"player": <int>, "asteroid": <int>}
        Returns:
          obs_dict, rew_dict, terminated_dict, truncated_dict, info_dict
        """
        # Process any quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return (
                    {},
                    {"player": 0.0, "asteroid": 0.0},
                    {"__all__": True},
                    {"__all__": False},
                    {},
                )

        if not self.game_over:
            self.dt = self.clock.tick(60) / 1000.0
            self.steps_elapsed += 1
        else:
            self.dt = 0

        # 1) Apply the player action
        player_act = action_dict.get("player", 0)
        if not self.game_over:
            self._apply_player_action(player_act)

        # 2) Apply the asteroid agent action
        asteroid_act = action_dict.get("asteroid", 0)
        if not self.game_over:
            self._apply_asteroid_action(asteroid_act)

        # 3) Update sprites
        for s in list(self.updatables):
            if hasattr(s, "update"):
                s.update(self.dt)
            if isinstance(s, Shot):
                s.position += s.velocity * self.dt
                if s.ttl <= 0:
                    s.kill()
            elif isinstance(s, Player):
                pass  # Movement already handled
            elif isinstance(s, Asteroid):
                s.position += s.velocity * self.dt

        # 4) Handle collisions
        destroyed = 0
        for asteroid in list(self.asteroids):
            if asteroid.collision_check(self.player):
                self.game_over = True
            for shot in list(self.shots):
                if asteroid.collision_check(shot):
                    shot.kill()
                    asteroid.kill()
                    destroyed += 1
                    break

        self.score += destroyed

        # 5) Count near misses
        self.near_miss_count = 0
        for asteroid in self.asteroids:
            dist = asteroid.position.distance_to(self.player.position)
            if dist < NEAR_MISS_DISTANCE and dist > (asteroid.radius + self.player.radius):
                self.near_miss_count += 1

        # 6) Check time limit
        if self.steps_elapsed >= self.max_steps:
            self.game_over = True

        # 7) Compute rewards
        player_reward = float(destroyed)
        asteroid_reward = self._compute_fun_reward()

        # 8) Set termination flags (using new Gymnasium API style)
        terminated = {"player": self.game_over, "asteroid": self.game_over, "__all__": self.game_over}
        truncated = {"player": False, "asteroid": False, "__all__": False}

        # 9) Build next observation and info dicts
        obs_dict = {"player": self._get_player_obs(), "asteroid": self._get_asteroid_obs()}
        rew_dict = {"player": player_reward, "asteroid": asteroid_reward}
        info_dict = {}

        return obs_dict, rew_dict, terminated, truncated, info_dict

    def render(self, mode="human"):
        if not self.render_mode or not self.screen:
            return
        self.screen.fill((0, 0, 0))
        # draw any sprites, text, etc.
        self.font.render_to(
            self.screen, (10, 10), f"Score: {self.score}", (255, 255, 255)
        )
        pygame.display.flip()

    def close(self):
        pygame.quit()

    # ----------------------------
    #   Koster-inspired "fun" reward
    # ----------------------------
    def _compute_fun_reward(self):
        reward = 0.0
        # 1) mastery
        reward += 0.1 * self.score
        # 2) near misses
        reward += 0.05 * self.near_miss_count
        # 3) variety
        unique_spawns = len(set(self.last_spawns))
        diversity_factor = unique_spawns / max(1, len(self.last_spawns))
        reward += 0.1 * diversity_factor
        # 4) death penalty
        if self.game_over:
            early_factor = 1.0 - (self.steps_elapsed / self.max_steps)
            reward -= 5.0 + 5.0 * early_factor
        # boredom
        if len(self.asteroids) < 1:
            reward -= 0.05
        # penalty if too many
        if len(self.asteroids) > MAX_ASTEROIDS_ONSCREEN:
            reward -= 0.1 * (len(self.asteroids) - MAX_ASTEROIDS_ONSCREEN)
        # step cost
        reward -= 0.01
        return reward

    # ----------------------------
    #   Action decoders
    # ----------------------------
    def _apply_player_action(self, a):
        """
        18 discrete combos => 3 (turn) x 3 (forward) x 2 (shoot)
           turn = -1, 0, +1
           forward = -1, 0, +1
           shoot = 0 or 1
        """
        turn_idx = a // 6  # 0..2
        rem = a % 6
        f_idx = rem // 2  # 0..2
        shoot_idx = rem % 2

        turn_val = {0: -1, 1: 0, 2: +1}[turn_idx]
        forward_val = {0: -1, 1: 0, 2: +1}[f_idx]
        shoot = shoot_idx == 1

        self.player.rotate(self.dt, turn_val)
        self.player.move(self.dt, forward_val)
        if shoot:
            new_shot = self.player.shoot()
            if new_shot:
                self.shots.add(new_shot)
                self.updatables.add(new_shot)

    def _apply_asteroid_action(self, a):
        """
        0 => no spawn
        1..108 => spawn with combo of (edge, radius, speed, angle).
        """
        if a == 0:
            return

        a -= 1  # now 0..107
        edge_idx = a // 27  # 0..3
        r = a % 27
        radius_idx = r // 9  # 0..2
        s = r % 9
        speed_idx = s // 3  # 0..2
        angle_idx = s % 3  # 0..2

        edge = ASTEROID_EDGES[edge_idx]
        radius = POSSIBLE_RADII[radius_idx]
        speed = POSSIBLE_SPEEDS[speed_idx]
        angle = POSSIBLE_ANGLES[angle_idx]

        frac = random.random()
        pos = edge[1](frac)
        direction = edge[0].rotate(angle)
        vel = direction * speed

        # spawn asteroid
        asteroid = Asteroid(pos.x, pos.y, radius)
        asteroid.velocity = vel

        self.asteroids.add(asteroid)
        self.updatables.add(asteroid)

        self.last_spawns.append((radius, speed, angle))
        if len(self.last_spawns) > 20:
            self.last_spawns.pop(0)


    # ----------------------------
    #   Observations
    # ----------------------------
    def _get_player_obs(self):
        """
        Return a 3D vector: [player_x, player_y, #asteroids].
        """
        return np.array(
            [self.player.position.x, self.player.position.y, len(self.asteroids)],
            dtype=np.float32,
        )

    def _get_asteroid_obs(self):
        """
        Return a 3D vector: [#asteroids, player_x, player_y].
        """
        return np.array(
            [len(self.asteroids), self.player.position.x, self.player.position.y],
            dtype=np.float32,
        )

    @property
    def observation_space(self):
        return Dict(
            {
                "player": self.observation_space_player,
                "asteroid": self.observation_space_asteroid,
            }
        )

    @property
    def action_space(self):
        return Dict(
            {"player": self.action_space_player, "asteroid": self.action_space_asteroid}
        )
