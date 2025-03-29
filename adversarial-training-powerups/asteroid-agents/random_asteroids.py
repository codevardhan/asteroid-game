import sys

sys.path.insert(0, "..")
from constants import *
import pygame
import random


class RandomAsteroidAgent:
    """
    Default adversarial agent controlling asteroid spawns.
    Each step, we can either spawn an asteroid or do nothing or spawn powerup

    The environment will interpret these actions.
    """

    def __init__(self, spawn_prob=0.5):
        """
        spawn_prob: Probability of spawning an asteroid at each step.
        """
        self.spawn_prob = spawn_prob
        self.spawn_timer = 0.0

    def get_action(self, obs):
        """
        Return a dict describing if/where to spawn an asteroid.
        For example:
          {
            'spawn': bool,
            'radius': <int>,
            'position': pygame.Vector2,
            'velocity': pygame.Vector2
          }
        """
        dt = obs.get("dt", 0.016)
        self.spawn_timer += dt

        # Enforce a spawn_rate limit if we want the "once every 0.8s" style.
        # If you'd like to spawn on every step or vary it, you can omit this.
        do_spawn = False
        if self.spawn_timer > ASTEROID_SPAWN_RATE:
            self.spawn_timer = 0
            # Now randomly decide if we spawn or not:
            if random.random() < self.spawn_prob:
                do_spawn = True

        if not do_spawn:
            return {"spawn": False}

        # Choose a random edge, random velocity, etc.
        edges = [
            [
                pygame.Vector2(1, 0),
                lambda y: pygame.Vector2(-ASTEROID_MAX_RADIUS, y * SCREEN_HEIGHT),
            ],
            [
                pygame.Vector2(-1, 0),
                lambda y: pygame.Vector2(
                    SCREEN_WIDTH + ASTEROID_MAX_RADIUS, y * SCREEN_HEIGHT
                ),
            ],
            [
                pygame.Vector2(0, 1),
                lambda x: pygame.Vector2(x * SCREEN_WIDTH, -ASTEROID_MAX_RADIUS),
            ],
            [
                pygame.Vector2(0, -1),
                lambda x: pygame.Vector2(
                    x * SCREEN_WIDTH, SCREEN_HEIGHT + ASTEROID_MAX_RADIUS
                ),
            ],
        ]

        edge = random.choice(edges)
        speed = random.randint(40, 100)
        velocity = edge[0] * speed
        velocity = velocity.rotate(random.randint(-30, 30))
        position = edge[1](random.uniform(0, 1))
        kind = random.randint(1, ASTEROID_KINDS)
        radius = ASTEROID_MIN_RADIUS * kind

        return {
            "spawn": True,
            "radius": radius,
            "position": position,
            "velocity": velocity,
        }
