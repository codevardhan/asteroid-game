import sys

sys.path.insert(0, "..")
from constants import *

import pygame

class HumanPlayerAgent:
    """
    Default “agent” that uses the keyboard.
    The environment will query pygame events and pass them along if needed.
    """

    def get_action(self, obs):
        """
        obs could be any observation from the environment – we won’t use it
        here. Instead, we read from pygame's key states.

        Return a dict describing the player's intended controls:
          {
            'turn': +1 (turn right), -1 (turn left), 0 (no turn),
            'forward': 1 (move forward), -1 (back), 0 (none),
            'shoot': True/False
          }
        """
        keys = pygame.key.get_pressed()

        turn = 0
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            turn = -1
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            turn = +1

        forward = 0
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            forward = +1
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            forward = -1

        shoot = False
        if keys[pygame.K_SPACE]:
            shoot = True

        return {
            "turn": turn,
            "forward": forward,
            "shoot": shoot,
        }
