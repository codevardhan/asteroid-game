import random
from powerups import LifePowerUp, ShotPowerUp, SpeedPowerUp
import pygame
from constants import *

class PowerUpManager(pygame.sprite.Sprite):
    def __init__(self):
        self.powerup_timer = 0
        self.action = None
        self.powerups = pygame.sprite.Group()
    def update(self, dt):
        self.powerup_timer += dt
        self.action = None

    def spawn_from_asteroid(self,asteroid):
            powerup_type = random.choice(["speed", "shot", "life"])
            vector3 = pygame.math.Vector2.rotate(asteroid.velocity,random.uniform(20, 50))
            if powerup_type == "shot":
                powerup = ShotPowerUp(asteroid.position.x,asteroid.position.y,2)
                powerup.velocity = vector3
                self.action = "PowerUp_Spawned_Shot"
            elif powerup_type == "speed":
                powerup = SpeedPowerUp(asteroid.position.x,asteroid.position.y,20)
                powerup.velocity = vector3
                self.action = "PowerUp_Spawned_Speed"
            elif powerup_type == "life":
                powerup = LifePowerUp(asteroid.position.x,asteroid.position.y,5)
                powerup.velocity = vector3
                self.action = "PowerUp_Spawned_Life"


