import random
from powerups import LifePowerUp, ShotPowerUp, SpeedPowerUp
import pygame
from constants import *

class PowerUpManager(pygame.sprite.Sprite):
    def __init__(self,difficulty_manager):
        self.powerup_timer = 0
        self.action = None
        self.powerups = pygame.sprite.Group()
        self.difficulty_manager = difficulty_manager
    def update(self, dt):
        self.powerup_timer += dt
        self.action = None

    def spawn_from_asteroid(self,asteroid):
            engagement = self.difficulty_manager.get_engagement_score()
            if engagement < 0.3:
                # Increase spawn chances
                spawn_chance = 0.3
                # Spawn smaller, faster asteroids for more action
            elif engagement > 0.7:
                # Engagement is high, increase challenge
                spawn_chance = 0.7
            else:
                # Normal engagement
                spawn_chance = 0.5
            if random.random() >= spawn_chance:
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


