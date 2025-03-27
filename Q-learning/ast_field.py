import pygame
import random
from asteroid import Asteroid
from constants import *

class AsteroidField(pygame.sprite.Sprite):
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

    def __init__(self, difficulty_manager):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.spawn_timer = 0.0
        self.difficulty_manager = difficulty_manager
        
        # Track metrics for RL
        self.asteroids_spawned = 0
        self.asteroids_destroyed = 0

    def spawn(self, radius, position, velocity):
        asteroid = Asteroid(position.x, position.y, radius)
        asteroid.velocity = velocity
        self.asteroids_spawned += 1

    def update(self, dt):
        self.spawn_timer += dt
        spawn_rate = self.difficulty_manager.get_spawn_rate()
        
        if self.spawn_timer > spawn_rate:
            self.spawn_timer = 0

            # Get current difficulty parameters
            speed_min, speed_max = self.difficulty_manager.get_speed_range()
            asteroid_kinds = self.difficulty_manager.get_asteroid_kinds()
            
            # Calculate engagement-based adjustments
            engagement = self.difficulty_manager.get_engagement_score()
            
            # If engagement is low, make things more interesting
            if engagement < 0.3:
                # Increase spawn chances
                spawn_chance = 1.0
                # Spawn smaller, faster asteroids for more action
                size_factor = 0.8
                speed_factor = 1.2
            elif engagement > 0.7:
                # Engagement is high, increase challenge
                spawn_chance = 1.0
                size_factor = 1.2
                speed_factor = 1.1
            else:
                # Normal engagement
                spawn_chance = 1.0
                size_factor = 1.0
                speed_factor = 1.0
            
            # Decide whether to spawn based on probability
            if random.random() < spawn_chance:
                # spawn a new asteroid at a random edge
                edge = random.choice(self.edges)
                
                # Calculate speed with engagement factor
                speed = random.randint(
                    int(speed_min * speed_factor),
                    int(speed_max * speed_factor)
                )
                
                velocity = edge[0] * speed
                velocity = velocity.rotate(random.randint(-30, 30))
                position = edge[1](random.uniform(0, 1))
                
                # Calculate size with engagement factor
                kind = random.randint(1, int(asteroid_kinds))
                radius = int(ASTEROID_MIN_RADIUS * kind * size_factor)
                
                self.spawn(radius, position, velocity)
                
                # Possible chance for a bonus asteroid based on engagement
                if random.random() < engagement * 0.2:
                    edge = random.choice(self.edges)
                    speed = random.randint(
                        int(speed_min * speed_factor),
                        int(speed_max * speed_factor)
                    )
                    
                    velocity = edge[0] * speed
                    velocity = velocity.rotate(random.randint(-30, 30))
                    position = edge[1](random.uniform(0, 1))
                    
                    # Make bonus asteroid slightly smaller
                    kind = random.randint(1, int(asteroid_kinds * 0.8))
                    radius = int(ASTEROID_MIN_RADIUS * kind * size_factor)
                    
                    self.spawn(radius, position, velocity)