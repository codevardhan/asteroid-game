from circleshape import *
from constants import ASTEROID_KINDS, ASTEROID_MAX_RADIUS, ASTEROID_MIN_RADIUS, ASTEROID_SPAWN_RATE
import random
import pygame
import math

class Asteroid(CircleShape):
    def __init__(self, x, y, radius):
        super().__init__(x,y,radius)
        self.position = pygame.Vector2(x,y)
        self.radius = radius
        self.velocity = pygame.Vector2(0,0)

        self.sides = random.randint(5,10)
        self.points  = self.generate_polygons()

        self.image = pygame.Surface((radius*2,radius*2),pygame.SRCALPHA)
        self.rect = self.image.get_rect(center=self.position)

    def generate_polygons(self):
        p = []
        angle_step = 2 * math.pi / self.sides

        for i in range(self.sides):
            angle = i * angle_step
            rand_off = random.uniform(0.8,1.2)
            x = self.radius * rand_off * math.cos(angle)
            y = self.radius * rand_off * math.sin(angle)
            p.append((x+self.radius,y+self.radius))
        return p
    
    def draw(self, screen):
        pygame.draw.polygon(screen,(255,255,255),[(p[0] + self.position.x - self.radius, p[1] + self.position.y - self.radius) for p in self.points],2)

    
    def update(self, dt):
        self.position += self.velocity * dt

    def split(self):
        self.kill()
        if self.radius <= ASTEROID_MIN_RADIUS:
            return
        
        #spawning asteroids
        random_degrees = random.uniform(20, 50)
        vector1 = pygame.math.Vector2.rotate(self.velocity, random_degrees)
        vector2 = pygame.math.Vector2.rotate(self.velocity, -random_degrees)

        new_radius = self.radius - ASTEROID_MIN_RADIUS
        
        asteroid1 = Asteroid(self.position.x, self.position.y, new_radius)
        asteroid2 = Asteroid(self.position.x, self.position.y, new_radius)

        asteroid1.velocity = vector1 * 1.5 
        asteroid2.velocity = vector2 * 1.5

    def collision_check(self, other):
        r1 = self.radius 
        r2 = other.radius
        if self.position.distance_to(other.position) <= r1 + r2:
            return True
        return False