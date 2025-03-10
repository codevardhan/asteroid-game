import pygame
from circleshape import *
NEON_GREEN = (57, 255, 20)
NEON_PINK = (255,20,147)
triangle_points = [(250, 100), (150, 400), (350, 400)]

class PowerUp(CircleShape):
    def __init__(self, x, y, radius,type):
        super().__init__(x,y,radius)
        self.type = type
        self.position = pygame.Vector2(x,y)
        self.radius = radius
        self.velocity = pygame.Vector2(0,0)

    def update(self,dt):
        self.position += self.velocity * dt
    
    def collision_check(self, other):
        r1 = self.radius 
        r2 = other.radius
        if self.position.distance_to(other.position) <= r1 + r2:
            return True
        return False
    
    def remove(self):
        self.kill()
    
    def apply_effect(self):
        match(self.type):
            case 'speed_power_up':
                PLAYER_TURN_SPEED += 100
                PLAYER_SPEED += 50
                return
            case 'shot_power_up':
                PLAYER_SHOOT_SPEED += 100
                PLAYER_SHOOT_COOLDOWN -= 0.05
                return
            
class SpeedPowerUp(PowerUp):

    #initializs powerup sprite
    def __init__(self, x, y, radius):
        super().__init__(x,y,radius,"speed_power_up")

    def draw(self,screen):
        pygame.draw.circle(screen,NEON_GREEN,(int(self.position.x), int(self.position.y)), self.radius,2)

class ShotPowerUp(PowerUp):
        #initializs powerup sprite
    def __init__(self, x, y, radius):
        super().__init__(x,y,radius,"shot_power_up")

    def draw(self,screen):
        pygame.draw.polygon(screen, NEON_PINK, triangle_points, 3)
