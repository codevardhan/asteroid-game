import pygame
from circleshape import *
NEON_GREEN = (57, 255, 20)
NEON_PINK = (255,20,147)
NEON_RED = (255, 30, 30)
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
    
    def apply_effect(self,player):
        match(self.type):
            case 'speed_power_up':
                player.player_turn_speed += 100
                player.player_speed += 50
                return
            case 'shot_power_up':
                player.player_shoot_speed += 100
                player.player_shoot_cooldown -= 0.05
                return
            case 'life_power_up':
                player.player_lives += 1
                return
class LifePowerUp(PowerUp):
    def __init__(self, x, y, radius):
        super().__init__(x,y,radius,"life_power_up")
        self.size = radius * 10
    def draw(self,screen):
        line_weight = self.size // 5
        points = [
            (self.position.x - line_weight, self.position.y - self.size // 2),
            (self.position.x + line_weight, self.position.y - self.size // 2),
            (self.position.x  + line_weight, self.position.y - line_weight),
            (self.position.x  + self.size // 2, self.position.y - line_weight),  
            (self.position.x  + self.size // 2, self.position.y + line_weight),  
            (self.position.x  + line_weight, self.position.y + line_weight),  
            (self.position.x  + line_weight, self.position.y + self.size // 2),  
            (self.position.x  - line_weight, self.position.y + self.size // 2),  
            (self.position.x  - line_weight, self.position.y + line_weight),  
            (self.position.x  - self.size // 2, self.position.y + line_weight), 
            (self.position.x  - self.size // 2, self.position.y - line_weight),  
            (self.position.x  - line_weight, self.position.y - line_weight)
        ]
        pygame.draw.polygon(screen,NEON_RED,points)
    def collision_check(self, other):
        line_weight = self.size // 5
        vertical_rect = pygame.Rect(self.position.x - line_weight, self.position.y - self.size // 2, 2 * line_weight, self.size)
        horizontal_rect = pygame.Rect(self.position.x - self.size // 2, self.position.y - line_weight, self.size, 2 * line_weight)
        return vertical_rect.collidepoint(other.position.x,other.position.y) or horizontal_rect.collidepoint(other.position.x,other.position.y)
    
class SpeedPowerUp(PowerUp):

    #initializs powerup sprite
    def __init__(self, x, y, radius):
        super().__init__(x,y,radius,"speed_power_up")

    def draw(self,screen):
        pygame.draw.circle(screen,NEON_GREEN,(int(self.position.x), int(self.position.y)), self.radius,2)
    def collision_check(self, other):
        return super().collision_check(other)
class ShotPowerUp(PowerUp):
        #initializs powerup sprite
    def __init__(self, x, y, radius):
        super().__init__(x,y,radius,"shot_power_up")
        self.size = radius * 10

    def draw(self,screen):
        triangle_points = [
            (self.position.x, self.position.y - self.size),  # Top point
            (self.position.x - self.size * 0.6, self.position.y + self.size * 0.6),  # Bottom-left
            (self.position.x + self.size * 0.6, self.position.y + self.size * 0.6)   # Bottom-right
        ]
        pygame.draw.polygon(screen, NEON_PINK, triangle_points, 3)
    
    def collision_check(self, other):
        min_x = self.position.x - self.size * 0.6
        min_y = self.position.y - self.size
        max_x = self.position.x + self.size * 0.6
        max_y = self.position.y + self.size * 0.6

        if min_x <= other.position.x <= max_x and min_y <= other.position.y <= max_y:
            return True
        return False
