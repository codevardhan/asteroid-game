import pygame
from circleshape import *
from constants import *
from shot import Shot

class Player(CircleShape):
    def __init__(self, x, y):
        super().__init__(x, y, PLAYER_RADIUS)
        self.rotation = 0
        self.timer = 0
        self.current_action = 0 #player starts in middle still
        self.player_shoot_cooldown = PLAYER_SHOOT_COOLDOWN
        self.player_speed = PLAYER_SPEED
        self.player_shoot_speed = PLAYER_SHOOT_SPEED
        self.player_turn_speed = PLAYER_TURN_SPEED
        self.player_lives = PLAYER_LIVES
        self.player_powerups = {"speed_power_up":0,"shot_power_up":0,"life_power_up":0}
        self.active_effects = []
    #player will look like triangle
    #hitbox logic - circle will be used
    def triangle(self):
        forward = pygame.Vector2(0, 1).rotate(self.rotation)
        right = pygame.Vector2(0, 1).rotate(self.rotation + 90) * self.radius / 1.5
        a = self.position + forward * self.radius
        b = self.position - forward * self.radius - right
        c = self.position - forward * self.radius + right
        return [a, b, c]
    
    def draw(self, screen):
        pygame.draw.polygon(screen, (255,255,255), self.triangle(), 2)

    def rotate(self, dt):
        self.rotation += self.player_turn_speed * dt

    def update_effects(self):
        current_time = pygame.time.get_ticks()
        expired_powerups = []
        for powerup,expiration in self.active_effects:
            if current_time >= expiration:
                expired_powerups.append(powerup)
                self.active_effects.remove((powerup,expiration))
                
        for powerup in expired_powerups:
            if powerup == "speed_power_up":
                self.player_powerups['speed_power_up']+=1
                self.player_turn_speed += 100
                self.player_speed += 50
            if powerup == "shot_power_up":
                self.player_powerups['shot_power_up']-=1
                self.player_shoot_speed -= 100
                self.player_shoot_cooldown += 0.05
            


    def update(self, dt):
        keys = pygame.key.get_pressed()
        self.current_action = 0
        #w,a,s,d and arrow keys
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            invert_dt = dt * -1
            self.rotate(invert_dt)
            self.current_action = 1

        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            self.rotate(dt)
            self.current_action = 2
            
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            self.move(dt)
            self.current_action = 3

        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            self.move(-dt)
            self.current_action = 4
            
        if keys[pygame.K_SPACE]:
            if self.timer <= 0:
                self.shoot()
                self.timer = self.player_shoot_cooldown
            self.current_action = 5

        self.timer -= dt
            
    def move(self, dt):
        forward = pygame.Vector2(0, 1).rotate(self.rotation)
        self.position += forward * self.player_speed * dt 

    def shoot(self):
        shot = Shot(self.position.x, self.position.y, SHOT_RADIUS)
        shot.velocity = pygame.Vector2(0,1).rotate(self.rotation) * self.player_shoot_speed