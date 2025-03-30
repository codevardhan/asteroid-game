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
NEON_GREEN = (57, 255, 20)
NEON_PINK = (255,20,147)
NEON_RED = (255, 30, 30)
EFFECT_DURATION = 5000

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
        gamma = 0.4
        #spawning asteroids
        random_degrees = random.uniform(20, 50)
        vector1 = pygame.math.Vector2.rotate(self.velocity, random_degrees)
        vector2 = pygame.math.Vector2.rotate(self.velocity, -random_degrees)

        new_radius = self.radius - ASTEROID_MIN_RADIUS
        
        asteroid1 = Asteroid(self.position.x, self.position.y, new_radius)
        asteroid2 = Asteroid(self.position.x, self.position.y, new_radius)
        # RL must decide when to let asteroids spawn powerups upon destruction
        # random math as placeholder
        random_degrees = random.uniform(20, 50)
        # instead of creating asteroids here, extract out, and spawn in main gameplay loop
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
        self.current_action = 0 #player starts in middle still
        self.player_shoot_cooldown = 0.4
        self.player_speed = PLAYER_SPEED
        self.player_shoot_speed = PLAYER_SHOOT_SPEED
        self.player_turn_speed = PLAYER_TURN_SPEED
        self.player_lives = 1
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
            
class PowerUp(CircleShape):
    def __init__(self, x, y, radius,type,ttl=5000):
        super().__init__(x,y,radius)
        self.type = type
        self.position = pygame.Vector2(x,y)
        self.radius = radius
        self.velocity = pygame.Vector2(0,0)
        self.spawn_time = pygame.time.get_ticks()
        self.ttl = 5000
    def update(self,dt):
        self.position += self.velocity * dt
        self.ttl -= 1
        if pygame.time.get_ticks() - self.spawn_time >= self.ttl:
            self.kill()
    
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
                player.player_powerups['speed_power_up']+=1
                player.player_turn_speed += 100
                player.player_speed += 50
                player.active_effects.append(("speed_power_up", pygame.time.get_ticks() + EFFECT_DURATION))
                return
            case 'shot_power_up':
                player.player_powerups['shot_power_up']+=1
                player.player_shoot_speed += 100
                player.player_shoot_cooldown -= 0.05
                player.active_effects.append(("shot_power_up", pygame.time.get_ticks() + EFFECT_DURATION))
                return
            case 'life_power_up':
                player.player_powerups['life_power_up']+=1
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
    
class AsteroidsRLLibEnv(MultiAgentEnv):
    """
    RLlib-compatible multi-agent environment.
    Agents:
      "player" -> discrete action ∈ [0..17] this creates action space problem, too many options
      "asteroid" -> discrete action ∈ [0..108] similar problem here, can be simplified
    Observations:
      - For the player: e.g. (playerX, playerY, num_asteroids)
      - For the asteroid: e.g. (num_asteroids, playerX, playerY)

    Each step => returns obs, rew, done, info for both agents.

    Making changes for player to be 5 actions
    Making changes for EnvRL to be 3 actions
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
        self.powerups = pygame.sprite.Group()
        self.updatables = pygame.sprite.Group()

        self.score = 0
        self.lives = 1
        self.collected = 0
        self.game_over = False
        self.steps_elapsed = 0

        self.last_asteroid_destroyed = None
        self.near_miss_count = 0
        self.last_spawns = []
        self.dt = 0.016

        self.max_steps = MAX_STEPS  # you can override in config

        # RLlib requires specifying the spaces
        # (could be gym.spaces.Box, Discrete, etc.)
        self.observation_space_player = Box(
            low=-1e5, high=1e5, shape=(5,), dtype=np.float32
            )
        self.observation_space_asteroid = Box(
        low=-1e5, high=1e5, shape=(3,), dtype=np.float32
        )

# Updated action spaces
        self.action_space_player = Discrete(5)  # Move, move left, move right, Shoot, Stop
        self.action_space_asteroid = Discrete(3)  # No Spawn, Spawn Asteroid, Spawn PowerUp
        self.possible_agents = ["player", "asteroid"]
        self.agents = self.possible_agents.copy()

    def reset(self, seed=None, options=None):
        # Clear old state
        self.asteroids.empty()
        self.shots.empty()
        self.updatables.empty()
        self.powerups.empty(0)
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
            elif isinstance(s,PowerUp):
                s.position += s.velocity * self.dt
        # 4) Handle collisions
        destroyed = 0
        for asteroid in list(self.asteroids):
            if asteroid.collision_check(self.player):
                self.player.player_lives -= 1
                if self.player.player_lives == 0:
                    self.game_over = True
            for shot in list(self.shots):
                if asteroid.collision_check(shot):
                    self.last_asteroid_destroyed = asteroid
                    shot.kill()
                    asteroid.kill()
                    destroyed += 1
                    break
        for powerup in list(self.powerups):
            if powerup.collision_check(self.player):
                self.collected += 1   
                powerup.apply_effect(self.player)
                powerup.remove()         
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
        player_reward = float(destroyed) + float(self.collected)
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
    #Adding reward system for powerups
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
        4 possible player actions
        """
        if a == 0:  # Move forward and turn left
            self.player.rotate(self.dt, -1)  # Turn left
            self.player.move(self.dt, 1)  # Move forward
        
        elif a == 1:  # Shoot
            new_shot = self.player.shoot()  # Shoot a projectile
        if new_shot:
            self.shots.add(new_shot)  # Add shot to the game
            self.updatables.add(new_shot)  # Add shot to the updatable objects
    
        elif a == 2:  # Move forward and turn right
            self.player.rotate(self.dt, 1)  # Turn right
            self.player.move(self.dt, 1)  # Move forward
        elif a == 3:
            self.player.move(self.dt, 1)
        elif a == 4:  # Stop (no action)
            self.player.move(self.dt, 0)

    def _apply_asteroid_action(self, a):
        """
        0 => no spawn
        1 => spawn asteroid
        2 => spawn powerup
        """
        if a == 0:
            return
        if a==2 and self.last_asteroid_destroyed == None:
            return
        if a==2 and self.last_asteroid_destroyed != None:
            self.spawn_from_asteroid(self.last_asteroid_destroyed)
            return

        edge = random.choice(ASTEROID_EDGES)
        radius = random.choice(POSSIBLE_RADII)
        speed = random.choice(POSSIBLE_SPEEDS)
        angle = random.choice(POSSIBLE_ANGLES)

        # Generate position and velocity
        frac = random.random()  # This will be used to calculate the spawn position along the edge
        pos = edge[1](frac)  # Get position from edge definition (lambda function)
        direction = edge[0].rotate(angle)  # Rotate direction vector by angle
        vel = direction * speed  # Compute velocity vector

    # Create the asteroid (you should have an Asteroid class to instantiate)
        asteroid = Asteroid(pos.x, pos.y, radius)
        asteroid.velocity = vel
    
        self.asteroids.add(asteroid)
        self.updatables.add(asteroid)

        self.last_spawns.append((radius, speed, angle))
        if len(self.last_spawns) > 20:
            self.last_spawns.pop(0)

    def spawn_from_asteroid(self,asteroid):
            
            life_weight = 0.2
            shot_weight = 0.3
            speed_weight = 0.5

            if self.player.player_lives < 2: 
                life_weight += 0.4
            if len(self.asteroids) > 10: 
                shot_weight += 0.3 
            if len(self.asteroids) <= 5 and self.player.player_lives >= 2:  
                speed_weight += 0.3 
            total = life_weight + shot_weight + speed_weight
            life_weight /= total
            shot_weight /= total
            speed_weight /= total
            powerup_type = random.choice(["speed", "shot", "life"],weights=[speed_weight,shot_weight,life_weight])
            vector3 = pygame.math.Vector2.rotate(asteroid.velocity,random.uniform(20, 50))
            if powerup_type == "shot":
                powerup = ShotPowerUp(asteroid.position.x,asteroid.position.y,2)
                powerup.velocity = vector3
                self.action = "PowerUp_Spawned_Shot"
                self.powerups.add(powerup)
                self.updatables.add(powerup)
            elif powerup_type == "speed":
                powerup = SpeedPowerUp(asteroid.position.x,asteroid.position.y,20)
                powerup.velocity = vector3
                self.action = "PowerUp_Spawned_Speed"
                self.powerups.add(powerup)
                self.updatables.add(powerup)
            elif powerup_type == "life":
                powerup = LifePowerUp(asteroid.position.x,asteroid.position.y,5)
                powerup.velocity = vector3
                self.action = "PowerUp_Spawned_Life"
                self.powerups.add(powerup)
                self.updatables.add(powerup)
                
    def _update_asteroids_buckets(self):
        asteroid_buckets = {}
        for asteroid in self.asteroids:
            bx,by = self._get_asteroid_bucket(asteroid)
            if (bx,by) not in asteroid_buckets:
                asteroid_buckets[(bx, by)] = []
            asteroid_buckets[(bx, by)].append(asteroid) 
        return asteroid_buckets  
    #SCREEN_WIDTH, SCREEN_HEIGHT
    def _get_player_bucket(self):
        px, py = self.player.position.x, self.player.position.y
        bucket_size = min(SCREEN_WIDTH // 8, SCREEN_HEIGHT // 6)
        bucket_x = px // bucket_size
        bucket_y = py // bucket_size
        return bucket_x,bucket_y
    
    def _get_asteroid_bucket(self,asteroid):
        ax, ay = asteroid.position.x, asteroid.position.y
        bucket_size = min(SCREEN_WIDTH // 8, SCREEN_HEIGHT // 6)
        bucket_x = ax // bucket_size
        bucket_y = ax // bucket_size
        bucket_x = max(0, min(bucket_x, SCREEN_WIDTH // 8 - 1))
        bucket_y = max(0, min(bucket_y, SCREEN_HEIGHT // 6 - 1))

        return bucket_x, bucket_y
    # ----------------------------
    #   Observations
    # ----------------------------
    def _get_player_obs(self):
        """
        Return a 3D vector: [player_x, player_y, #asteroids, powerups collected,player lives, # of active powerups].
        """
                #player lives bucket
        
        p_lives = self.player.player_lives
        if p_lives == 1:
            p_lives_bucket = 0
        elif p_lives > 2 and p_lives < 5:
            p_lives_bucket = 1
        else:
            p_lives_bucket = 2

        num_asts = len(self.asteroids)
        if num_asts < 10:
            num_asts_bucket = 0
        elif num_asts >= 10 and num_asts < 20:
            num_asts_bucket = 1
        elif num_asts >= 20 and num_asts < 40:
            num_asts_bucket = 2
        elif num_asts >= 40:
            num_asts_bucket = 3
        
        num_active_effects = len(self.player.active_effects)
        if num_active_effects == 0:
            num_act_effects_bucket = 0
        elif num_active_effects >= 1 and num_active_effects < 4:
            num_act_effects_bucket = 1
        else:
            num_act_effects_bucket = 2

        if self.collected == 0:
            collected_buckets = 0
        elif self.collected >= 1 and self.collected < 3:
            collected_buckets = 1
        elif self.collected >= 3 and self.collected < 6:
            collected_buckets = 2
        else:
            collected_buckets = 3

        return np.array(
            [self.player.position.x, self.player.position.y,self.player.velocity.x,self.player.velocity.x,
              num_asts_bucket,collected_buckets,p_lives_bucket,num_act_effects_bucket],
            dtype=np.float32,
        )

    def _get_asteroid_obs(self):
        """
        Return a 3D vector: [#asteroids, player_x, player_y, # of powerups on board].
        """
        num_asts = len(self.asteroids)
        if num_asts < 10:
            num_asts_bucket = 0
        elif num_asts >= 10 and num_asts < 20:
            num_asts_bucket = 1
        elif num_asts >= 20 and num_asts < 40:
            num_asts_bucket = 2
        elif num_asts >= 40:
            num_asts_bucket = 3

        px, py = self.player.position.x, self.player.position.y

        num_pup = len(self.powerups)
        if num_pup < 3:
            num_pup_bucket = 0
        elif num_pup >=3 and num_pup < 5:
            num_pup_bucket = 1
        elif num_pup >=5:
            num_pup_bucket = 2

        #bucket for near miss count
        if self.near_miss_count < 5:
            near_miss_bucket = 0  # Low risk
        elif self.near_miss_count < 15:
            near_miss_bucket = 1  # Medium risk
        else:
            near_miss_bucket = 2 

        return np.array(
            [num_asts_bucket, px,
              py,self.player.velocity.x,self.player.velocity.x
              ,num_pup_bucket],
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
