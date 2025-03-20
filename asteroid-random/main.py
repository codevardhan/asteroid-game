import json
import pygame # type: ignore
import pygame.freetype # type: ignore
from constants import *
from player import Player
from asteroid import Asteroid
from asteroidfield import *
from shot import Shot
from powerups import PowerUp
import numpy as np
training_data = []
def collect_data(player,shots,score,elapsed_time,asteroids,powerups):
    state = {
        "player_x":player.position.x,
        "player_y":player.position.y,
        "player_velocity_x":player.velocity.x,
        "player_velocity_y":player.velocity.y,
        "player_turn_speed":player.player_turn_speed,
        "player_speed":player.player_speed,
        "player_lives":player.player_lives,
        "player_score":score,
        "time":elapsed_time,
        "num_shots": len(shots),
        "num_speed_power_up_taken":player.player_powerups['speed_power_up'],
        "num_shot_power_up_taken":player.player_powerups['shot_power_up'],
        "num_life_power_up_taken":player.player_powerups['life_power_up'],
        "num_active_powerups":len(player.active_effects),
        "num_powerups_on_board":len(powerups),
        "num_asteroids_in_field":len(asteroids),
        "avg_speed_asteroids":np.mean([a.velocity for a in asteroids] if asteroids else 0),
        "asteroid_spawn_rate": ASTEROID_SPAWN_RATE # type: ignore
    }
    training_data.append(state)

def save_data():
    with open("training_data.json","w") as e:
        json.dump(training_data,e,indent=4)

def main():
    #initialzing  game and screen
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    font = pygame.freetype.SysFont(None, 36)
    start_time = pygame.time.get_ticks()
    #groups
    updatables = pygame.sprite.Group()
    drawables = pygame.sprite.Group()
    asteroids = pygame.sprite.Group()
    shots = pygame.sprite.Group()
    powerups = pygame.sprite.Group()
    #containers
    Asteroid.containers = (asteroids, updatables, drawables)
    Shot.containers = (shots, updatables, drawables)
    PowerUp.containers = (powerups,updatables,drawables)

    AsteroidField.containers = updatables
    asteroid_field = AsteroidField()


    dt = 0
    game_over = False
    score = 0
    lives = 1
    #instances
    #rendering player and asteroids
    Player.containers = (updatables, drawables)
    x = SCREEN_WIDTH / 2
    y = SCREEN_HEIGHT / 2
    player = Player(x, y)
     
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            
            elif event.type == pygame.KEYDOWN and game_over:
                if event.key == pygame.K_r:
                    game_over = False
                    score = 0
                    updatables.empty()
                    drawables.empty()
                    asteroids.empty()
                    shots.empty()
                    player = Player(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)
                    lives = player.player_lives
                    asteroid_field = AsteroidField()
            
            
        screen.fill((0,0,0)) 
        
        for drawable in drawables:
            drawable.draw(screen)

        if not game_over:
            dt = clock.tick(60) / 1000.0
        
        for updateable in updatables:
            updateable.update(dt)
            
        for asteroid in asteroids:
            if asteroid.collision_check(player):
                # print("GAME OVER")
                player.player_lives -= 1
                player.position.x = SCREEN_WIDTH / 2
                player.position.y = SCREEN_HEIGHT / 2
                lives = player.player_lives
                if lives == 0:
                    game_over = True
                break
                # return
        
            for shot in shots:
                if asteroid.collision_check(shot):
                    asteroid.split()
                    shot.kill()
                    score += 1
                    # print("hit")

        # check for player collison or shot collision with a powerup
        for powerup in powerups:
            if powerup.collision_check(player):
                powerup.apply_effect(player)
                lives = player.player_lives
                powerup.remove()

        elapsed_time = (pygame.time.get_ticks() - start_time) // 1000
        font.render_to(screen, (10,10), f"Score: {score}", (255,255,255))
        font.render_to(screen, (180,10),f"Lives: {lives}",(255,255,255))
        font.render_to(screen,(SCREEN_WIDTH - 300,10),f"Time: {elapsed_time}",(255,255,255))
        collect_data(player,shots,score,elapsed_time,asteroids,powerups)
        if game_over:
            font.render_to(screen, (SCREEN_WIDTH //2 -100, SCREEN_HEIGHT // 2), "GAME OVER", (255,255,255))
            font.render_to(screen, (SCREEN_WIDTH //2 -140, SCREEN_HEIGHT //2 +50),"Press R to restart", (255,255,255))
            player.kill()
            save_data()

        pygame.display.flip()
            
if __name__ == "__main__":
    main()