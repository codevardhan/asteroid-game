import pygame
import pygame.freetype
from constants import *
from player import Player
from asteroid import Asteroid
from asteroidfield import *
from shot import Shot

def main():
    #initialzing  game and screen
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    font = pygame.freetype.SysFont(None, 36)

    #groups
    updatables = pygame.sprite.Group()
    drawables = pygame.sprite.Group()
    asteroids = pygame.sprite.Group()
    shots = pygame.sprite.Group()
    #containers
    Asteroid.containers = (asteroids, updatables, drawables)
    Shot.containers = (shots, updatables, drawables)

    AsteroidField.containers = updatables
    asteroid_field = AsteroidField()


    dt = 0
    game_over = False
    score = 0

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
                game_over = True
                break
                # return
        
            for shot in shots:
                if asteroid.collision_check(shot):
                    asteroid.split()
                    shot.kill()
                    score += 1
                    # print("hit")

        font.render_to(screen, (10,10), f"Score: {score}", (255,255,255))
        
        if game_over:
            font.render_to(screen, (SCREEN_WIDTH //2 -100, SCREEN_HEIGHT // 2), "GAME OVER", (255,255,255))
            font.render_to(screen, (SCREEN_WIDTH //2 -140, SCREEN_HEIGHT //2 +50),"Press R to restart", (255,255,255))
            player.kill()

        pygame.display.flip()
            
if __name__ == "__main__":
    main()