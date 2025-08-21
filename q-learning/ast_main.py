import pygame
import pygame.freetype
from constants import *
from player import Player
from asteroid import Asteroid
from ast_field import AsteroidField
from shot import Shot
from powerups import PowerUp
from RLlevel import RLDifficultyManager
from powerup_manager import PowerUpManager


def main():
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    font = pygame.freetype.SysFont(None, 36)
    small_font = pygame.freetype.SysFont(None, 24)

    # Sprite groups
    updatables = pygame.sprite.Group()
    drawables = pygame.sprite.Group()
    asteroids = pygame.sprite.Group()
    shots = pygame.sprite.Group()
    powerups = pygame.sprite.Group()

    # Container setup
    Asteroid.containers = (asteroids, updatables, drawables)
    Shot.containers = (shots, updatables, drawables)
    AsteroidField.containers = updatables
    Player.containers = (updatables, drawables)
    PowerUp.containers = (powerups, updatables, drawables)

    # Difficulty manager
    difficulty_manager = RLDifficultyManager()
    difficulty_manager.verbose = True  # <- Turn this ON to see debug prints

    try:
        difficulty_manager.load_model("difficulty_model.json")
    except:
        print("Starting with a new model.")
    PowerUpManager.containers = updatables
    asteroid_field = AsteroidField(difficulty_manager)
    powerup_m = PowerUpManager(difficulty_manager)
    updatables.add(asteroid_field)

    # Game state
    player = Player(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)
    player.shots_fired_count = 0

    game_over = False
    score = 0
    dt = 0
    lives = 1
    # RL metrics
    shots_fired = 0
    shots_hit = 0
    near_misses = 0
    near_miss_distance = PLAYER_RADIUS * 2
    collected = 0
    difficulty_message = ""
    difficulty_timer = 0

    games_played = 0
    high_score = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                difficulty_manager.save_model(
                    "../outputs/q-learning/difficulty_model.json"
                )
                difficulty_manager.save_training_log(
                    "../outputs/q-learning/training_log.csv"
                )
                difficulty_manager.plot_training_progress(
                    "../outputs/q-learning/training_plot.png"
                )
                return

            elif event.type == pygame.KEYDOWN:
                if game_over and event.key == pygame.K_r:
                    game_over = False
                    score = 0
                    shots_fired = 0
                    shots_hit = 0
                    near_misses = 0
                    games_played += 1

                    updatables.empty()
                    drawables.empty()
                    asteroids.empty()
                    shots.empty()
                    powerups.empty()

                    player = Player(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)
                    player.shots_fired_count = 0
                    lives = player.player_lives
                    asteroid_field = AsteroidField(difficulty_manager)
                    powerup_m = PowerUpManager(difficulty_manager)
                    updatables.add(asteroid_field)

                elif event.key == pygame.K_SPACE:
                    shots_fired += 1
                    player.shots_fired_count += 1

        screen.fill((0, 0, 0))

        for drawable in drawables:
            drawable.draw(screen)

        if not game_over:
            dt = clock.tick(60) / 1000.0
            # print(len(player.active_effects))
            player_data = {
                "near_misses": near_misses,
                "shots_fired": shots_fired,
                "shots_hit": shots_hit,
                "player_alive": True,
                "collected_powerups": len(player.active_effects),
                "score": score,
            }

            shots_fired = 0
            shots_hit = 0
            near_misses = 0
            collected = 0
            episode_ended = difficulty_manager.update(dt, player_data)

            if difficulty_timer > 0:
                difficulty_timer -= dt

        for updateable in updatables:
            updateable.update(dt)
            player.update_effects()

        for asteroid in asteroids:
            dist = asteroid.position.distance_to(player.position)
            if near_miss_distance < dist < near_miss_distance * 1.5:
                near_misses += 1

            if asteroid.collision_check(player):
                player.player_lives -= 1
                player.position.x = SCREEN_WIDTH / 2
                player.position.y = SCREEN_HEIGHT / 2
                lives = player.player_lives
                if player.player_lives == 0:
                    game_over = True

                player_data = {
                    "near_misses": near_misses,
                    "shots_fired": shots_fired,
                    "shots_hit": shots_hit,
                    "player_alive": False,
                    "collected_powerups": len(player.active_effects),
                    "score": score,
                }

                difficulty_manager.update(dt, player_data)
                difficulty_manager.save_model("difficulty_model.json")

                high_score = max(high_score, score)
                break

            for shot in shots:
                if asteroid.collision_check(shot):
                    asteroid.split()
                    powerup_m.spawn_from_asteroid(asteroid)
                    shot.kill()
                    score += 1
                    shots_hit += 1

            for powerup in powerups:
                if powerup.collision_check(player):
                    powerup.apply_effect(player)
                    collected += 1
                    powerup.remove()

        # Draw UI
        font.render_to(screen, (10, 10), f"Score: {score}", (255, 255, 255))

        if not game_over:
            engagement = difficulty_manager.get_engagement_score()

            if engagement < 0.3:
                engagement_color = (255, 100, 100)
            elif engagement < 0.7:
                engagement_color = (255, 255, 100)
            else:
                engagement_color = (100, 255, 100)

            pygame.draw.rect(screen, (50, 50, 50), (10, 55, 200, 20))
            pygame.draw.rect(screen, engagement_color, (10, 55, 200 * engagement, 20))
            small_font.render_to(screen, (220, 55), f"Engagement", (200, 200, 200))

            difficulty_info = f"Spawn Rate: {difficulty_manager.get_spawn_rate():.2f}s | Speed: {difficulty_manager.get_speed_range()[0]:.0f}-{difficulty_manager.get_speed_range()[1]:.0f}"
            small_font.render_to(
                screen, (10, SCREEN_HEIGHT - 60), difficulty_info, (200, 200, 200)
            )
            font.render_to(
                screen, (180, 10), f"Lives: {player.player_lives}", (255, 255, 255)
            )
            explore_info = (
                f"AI Learning: {difficulty_manager.get_exploration_rate():.2f}"
            )
            small_font.render_to(
                screen, (10, SCREEN_HEIGHT - 30), explore_info, (180, 180, 220)
            )

            hit_ratio = shots_hit / max(1, player.shots_fired_count) * 100
            # small_font.render_to(screen, (SCREEN_WIDTH - 200, 10), f"Hit Ratio: {hit_ratio:.1f}%", (200, 200, 200))
            small_font.render_to(
                screen,
                (SCREEN_WIDTH - 200, 40),
                f"Games: {games_played}",
                (200, 200, 200),
            )
            small_font.render_to(
                screen, (SCREEN_WIDTH - 200, 70), f"Best: {high_score}", (200, 200, 200)
            )

        if difficulty_timer > 0:
            text_surface, rect = font.render(difficulty_message, (255, 255, 0))
            screen.blit(text_surface, (SCREEN_WIDTH // 2 - rect.width // 2, 100))

        if game_over:
            font.render_to(
                screen,
                (SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2),
                "GAME OVER",
                (255, 255, 255),
            )
            font.render_to(
                screen,
                (SCREEN_WIDTH // 2 - 140, SCREEN_HEIGHT // 2 + 50),
                "Press R to restart",
                (255, 255, 255),
            )

            small_font.render_to(
                screen,
                (SCREEN_WIDTH // 2 - 140, SCREEN_HEIGHT // 2 + 100),
                f"Engagement: {difficulty_manager.get_engagement_score():.2f}",
                (200, 200, 200),
            )

            small_font.render_to(
                screen,
                (SCREEN_WIDTH // 2 - 140, SCREEN_HEIGHT // 2 + 130),
                f"Survival Time: {difficulty_manager.episode_length:.1f}s",
                (200, 200, 200),
            )

            player.kill()

        pygame.display.flip()


if __name__ == "__main__":
    main()
