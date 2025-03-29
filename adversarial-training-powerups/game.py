import random
import pygame

import pygame.freetype

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

ASTEROID_MIN_RADIUS = 20
ASTEROID_KINDS = 3
ASTEROID_SPAWN_RATE = 0.8  # seconds
ASTEROID_MAX_RADIUS = ASTEROID_MIN_RADIUS * ASTEROID_KINDS

PLAYER_RADIUS = 20
PLAYER_TURN_SPEED = 300
PLAYER_SPEED = 200

SHOT_RADIUS = 5
PLAYER_SHOOT_SPEED = 500
PLAYER_SHOOT_COOLDOWN = 0.4


# Base class for game objects
class CircleShape(pygame.sprite.Sprite):
    def __init__(self, x, y, radius):
        # will use later
        if hasattr(self, "containers"):
            super().__init__(self.containers)
        else:
            super().__init__()

        self.position = pygame.Vector2(x, y)
        self.velocity = pygame.Vector2(0, 0)
        self.radius = radius

    def draw(self, screen):
        # sub-classes must override
        pass

    def update(self, dt):
        # sub-classes must override
        pass

    def collision_check(self, other):
        r1 = self.radius
        r2 = other.radius
        if self.position.distance_to(other.position) <= r1 + r2:
            return True
        return False


class Asteroid(CircleShape):
    def __init__(self, x, y, radius):
        super().__init__(x, y, radius)

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 255, 255), self.position, self.radius, 2)

    def update(self, dt):
        self.position += self.velocity * dt

    def split(self):
        self.kill()
        if self.radius <= ASTEROID_MIN_RADIUS:
            return

        # spawning asteroids
        random_degrees = random.uniform(20, 50)
        vector1 = pygame.math.Vector2.rotate(self.velocity, random_degrees)
        vector2 = pygame.math.Vector2.rotate(self.velocity, -random_degrees)

        new_radius = self.radius - ASTEROID_MIN_RADIUS

        asteroid1 = Asteroid(self.position.x, self.position.y, new_radius)
        asteroid2 = Asteroid(self.position.x, self.position.y, new_radius)

        asteroid1.velocity = vector1 * 1.5
        asteroid2.velocity = vector2 * 1.5


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

    def __init__(self):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.spawn_timer = 0.0

    def spawn(self, radius, position, velocity):
        asteroid = Asteroid(position.x, position.y, radius)
        asteroid.velocity = velocity

    def update(self, dt):
        self.spawn_timer += dt
        if self.spawn_timer > ASTEROID_SPAWN_RATE:
            self.spawn_timer = 0

            # spawn a new asteroid at a random edge
            edge = random.choice(self.edges)
            speed = random.randint(40, 100)
            velocity = edge[0] * speed
            velocity = velocity.rotate(random.randint(-30, 30))
            position = edge[1](random.uniform(0, 1))
            kind = random.randint(1, ASTEROID_KINDS)
            self.spawn(ASTEROID_MIN_RADIUS * kind, position, velocity)


class Player(CircleShape):
    def __init__(self, x, y):
        super().__init__(x, y, PLAYER_RADIUS)
        self.rotation = 0
        self.timer = 0

    # player will look like triangle
    # hitbox logic - circle will be used
    def triangle(self):
        forward = pygame.Vector2(0, 1).rotate(self.rotation)
        right = pygame.Vector2(0, 1).rotate(self.rotation + 90) * self.radius / 1.5
        a = self.position + forward * self.radius
        b = self.position - forward * self.radius - right
        c = self.position - forward * self.radius + right
        return [a, b, c]

    def draw(self, screen):
        pygame.draw.polygon(screen, (255, 255, 255), self.triangle(), 2)

    def rotate(self, dt):
        self.rotation += PLAYER_TURN_SPEED * dt

    def update(self, dt):
        keys = pygame.key.get_pressed()

        # w,a,s,d and arrow keys
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            invert_dt = dt * -1
            self.rotate(invert_dt)

        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            self.rotate(dt)

        if keys[pygame.K_w] or keys[pygame.K_UP]:
            self.move(dt)

        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            self.move(-dt)

        if keys[pygame.K_SPACE]:
            if self.timer <= 0:
                self.shoot()
                self.timer = PLAYER_SHOOT_COOLDOWN

        self.timer -= dt

    def move(self, dt):
        forward = pygame.Vector2(0, 1).rotate(self.rotation)
        self.position += forward * PLAYER_SPEED * dt

    def shoot(self):
        shot = Shot(self.position.x, self.position.y, SHOT_RADIUS)
        shot.velocity = pygame.Vector2(0, 1).rotate(self.rotation) * PLAYER_SHOOT_SPEED


class Shot(CircleShape):
    def __init__(self, x, y, SHOT_RADIUS):
        super().__init__(x, y, SHOT_RADIUS)

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 255, 255), self.position, self.radius, 2)

    def update(self, dt):
        self.position += self.velocity * dt


def main():
    # initialzing  game and screen
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    font = pygame.freetype.SysFont(None, 36)

    # groups
    updatables = pygame.sprite.Group()
    drawables = pygame.sprite.Group()
    asteroids = pygame.sprite.Group()
    shots = pygame.sprite.Group()
    # containers
    Asteroid.containers = (asteroids, updatables, drawables)
    Shot.containers = (shots, updatables, drawables)

    AsteroidField.containers = updatables
    asteroid_field = AsteroidField()

    dt = 0
    game_over = False
    score = 0

    # instances
    # rendering player and asteroids
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

        screen.fill((0, 0, 0))

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

        font.render_to(screen, (10, 10), f"Score: {score}", (255, 255, 255))

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
            player.kill()

        pygame.display.flip()


if __name__ == "__main__":
    main()
