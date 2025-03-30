import pygame

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

PLAYER_RADIUS = 20
PLAYER_SPEED = 200
PLAYER_TURN_SPEED = 300

SHOT_RADIUS = 5
SHOT_SPEED = 500
SHOT_COOLDOWN = 0.4

ASTEROID_MIN_RADIUS = 20
ASTEROID_KINDS = 3
ASTEROID_MAX_RADIUS = ASTEROID_MIN_RADIUS * ASTEROID_KINDS

MAX_ASTEROIDS_ONSCREEN = 15
NEAR_MISS_DISTANCE = 50
MAX_STEPS = 1000
PLAYER_SHOOT_SPEED = 500

# ---------------------------------------------------
#  Two agents: "player" and "asteroid"
#    We define discrete action spaces for both,
#    but you could also do continuous.
# ---------------------------------------------------
PLAYER_ACTION_SIZE = 18
ASTEROID_ACTION_SIZE = 109

# Predefine spawn parameters for the asteroid agent
ASTEROID_EDGES = [
    [
        pygame.Vector2(1, 0),
        lambda y: pygame.Vector2(-ASTEROID_MAX_RADIUS, y * SCREEN_HEIGHT),
    ],
    [
        pygame.Vector2(-1, 0),
        lambda y: pygame.Vector2(SCREEN_WIDTH + ASTEROID_MAX_RADIUS, y * SCREEN_HEIGHT),
    ],
    [
        pygame.Vector2(0, 1),
        lambda x: pygame.Vector2(x * SCREEN_WIDTH, -ASTEROID_MAX_RADIUS),
    ],
    [
        pygame.Vector2(0, -1),
        lambda x: pygame.Vector2(x * SCREEN_WIDTH, SCREEN_HEIGHT + ASTEROID_MAX_RADIUS),
    ],
]
POSSIBLE_RADII = [ASTEROID_MIN_RADIUS * k for k in [1, 2, 3]]  # [20, 40, 60]
POSSIBLE_SPEEDS = [40, 70, 100]
POSSIBLE_ANGLES = [-30, 0, 30]
ASTEROID_SPAWN_RATE = 0.8  # seconds