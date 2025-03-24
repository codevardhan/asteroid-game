import sys
sys.path.insert(0, '../asteroid-random')
from player import Player
from shot import Shot
from asteroid import Asteroid
from asteroidfield import AsteroidField
from powerups import PowerUp
import pygame
import pygame.freetype
import random

import gymnasium as gym
from gymnasium import spaces
import numpy as np

import math
import heapq



from constants import *


class AStarAgent:
    def __init__(
        self,
        grid_size=(64, 36),
        safe_distance=70,
        replan_interval=0.4,
        shoot_distance=500,    # Max distance to attempt shooting
        # Angle threshold (degrees) to consider an asteroid "in front"
        shoot_angle_thresh=40,
    ):
        self.grid_cols, self.grid_rows = grid_size
        self.cell_width = SCREEN_WIDTH / self.grid_cols
        self.cell_height = SCREEN_HEIGHT / self.grid_rows

        self.safe_distance = safe_distance
        self.replan_interval = replan_interval
        self.time_since_replan = 0.0

        self.current_path = []

        # Combat params
        self.shoot_distance = shoot_distance
        self.shoot_angle_thresh = shoot_angle_thresh

    def update(self, dt, player, asteroids,powerups):
        """
        AI update step. 
        1) Possibly replan path with A*
        2) Attempt to shoot if an asteroid is in line of fire
        3) Follow the path from A*
        """
        self.time_since_replan += dt

        # 1) Replan if needed
        if self.time_since_replan > self.replan_interval:
            self.current_path = self.plan_path(player, asteroids,powerups)
            self.time_since_replan = 0.0

        # 2) Combat check: see if we can shoot an asteroid
        self.shoot_if_possible(player, asteroids)

        # 3) Follow the path
        self.follow_path(dt, player)

    def shoot_if_possible(self, player, asteroids):
        """
        Find if there's an asteroid in front of the player's ship
        within some distance and angle. If yes, shoot.
        """
        # If there's no asteroids, do nothing
        if not asteroids:
            return

        # Player forward vector
        forward_vec = pygame.Vector2(0, 1).rotate(player.rotation)

        # We'll track the best candidate to shoot (closest in front)
        best_asteroid = None
        best_dist = float("inf")

        for asteroid in asteroids:
            ax, ay = asteroid.position.x, asteroid.position.y
            dx = ax - player.position.x
            dy = ay - player.position.y

            dist = math.hypot(dx, dy)
            if dist > self.shoot_distance:
                continue  # too far to shoot

            # angle between player's forward direction and vector to the asteroid
            angle_diff = self.angle_between(forward_vec, (dx, dy))

            # If it's roughly in front
            if abs(angle_diff) < self.shoot_angle_thresh:
                # Check if it's the closest in front
                if dist < best_dist:
                    best_dist = dist
                    best_asteroid = asteroid

        # If we found an asteroid in front, try to shoot
        if best_asteroid is not None:
            # The player class might have a cooldown or timer.
            # Typically in your code, you do something like:
            if player.timer <= 0:
                player.shoot()
                player.timer = player.player_shoot_cooldown

    def angle_between(self, vecA, vecB):
        """
        Returns the angle in degrees between two vectors (vecA, vecB).
        If it's 0, they point the same way; 180 means opposite directions, etc.
        """
        vA = pygame.Vector2(vecA)
        vB = pygame.Vector2(vecB)
        angle = math.degrees(math.atan2(vB.y, vB.x) - math.atan2(vA.y, vA.x))
        # normalize angle to (-180, 180)
        if angle > 180:
            angle -= 360
        elif angle < -180:
            angle += 360
        return angle

    def plan_path(self, player, asteroids,powerups):
        """
        1. Build a grid representation (passable / blocked).
        2. Decide on a goal cell (e.g., the cell that is farthest from all asteroids).
        3. Run A* from player's cell to the goal cell.
        4. Return the path (list of grid cells).
        """
        grid = self.build_grid(asteroids)

        # Convert player's position to a grid cell
        start_cell = self.world_to_grid(player.position.x, player.position.y)

        goal_cell = self.find_best_powerup(grid,powerups,player)
        #goal_cell = self.find_safest_cell(grid, asteroids)

        if goal_cell is None:
            # Choose a goal cell. For example, pick the cell that
            # is farthest from all asteroids (in grid space). if unable to find powerups
            goal_cell = self.find_safest_cell(grid, asteroids)
        if goal_cell is None:
            # No safe place found. Return empty path, might just drift or try to shoot.
            return []

        path = self.a_star_search(grid, start_cell, goal_cell)
        return path

    def build_grid(self, asteroids):
        """
        Return a 2D list (rows x cols) marking True if passable, False if blocked.
        We block cells that are within `safe_distance` of any asteroid.
        """
        # Initialize all cells as passable
        grid = [[True for _ in range(self.grid_cols)]
                for _ in range(self.grid_rows)]

        for asteroid in asteroids:
            ax, ay = asteroid.position.x, asteroid.position.y
            ar = asteroid.radius

            # For each asteroid, we mark cells as blocked if they're within (ar + safe_distance).
            block_radius = ar + self.safe_distance

            # Determine which cells in a bounding box around the asteroid might be blocked
            min_col = max(0, int((ax - block_radius) // self.cell_width))
            max_col = min(self.grid_cols - 1,
                          int((ax + block_radius) // self.cell_width))
            min_row = max(0, int((ay - block_radius) // self.cell_height))
            max_row = min(self.grid_rows - 1,
                          int((ay + block_radius) // self.cell_height))

            for row in range(min_row, max_row + 1):
                for col in range(min_col, max_col + 1):
                    cell_center_x = (col + 0.5) * self.cell_width
                    cell_center_y = (row + 0.5) * self.cell_height
                    dist = math.hypot(cell_center_x - ax, cell_center_y - ay)
                    if dist < block_radius:
                        grid[row][col] = False

        return grid

    def find_safest_cell(self, grid, asteroids):
        """
        Example approach: find the cell in 'grid' that is passable and
        has the maximum distance to the nearest asteroid center.
        """
        best_cell = None
        best_dist = -1

        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0

        for row in range(rows):
            for col in range(cols):
                if not grid[row][col]:
                    continue
                # Center of cell in world coords
                cx, cy = self.grid_to_world(col, row)
                # measure distance to nearest asteroid
                nearest_asteroid_dist = float("inf")
                for asteroid in asteroids:
                    dist = math.hypot(cx - asteroid.position.x,
                                      cy - asteroid.position.y)
                    if dist < nearest_asteroid_dist:
                        nearest_asteroid_dist = dist

                if nearest_asteroid_dist > best_dist:
                    best_dist = nearest_asteroid_dist
                    best_cell = (col, row)

        return best_cell

    def find_best_powerup(self,grid,powerups,player):
        best_cell = None
        best_dist = float("inf")
        for powerup in powerups:
            powerup_cell = self.world_to_grid(powerup.position.x, powerup.position.y)
            if (
                powerup_cell[1] < 0 or powerup_cell[1] >= len(grid) or
                powerup_cell[0] < 0 or powerup_cell[0] >= len(grid[0])
                ):
                print(f"Skipping out-of-bounds powerup cell: {powerup_cell}")
                continue
            if not grid[powerup_cell[1]][powerup_cell[0]]:
                continue

            player_cell = self.world_to_grid(player.position.x, player.position.y)
            dist = math.hypot(powerup_cell[0] - player_cell[0], powerup_cell[1] - player_cell[1])

            if dist < best_dist:
                best_dist = dist
                best_cell = powerup_cell

        return best_cell

    def a_star_search(self, grid, start_cell, goal_cell):
        """
        A standard A* search on a 2D grid.
        - grid[row][col] indicates passability
        - start_cell and goal_cell are (col, row) in grid coordinates
        - returns a list of (col, row) cells from start to goal (including goal).
        """

        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0

        # If start or goal is blocked, abort
        sr, sc = start_cell[1], start_cell[0]
        gr, gc = goal_cell[1], goal_cell[0]
        if not (0 <= sr < rows and 0 <= sc < cols and 0 <= gr < rows and 0 <= gc < cols):
            return []
        if (not grid[sr][sc]) or (not grid[gr][gc]):
            return []

        # We'll store: (f_cost, g_cost, (col, row), parent)
        open_set = []
        heapq.heappush(open_set, (0, 0, start_cell, None))
        visited = set()
        came_from = dict()

        # G cost dictionary
        g_costs = {start_cell: 0}

        while open_set:
            f, g, current, parent = heapq.heappop(open_set)
            if current in visited:
                continue
            visited.add(current)
            came_from[current] = parent

            if current == goal_cell:
                # found path
                return self._reconstruct_path(came_from, current)

            # Explore neighbors (8 directions or 4 directions).
            neighbors = self.get_neighbors(current, grid)
            for ncol, nrow in neighbors:
                if (ncol, nrow) in visited:
                    continue
                tentative_g = g + self.distance(current, (ncol, nrow))
                old_g = g_costs.get((ncol, nrow), float("inf"))
                if tentative_g < old_g:
                    g_costs[(ncol, nrow)] = tentative_g
                    h = self.distance((ncol, nrow), goal_cell)
                    f_cost = tentative_g + h
                    heapq.heappush(
                        open_set, (f_cost, tentative_g, (ncol, nrow), current))

        # No path found
        return []

    def get_neighbors(self, cell, grid):
        """
        Return the valid passable neighbors for the given cell in 2D grid.
        We can use 8-direction adjacency for a smoother path.
        """
        (col, row) = cell
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0

        neighbors = []
        for dcol in [-1, 0, 1]:
            for drow in [-1, 0, 1]:
                if dcol == 0 and drow == 0:
                    continue
                ncol = col + dcol
                nrow = row + drow
                if 0 <= ncol < cols and 0 <= nrow < rows:
                    if grid[nrow][ncol]:
                        neighbors.append((ncol, nrow))

        return neighbors

    def distance(self, cell_a, cell_b):
        """
        Euclidean distance in grid space (col, row).
        You could also do a simpler Manhattan or diagonal distance for faster pathfinding.
        """
        (colA, rowA) = cell_a
        (colB, rowB) = cell_b
        return math.hypot(colA - colB, rowA - rowB)

    def _reconstruct_path(self, came_from, current):
        """
        Reconstruct the path from A* came_from map.
        """
        path = []
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path

    def world_to_grid(self, wx, wy):
        """
        Convert world (pixel) coordinates to (col, row) in the grid.
        """
        col = int(wx // self.cell_width)
        row = int(wy // self.cell_height)
        return (col, row)

    def grid_to_world(self, col, row):
        """
        Convert a grid cell to the center pixel coords in the world.
        """
        x = (col + 0.5) * self.cell_width
        y = (row + 0.5) * self.cell_height
        return (x, y)

    def follow_path(self, dt, player):
        """
        Move along A* path if we have one. 
        Possibly mix in logic to chase an asteroid if you want to be aggressive.
        """
        if not self.current_path:
            return
        
        next_cell = self.current_path[0]
        next_world_x, next_world_y = self.grid_to_world(*next_cell)

        dist = math.hypot(player.position.x - next_world_x,
                          player.position.y - next_world_y)
        if dist < 10:
            self.current_path.pop(0)
            return

        desired_angle = math.degrees(math.atan2(
            next_world_y - player.position.y,
            next_world_x - player.position.x
        )) - 90
        desired_angle %= 360
        current_angle = player.rotation % 360
        diff = (desired_angle - current_angle) % 360
        
        if diff > 180:
            player.rotation -= player.player_speed * dt
        else:
            player.rotation += player.player_speed * dt

        self.thrust_forward(dt, player)

    def thrust_forward(self, dt, player):
        """
        Just like pressing W or UP, we move the player forward at some speed.
        We'll re-use the same logic as in your Player's .move() if possible.
        """
        forward = pygame.Vector2(0, 1).rotate(player.rotation)
        move_speed = player.player_speed  # e.g. 200
        player.position += forward * move_speed * dt


def main():
    # Initialize game
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    font = pygame.freetype.SysFont(None, 36)

    # Groups
    updatables = pygame.sprite.Group()
    drawables = pygame.sprite.Group()
    asteroids = pygame.sprite.Group()
    shots = pygame.sprite.Group()
    powerups = pygame.sprite.Group()

    # Containers
    Asteroid.containers = (asteroids, updatables, drawables)
    Shot.containers = (shots, updatables, drawables)
    AsteroidField.containers = updatables
    PowerUp.containers = (powerups,updatables,drawables)
    asteroid_field = AsteroidField()

    dt = 0
    game_over = False
    score = 0
    lives = 0
    # Instantiate the player and the AI agent
    Player.containers = (updatables, drawables)
    player = Player(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)
    # Expose player's turning/thrust speeds if needed (or just rely on constants)
    player.player_turn_speed = PLAYER_TURN_SPEED
    player.player_speed = PLAYER_SPEED

    # AStar agent
    agent = AStarAgent()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        screen.fill((0, 0, 0))

        # Update time
        dt = clock.tick(60) / 1000.0

        # A* agent decides how to move the player
        agent.update(dt, player, asteroids,powerups)

        # Update game objects
        for updateable in updatables:
            updateable.update(dt)

        # Collisions
        for asteroid in asteroids:
            if asteroid.collision_check(player):
                player.player_lives -= 1
                player.position.x = SCREEN_WIDTH / 2
                player.position.y = SCREEN_HEIGHT / 2
                lives = player.player_lives
                if lives == 0:
                    game_over = True
            for shot in shots:
                if asteroid.collision_check(shot):
                    asteroid.split()
                    shot.kill()
                    score += 1

        for powerup in powerups:
            if powerup.collision_check(player):
                powerup.apply_effect(player)
                lives = player.player_lives
                powerup.remove()
        # Draw
        for drawable in drawables:
            drawable.draw(screen)

        font.render_to(screen, (10, 10), f"Score: {score}", (255, 255, 255))
        font.render_to(screen, (180,10),f"Lives: {lives}",(255,255,255))
        if game_over:
            font.render_to(screen, (SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2),
                           "GAME OVER", (255, 255, 255))
            # Optionally freeze the agent or reset
        else:
            wrap_sprites(updatables)

        pygame.display.flip()


def wrap_sprites(group):
    for sprite in group:
        if hasattr(sprite, 'position'):
            if sprite.position.x < 0:
                sprite.position.x = SCREEN_WIDTH
            elif sprite.position.x > SCREEN_WIDTH:
                sprite.position.x = 0
            if sprite.position.y < 0:
                sprite.position.y = SCREEN_HEIGHT
            elif sprite.position.y > SCREEN_HEIGHT:
                sprite.position.y = 0


if __name__ == "__main__":
    main()
