# Procedurally Generated Content Using Asteroids

This project is a research playground that fuses procedural content generation (PCG) with reinforcement learning (RL) to create a version of the classic Asteroids that never stops learning. Instead of pre-scripted levels, an RL “level designer” studies how the player (human, heuristic, or RL) flies and dynamically spawns new asteroid waves to keep the challenge fresh, fair, and fun.

- Why this matters: Most games plateau once you master them. By coupling PCG with RL we aim for self-balancing gameplay that scales in difficulty and variety to match any skill level.

- What’s inside: Three escalating approaches—from a heuristic-versus-RL baseline to a full adversarial PPO-vs-PPO duel—let you explore the trade-offs between simplicity, player experience, and computational heft.

Whether you’re into game AI research, RL experimentation, or just want a smarter way to blow up space rocks, this project gives you the code and scaffolding to dive in quickly and iterate fast.

## Original Game

- This is a Python-based recreation of the classic Asteroids game built using the Pygame library. Control a spaceship, navigate through space, and shoot asteroids to survive as long as possible.


- Gameplay Controls:
    - Use W, A, S, D or arrow keys to maneuver your spaceship.
    - Press Spacebar to fire projectiles and destroy incoming asteroids.

- Running the Game:

```bash
pip install -r game/requirements.txt
python game/original/main.py
```

## Modified Game

- A juiced-up Asteroids clone that adds polygonal asteroids, collectible power-ups, player lives, and automatic gameplay logging for future machine-learning experiments.

What’s New vs. the Original:

- Irregular Asteroids: Each rock is a 5- to 10-sided polygon generated on the fly, making trajectories less predictable.

- Power-Ups:
- 🚀 Speed Boost – faster thrust & turning.
- 💥 Rapid Fire – higher projectile speed & shorter cooldown.
- ❤️ Extra Life – grants an additional ship.

- Data Collection: Every second the game records player/asteroid stats and writes them to training_data.json for RL or analytics.
- Lives System: You now start with one life (expandable via power-ups) instead of instant game-over on first hit.

```bash
pip install -r game/requirements.txt
python game/modified/main.py
```
## Approaches in a Nutshell

- ### Koster-Inspired PPO Agents (Fun-Tuned RL)
The generator’s action space expands (radius, speed, angle, count), and a reward scheme rooted in mastery, challenge, diversity, and fairness balances excitement: near-miss bonuses, variety incentives, and penalties for boring or unfair waves.

- ### Adversarial Dual-Agent RL (PPO vs PPO)
Both the "player" and the "level designer" are PPO agents learning together. The pilot optimizes survival and accuracy; the generator adapts spawns in real time. This co-evolution yields highly dynamic, human-relevant difficulty—at the cost of heavier training and occasional instability.

