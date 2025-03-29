from environment import AsteroidsEnv
from human import HumanPlayerAgent
from random_asteroids import RandomAsteroidAgent


if __name__ == "__main__":

    # by default use human agent as the player and a random asteroid spawner
    human_agent = HumanPlayerAgent()
    random_asteroid_agent = RandomAsteroidAgent()
    env = AsteroidsEnv(human_agent, random_asteroid_agent)

    done = False
    obs = env.reset()

    while not done:
        obs, reward, done, info = env.step()
        env.render()

    env.close()
