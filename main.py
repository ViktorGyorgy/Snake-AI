import gym
env = gym.make('snake_game:snake_game/SnakeWorld-v0', render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step((1, 3))

    if terminated or truncated:
        observation, info = env.reset()

env.close()