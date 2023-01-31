from gym.envs.registration import register

register(
    id="snake_game/SnakeWorld-v0",
    entry_point="snake_game.envs:SnakeWorldEnv",
)