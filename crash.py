import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")  # create the game, with a window
obs, info = env.reset()                                 # start an attempt

for step in range(1000):
    action = env.action_space.sample()                  # pick a random action (0-3)
    obs, reward, terminated, truncated, info = env.step(action)  # do it, get results
    if step % 20 == 0:
        print([round(float(x), 2) for x in obs], "reward:", round(float(reward), 2))
    if terminated or truncated:                         # crashed, landed, or ran out of time
        obs, info = env.reset()                         # start a new attempt

env.close()