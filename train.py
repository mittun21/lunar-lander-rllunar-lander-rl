import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Run 16 copies of the game at once so it learns faster (no window = much quicker)
env = make_vec_env("LunarLander-v3", n_envs=16)

# The AI brain + PPO training method, with settings known to work well for this game
model = PPO("MlpPolicy", env, n_steps=1024, batch_size=64, n_epochs=4,
            gamma=0.999, gae_lambda=0.98, ent_coef=0.01, verbose=1)

model.learn(total_timesteps=1_000_000)  # practice for 1 million frames
model.save("ppo_lander")                 # save the trained brain to a file
print("Training done, saved as ppo_lander.zip")